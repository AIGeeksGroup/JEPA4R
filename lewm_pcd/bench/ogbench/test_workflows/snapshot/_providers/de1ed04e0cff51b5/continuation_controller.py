"""Persistent, single-owner G05-to-G06 Native SR handoff controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def proc_identity(pid: int):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
        return {"pid": pid, "start_ticks": fields[21], "cmdline": cmdline,
                "cmdline_sha256": hashlib.sha256(cmdline.encode()).hexdigest()}
    except FileNotFoundError:
        return None


def run(cmd, *, cwd: Path, env=None, log: Path | None = None) -> int:
    if log is None:
        return subprocess.run(cmd, cwd=cwd, env=env, check=False).returncode
    with log.open("wb") as stream:
        return subprocess.run(cmd, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                              check=False).returncode


def verify_freeze(base: Path, receipt: Path) -> None:
    check = subprocess.run(["sha256sum", "-c", str(receipt / "source_freeze.sha256")],
                           cwd=base, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (receipt / "controller_freeze_check.stdout").write_text(check.stdout)
    if check.returncode:
        raise RuntimeError("frozen source/config/input hash verification failed")


def verify_report(base: Path, group: str, mode: str, output: Path, expected: int) -> dict:
    report = json.loads((output / "report.json").read_text())
    if (int((output / "actual_exit_code").read_text()) != 0 or
            report.get("actual_exit_code") != 0 or report.get("stage") not in (None, "complete") or
            report.get("group_id") != group or report.get("mode") != mode or
            report.get("episodes") != expected or report.get("unique_seeds") != expected or
            report.get("current_dim") != 211 or report.get("goal_dim") != 198 or
            report.get("online_encoder") is not True or
            report.get("offline_embedding_cache_used_for_policy") is not False or
            report.get("synthetic_zero_padding") is not False):
        raise RuntimeError(f"{group} {mode} aggregate report failed contract checks")
    successes = 0
    successes_by_task = {}
    seeds = set()
    for task in range(1, 6):
        count = min(50, expected - 50 * (task - 1))
        if count <= 0:
            break
        task_successes = 0
        for index in range(count):
            episode = output / "episodes" / f"task{task}" / f"episode_{index:03d}"
            item = json.loads((episode / "result.json").read_text())
            seed = 970000000 + task * 100000 + index
            if (int((episode / "attempt_exit_code").read_text()) != 0 or
                    item.get("actual_exit_code") != 0 or item.get("group_id") != group or
                    item.get("task_id") != task or item.get("episode_index") != index or
                    item.get("seed") != seed or item.get("steps", 0) < 1 or
                    item.get("current_goal_contract") != {"current_dim": 211, "goal_dim": 198,
                                                           "synthetic_zero_padding": False} or
                    item.get("temperature") != 0.0 or item.get("gaussian_noise") is not None or
                    item.get("goal_online_encoder_calls") != 1 or
                    item.get("original_evaluate_reused") is not True):
                raise RuntimeError(f"{group} episode contract failed: task={task} index={index}")
            if len(item.get("policy_input_contracts", [])) != item["steps"]:
                raise RuntimeError(f"{group} input-contract count mismatch: task={task} index={index}")
            for row in item["policy_input_contracts"]:
                if (row.get("current_dim") != 211 or row.get("goal_dim") != 198 or
                        row.get("current_finite") is not True or row.get("goal_finite") is not True):
                    raise RuntimeError(f"{group} non-finite/wrong-width policy input")
            if group == "G06":
                proof = item.get("goal_capture_proof", {})
                if (proof.get("physics_unchanged") is not True or
                        proof.get("env_rng_unchanged") is not True or
                        proof.get("geometry_offsamples") != 0 or
                        item.get("camera", {}).get("geometry_offsamples") != 0):
                    raise RuntimeError("G06 PCD goal-capture invariants failed")
            seeds.add(seed)
            task_successes += int(item["success"])
            successes += int(item["success"])
        successes_by_task[str(task)] = task_successes
    if len(seeds) != expected or successes != report.get("successes"):
        raise RuntimeError(f"{group} per-episode count/seed/success summary mismatch")
    if abs(successes / expected - float(report.get("success_rate", -1))) > 1e-12:
        raise RuntimeError(f"{group} success-rate summary mismatch")
    return {"episodes": expected, "successes": successes,
            "success_rate": successes / expected, "unique_seeds": len(seeds),
            "successes_by_task": successes_by_task}


def gpu_drained() -> tuple[bool, str]:
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                                 "--format=csv,noheader,nounits"],
                                check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=15)
        pids = [x.strip() for x in result.stdout.splitlines() if x.strip()]
        mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, timeout=15)
        used = [int(x.strip().split()[0]) for x in mem.stdout.splitlines() if x.strip()]
        return not pids and bool(used) and max(used) <= 512, f"pids={pids}, memory_mib={used}"
    except Exception as exc:
        return False, repr(exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--g05-pid", type=int, required=True)
    parser.add_argument("--stale-seconds", type=int, default=900)
    args = parser.parse_args()
    base = args.base.resolve()
    receipt = base / "runtime/authorized_20261020_v1"
    controller = receipt / "controller"
    controller.mkdir(parents=True, exist_ok=True)
    lock = controller / "lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(fd, f"pid={os.getpid()}\nstarted={time.time()}\n".encode())
        os.close(fd)
    except FileExistsError:
        raise RuntimeError("controller lock already exists; refusing duplicate handoff")

    status_path = controller / "status.json"
    status = {"stage": "starting", "g05_pid": args.g05_pid,
              "started": time.time(), "actual_exit_code": None}

    def update(**kwargs):
        status.update(kwargs, updated=time.time())
        atomic_json(status_path, status)

    try:
        verify_freeze(base, receipt)
        ident = proc_identity(args.g05_pid)
        expected_tokens = ["visual_sr.py", "--group G05", "--mode full",
                           "--output full/G05_current_20261020_v1"]
        if ident is None or any(token not in ident["cmdline"] for token in expected_tokens):
            raise RuntimeError("specified G05 PID is absent or argv does not match this frozen run")
        atomic_json(controller / "g05_process_identity.json", ident)
        update(stage="waiting_for_G05", process=ident)

        last_completed = 0
        while True:
            current = proc_identity(args.g05_pid)
            if current is None:
                break
            if current["start_ticks"] != ident["start_ticks"] or current["cmdline_sha256"] != ident["cmdline_sha256"]:
                raise RuntimeError("G05 PID identity changed while waiting")
            run_status_path = base / "full/G05_current_20261020_v1/status.json"
            if run_status_path.exists():
                run_status = json.loads(run_status_path.read_text())
                completed = int(run_status.get("completed_episodes", 0))
                last_completed = max(last_completed, completed)
                age = time.time() - float(run_status.get("updated", 0))
                update(stage="waiting_for_G05", completed_episodes=completed,
                       successes=run_status.get("successes"), last_updated=run_status.get("updated"))
                if age > args.stale_seconds:
                    raise RuntimeError(f"G05 status stale for {age:.0f}s")
            time.sleep(30)

        g05_exit = base / "full/G05_current_20261020_v1/actual_exit_code"
        if not g05_exit.is_file() or int(g05_exit.read_text()) != 0:
            raise RuntimeError("G05 did not finish with actual_exit_code=0; G06 is blocked")
        g05_summary = verify_report(base, "G05", "full", base / "full/G05_current_20261020_v1", 250)
        verify_freeze(base, receipt)
        update(stage="waiting_for_GPU_drain", g05_summary=g05_summary)

        deadline = time.time() + 180
        while time.time() < deadline:
            drained, evidence = gpu_drained()
            update(stage="waiting_for_GPU_drain", gpu=evidence)
            if drained:
                break
            time.sleep(10)
        else:
            raise RuntimeError("GPU did not drain within 180s; G06 is blocked")

        env = os.environ.copy()
        env.update({"LD_LIBRARY_PATH": "/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/runtime/lib:/usr/lib/x86_64-linux-gnu",
                    "MUJOCO_GL": "egl", "JAX_PLATFORMS": "cpu"})
        diag = base / "diagnostic/G06_current_20261020_v1"
        if diag.exists():
            raise RuntimeError("G06 diagnostic output exists; refusing overwrite")
        update(stage="G06_diagnostic_running", gpu_drain=evidence)
        diag_rc = run([sys.executable, "-u", "visual_sr.py", "--group", "G06",
                       "--mode", "diagnostic", "--output", str(diag)], cwd=base,
                      env=env, log=controller / "G06_diagnostic.stdout")
        (controller / "G06_diagnostic.exit_code").write_text(f"{diag_rc}\n")
        if diag_rc != 0:
            raise RuntimeError(f"G06 diagnostic failed with exit code {diag_rc}; full run blocked")
        diag_summary = verify_report(base, "G06", "diagnostic", diag, 1)
        diag_ep = json.loads((diag / "episodes/task1/episode_000/result.json").read_text())
        if diag_ep.get("seed") != 970100000:
            raise RuntimeError("G06 diagnostic seed mismatch")
        verify_freeze(base, receipt)

        full = base / "full/G06_current_20261020_v1"
        if full.exists():
            raise RuntimeError("G06 full output exists; refusing overwrite")
        update(stage="G06_full_running", g05_summary=g05_summary,
               g06_diagnostic=diag_summary)
        log = (controller / "G06_full.stdout").open("wb")
        proc = subprocess.Popen([sys.executable, "-u", "visual_sr.py", "--group", "G06",
                                 "--mode", "full", "--output", str(full)],
                                cwd=base, env=env, stdout=log, stderr=subprocess.STDOUT)
        atomic_json(controller / "G06_full_process.json", {"pid": proc.pid,
                    "argv": proc.args, "started": time.time()})
        while proc.poll() is None:
            s = {}
            if (full / "status.json").exists():
                s = json.loads((full / "status.json").read_text())
            update(stage="G06_full_running", pid=proc.pid,
                   completed_episodes=s.get("completed_episodes", 0),
                   successes=s.get("successes"), last_updated=s.get("updated"))
            time.sleep(30)
        log.close()
        (controller / "G06_full.exit_code").write_text(f"{proc.returncode}\n")
        if proc.returncode != 0:
            raise RuntimeError(f"G06 full failed with exit code {proc.returncode}")
        g06_summary = verify_report(base, "G06", "full", full, 250)
        update(stage="complete", g05_summary=g05_summary,
               g06_diagnostic=diag_summary, g06_summary=g06_summary,
               actual_exit_code=0, completed=time.time())
        (controller / "controller.exit_code").write_text("0\n")
        return 0
    except BaseException as exc:
        update(stage="failed", error=repr(exc), actual_exit_code=1)
        (controller / "controller.exit_code").write_text("1\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
