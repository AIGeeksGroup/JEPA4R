"""Unarmed serial controller for the Cube3 G08 -> G09 Native SR successor.

The controller deliberately does not connect to a remote host.  When copied to
the evaluation checkout, it waits on the existing G05 -> G06 controller,
checks the upstream receipts, and invokes the shared ``visual_sr.py`` through a
thin Cube3 registration shim.  Every stage has its own output, log, exit-code,
and status records; an existing path is never reused.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CORE = HERE.parent
GROUPS = ("G08", "G09")
MODES = ("diagnostic", "full")
EXPECTED_EPISODES = {"diagnostic": 1, "full": 250}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def require_upstream_complete(runtime: Path) -> dict:
    """Require the durable G05 -> G06 controller receipt and exit code."""
    runtime = Path(runtime)
    status_path = runtime / "status.json"
    exit_path = runtime / "controller.exit_code"
    if not status_path.is_file() or not exit_path.is_file():
        raise RuntimeError("G05->G06 controller receipt is incomplete")
    status = json.loads(status_path.read_text())
    if status.get("stage") != "complete" or status.get("actual_exit_code") != 0:
        raise RuntimeError("G05->G06 controller is not complete with exit 0")
    if exit_path.read_text().strip() != "0":
        raise RuntimeError("G05->G06 controller.exit_code is not zero")
    return status


def _continuation_module():
    try:
        import continuation_controller
        return continuation_controller
    except ModuleNotFoundError:
        import importlib.util
        candidates = (
            CORE / "continuation_controller.py",
            CORE / "runtime/authorized_20261020_v1/controller/continuation_controller.py",
        )
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise RuntimeError("shared continuation_controller.py is unavailable")
        spec = importlib.util.spec_from_file_location("_continuation_controller", source)
        if spec is None or spec.loader is None:
            raise RuntimeError("shared continuation_controller.py is unavailable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def verify_upstream_reports(status: dict, base: Path | None = None) -> None:
    """Validate the summaries emitted by the existing G05 -> G06 controller."""
    for key in ("g05_summary", "g06_summary"):
        summary = status.get(key)
        if not isinstance(summary, dict) or summary.get("episodes") != 250 or summary.get("unique_seeds") != 250:
            raise RuntimeError(f"missing complete upstream {key} report")
    if base is not None:
        verifier = _continuation_module().verify_report
        for group in ("G05", "G06"):
            output = Path(base) / "full" / f"{group}_current_20261020_v1"
            actual = verifier(Path(base), group, "full", output, 250)
            if actual.get("episodes") != 250 or actual.get("unique_seeds") != 250:
                raise RuntimeError(f"{group} report audit did not close")


def _read_report(output: Path, group: str, mode: str) -> dict:
    expected = EXPECTED_EPISODES[mode]
    report_path = output / "report.json"
    exit_path = output / "actual_exit_code"
    if not report_path.is_file() or not exit_path.is_file():
        raise RuntimeError(f"{group} {mode} report/exit receipt is missing")
    report = json.loads(report_path.read_text())
    if (exit_path.read_text().strip() != "0" or report.get("actual_exit_code") != 0 or
            report.get("stage") not in (None, "complete") or report.get("group_id") != group or
            report.get("mode") != mode or report.get("episodes") != expected or
            report.get("unique_seeds") != expected or report.get("current_dim") != 211 or
            report.get("goal_dim") != 201 or report.get("max_episode_steps") != 1000 or
            report.get("online_encoder") is not True or
            report.get("offline_embedding_cache_used_for_policy") is not False or
            report.get("synthetic_zero_padding") is not False):
        raise RuntimeError(f"{group} {mode} aggregate report failed Cube3 contract")
    return report


def audit_output(output: Path, group: str, mode: str, expected: int | None = None) -> dict:
    """Audit report and every episode produced by the shared runner."""
    output = Path(output)
    expected = EXPECTED_EPISODES[mode] if expected is None else expected
    if expected != EXPECTED_EPISODES[mode]:
        raise ValueError("Cube3 audit expected count must match diagnostic/full mode")
    report = _read_report(output, group, mode)
    successes = 0
    successes_by_task: dict[str, int] = {}
    seeds: set[int] = set()
    for task, index in ((1, 0),) if mode == "diagnostic" else (
            (task, index) for task in range(1, 6) for index in range(50)):
        episode = output / "episodes" / f"task{task}" / f"episode_{index:03d}"
        result_path = episode / "result.json"
        exit_path = episode / "attempt_exit_code"
        if not result_path.is_file() or not exit_path.is_file():
            raise RuntimeError(f"{group} {mode} episode receipt is missing: task={task} index={index}")
        item = json.loads(result_path.read_text())
        seed = 970000000 + task * 100000 + index
        contract = {"current_dim": 211, "goal_dim": 201, "synthetic_zero_padding": False}
        if (exit_path.read_text().strip() != "0" or item.get("actual_exit_code") != 0 or
                item.get("group_id") != group or item.get("task_id") != task or
                item.get("episode_index") != index or item.get("seed") != seed or
                not 1 <= int(item.get("steps", 0)) <= 1000 or
                item.get("current_goal_contract") != contract or
                item.get("temperature") != 0.0 or item.get("gaussian_noise") is not None or
                item.get("goal_online_encoder_calls") != 1 or
                item.get("original_evaluate_reused") is not True):
            raise RuntimeError(f"{group} {mode} episode contract failed: task={task} index={index}")
        rows = item.get("policy_input_contracts", [])
        if len(rows) != item["steps"]:
            raise RuntimeError(f"{group} {mode} input-contract count mismatch")
        if any(row.get("current_dim") != 211 or row.get("goal_dim") != 201 or
               row.get("current_finite") is not True or row.get("goal_finite") is not True
               for row in rows):
            raise RuntimeError(f"{group} {mode} non-finite/wrong-width policy input")
        if group == "G09":
            proof = item.get("goal_capture_proof", {})
            camera = item.get("camera", {})
            if (proof.get("physics_unchanged") is not True or
                    proof.get("env_rng_unchanged") is not True or
                    proof.get("global_rng_unchanged") is not True or
                    proof.get("geometry_offsamples") != 0 or
                    camera.get("geometry_offsamples", 0) != 0 or
                    not proof.get("captures")):
                raise RuntimeError("G09 PCD goal-capture invariants failed")
        seed_value = int(item["seed"])
        if seed_value in seeds:
            raise RuntimeError(f"{group} {mode} repeated seed {seed_value}")
        seeds.add(seed_value)
        successes += int(item["success"])
        successes_by_task[str(task)] = successes_by_task.get(str(task), 0) + int(item["success"])
    if len(seeds) != expected or successes != report.get("successes"):
        raise RuntimeError(f"{group} {mode} per-episode count/seed/success mismatch")
    success_rate = successes / expected
    if abs(success_rate - float(report.get("success_rate", -1))) > 1e-12:
        raise RuntimeError(f"{group} {mode} success-rate summary mismatch")
    return {"episodes": expected, "successes": successes,
            "success_rate": success_rate, "unique_seeds": len(seeds),
            "successes_by_task": successes_by_task}


def cube3_contract(base: Path) -> dict:
    """Load the thin adapter and assert the explicit Cube3 contract."""
    sys.path.insert(0, str(base))
    sys.path.insert(0, str(base / "cube3_successor_v1"))
    import visual_sr
    import cube3_successor_v1.visual_sr_cube3 as adapter

    adapter.install(visual_sr)
    expected = {
        "G08": {"cube_count": 3, "state_dim": 46, "current_dim": 211,
                 "goal_dim": 201, "max_episode_steps": 1000, "mode": "rgb"},
        "G09": {"cube_count": 3, "state_dim": 46, "current_dim": 211,
                 "goal_dim": 201, "max_episode_steps": 1000, "mode": "pcd"},
    }
    for group, contract in expected.items():
        actual = visual_sr.group_contract(group)
        if any(actual.get(key) != value for key, value in contract.items()):
            raise RuntimeError(f"{group} Cube3 contract mismatch")
    return expected


def _shared_runner_script(base: Path, group: str, mode: str, output: Path) -> str:
    # This is the only adaptation: register Cube3 groups, then call the shared
    # core main() so the evaluator/online encoder path remains unchanged.
    return (
        "import sys; "
        f"sys.path.insert(0, {str(base)!r}); "
        f"sys.path.insert(0, {str(base / 'cube3_successor_v1')!r}); "
        "import visual_sr; import cube3_successor_v1.visual_sr_cube3 as adapter; "
        "adapter.install(visual_sr); "
        f"visual_sr.main({group!r}, {mode!r}, __import__('pathlib').Path({str(output)!r}))"
    )


def gpu_drained() -> tuple[bool, str]:
    """Reuse the proven G05/G06 drain helper without importing its CLI."""
    return _continuation_module().gpu_drained()


def wait_for_upstream(runtime: Path, *, poll_seconds: int = 30, timeout_seconds: int = 0) -> dict:
    started = time.time()
    while True:
        try:
            return require_upstream_complete(runtime)
        except RuntimeError:
            if timeout_seconds and time.time() - started >= timeout_seconds:
                raise
            time.sleep(poll_seconds)


def run_serial(base: Path, runtime: Path, python: str, *, poll_seconds: int = 30,
               upstream_runtime: Path | None = None,
               output_tag: str = "20261020_v1") -> int:
    """Run the unarmed successor when explicitly invoked on a deployment host."""
    base = Path(base).resolve()
    runtime = Path(runtime).resolve()
    if upstream_runtime is None:
        upstream_runtime = base / "runtime/authorized_20261020_v1/controller"
    upstream_runtime = Path(upstream_runtime).resolve()
    if runtime.exists():
        raise RuntimeError(f"refusing existing successor runtime: {runtime}")
    runtime.mkdir(parents=True)
    lock = runtime / "lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, f"pid={os.getpid()}\nstarted={time.time()}\n".encode())
    os.close(fd)
    status = {"stage": "waiting_for_G05_G06", "started": time.time(),
              "groups": [], "actual_exit_code": None}
    atomic_json(runtime / "status.json", status)

    def update(**values):
        status.update(values, updated=time.time())
        atomic_json(runtime / "status.json", status)

    try:
        contract = cube3_contract(base)
        atomic_json(runtime / "cube3_contract.json", contract)
        update(stage="waiting_for_G05_G06", cube3_contract=contract)
        upstream = wait_for_upstream(upstream_runtime, poll_seconds=poll_seconds)
        verify_upstream_reports(upstream, base)
        update(stage="waiting_for_GPU_drain", upstream=upstream)
        while True:
            drained, evidence = gpu_drained()
            update(stage="waiting_for_GPU_drain", gpu=evidence)
            if drained:
                break
            time.sleep(poll_seconds)
        env = os.environ.copy()
        env.update({"MUJOCO_GL": "egl", "JAX_PLATFORMS": "cpu"})
        for group in GROUPS:
            for mode in MODES:
                name = f"{group}_{mode}"
                output = base / ("diagnostic" if mode == "diagnostic" else "full") / f"{group}_current_{output_tag}"
                log = runtime / f"{name}.stdout"
                exit_path = runtime / f"{name}.exit_code"
                if output.exists() or log.exists() or exit_path.exists():
                    raise RuntimeError(f"refusing existing {name} output/log/exit path")
                output.parent.mkdir(parents=True, exist_ok=True)
                update(stage=f"{name}_running", group=group, mode=mode, output=str(output))
                stream = log.open("wb")
                try:
                    proc = subprocess.Popen([python, "-u", "-c", _shared_runner_script(base, group, mode, output)],
                                            cwd=base, env=env, stdout=stream, stderr=subprocess.STDOUT)
                except BaseException:
                    stream.close()
                    raise
                atomic_json(runtime / f"{name}.process.json",
                            {"pid": proc.pid, "argv": proc.args, "started": time.time()})
                while proc.poll() is None:
                    progress = {}
                    if (output / "status.json").exists():
                        progress = json.loads((output / "status.json").read_text())
                    update(stage=f"{name}_running", group=group, mode=mode,
                           completed_episodes=progress.get("completed_episodes", 0),
                           successes=progress.get("successes"), last_updated=progress.get("updated"))
                    time.sleep(poll_seconds)
                stream.close()
                (runtime / f"{name}.exit_code").write_text(f"{proc.returncode}\n")
                if proc.returncode != 0:
                    raise RuntimeError(f"{name} failed with exit code {proc.returncode}")
                summary = audit_output(output, group, mode)
                status.setdefault("groups", []).append({"group": group, "mode": mode,
                                                         "output": str(output), "audit": summary})
                update(stage=f"{name}_complete")
                if mode == "full" and not (group == GROUPS[-1] and mode == MODES[-1]):
                    while True:
                        drained, evidence = gpu_drained()
                        update(stage=f"{name}_waiting_for_GPU_drain", gpu=evidence)
                        if drained:
                            break
                        time.sleep(poll_seconds)
        update(stage="complete", actual_exit_code=0, completed=time.time())
        (runtime / "controller.exit_code").write_text("0\n")
        return 0
    except BaseException as exc:
        update(stage="failed", error=repr(exc), actual_exit_code=1)
        (runtime / "controller.exit_code").write_text("1\n")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--upstream-runtime", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--output-tag", default="20261020_v1")
    args = parser.parse_args()
    return run_serial(args.base, args.runtime, args.python,
                      poll_seconds=args.poll_seconds, upstream_runtime=args.upstream_runtime,
                      output_tag=args.output_tag)


if __name__ == "__main__":
    raise SystemExit(main())
