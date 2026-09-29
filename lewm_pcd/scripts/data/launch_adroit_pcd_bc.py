"""Persistent, serial, auditable launcher for the Adroit PCD BC extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def ps_resource(pid: int) -> dict:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pcpu=,rss=,etime=,stat="],
            capture_output=True,
            text=True,
            check=False,
        )
        fields = result.stdout.strip().split(None, 3)
        if len(fields) == 4:
            return {"pcpu": float(fields[0]), "rss_kib": int(fields[1]), "etime": fields[2], "stat": fields[3]}
    except (OSError, ValueError):
        pass
    return {"pcpu": None, "rss_kib": None, "etime": None, "stat": None}


def gpu_resource() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        rows = []
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 3:
                rows.append({"utilization_gpu": float(fields[0]), "memory_used_mib": float(fields[1]), "memory_total_mib": float(fields[2])})
        return {"gpus": rows}
    except (OSError, ValueError):
        return {"gpus": []}


def append_resource(path: Path, pid: int) -> None:
    record = {"timestamp": now(), "pid": pid, **ps_resource(pid), **gpu_resource()}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def validate_seed(seed_dir: Path, config: dict) -> dict:
    metrics_path = seed_dir / "metrics.jsonl"
    job_config_path = seed_dir / "job_config.json"
    policy_path = seed_dir / "adroit_cortex_pcd" / "iterations" / "best_policy.pickle"
    if not metrics_path.exists() or not job_config_path.exists() or not policy_path.exists():
        raise RuntimeError("missing metrics.jsonl, job_config.json, or best_policy.pickle")

    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    losses = [record for record in records if "epoch_loss" in record]
    evaluations = [record for record in records if "eval/epoch" in record]
    if len(losses) != config["epochs"]:
        raise RuntimeError(f"expected {config['epochs']} epoch losses, found {len(losses)}")
    expected_evals = config["epochs"] // config["eval_frequency"]
    if len(evaluations) != expected_evals:
        raise RuntimeError(f"expected {expected_evals} evaluator records, found {len(evaluations)}")
    if evaluations[-1]["eval/epoch"] != float(config["epochs"] - 1):
        raise RuntimeError(f"last evaluator epoch is {evaluations[-1]['eval/epoch']}, expected {config['epochs'] - 1}")
    if not all(math.isfinite(float(record["epoch_loss"])) for record in losses):
        raise RuntimeError("epoch loss contains NaN or Inf")
    for record in evaluations:
        for key in ("eval/score_mean", "eval/success", "eval/highest_success", "eval/highest_score"):
            if key not in record or not math.isfinite(float(record[key])):
                raise RuntimeError(f"evaluation record has missing/non-finite {key}")

    try:
        import yaml

        job = yaml.safe_load(job_config_path.read_text(encoding="utf-8"))
        checks = {
            "env": "pen-v0",
            "embedding": "lewm_adroit_pen_pcd",
            "epochs": config["epochs"],
            "eval_frequency": config["eval_frequency"],
            "eval_num_traj": config["eval_num_traj"],
            "num_demos": config["num_demos"],
            "device": "cpu",
        }
        for key, expected in checks.items():
            if job.get(key) != expected:
                raise RuntimeError(f"job_config mismatch for {key}: {job.get(key)!r} != {expected!r}")
        if job.get("env_kwargs", {}).get("add_proprio") is not True or job.get("env_kwargs", {}).get("proprio_key") != "proprio":
            raise RuntimeError("job_config proprio contract mismatch")
    except ImportError:
        job = None

    with policy_path.open("rb") as handle:
        policy = pickle.load(handle)
    resource_path = seed_dir / "resource.jsonl"
    resources = [json.loads(line) for line in resource_path.read_text(encoding="utf-8").splitlines() if line.strip()] if resource_path.exists() else []
    cpu_values = [row["pcpu"] for row in resources if row.get("pcpu") is not None]
    rss_values = [row["rss_kib"] for row in resources if row.get("rss_kib") is not None]
    gpu_values = [gpu["utilization_gpu"] for row in resources for gpu in row.get("gpus", [])]
    validation = {
        "status": "PASS",
        "track": "3D PCD extension; not official RGB CortexBench score",
        "seed": config["seed"],
        "loss_records": len(losses),
        "eval_records": len(evaluations),
        "final_eval_success_percent": evaluations[-1].get("eval/success"),
        "highest_eval_success_percent": evaluations[-1].get("eval/highest_success"),
        "final_eval_score": evaluations[-1].get("eval/score_mean"),
        "policy_type": type(policy).__name__,
        "policy_sha256": sha256(policy_path),
        "policy_bytes": policy_path.stat().st_size,
        "demo_pickle_sha256": config["demo_sha256"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "resource_samples": len(resources),
        "cpu_percent_max": max(cpu_values) if cpu_values else None,
        "rss_mib_max": max(rss_values) / 1024.0 if rss_values else None,
        "gpu_utilization_percent_max": max(gpu_values) if gpu_values else None,
        "job_config_checked": job is not None,
    }
    atomic_json(seed_dir / "validation.json", validation)
    return validation


def run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    data_dir = args.data_dir.resolve()
    demo_path = data_dir / "pen-v0.pickle"
    checkpoint_path = Path(args.checkpoint).resolve()
    if not demo_path.exists():
        raise FileNotFoundError(demo_path)
    if not (checkpoint_path / "weights_epoch_10.pt").exists():
        raise FileNotFoundError(checkpoint_path / "weights_epoch_10.pt")

    config = {
        "track": "3D PCD extension; not official RGB CortexBench score",
        "env": "pen-v0",
        "embedding": "lewm_adroit_pen_pcd",
        "data_dir": str(data_dir),
        "demo_sha256": sha256(demo_path),
        "checkpoint_dir": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path / "weights_epoch_10.pt"),
        "num_demos": args.num_demos,
        "epochs": args.epochs,
        "eval_frequency": args.eval_frequency,
        "eval_num_traj": args.eval_num_traj,
        "num_cpu": args.num_cpu,
        "device": "cpu",
        "seeds": args.seeds,
    }
    state_path = root / "launcher.state.json"
    exit_path = root / "exit_code"
    if root.exists() and state_path.exists():
        state = read_json(state_path)
        if state.get("config") != config:
            raise RuntimeError("existing launcher config does not match requested run")
        if state.get("stage") == "finished":
            return int(state.get("exit_code", 0))
    elif root.exists() and any(path.name not in {"launcher.log", "launcher.pid"} for path in root.iterdir()):
        raise FileExistsError(f"nonempty output root without matching launcher state: {root}")
    else:
        root.mkdir(parents=True, exist_ok=True)
        state = {
            "stage": "initialized",
            "started_at": now(),
            "updated_at": now(),
            "current_seed": None,
            "completed_seeds": [],
            "failed_seeds": [],
            "exit_code": None,
            "config": config,
        }
        atomic_json(state_path, state)

    env = dict(os.environ)
    env.update({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    repo = Path(__file__).resolve().parents[2]
    runner = repo / "bench" / "cortexbench" / "run_bc3d_adroit.py"
    python = sys.executable

    for seed in args.seeds:
        seed_dir = root / f"seed_{seed}"
        seed_state_path = seed_dir / "state.json"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_config = {**config, "seed": seed}
        previous = read_json(seed_state_path)
        if previous and previous.get("stage") == "finished" and previous.get("exit_code") == 0:
            validation = validate_seed(seed_dir, seed_config)
            state["completed_seeds"] = sorted(set(state.get("completed_seeds", [])) | {seed})
            state["last_validation"] = validation
            state["updated_at"] = now()
            atomic_json(state_path, state)
            continue
        if previous and previous.get("stage") == "failed":
            raise RuntimeError(f"seed {seed} already failed; refusing silent rerun")

        command = [
            python, str(runner), "--config-name", "Adroit_BC_config",
            "env=pen-v0", "embedding=lewm_adroit_pen_pcd", f"seed={seed}", f"data_dir={data_dir}/",
            f"num_demos={args.num_demos}", f"epochs={args.epochs}", f"eval_frequency={args.eval_frequency}",
            "save_frequency=10", f"eval_num_traj={args.eval_num_traj}", f"num_cpu={args.num_cpu}",
            "device=cpu", "data_parallel=false", "wandb.project=local", "wandb.entity=local",
            "job_name=adroit_cortex_pcd", f"hydra.run.dir={seed_dir}",
        ]
        (seed_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        atomic_json(seed_state_path, {"stage": "running", "seed": seed, "started_at": now(), "updated_at": now(), "exit_code": None, "config": seed_config})
        state.update(stage="running", current_seed=seed, updated_at=now())
        atomic_json(state_path, state)
        resource_path = seed_dir / "resource.jsonl"
        with (seed_dir / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=seed_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                append_resource(resource_path, process.pid)
                state.update(updated_at=now(), pid=process.pid)
                atomic_json(state_path, state)
                time.sleep(args.poll_seconds)
            return_code = int(process.returncode)
        append_resource(resource_path, process.pid)
        atomic_json(seed_state_path, {"stage": "process_finished", "seed": seed, "finished_at": now(), "updated_at": now(), "exit_code": return_code, "config": seed_config})
        (seed_dir / "exit_code").write_text(f"{return_code}\n", encoding="utf-8")
        if return_code != 0:
            state.update(stage="failed", current_seed=seed, failed_seeds=[{"seed": seed, "exit_code": return_code}], exit_code=return_code, error="official BC process failed", updated_at=now())
            atomic_json(state_path, state)
            exit_path.write_text(f"{return_code}\n", encoding="utf-8")
            return return_code
        try:
            validation = validate_seed(seed_dir, seed_config)
        except Exception as exc:
            atomic_json(seed_state_path, {"stage": "failed", "seed": seed, "finished_at": now(), "updated_at": now(), "exit_code": 1, "error": repr(exc), "config": seed_config})
            state.update(stage="failed", current_seed=seed, failed_seeds=[{"seed": seed, "exit_code": 1, "error": repr(exc)}], exit_code=1, updated_at=now())
            atomic_json(state_path, state)
            exit_path.write_text("1\n", encoding="utf-8")
            return 1
        atomic_json(seed_state_path, {"stage": "finished", "seed": seed, "finished_at": now(), "updated_at": now(), "exit_code": 0, "validation": validation, "config": seed_config})
        state["completed_seeds"] = sorted(set(state.get("completed_seeds", [])) | {seed})
        state.update(current_seed=None, updated_at=now())
        atomic_json(state_path, state)

    state.update(stage="finished", current_seed=None, exit_code=0, finished_at=now(), updated_at=now())
    atomic_json(state_path, state)
    exit_path.write_text("0\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 200, 300])
    parser.add_argument("--num-demos", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-frequency", type=int, default=5)
    parser.add_argument("--eval-num-traj", type=int, default=25)
    parser.add_argument("--num-cpu", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        root = args.output_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        state_path = root / "launcher.state.json"
        state = read_json(state_path) or {"started_at": now(), "completed_seeds": [], "failed_seeds": []}
        state.update(stage="failed", updated_at=now(), finished_at=now(), exit_code=1, error=repr(exc), traceback=traceback.format_exc())
        atomic_json(state_path, state)
        (root / "exit_code").write_text("1\n", encoding="utf-8")
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
