"""Persistent serial launcher for official Adroit RGB BC and frozen baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm_pcd.bench.cortexbench.protocol import validate_metric_records


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ps_resource(pid: int) -> dict:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pcpu=,rss=,etime=,stat="],
            capture_output=True, text=True, check=False,
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
            capture_output=True, text=True, check=False,
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
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": now(), "pid": pid, **ps_resource(pid), **gpu_resource()}) + "\n")


def validate_resolved_config(resolved: dict, expected_demos: int, expected_epochs: int,
                             expected_eval_frequency: int, expected_eval_traj: int) -> None:
    expected = {
        "env": "pen-v0",
        "num_demos": expected_demos,
        "epochs": expected_epochs,
        "eval_frequency": expected_eval_frequency,
        "eval_num_traj": expected_eval_traj,
    }
    for key, value in expected.items():
        if resolved.get(key) != value:
            raise ValueError(f"resolved config mismatch for {key}: {resolved.get(key)!r} != {value!r}")
    env_kwargs = resolved.get("env_kwargs", {})
    if env_kwargs.get("add_proprio") is not True or env_kwargs.get("proprio_key") != "proprio":
        raise ValueError("resolved config proprio contract mismatch")
    if resolved.get("camera") != "vil_camera" or resolved.get("pixel_based") is not True:
        raise ValueError("resolved config RGB camera/pixel contract mismatch")


def inspect_demo(path: Path, expected_count: int) -> dict:
    with path.open("rb") as handle:
        demos = pickle.load(handle)
    if len(demos) != expected_count:
        raise ValueError(f"official Adroit demo count is {len(demos)}, expected {expected_count}")
    if not demos:
        raise ValueError("official Adroit demo file is empty")
    action_shapes = sorted({tuple(path["actions"].shape[1:]) for path in demos})
    proprio_shapes = sorted({tuple(path["env_infos"]["proprio"].shape[1:]) for path in demos})
    if action_shapes != [(24,)] or proprio_shapes != [(24,)]:
        raise ValueError(f"Adroit action/proprio contract mismatch: actions={action_shapes}, proprio={proprio_shapes}")
    image_shapes = sorted({tuple(path["images"].shape[1:]) for path in demos})
    if image_shapes != [(256, 256, 3)]:
        raise ValueError(f"Adroit RGB image contract mismatch: images={image_shapes}")
    mismatched_lengths = [
        index for index, path in enumerate(demos)
        if len(path["actions"]) != len(path["images"])
        or len(path["env_infos"]["proprio"]) != len(path["images"])
    ]
    if mismatched_lengths:
        raise ValueError(f"Adroit demo frame alignment mismatch at demos {mismatched_lengths[:5]}")
    return {
        "status": "PASS",
        "demo_count": len(demos),
        "action_shapes": [list(shape) for shape in action_shapes],
        "proprio_shapes": [list(shape) for shape in proprio_shapes],
        "image_shapes": [list(shape) for shape in image_shapes],
        "episode_lengths": {"min": min(len(path["actions"]) for path in demos), "max": max(len(path["actions"]) for path in demos)},
    }


def validate_seed(seed_dir: Path, config: dict) -> dict:
    metrics_path = seed_dir / "metrics.jsonl"
    config_path = seed_dir / ".hydra" / "config.yaml"
    policies = list(seed_dir.rglob("best_policy.pickle"))
    if not metrics_path.exists() or not config_path.exists() or len(policies) != 1:
        raise RuntimeError("missing metrics, resolved config, or unique best_policy.pickle")
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = validate_metric_records(records, config["epochs"], config["eval_frequency"])
    import yaml
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_resolved_config(resolved, config["num_demos"], config["epochs"], config["eval_frequency"], config["eval_num_traj"])
    with policies[0].open("rb") as handle:
        policy = pickle.load(handle)
    resource_path = seed_dir / "resource.jsonl"
    resources = [json.loads(line) for line in resource_path.read_text(encoding="utf-8").splitlines() if line.strip()] if resource_path.exists() else []
    cpu = [row["pcpu"] for row in resources if row.get("pcpu") is not None]
    rss = [row["rss_kib"] for row in resources if row.get("rss_kib") is not None]
    gpu = [item["utilization_gpu"] for row in resources for item in row.get("gpus", [])]
    result = {
        "status": "PASS",
        "track": "official RGB CortexBench Adroit Pen",
        "model": config["model"],
        "seed": config["seed"],
        "num_demos": config["num_demos"],
        "loss_records": metrics["loss_records"],
        "eval_records": metrics["eval_records"],
        "best_of_20_success_percent": metrics["best_success"],
        "final_success_percent": metrics["final_success"],
        "policy_type": type(policy).__name__,
        "policy_sha256": sha256(policies[0]),
        "policy_path": str(policies[0]),
        "demo_sha256": config["demo_sha256"],
        "resource_samples": len(resources),
        "cpu_percent_max": max(cpu) if cpu else None,
        "rss_mib_max": max(rss) / 1024.0 if rss else None,
        "gpu_utilization_percent_max": max(gpu) if gpu else None,
        "resolved_config_checked": True,
    }
    atomic_json(seed_dir / "validation.json", result)
    return result


def run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    data_dir = args.data_dir.resolve()
    demo_path = data_dir / "pen-v0.pickle"
    if not demo_path.exists():
        raise FileNotFoundError(demo_path)
    config = {
        "track": "official RGB CortexBench Adroit Pen",
        "model": args.model,
        "env": "pen-v0",
        "data_dir": str(data_dir),
        "demo_sha256": sha256(demo_path),
        "num_demos": args.num_demos,
        "epochs": args.epochs,
        "eval_frequency": args.eval_frequency,
        "eval_num_traj": args.eval_num_traj,
        "num_cpu": args.num_cpu,
        "device": "cpu",
        "seeds": args.seeds,
    }
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "launcher.state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config") != config:
            raise RuntimeError("existing launcher config does not match requested run")
        if state.get("stage") == "finished":
            return int(state.get("exit_code", 0))
    elif any(path.name not in {"launcher.log", "launcher.pid"} for path in root.iterdir()):
        raise FileExistsError(f"nonempty output root without matching launcher state: {root}")
    else:
        state = {"stage": "initialized", "started_at": now(), "updated_at": now(), "completed_seeds": [], "failed_seeds": [], "exit_code": None, "config": config}
        atomic_json(state_path, state)

    atomic_json(root / "dataset_contract.json", inspect_demo(demo_path, args.num_demos))
    env = dict(os.environ)
    env.update({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    repo = Path(__file__).resolve().parents[2]
    runner = repo / "bench" / "cortexbench" / "run_bc.py"
    for seed in args.seeds:
        seed_dir = root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_state_path = seed_dir / "state.json"
        seed_config = {**config, "seed": seed}
        previous = json.loads(seed_state_path.read_text(encoding="utf-8")) if seed_state_path.exists() else None
        if previous and previous.get("stage") == "finished" and previous.get("exit_code") == 0:
            validation = validate_seed(seed_dir, seed_config)
            state["completed_seeds"] = sorted(set(state.get("completed_seeds", [])) | {seed})
            state["last_validation"] = validation
            state["updated_at"] = now()
            atomic_json(state_path, state)
            continue
        if previous:
            raise RuntimeError(f"seed {seed} has an existing non-finished state; refusing overwrite")
        if any(seed_dir.iterdir()):
            raise FileExistsError(f"nonempty seed output root: {seed_dir}")
        command = [
            sys.executable, str(runner), "--config-name", "Adroit_BC_config",
            "env=pen-v0", f"embedding={args.model}", f"seed={seed}", f"data_dir={data_dir}/",
            f"num_demos={args.num_demos}", f"epochs={args.epochs}", f"eval_frequency={args.eval_frequency}",
            "save_frequency=10", f"eval_num_traj={args.eval_num_traj}", f"num_cpu={args.num_cpu}",
            "device=cpu", "data_parallel=false", "wandb.project=local", "wandb.entity=local",
            f"hydra.run.dir={seed_dir}",
        ]
        (seed_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        atomic_json(seed_state_path, {"stage": "running", "seed": seed, "started_at": now(), "updated_at": now(), "exit_code": None, "config": seed_config})
        state.update(stage="running", current_seed=seed, updated_at=now())
        atomic_json(state_path, state)
        with (seed_dir / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=seed_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                append_resource(seed_dir / "resource.jsonl", process.pid)
                state.update(updated_at=now(), pid=process.pid)
                atomic_json(state_path, state)
                time.sleep(args.poll_seconds)
            return_code = int(process.returncode)
        append_resource(seed_dir / "resource.jsonl", process.pid)
        (seed_dir / "exit_code").write_text(f"{return_code}\n", encoding="utf-8")
        if return_code != 0:
            atomic_json(seed_state_path, {"stage": "failed", "seed": seed, "finished_at": now(), "updated_at": now(), "exit_code": return_code, "config": seed_config})
            state.update(stage="failed", current_seed=seed, failed_seeds=[{"seed": seed, "exit_code": return_code}], exit_code=return_code, updated_at=now())
            atomic_json(state_path, state)
            (root / "exit_code").write_text(f"{return_code}\n", encoding="utf-8")
            return return_code
        try:
            validation = validate_seed(seed_dir, seed_config)
        except Exception as exc:
            atomic_json(seed_state_path, {"stage": "failed", "seed": seed, "finished_at": now(), "updated_at": now(), "exit_code": 1, "error": repr(exc), "config": seed_config})
            state.update(stage="failed", current_seed=seed, failed_seeds=[{"seed": seed, "exit_code": 1, "error": repr(exc)}], exit_code=1, updated_at=now())
            atomic_json(state_path, state)
            (root / "exit_code").write_text("1\n", encoding="utf-8")
            return 1
        atomic_json(seed_state_path, {"stage": "finished", "seed": seed, "finished_at": now(), "updated_at": now(), "exit_code": 0, "validation": validation, "config": seed_config})
        state["completed_seeds"] = sorted(set(state.get("completed_seeds", [])) | {seed})
        state.update(current_seed=None, updated_at=now())
        atomic_json(state_path, state)
    state.update(stage="finished", current_seed=None, exit_code=0, finished_at=now(), updated_at=now())
    atomic_json(state_path, state)
    (root / "exit_code").write_text("0\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 200, 300])
    parser.add_argument("--num-demos", type=int, default=100)
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
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"started_at": now(), "completed_seeds": [], "failed_seeds": []}
        state.update(stage="failed", updated_at=now(), finished_at=now(), exit_code=1, error=repr(exc), traceback=traceback.format_exc())
        atomic_json(state_path, state)
        (root / "exit_code").write_text("1\n", encoding="utf-8")
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
