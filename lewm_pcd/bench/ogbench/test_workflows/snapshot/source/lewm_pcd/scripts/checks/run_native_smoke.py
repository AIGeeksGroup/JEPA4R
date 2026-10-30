"""Run a bounded OGBench-native point-cloud random-policy smoke test."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=91000)
    parser.add_argument("--action-seed", type=int, default=91042)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    status = {
        "stage": "starting",
        "status": "running",
        "started_at": _now(),
        "updated_at": _now(),
        "completed_episodes": 0,
        "failed_episodes": 0,
        "exit_code": None,
        "error": None,
    }
    _write_json(status_path, status)

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root / "lewm_pcd"))
    env = None
    try:
        import gymnasium as gym
        import torch

        import envs  # noqa: F401
        from envs import PointCloudWrapper
        from eval_ogbench import RandomPolicy, evaluate_native, summarize_results

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        config = {
            "host": socket.gethostname(),
            "repo_root": str(repo_root),
            "commit": commit,
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "task_ids": [1, 2, 3, 4, 5],
            "episodes_per_task": args.episodes_per_task,
            "max_episode_steps": args.max_episode_steps,
            "seed_manifest": {
                str(task_id): [args.seed_base + task_id]
                for task_id in range(1, 6)
            },
            "action_seed": args.action_seed,
            "ob_type": "states",
            "observation_adapter": "single front_pixels depth PCD",
            "num_points": 1200,
            "camera": "front_pixels",
        }
        _write_json(output_dir / "config.json", config)

        status.update({"stage": "environment_reset_step", "updated_at": _now()})
        _write_json(status_path, status)
        base_env = gym.make(
            "swm/OGBCubePCD-v0",
            ob_type="states",
            env_type="single",
            max_episode_steps=args.max_episode_steps,
            terminate_at_goal=False,
            visualize_info=False,
        )
        env = PointCloudWrapper(
            base_env,
            num_points=1200,
            cameras=("front_pixels",),
            crop_box=((0.10, -0.50, -0.02), (0.75, 0.50, 0.60)),
            voxel_size=0.005,
            cube_fraction=0.25,
        )
        env.action_space.seed(args.action_seed)
        seed_manifest = {
            task_id: [args.seed_base + task_id]
            for task_id in range(1, 6)
        }
        policy = RandomPolicy(env.action_space)
        records = evaluate_native(
            env,
            policy,
            task_ids=(1, 2, 3, 4, 5),
            episodes_per_task=args.episodes_per_task,
            seed_manifest=seed_manifest,
        )
        summary = summarize_results(records)
        _write_json(output_dir / "results.json", {"records": records, "summary": summary})
        with (output_dir / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

        status.update(
            {
                "stage": "completed",
                "status": "completed",
                "updated_at": _now(),
                "completed_episodes": len(records),
                "failed_episodes": 0,
                "exit_code": 0,
            }
        )
        _write_json(status_path, status)
        print(json.dumps({"summary": summary, "output_dir": str(output_dir)}))
        return 0
    except Exception as exc:
        error = "".join(traceback.format_exception(exc))
        (output_dir / "error.txt").write_text(error)
        status.update(
            {
                "stage": "failed",
                "status": "failed",
                "updated_at": _now(),
                "exit_code": 1,
                "error": repr(exc),
            }
        )
        _write_json(status_path, status)
        print(error, file=sys.stderr)
        return 1
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
