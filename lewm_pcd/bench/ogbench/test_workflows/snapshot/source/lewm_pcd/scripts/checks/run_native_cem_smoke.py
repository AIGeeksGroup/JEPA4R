"""Run one bounded native OGBench PCD checkpoint/CEM smoke step."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_action_scaler(dataset_path: Path, chunk_size: int = 100_000):
    import h5py
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    with h5py.File(dataset_path, "r") as dataset:
        actions = dataset["action"]
        for start in range(0, len(actions), chunk_size):
            chunk = np.asarray(actions[start : start + chunk_size])
            chunk = chunk[~np.isnan(chunk).any(axis=1)]
            scaler.partial_fit(chunk)
    return scaler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=92001)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    status = {
        "stage": "starting",
        "status": "running",
        "started_at": _now(),
        "updated_at": _now(),
        "exit_code": None,
        "error": None,
    }
    _write_json(status_path, status)

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root / "lewm_pcd"))
    world = None
    try:
        import torch
        from omegaconf import OmegaConf
        import stable_worldmodel as swm
        from stable_worldmodel.planning.solver import CEMSolver
        from stable_worldmodel.policy import PlanConfig

        import envs  # noqa: F401
        from envs import PCDWorldModelPolicy, PointCloudScaler, PointCloudWrapper

        checkpoint = Path(args.checkpoint).resolve()
        dataset_path = Path(args.dataset).resolve()
        checkpoint_config = OmegaConf.load(checkpoint.parent / "config.yaml")
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        config = {
            "host": socket.gethostname(),
            "repo_root": str(repo_root),
            "commit": commit,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_dataset": str(checkpoint_config.data.dataset.name),
            "action_scaler_dataset": str(dataset_path),
            "action_scaler_dataset_size": dataset_path.stat().st_size,
            "checkpoint_observation_protocol": "legacy six-camera fused PCD",
            "runtime_observation_protocol": "single fixed front_pixels eye-to-hand PCD",
            "observation_protocol_match": False,
            "intended_use": "interface smoke only; not a performance evaluation",
            "task_id": args.task_id,
            "seed": args.seed,
            "max_episode_steps": 2,
            "pcd": {"num_points": 1200, "camera": "front_pixels"},
            "plan": {
                "horizon": 1,
                "receding_horizon": 1,
                "action_block": 5,
                "num_samples": 4,
                "n_steps": 1,
                "topk": 2,
            },
        }
        _write_json(output_dir / "config.json", config)

        status.update({"stage": "checkpoint_reload", "updated_at": _now()})
        _write_json(status_path, status)
        model = swm.wm.utils.load_pretrained(str(checkpoint), cache_dir="/")
        model = model.to("cuda").eval().requires_grad_(False)
        if model.obs_key != "pcd":
            raise ValueError(f"expected obs_key='pcd', got {model.obs_key!r}")

        solver = CEMSolver(
            cost=model,
            batch_size=1,
            num_samples=4,
            var_scale=1.0,
            n_steps=1,
            topk=2,
            device="cuda",
            seed=args.seed,
        )
        plan_config = PlanConfig(
            horizon=1,
            receding_horizon=1,
            action_block=5,
        )
        scaler = PointCloudScaler(
            center=(0.425, 0.0, 0.2), scale=4.0, use_rgb=True
        )
        action_scaler = _fit_action_scaler(dataset_path)
        policy = PCDWorldModelPolicy(
            solver=solver,
            config=plan_config,
            process={"pcd": scaler, "goal": scaler, "action": action_scaler},
            transform={},
            use_rgb=True,
        )
        world = swm.World(
            env_name="swm/OGBCubePCD-v0",
            num_envs=1,
            max_episode_steps=2,
            add_pixels=False,
            pre_wrappers=[
                partial(
                    PointCloudWrapper,
                    num_points=1200,
                    cameras=("front_pixels",),
                    crop_box=((0.10, -0.50, -0.02), (0.75, 0.50, 0.60)),
                    voxel_size=0.005,
                    cube_fraction=0.25,
                )
            ],
            ob_type="states",
            env_type="single",
            terminate_at_goal=False,
            visualize_info=False,
        )
        world.set_policy(policy)

        status.update({"stage": "native_reset", "updated_at": _now()})
        _write_json(status_path, status)
        world.reset(
            seed=args.seed,
            options={"task_id": args.task_id, "render_goal": True},
        )
        reset_shapes = {
            key: list(np.asarray(world.infos[key]).shape)
            for key in ("pcd_xyz", "pcd_rgb", "goal_pcd_xyz", "goal_pcd_rgb")
        }
        for key in ("pcd_xyz", "goal_pcd_xyz"):
            if not np.isfinite(world.infos[key]).all():
                raise ValueError(f"non-finite values in {key}")

        status.update({"stage": "cem_gpu_forward", "updated_at": _now()})
        _write_json(status_path, status)
        started = time.perf_counter()
        action = policy.get_action(dict(world.infos))
        cem_wall_time = time.perf_counter() - started
        if action.shape != (1, 5):
            raise ValueError(f"expected action shape (1, 5), got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("CEM returned a non-finite action")
        if not world.envs.single_action_space.contains(action[0]):
            raise ValueError(f"CEM returned action outside env bounds: {action[0]}")

        _, rewards, terminated, truncated, stepped_info = world.envs.step(action)
        result = {
            "model_class": f"{type(model).__module__}.{type(model).__name__}",
            "encoder_class": type(model.encoder).__name__,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "model_device": str(next(model.parameters()).device),
            "model_dtype": str(next(model.parameters()).dtype),
            "reset_shapes": reset_shapes,
            "action_shape": list(action.shape),
            "action": action.tolist(),
            "action_finite": bool(np.isfinite(action).all()),
            "cem_wall_time": cem_wall_time,
            "reward": float(rewards[0]),
            "terminated": bool(terminated[0]),
            "truncated": bool(truncated[0]),
            "success": bool(np.asarray(stepped_info["success"])[0, 0]),
            "step_pcd_shape": list(np.asarray(stepped_info["pcd_xyz"]).shape),
            "step_pcd_finite": bool(np.isfinite(stepped_info["pcd_xyz"]).all()),
        }
        _write_json(output_dir / "result.json", result)

        status.update(
            {
                "stage": "completed",
                "status": "completed",
                "updated_at": _now(),
                "exit_code": 0,
            }
        )
        _write_json(status_path, status)
        print(json.dumps({"result": result, "output_dir": str(output_dir)}))
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
        if world is not None:
            world.close()


if __name__ == "__main__":
    raise SystemExit(main())
