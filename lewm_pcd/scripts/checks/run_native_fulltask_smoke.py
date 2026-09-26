"""Bounded full-task OGBench native evaluation with auditable artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


TASK_SEED_BASE = {1: 10000, 2: 11000, 3: 12000, 4: 13000, 5: 14000}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value) -> None:
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


def _save_rgb(path: Path, image) -> None:
    from PIL import Image

    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def _save_pcd(path: Path, xyz, rgb, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = np.asarray(xyz)
    colors = np.asarray(rgb, dtype=np.float32) / 255.0
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(xyz[:, 0], xyz[:, 1], c=colors, s=3)
    axes[0].set(xlabel="x", ylabel="y", title=f"{title}: top view")
    axes[0].set_aspect("equal", adjustable="box")
    axes[1].scatter(xyz[:, 0], xyz[:, 2], c=colors, s=3)
    axes[1].set(xlabel="x", ylabel="z", title=f"{title}: side view")
    axes[1].set_aspect("equal", adjustable="box")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_episode_animation(artifact: Path, frames, fps: int = 20) -> str:
    try:
        import imageio_ffmpeg  # noqa: F401
        import imageio.v2 as imageio

        path = artifact / "episode.mp4"
        imageio.mimsave(path, frames, fps=fps)
    except ImportError:
        from PIL import Image

        path = artifact / "episode.gif"
        images = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=max(1, round(1000 / fps)),
            loop=0,
            optimize=False,
        )
    return path.name


class NativeCubeOraclePolicy:
    """Use the OGBench Cube plan oracle without changing the official goal."""

    def __init__(self, env, *, noise=0.0, seed=0):
        from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle

        self.env = env
        self.oracle = CubePlanOracle(
            env=env.unwrapped,
            noise=noise,
            noise_smoothing=0.5,
        )
        np.random.seed(seed)
        self._needs_reset = True

    def reset(self):
        self._needs_reset = True

    def act(self, obs, info):
        if self._needs_reset:
            self.oracle.reset(None, info)
            self._needs_reset = False
        return np.clip(np.asarray(self.oracle.select_action(None, info)), -1, 1)


def _build_checkpoint_policy(args, env, output_dir):
    import torch
    import stable_worldmodel as swm
    from stable_worldmodel.planning.solver import CEMSolver
    from stable_worldmodel.policy import PlanConfig

    from envs import PCDWorldModelPolicy, PointCloudScaler
    from eval_ogbench import NativeBatchedPolicy

    checkpoint = Path(args.checkpoint).resolve()
    dataset = Path(args.dataset).resolve()
    model = swm.wm.utils.load_pretrained(str(checkpoint), cache_dir="/")
    model = model.to("cuda").eval().requires_grad_(False)
    if model.obs_key != "pcd":
        raise ValueError(f"expected PCD checkpoint, got obs_key={model.obs_key!r}")

    class RecordingCEMSolver(CEMSolver):
        def reset_episode(self):
            self.first_costs = None

        def solve(self, *solver_args, **solver_kwargs):
            result = super().solve(*solver_args, **solver_kwargs)
            if self.first_costs is None:
                self.first_costs = [float(value) for value in result["costs"]]
            return result

    solver = RecordingCEMSolver(
        cost=model,
        batch_size=1,
        num_samples=args.num_samples,
        var_scale=1.0,
        n_steps=args.cem_steps,
        topk=args.topk,
        device="cuda",
        seed=args.solver_seed,
    )
    plan = PlanConfig(
        horizon=args.horizon,
        receding_horizon=1,
        action_block=5,
    )
    pcd_scaler = PointCloudScaler(
        center=(0.425, 0.0, 0.2), scale=4.0, use_rgb=True
    )
    planner = PCDWorldModelPolicy(
        solver=solver,
        config=plan,
        process={
            "pcd": pcd_scaler,
            "goal": pcd_scaler,
            "action": _fit_action_scaler(dataset),
        },
        transform={},
        use_rgb=True,
    )
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "action_scaler_dataset": str(dataset),
        "action_scaler_dataset_size": dataset.stat().st_size,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "encoder_class": type(model.encoder).__name__,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "model_device": str(next(model.parameters()).device),
        "checkpoint_observation_protocol": "legacy six-camera fused PCD",
        "runtime_observation_protocol": "single fixed front_pixels eye-to-hand PCD",
        "observation_protocol_match": False,
        "intended_use": "engineering full-task smoke only; not performance evaluation",
    }
    _write_json(output_dir / "checkpoint.json", metadata)
    return NativeBatchedPolicy(planner, env), solver


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy", choices=("random", "oracle", "checkpoint"), required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset")
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--cem-steps", type=int, default=2)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--solver-seed", type=int, default=42)
    parser.add_argument("--video-stride", type=int, default=2)
    args = parser.parse_args()
    if args.policy == "checkpoint" and not (args.checkpoint and args.dataset):
        parser.error("--checkpoint and --dataset are required for checkpoint policy")

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

        seed_manifest = {
            task_id: [TASK_SEED_BASE[task_id] + i for i in range(args.episodes_per_task)]
            for task_id in args.task_ids
        }
        config = {
            "host": socket.gethostname(),
            "repo_root": str(repo_root),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
            ).strip(),
            "policy": args.policy,
            "task_ids": args.task_ids,
            "episodes_per_task": args.episodes_per_task,
            "max_episode_steps": args.max_episode_steps,
            "official_ogbench_cube_time_limit": 200,
            "seed_manifest": {str(k): v for k, v in seed_manifest.items()},
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "pcd_camera": "front_pixels",
            "pcd_points": 1200,
            "cem": None if args.policy != "checkpoint" else {
                "horizon": args.horizon,
                "receding_horizon": 1,
                "action_block": 5,
                "num_samples": args.num_samples,
                "n_steps": args.cem_steps,
                "topk": args.topk,
                "solver_seed": args.solver_seed,
                "mode": "reduced engineering smoke; official 300/30/10 not used",
            },
        }
        _write_json(output_dir / "config.json", config)
        _write_json(output_dir / "seed_manifest.json", config["seed_manifest"])

        base_env = gym.make(
            "swm/OGBCubePCD-v0",
            ob_type="states",
            env_type="single",
            max_episode_steps=args.max_episode_steps,
            terminate_at_goal=True,
            visualize_info=False,
        )
        env = PointCloudWrapper(
            base_env,
            num_points=1200,
            cameras=("front_pixels",),
            crop_box=((0.10, -0.50, -0.02), (0.75, 0.50, 0.60)),
            voxel_size=0.005,
            cube_fraction=0.25,
            keep_rgb=True,
        )
        env.action_space.seed(args.solver_seed)
        solver = None
        if args.policy == "random":
            policy = RandomPolicy(env.action_space)
        elif args.policy == "oracle":
            policy = NativeCubeOraclePolicy(env, noise=0.0, seed=args.solver_seed)
        else:
            policy, solver = _build_checkpoint_policy(args, env, output_dir)

        frames = {}
        artifact_dirs = {}

        def on_reset(**event):
            if event["episode_idx"] != 0:
                return
            task_id = event["task_id"]
            info = event["info"]
            artifact = output_dir / "artifacts" / f"task{task_id}_episode0"
            artifact.mkdir(parents=True, exist_ok=True)
            artifact_dirs[task_id] = artifact
            _save_rgb(artifact / "initial_rgb.png", info["pixels/front_pixels"])
            _save_rgb(artifact / "goal_rgb.png", env.unwrapped._cur_goal_rendered)
            _save_pcd(
                artifact / "initial_pcd.png", info["pcd_xyz"], info["pcd_rgb"], "initial PCD"
            )
            _save_pcd(
                artifact / "goal_pcd.png", info["goal_pcd_xyz"], info["goal_pcd_rgb"], "goal PCD"
            )
            np.savez_compressed(
                artifact / "initial_goal_pcd.npz",
                pcd_xyz=info["pcd_xyz"],
                pcd_rgb=info["pcd_rgb"],
                goal_pcd_xyz=info["goal_pcd_xyz"],
                goal_pcd_rgb=info["goal_pcd_rgb"],
            )
            frames[task_id] = [np.asarray(info["pixels/front_pixels"]).copy()]

        def on_step(**event):
            if event["episode_idx"] != 0:
                return
            task_id = event["task_id"]
            if event["steps"] % args.video_stride == 0 or event["terminated"] or event["truncated"]:
                frames[task_id].append(np.asarray(event["info"]["pixels/front_pixels"]).copy())

        def on_episode_end(**event):
            record = event["record"]
            if record["episode_idx"] != 0:
                return
            task_id = record["task_id"]
            artifact = artifact_dirs[task_id]
            animation = _save_episode_animation(artifact, frames[task_id], fps=20)
            debug = {"record": record, "episode_animation": animation}
            if solver is not None and solver.first_costs is not None:
                debug["first_cem_final_elite_mean_costs"] = solver.first_costs
            _write_json(artifact / "debug.json", debug)

        status.update({"stage": "native_fulltask_rollout", "updated_at": _now()})
        _write_json(status_path, status)
        records = evaluate_native(
            env,
            policy,
            task_ids=args.task_ids,
            episodes_per_task=args.episodes_per_task,
            seed_manifest=seed_manifest,
            on_reset=on_reset,
            on_step=on_step,
            on_episode_end=on_episode_end,
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
