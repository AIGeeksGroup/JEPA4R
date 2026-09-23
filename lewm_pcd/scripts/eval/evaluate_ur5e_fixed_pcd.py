"""Replay a fixed UR5e Cube holdout episode with a DexArt point-cloud CEM planner."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import hashlib
import json
import time
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
from stable_worldmodel.planning.solver import CEMSolver

import envs  # noqa: F401 - registers swm/OGBCubePCD-v0
from envs.eval_policy import PCDWorldModelPolicy, PointCloudScaler
from envs.pcd_wrapper import DEFAULT_CAMERAS, DEFAULT_CROP_BOX, PointCloudWrapper
from envs.ur5e_cube_tasks import TASKS
from scripts.eval.ur5e_cem_contract import atomic_json, inspect_holdout


DEFAULT_RUNTIME = Path("/root/autodl-tmp/workspace/users/lewm/ogbench-ur5e-multitask")
EVAL_KEYS = ["qpos", "qvel", "pcd_xyz", "pcd_rgb", "action"]
TASK_SLUGS = tuple(task.slug for task in TASKS)
PROTOCOL = "ur5e-fixed-dexart-pcd-cem300-top30-iter10-h5-b5-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_completed_record(path: Path, identity: dict) -> dict | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    for key, value in identity.items():
        if record.get(key) != value:
            raise ValueError(f"existing CEM record mismatch: {path}: {key}")
    if type(record.get("success")) is not bool or type(record.get("episode_index")) is not int:
        raise ValueError(f"incomplete CEM record: {path}")
    return record


def summarize_records(records: list[dict]) -> dict:
    indices = [r["episode_index"] for r in records]
    if len(indices) != len(set(indices)) or any(type(r["success"]) is not bool for r in records):
        raise ValueError("duplicate episode or invalid success in CEM records")
    completed = len(records)
    successes = sum(r["success"] for r in records)
    rate = successes / completed if completed else 0.0
    return {"completed": completed, "successes": successes,
            "success_rate": rate, "success_percent": 100.0 * rate}


def make_world_kwargs(task) -> dict:
    return {
        "env_name": "swm/OGBCubePCD-v0",
        "num_envs": 1,
        "max_episode_steps": 2 * task.max_episode_steps,
        "image_shape": (224, 224),
        "env_type": task.family,
        "reward_task_id": task.task_id,
        "mode": "task",
        "permute_blocks": False,
        "terminate_at_goal": True,
        "ob_type": "states",
        "multiview": False,
        "visualize_info": False,
        "pre_wrappers": [
            partial(
                PointCloudWrapper,
                num_points=1200,
                cameras=DEFAULT_CAMERAS,
                foreground_only=True,
                crop_box=DEFAULT_CROP_BOX,
                voxel_size=0.005,
                cube_fraction=0.25,
                keep_rgb=False,
            )
        ],
    }


def make_cem_kwargs(seed: int) -> dict:
    return {
        "batch_size": 1,
        "num_samples": 300,
        "var_scale": 1.0,
        "n_steps": 10,
        "topk": 30,
        "device": "cuda",
        "seed": int(seed),
    }


def make_state_callables() -> list[dict]:
    return [{"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}}]


class CheckedPCDPolicy(PCDWorldModelPolicy):
    def get_action(self, info_dict, **kwargs):
        action = super().get_action(info_dict, **kwargs)
        if not np.isfinite(action).all():
            raise FloatingPointError("CEM produced a nonfinite action")
        return action


def load_task_model(runtime: Path, task):
    run = runtime / "checkpoints" / f"ur5e_{task.slug}_pcd_dexart_10ep"
    checkpoint = run / "weights_epoch_10.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = OmegaConf.load(run / "config.yaml")
    if config.model.encoder._target_ != "encoders.DexArtPointNet":
        raise ValueError("checkpoint is not the DexArt baseline")
    pre = config.data.preprocessor
    expected = {"center": [0.425, 0.0, 0.2], "scale": 4.0, "use_rgb": True}
    for key, value in expected.items():
        if OmegaConf.to_container(pre, resolve=True)[key] != value:
            raise ValueError(f"checkpoint PCD preprocessor mismatch: {key}")
    train_path = runtime / "full" / task.slug / "train.h5"
    if Path(config.data.dataset.name).resolve() != train_path.resolve():
        raise ValueError("checkpoint training dataset does not match this task")
    model = swm.wm.utils.load_pretrained(str(checkpoint)).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model, checkpoint, train_path


def validate_external_checkpoint_source(
    checkpoint: Path, train_path: Path
) -> tuple[Path, Path]:
    checkpoint, train_path = Path(checkpoint).resolve(), Path(train_path).resolve()
    for path in (checkpoint, train_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = OmegaConf.load(checkpoint.parent / "config.yaml")
    if config.model.encoder._target_ != "encoders.DexArtPointNet":
        raise ValueError("checkpoint is not the DexArt baseline")
    pre = OmegaConf.to_container(config.data.preprocessor, resolve=True)
    for key, value in {"center": [0.425, 0.0, 0.2], "scale": 4.0, "use_rgb": True}.items():
        if pre[key] != value:
            raise ValueError(f"checkpoint PCD preprocessor mismatch: {key}")
    if Path(config.data.dataset.name).resolve() != train_path:
        raise ValueError("checkpoint training dataset does not match action scaler source")
    return checkpoint, train_path


def load_external_checkpoint(checkpoint: Path, train_path: Path):
    checkpoint, train_path = validate_external_checkpoint_source(checkpoint, train_path)
    model = swm.wm.utils.load_pretrained(str(checkpoint)).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model, checkpoint, train_path


def fit_action_scaler(train_path: Path) -> StandardScaler:
    with h5py.File(train_path, "r") as h5:
        actions = h5["action"][:]
    valid = np.isfinite(actions).all(axis=1)
    if not valid.any() or actions.shape[1] != 5:
        raise ValueError("train actions have no finite 5-D rows")
    return StandardScaler().fit(actions[valid])


def _find_point_wrapper(env):
    while env is not None:
        if isinstance(env, PointCloudWrapper):
            return env
        env = getattr(env, "env", None)
    raise RuntimeError("PointCloudWrapper is missing from CEM world")


def preflight_state(world, holdout_path: Path, spec, seed: int) -> dict:
    """Verify physical restoration and a finite fresh sensor sample before planning."""
    with h5py.File(holdout_path, "r") as h5:
        qpos = h5["qpos"][spec.offset]
        qvel = h5["qvel"][spec.offset]
    world.reset(seed=seed)
    raw = world.envs.envs[0].unwrapped
    raw.set_state(qpos, qvel)
    qpos_error = float(np.max(np.abs(raw._data.qpos - qpos)))
    qvel_error = float(np.max(np.abs(raw._data.qvel - qvel)))
    if qpos_error > 1e-7 or qvel_error > 1e-7:
        raise AssertionError(f"holdout state restoration failed: {qpos_error}, {qvel_error}")
    cloud = _find_point_wrapper(world.envs.envs[0])._point_cloud()
    if cloud["pcd_xyz"].shape != (1200, 3) or not np.isfinite(cloud["pcd_xyz"]).all():
        raise AssertionError("live point-cloud observation is invalid")
    return {"qpos_max_abs_error": qpos_error, "qvel_max_abs_error": qvel_error,
            "live_pcd_points": int(cloud["pcd_xyz"].shape[0])}


def evaluate_episode(task, spec, model, dataset, action_scaler, seed: int, holdout_path: Path) -> dict:
    world = swm.World(**make_world_kwargs(task))
    try:
        preflight = preflight_state(world, holdout_path, spec, seed)
        process = {
            "action": action_scaler,
            "pcd": PointCloudScaler(center=(0.425, 0.0, 0.2), scale=4.0, use_rgb=True),
            "goal": PointCloudScaler(center=(0.425, 0.0, 0.2), scale=4.0, use_rgb=True),
        }
        solver = CEMSolver(cost=model, **make_cem_kwargs(seed))
        policy = CheckedPCDPolicy(
            solver=solver,
            config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5),
            process=process,
            transform={},
            use_rgb=True,
        )
        world.set_policy(policy)
        start = time.monotonic()
        metrics = world.evaluate(
            dataset=dataset,
            episodes_idx=[spec.episode_index],
            start_steps=[0],
            goal_offset=spec.goal_step,
            eval_budget=task.max_episode_steps,
            callables=make_state_callables(),
            video=None,
        )
        elapsed = time.monotonic() - start
        successes = np.asarray(metrics["episode_successes"])
        if successes.shape != (1,):
            raise AssertionError("CEM returned an unexpected success vector")
        return {"success": bool(successes[0]), "elapsed_seconds": elapsed, "preflight": preflight}
    finally:
        world.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASK_SLUGS)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scaler-train", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = next(task for task in TASKS if task.slug == args.task)
    runtime = args.runtime.resolve()
    holdout_path = runtime / "full" / task.slug / "holdout.h5"
    specs = inspect_holdout(holdout_path, task)
    if not 1 <= args.episodes <= len(specs):
        raise ValueError("episodes must be between 1 and 100")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this CEM protocol")
    os.environ.setdefault("STABLEWM_HOME", str(runtime))
    external_source = args.checkpoint is not None or args.scaler_train is not None
    if external_source and (args.checkpoint is None or args.scaler_train is None):
        raise ValueError("--checkpoint and --scaler-train must be provided together")
    if external_source:
        model, checkpoint, train_path = load_external_checkpoint(
            args.checkpoint, args.scaler_train
        )
    else:
        model, checkpoint, train_path = load_task_model(runtime, task)
    scaler = fit_action_scaler(train_path)
    dataset = swm.data.HDF5Dataset(path=holdout_path, keys_to_load=EVAL_KEYS)
    output = args.output or runtime / "cem_dexart_100" / task.slug
    output.mkdir(parents=True, exist_ok=True)
    identity = {"task": task.slug, "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "holdout": str(holdout_path), "holdout_bytes": holdout_path.stat().st_size,
                "protocol": (
                    "ur5e-mixed-fixed-dexart-pcd-cem300-top30-iter10-h5-b5-v1"
                    if external_source else PROTOCOL
                ), "base_seed": args.seed}
    records = []
    for spec in specs[:args.episodes]:
        record_path = output / f"episode_{spec.episode_index:03d}.json"
        seed = args.seed + spec.episode_index
        existing = read_completed_record(record_path, identity)
        if existing is not None:
            if existing["episode_index"] != spec.episode_index or existing.get("seed") != seed or existing.get("goal_step") != spec.goal_step:
                raise ValueError(f"existing CEM record mismatch: {record_path}: episode selection")
            records.append(existing)
            print(f"resume episode {spec.episode_index}: success={existing['success']}", flush=True)
            continue
        result = evaluate_episode(task, spec, model, dataset, scaler, seed, holdout_path)
        record = {**identity, "episode_index": spec.episode_index,
                  "seed": seed, "goal_step": spec.goal_step, **result}
        atomic_json(record_path, record)
        records.append(record)
        atomic_json(output / "summary.json", {"task": task.slug, "requested": args.episodes,
                                             **summarize_records(records), "protocol": identity["protocol"]})
        print(json.dumps(record, ensure_ascii=False), flush=True)
    atomic_json(output / "summary.json", {"task": task.slug, "requested": args.episodes,
                                         **summarize_records(records), "protocol": identity["protocol"]})


if __name__ == "__main__":
    main()
