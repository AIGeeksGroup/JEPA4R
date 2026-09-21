"""Evaluate a fixed UR5e Cube holdout with a seeded random action policy."""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import stable_worldmodel as swm

os.environ.setdefault("MUJOCO_GL", "egl")

import envs  # noqa: F401
from envs.pcd_wrapper import DEFAULT_CAMERAS, DEFAULT_CROP_BOX, PointCloudWrapper
from envs.ur5e_cube_tasks import TASKS
from scripts.eval.ur5e_cem_contract import atomic_json, inspect_holdout

DEFAULT_RUNTIME = Path("/root/autodl-tmp/workspace/users/lewm/ogbench-ur5e-multitask")
EVAL_KEYS = ["qpos", "qvel", "pcd_xyz", "pcd_rgb", "action"]
PROTOCOL = "ur5e-fixed-random-pcd-v1"


def make_random_world_kwargs(task) -> dict:
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


def make_random_policy(seed: int):
    return swm.policy.RandomPolicy(seed=int(seed))


def summarize(records: list[dict]) -> dict:
    successes = sum(bool(record["success"]) for record in records)
    completed = len(records)
    rate = successes / completed if completed else 0.0
    return {"completed": completed, "successes": successes,
            "success_rate": rate, "success_percent": 100.0 * rate}


def evaluate_episode(task, spec, dataset, seed: int) -> dict:
    world = swm.World(**make_random_world_kwargs(task))
    try:
        world.set_policy(make_random_policy(seed))
        start = time.monotonic()
        metrics = world.evaluate(
            dataset=dataset,
            episodes_idx=[spec.episode_index],
            start_steps=[0],
            goal_offset=spec.goal_step,
            eval_budget=task.max_episode_steps,
            callables=[{"method": "set_state", "args": {
                "qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}}],
            video=None,
        )
        successes = np.asarray(metrics["episode_successes"])
        if successes.shape != (1,):
            raise AssertionError("random evaluator returned an unexpected success vector")
        return {"success": bool(successes[0]), "elapsed_seconds": time.monotonic() - start}
    finally:
        world.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=[task.slug for task in TASKS])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=52000)
    args = parser.parse_args()
    task = next(task for task in TASKS if task.slug == args.task)
    runtime = args.runtime.resolve()
    holdout = runtime / "full" / task.slug / "holdout.h5"
    specs = inspect_holdout(holdout, task)
    if not 1 <= args.episodes <= len(specs):
        raise ValueError("episodes must be between 1 and 100")
    dataset = swm.data.HDF5Dataset(path=holdout, keys_to_load=EVAL_KEYS)
    output = args.output or runtime / "cem_random_100" / task.slug
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in specs[:args.episodes]:
        seed = args.seed + spec.episode_index
        result = evaluate_episode(task, spec, dataset, seed)
        record = {"task": task.slug, "episode_index": spec.episode_index,
                  "seed": seed, "goal_step": spec.goal_step,
                  "holdout": str(holdout), "protocol": PROTOCOL, **result}
        atomic_json(output / f"episode_{spec.episode_index:03d}.json", record)
        records.append(record)
        atomic_json(output / "summary.json", {"task": task.slug,
                                             "requested": args.episodes,
                                             **summarize(records), "protocol": PROTOCOL})
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
