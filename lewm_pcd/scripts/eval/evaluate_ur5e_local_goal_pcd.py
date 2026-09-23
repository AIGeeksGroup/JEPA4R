"""Run a mainline-aligned 25-step CEM diagnostic on fixed UR5e Cube data."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
from stable_worldmodel.planning.solver import CEMSolver

from envs.eval_policy import PointCloudScaler
from envs.ur5e_cube_tasks import TASKS
from scripts.eval.evaluate_ur5e_fixed_pcd import (
    CheckedPCDPolicy,
    DEFAULT_RUNTIME,
    file_sha256,
    fit_action_scaler,
    load_external_checkpoint,
    load_task_model,
    make_cem_kwargs,
    make_state_callables,
    make_world_kwargs,
    preflight_state,
    read_completed_record,
)
from scripts.eval.ur5e_cem_contract import (
    EpisodeSpec,
    atomic_json,
    sample_local_goal_specs,
)

CEM_PROTOCOL = "ur5e-fixed-dexart-pcd-local-goal-mainline-cem-v2"
RANDOM_PROTOCOL = "ur5e-fixed-random-pcd-local-goal-mainline-v1"
TASK_SLUGS = tuple(task.slug for task in TASKS)


def make_local_cem_kwargs(seed: int, iterations: int) -> dict:
    if iterations <= 0:
        raise ValueError("CEM iterations must be positive")
    kwargs = make_cem_kwargs(seed)
    kwargs["n_steps"] = int(iterations)
    return kwargs


def make_local_policy(
    kind: str,
    seed: int,
    *,
    model=None,
    action_scaler=None,
    cem_iterations: int = 30,
):
    if kind == "random":
        return swm.policy.RandomPolicy(seed=int(seed))
    if kind != "cem":
        raise ValueError(f"unsupported local policy: {kind}")
    if model is None or action_scaler is None:
        raise ValueError("CEM policy requires a model and action scaler")
    process = {
        "action": action_scaler,
        "pcd": PointCloudScaler(
            center=(0.425, 0.0, 0.2), scale=4.0, use_rgb=True
        ),
        "goal": PointCloudScaler(
            center=(0.425, 0.0, 0.2), scale=4.0, use_rgb=True
        ),
    }
    solver = CEMSolver(cost=model, **make_local_cem_kwargs(seed, cem_iterations))
    return CheckedPCDPolicy(
        solver=solver,
        config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5),
        process=process,
        transform={},
        use_rgb=True,
    )


def local_eval_keys(task) -> list[str]:
    keys = ["qpos", "qvel", "pcd_xyz", "pcd_rgb", "action"]
    for cube_id in range(task.num_cubes):
        keys.extend(
            [
                f"privileged/block_{cube_id}_pos",
                f"privileged/block_{cube_id}_quat",
            ]
        )
    return keys


def make_local_goal_callables(task) -> list[dict]:
    callables = make_state_callables()
    for cube_id in range(task.num_cubes):
        callables.append(
            {
                "method": "set_target_pos",
                "args": {
                    "cube_id": {"value": cube_id, "in_dataset": False},
                    "target_pos": {
                        "value": f"goal_privileged/block_{cube_id}_pos"
                    },
                    "target_quat": {
                        "value": f"goal_privileged/block_{cube_id}_quat"
                    },
                },
            }
        )
    return callables


def _subset_summary(records: list[dict]) -> dict:
    count = len(records)
    successes = sum(record["success"] for record in records)
    return {
        "count": count,
        "successes": successes,
        "success_rate": successes / count if count else 0.0,
    }


def summarize_local_records(records: list[dict]) -> dict:
    sample_indices = [record["sample_index"] for record in records]
    if len(sample_indices) != len(set(sample_indices)):
        raise ValueError("duplicate local-goal sample")
    if any(type(record.get("success")) is not bool for record in records):
        raise ValueError("invalid local-goal success value")
    completed = len(records)
    successes = sum(record["success"] for record in records)
    immediate = [record for record in records if record["start_already_success"]]
    nontrivial = [record for record in records if record["nontrivial"]]
    return {
        "completed": completed,
        "successes": successes,
        "success_rate": successes / completed if completed else 0.0,
        "success_percent": 100.0 * successes / completed if completed else 0.0,
        "start_already_success": _subset_summary(immediate),
        "nontrivial": _subset_summary(nontrivial),
    }


def evaluate_local_goal(
    task,
    spec,
    model,
    dataset,
    action_scaler,
    *,
    policy_kind: str,
    seed: int,
    dataset_path: Path,
    eval_budget: int,
    cem_iterations: int,
) -> dict:
    world = swm.World(**make_world_kwargs(task))
    try:
        start_spec = EpisodeSpec(
            episode_index=spec.episode_index,
            offset=spec.start_row,
            length=spec.length - spec.start_step,
        )
        preflight = preflight_state(world, dataset_path, start_spec, seed)
        policy = make_local_policy(
            policy_kind,
            seed,
            model=model,
            action_scaler=action_scaler,
            cem_iterations=cem_iterations,
        )
        world.set_policy(policy)
        start = time.monotonic()
        metrics = world.evaluate(
            dataset=dataset,
            episodes_idx=[spec.episode_index],
            start_steps=[spec.start_step],
            goal_offset=spec.goal_offset,
            eval_budget=eval_budget,
            callables=make_local_goal_callables(task),
            video=None,
        )
        elapsed = time.monotonic() - start
        successes = np.asarray(metrics["episode_successes"])
        if successes.shape != (1,):
            raise AssertionError("local evaluator returned an unexpected success vector")
        return {
            "success": bool(successes[0]),
            "elapsed_seconds": elapsed,
            "preflight": preflight,
        }
    finally:
        world.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASK_SLUGS)
    parser.add_argument("--policy", choices=("cem", "random"), default="cem")
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scaler-train", type=Path)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--source", choices=("train", "holdout"), default="train")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--cem-seed", type=int, default=42)
    parser.add_argument("--cem-iterations", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--nontrivial-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = next(task for task in TASKS if task.slug == args.task)
    runtime = args.runtime.resolve()
    dataset_path = (
        args.dataset_path.resolve()
        if args.dataset_path is not None
        else runtime / "full" / task.slug / f"{args.source}.h5"
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    specs = sample_local_goal_specs(
        dataset_path,
        task,
        count=args.count,
        seed=args.sample_seed,
        goal_offset=args.goal_offset,
        nontrivial_only=args.nontrivial_only,
        expert_only=True,
    )
    if args.eval_budget <= 0:
        raise ValueError("eval budget must be positive")
    if args.policy == "cem" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this CEM protocol")

    external_source = args.checkpoint is not None or args.scaler_train is not None
    if external_source and (args.checkpoint is None or args.scaler_train is None):
        raise ValueError("--checkpoint and --scaler-train must be provided together")
    os.environ.setdefault("STABLEWM_HOME", str(runtime))
    if args.policy == "cem":
        if external_source:
            model, checkpoint, train_path = load_external_checkpoint(
                args.checkpoint, args.scaler_train
            )
        else:
            model, checkpoint, train_path = load_task_model(runtime, task)
        scaler = fit_action_scaler(train_path)
        base_seed = args.cem_seed
    else:
        model = checkpoint = scaler = None
        base_seed = args.random_seed
    dataset = swm.data.HDF5Dataset(
        path=dataset_path,
        keys_to_load=local_eval_keys(task),
    )
    selection = "nontrivial" if args.nontrivial_only else "all"
    output = args.output or (
        runtime / f"{args.policy}_local{args.goal_offset}_{args.source}_{selection}_{args.count}"
        / task.slug
    )
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "task": task.slug,
        "dataset": str(dataset_path),
        "dataset_bytes": dataset_path.stat().st_size,
        "protocol": (
            "ur5e-mixed-fixed-pcd-local-goal-cem-v1"
            if args.policy == "cem" and external_source
            else CEM_PROTOCOL if args.policy == "cem" else RANDOM_PROTOCOL
        ),
        "policy": args.policy,
        "source": args.source,
        "selection": selection,
        "goal_offset": args.goal_offset,
        "eval_budget": args.eval_budget,
        "sample_seed": args.sample_seed,
    }
    if args.policy == "cem":
        identity.update(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "cem": {
                    "num_samples": 300,
                    "topk": 30,
                    "iterations": args.cem_iterations,
                    "horizon": 5,
                    "receding_horizon": 5,
                    "action_block": 5,
                },
                "base_cem_seed": args.cem_seed,
            }
        )
    else:
        identity["base_random_seed"] = args.random_seed
    atomic_json(
        output / "selection.json",
        {**identity, "samples": [asdict(spec) for spec in specs]},
    )

    records = []
    for spec in specs:
        record_path = output / f"window_{spec.sample_index:03d}.json"
        seed = base_seed + spec.sample_index
        existing = read_completed_record(record_path, identity)
        if existing is not None:
            expected = {
                "sample_index": spec.sample_index,
                "episode_index": spec.episode_index,
                "start_step": spec.start_step,
                "goal_step": spec.goal_step,
                "seed": seed,
            }
            if any(existing.get(key) != value for key, value in expected.items()):
                raise ValueError(
                    f"existing local record mismatch: {record_path}: window selection"
                )
            records.append(existing)
            print(
                f"resume window {spec.sample_index}: success={existing['success']}",
                flush=True,
            )
            continue

        result = evaluate_local_goal(
            task,
            spec,
            model,
            dataset,
            scaler,
            policy_kind=args.policy,
            seed=seed,
            dataset_path=dataset_path,
            eval_budget=args.eval_budget,
            cem_iterations=args.cem_iterations,
        )
        record = {
            **identity,
            **asdict(spec),
            "start_row": spec.start_row,
            "goal_step": spec.goal_step,
            "goal_row": spec.goal_row,
            "seed": seed,
            **result,
        }
        atomic_json(record_path, record)
        records.append(record)
        atomic_json(
            output / "summary.json",
            {
                **identity,
                "requested": args.count,
                **summarize_local_records(records),
            },
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)

    atomic_json(
        output / "summary.json",
        {
            **identity,
            "requested": args.count,
            **summarize_local_records(records),
        },
    )


if __name__ == "__main__":
    main()
