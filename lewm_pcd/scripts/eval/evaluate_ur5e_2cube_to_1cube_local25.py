"""Evaluate a 2cube UR5e model on real 1cube local-goal windows."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import stable_worldmodel as swm
import torch

import envs  # noqa: F401 - registers swm/OGBCubePCD-v0
from envs.pcd_wrapper import DEFAULT_CAMERAS, DEFAULT_CROP_BOX, PointCloudWrapper
from scripts.eval.evaluate_ur5e_fixed_pcd import (
    file_sha256,
    fit_action_scaler,
    load_external_checkpoint,
    preflight_state,
    read_completed_record,
)
from scripts.eval.evaluate_ur5e_local_goal_pcd import (
    make_local_cem_kwargs,
    make_local_policy,
    summarize_local_records,
)
from scripts.eval.ur5e_cem_contract import EpisodeSpec, LocalGoalSpec, atomic_json


DEFAULT_RUNTIME = Path("/autodl-fs/data/lewm/ogbench-ur5e-task1-10000")
DEFAULT_TARGET_DATASET = Path(
    "/root/autodl-tmp/workspace/users/lewm/"
    "hf-upload-jepa4r-robotics-data-archive/ur5e/"
    "cube_single_expert_pcd_raw.h5"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_RUNTIME / "transfer_2cube_to_1cube_local25_corrected"
EXPECTED_CHECKPOINT_SHA256 = (
    "ce13e4bd3b50ca44aaedd9b109c7f2d5334189ae14a3c82b502bc17108605c17"
)
CEM_PROTOCOL = "ur5e-2cube-to-1cube-local25-cem300-top30-iter10-h5-b5-v2"
RANDOM_PROTOCOL = "ur5e-2cube-to-1cube-local25-random-v2"


def single_cube_eval_keys() -> list[str]:
    return [
        "qpos",
        "qvel",
        "pcd_xyz",
        "pcd_rgb",
        "action",
        "privileged_block_0_pos",
        "privileged_block_0_quat",
    ]


def single_cube_goal_callables() -> list[dict]:
    return [
        {
            "method": "set_state",
            "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}},
        },
        {
            "method": "set_target_pos",
            "args": {
                "cube_id": {"value": 0, "in_dataset": False},
                "target_pos": {"value": "goal_privileged_block_0_pos"},
                "target_quat": {"value": "goal_privileged_block_0_quat"},
            },
        },
    ]


def make_single_cube_world_kwargs() -> dict:
    return {
        "env_name": "swm/OGBCubePCD-v0",
        "num_envs": 1,
        "max_episode_steps": 100,
        "image_shape": (224, 224),
        "env_type": "single",
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


def source_artifacts(runtime: str | Path) -> tuple[Path, Path]:
    runtime = Path(runtime)
    checkpoint = (
        runtime
        / "checkpoints"
        / "ur5e_cube_double_task1_single_pnp_pcd_dexart_expert10k_10ep"
        / "weights_epoch_10.pt"
    )
    return checkpoint, runtime / "full" / "train.h5"


def make_transfer_cem_kwargs(seed: int) -> dict:
    return make_local_cem_kwargs(seed=int(seed), iterations=10)


def _canonical_sha256(value: dict) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sample_single_cube_specs(
    path: str | Path,
    *,
    count: int,
    seed: int,
    goal_offset: int,
    nontrivial_only: bool = False,
    success_threshold: float = 0.04,
    episode_start: int = 0,
    episode_stop: int | None = None,
) -> list[LocalGoalSpec]:
    """Sample unique fixed-offset windows from the flat 1cube dataset."""
    if count <= 0:
        raise ValueError("requested window count must be positive")
    if goal_offset <= 0:
        raise ValueError("goal offset must be positive")
    if success_threshold <= 0:
        raise ValueError("success threshold must be positive")

    with h5py.File(path, "r") as h5:
        lengths = np.asarray(h5["ep_len"][:], dtype=np.int64)
        offsets = np.asarray(h5["ep_offset"][:], dtype=np.int64)
        if lengths.ndim != 1 or offsets.shape != lengths.shape or not lengths.size:
            raise ValueError("invalid episode lengths or offsets")
        if np.any(lengths <= 0):
            raise ValueError("episode lengths must be positive")
        stop = len(lengths) if episode_stop is None else episode_stop
        if not 0 <= episode_start < stop <= len(lengths):
            raise ValueError("invalid episode range")
        expected_offsets = np.r_[0, np.cumsum(lengths[:-1])]
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError("episode offsets must be contiguous")
        total = int(lengths.sum())
        expected_shapes = {
            "qpos": (total, 21),
            "qvel": (total, 20),
            "pcd_xyz": (total, 1200, 3),
            "pcd_rgb": (total, 1200, 3),
            "action": (total, 5),
            "privileged_block_0_pos": (total, 3),
            "privileged_block_0_quat": (total, 4),
        }
        for key, shape in expected_shapes.items():
            if key not in h5 or h5[key].shape != shape:
                actual = None if key not in h5 else h5[key].shape
                raise ValueError(f"{key} shape mismatch: {actual} != {shape}")
        positions = np.asarray(h5["privileged_block_0_pos"][:], dtype=np.float64)
        if not np.isfinite(positions).all():
            raise ValueError("privileged_block_0_pos contains nonfinite values")

        episode_parts = []
        start_parts = []
        displacement_parts = []
        for episode_index, (offset, length) in enumerate(zip(offsets, lengths)):
            if not episode_start <= episode_index < stop:
                continue
            offset, length = int(offset), int(length)
            valid = length - goal_offset
            if valid <= 0:
                continue
            displacement = np.linalg.norm(
                positions[offset + goal_offset : offset + length]
                - positions[offset : offset + valid],
                axis=1,
            )
            episode_parts.append(np.full(valid, episode_index, dtype=np.int64))
            start_parts.append(np.arange(valid, dtype=np.int64))
            displacement_parts.append(displacement)

        if not episode_parts:
            raise ValueError("goal offset leaves no valid windows")
        episode_indices = np.concatenate(episode_parts)
        start_steps = np.concatenate(start_parts)
        displacements = np.concatenate(displacement_parts)
        eligible = displacements > success_threshold if nontrivial_only else np.ones(
            len(displacements), dtype=bool
        )
        eligible_indices = np.flatnonzero(eligible)
        if count > len(eligible_indices):
            raise ValueError(
                f"requested {count} windows but only {len(eligible_indices)} are eligible"
            )

        rng = np.random.default_rng(seed)
        chosen = rng.choice(eligible_indices, size=count, replace=False)
        start_rows = offsets[episode_indices] + start_steps
        chosen = chosen[np.argsort(start_rows[chosen])]

        specs = []
        finite_goal_keys = (
            "qpos",
            "qvel",
            "pcd_xyz",
            "pcd_rgb",
            "privileged_block_0_pos",
            "privileged_block_0_quat",
        )
        for sample_index, candidate in enumerate(chosen):
            episode_index = int(episode_indices[candidate])
            offset = int(offsets[episode_index])
            start_step = int(start_steps[candidate])
            start_row = offset + start_step
            goal_row = start_row + goal_offset
            for key in finite_goal_keys:
                if not np.isfinite(h5[key][start_row]).all() or not np.isfinite(
                    h5[key][goal_row]
                ).all():
                    raise ValueError(f"{key} contains nonfinite selected state")
            if not np.isfinite(h5["action"][start_row]).all():
                raise ValueError("action contains nonfinite selected start")
            displacement = float(displacements[candidate])
            specs.append(
                LocalGoalSpec(
                    sample_index=sample_index,
                    episode_index=episode_index,
                    offset=offset,
                    length=int(lengths[episode_index]),
                    start_step=start_step,
                    goal_offset=int(goal_offset),
                    max_cube_displacement=displacement,
                    start_already_success=bool(displacement <= success_threshold),
                    nontrivial=bool(displacement > success_threshold),
                )
            )
    return specs


def ensure_selection(
    path: str | Path, identity: dict, specs: list[LocalGoalSpec]
) -> tuple[dict, str]:
    """Create or validate one immutable selection shared by both policies."""
    path = Path(path)
    manifest = {**identity, "samples": [asdict(spec) for spec in specs]}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(f"selection manifest mismatch: {path}")
        manifest = existing
    else:
        atomic_json(path, manifest)
    return manifest, _canonical_sha256(manifest)


def _load_transfer_model(checkpoint: Path):
    model = swm.wm.utils.load_pretrained(str(checkpoint)).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


def evaluate_transfer_window(
    spec: LocalGoalSpec,
    model,
    dataset,
    action_scaler,
    *,
    policy_kind: str,
    seed: int,
    target_dataset: Path,
    eval_budget: int,
) -> dict:
    world = swm.World(**make_single_cube_world_kwargs())
    try:
        start_spec = EpisodeSpec(
            episode_index=spec.episode_index,
            offset=spec.start_row,
            length=spec.length - spec.start_step,
        )
        preflight = preflight_state(world, target_dataset, start_spec, seed)
        policy = make_local_policy(
            policy_kind,
            seed,
            model=model,
            action_scaler=action_scaler,
            cem_iterations=10,
        )
        world.set_policy(policy)
        started = time.monotonic()
        metrics = world.evaluate(
            dataset=dataset,
            episodes_idx=[spec.episode_index],
            start_steps=[spec.start_step],
            goal_offset=spec.goal_offset,
            eval_budget=eval_budget,
            callables=single_cube_goal_callables(),
            video=None,
        )
        elapsed = time.monotonic() - started
        successes = np.asarray(metrics["episode_successes"])
        if successes.shape != (1,):
            raise AssertionError("transfer evaluator returned an unexpected success vector")
        return {
            "success": bool(successes[0]),
            "elapsed_seconds": elapsed,
            "preflight": preflight,
        }
    finally:
        world.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, choices=("cem", "random"))
    parser.add_argument("--selection", required=True, choices=("all", "nontrivial"))
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--target-dataset", type=Path, default=DEFAULT_TARGET_DATASET)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scaler-train", type=Path)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--policy-seed", type=int, default=42)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-stop", type=int)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def _summary(identity: dict, requested: int, records: list[dict]) -> dict:
    return {**identity, "requested": requested, **summarize_local_records(records)}


def main() -> None:
    args = build_parser().parse_args()
    if args.eval_budget <= 0:
        raise ValueError("eval budget must be positive")
    runtime = args.runtime.resolve()
    target_dataset = args.target_dataset.resolve()
    output_root = args.output_root.resolve()
    if output_root == (
        DEFAULT_RUNTIME / "transfer_2cube_to_1cube_local25"
    ).resolve():
        raise ValueError("refusing to overwrite the historical invalid transfer output")
    if not target_dataset.is_file():
        raise FileNotFoundError(target_dataset)

    nontrivial_only = args.selection == "nontrivial"
    specs = sample_single_cube_specs(
        target_dataset,
        count=args.count,
        seed=args.sample_seed,
        goal_offset=args.goal_offset,
        nontrivial_only=nontrivial_only,
        episode_start=args.episode_start,
        episode_stop=args.episode_stop,
    )
    selection_identity = {
        "target_dataset": str(target_dataset),
        "target_dataset_bytes": target_dataset.stat().st_size,
        "count": args.count,
        "sample_seed": args.sample_seed,
        "goal_offset": args.goal_offset,
        "selection": args.selection,
        "nontrivial_only": nontrivial_only,
        "success_threshold": 0.04,
    }
    if args.episode_start != 0 or args.episode_stop is not None:
        selection_identity.update(
            episode_start=args.episode_start, episode_stop=args.episode_stop
        )
    selection_path = output_root / args.selection / "selection.json"
    _, selection_sha256 = ensure_selection(
        selection_path, selection_identity, specs
    )

    external_source = args.checkpoint is not None or args.scaler_train is not None
    if external_source and (args.checkpoint is None or args.scaler_train is None):
        raise ValueError("--checkpoint and --scaler-train must be provided together")
    checkpoint, train_path = source_artifacts(runtime)
    os.environ.setdefault("STABLEWM_HOME", str(runtime))
    if args.policy == "cem":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this CEM protocol")
        if external_source:
            model, checkpoint, train_path = load_external_checkpoint(
                args.checkpoint, args.scaler_train
            )
            checkpoint_sha256 = file_sha256(checkpoint)
        else:
            for path in (checkpoint, train_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            checkpoint_sha256 = file_sha256(checkpoint)
            if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
                raise ValueError("2cube checkpoint SHA-256 mismatch")
            model = _load_transfer_model(checkpoint)
        action_scaler = fit_action_scaler(train_path)
    else:
        model = action_scaler = None
        checkpoint_sha256 = None

    dataset = swm.data.HDF5Dataset(
        path=target_dataset,
        keys_to_load=single_cube_eval_keys(),
    )
    output = output_root / args.selection / args.policy
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "protocol": (
            "ur5e-mixed-to-1cube-local25-cem300-top30-iter10-h5-b5-v1"
            if args.policy == "cem" and external_source
            else CEM_PROTOCOL if args.policy == "cem" else RANDOM_PROTOCOL
        ),
        "policy": args.policy,
        "selection": args.selection,
        "selection_manifest": str(selection_path),
        "selection_sha256": selection_sha256,
        "target_dataset": str(target_dataset),
        "target_dataset_bytes": target_dataset.stat().st_size,
        "env_type": "single",
        "mode": "task",
        "terminate_at_goal": True,
        "goal_offset": args.goal_offset,
        "eval_budget": args.eval_budget,
        "base_policy_seed": args.policy_seed,
    }
    if args.episode_start != 0 or args.episode_stop is not None:
        identity.update(episode_start=args.episode_start, episode_stop=args.episode_stop)
    if args.policy == "cem":
        cem = make_transfer_cem_kwargs(args.policy_seed)
        identity.update(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "source_train_for_action_scaler": str(train_path),
                "cem": {
                    "samples": cem["num_samples"],
                    "topk": cem["topk"],
                    "iterations": cem["n_steps"],
                    "horizon": 5,
                    "receding_horizon": 5,
                    "action_block": 5,
                },
            }
        )

    records = []
    for spec in specs:
        record_path = output / f"window_{spec.sample_index:03d}.json"
        seed = args.policy_seed + spec.sample_index
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
                raise ValueError(f"existing transfer record window mismatch: {record_path}")
            records.append(existing)
            print(
                f"resume window {spec.sample_index}: success={existing['success']}",
                flush=True,
            )
            continue

        result = evaluate_transfer_window(
            spec,
            model,
            dataset,
            action_scaler,
            policy_kind=args.policy,
            seed=seed,
            target_dataset=target_dataset,
            eval_budget=args.eval_budget,
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
        atomic_json(output / "summary.json", _summary(identity, args.count, records))
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)

    summary = _summary(identity, args.count, records)
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
