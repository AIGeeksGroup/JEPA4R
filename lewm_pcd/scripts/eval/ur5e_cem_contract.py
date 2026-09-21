"""Validate fixed-task UR5e point-cloud holdouts before CEM replay."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class EpisodeSpec:
    episode_index: int
    offset: int
    length: int

    @property
    def goal_step(self) -> int:
        return self.length - 1


def atomic_json(path: str | Path, value: dict) -> None:
    """Write one complete JSON record using a temporary sibling file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def inspect_holdout(path: str | Path, task, *, require_full: bool = True) -> list[EpisodeSpec]:
    """Return episode bounds only when the holdout matches the fixed task."""
    with h5py.File(path, "r") as h5:
        lengths = np.asarray(h5["ep_len"][:], dtype=np.int64)
        offsets = np.asarray(h5["ep_offset"][:], dtype=np.int64)
        if lengths.ndim != 1 or offsets.shape != lengths.shape or not lengths.size:
            raise ValueError("invalid episode lengths or offsets")
        if require_full and lengths.size != 100:
            raise ValueError("full holdout must contain 100 episodes")
        if np.any(lengths < 2):
            raise ValueError("each episode needs a start and goal frame")
        expected_offsets = np.r_[0, np.cumsum(lengths[:-1])]
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError("episode offsets must be contiguous")

        total = int(lengths.sum())
        expected_shapes = {
            "fixed_task_id": (total,),
            "source_policy": (total,),
            "accepted_success": (total,),
            "qpos": (total, task.qpos_dim),
            "qvel": (total, task.qvel_dim),
            "pcd_xyz": (total, 1200, 3),
            "pcd_rgb": (total, 1200, 3),
            "action": (total, task.action_dim),
        }
        for key, shape in expected_shapes.items():
            if h5[key].shape != shape:
                raise ValueError(f"{key} shape mismatch: {h5[key].shape} != {shape}")
        if not np.all(h5["fixed_task_id"][:] == task.task_id):
            raise ValueError("holdout task ID mismatch")
        if not np.all(h5["source_policy"][:] == 1):
            raise ValueError("holdout must contain expert episodes only")
        if not np.all(h5["accepted_success"][:]):
            raise ValueError("holdout requires expert success for every episode")

        specs = [
            EpisodeSpec(i, int(offset), int(length))
            for i, (offset, length) in enumerate(zip(offsets, lengths))
        ]
        for spec in specs:
            for row in (spec.offset, spec.offset + spec.goal_step):
                if not np.isfinite(h5["pcd_xyz"][row]).all():
                    raise ValueError(f"nonfinite point cloud at row {row}")
                if not np.isfinite(h5["qpos"][row]).all() or not np.isfinite(h5["qvel"][row]).all():
                    raise ValueError(f"nonfinite robot state at row {row}")
        return specs


@dataclass(frozen=True)
class LocalGoalSpec:
    sample_index: int
    episode_index: int
    offset: int
    length: int
    start_step: int
    goal_offset: int
    max_cube_displacement: float
    start_already_success: bool
    nontrivial: bool

    @property
    def start_row(self) -> int:
        return self.offset + self.start_step

    @property
    def goal_step(self) -> int:
        return self.start_step + self.goal_offset

    @property
    def goal_row(self) -> int:
        return self.offset + self.goal_step


def sample_local_goal_specs(
    path: str | Path,
    task,
    *,
    count: int,
    seed: int = 42,
    goal_offset: int = 25,
    nontrivial_only: bool = False,
    success_threshold: float = 0.04,
    expert_only: bool = False,
) -> list[LocalGoalSpec]:
    """Sample mainline-style fixed-offset goals without crossing episodes."""
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
        expected_offsets = np.r_[0, np.cumsum(lengths[:-1])]
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError("episode offsets must be contiguous")
        total = int(lengths.sum())
        if h5["fixed_task_id"].shape != (total,) or not np.all(
            h5["fixed_task_id"][:] == task.task_id
        ):
            raise ValueError("dataset task ID mismatch")
        source_policy = None
        if expert_only:
            if "source_policy" not in h5 or h5["source_policy"].shape != (total,):
                raise ValueError("expert-only sampling requires source_policy")
            source_policy = np.asarray(h5["source_policy"][:])

        positions = []
        for cube_id in range(task.num_cubes):
            key = f"privileged/block_{cube_id}_pos"
            if key not in h5 or h5[key].shape != (total, 3):
                raise ValueError(f"{key} shape mismatch")
            value = np.asarray(h5[key][:], dtype=np.float64)
            if not np.isfinite(value).all():
                raise ValueError(f"{key} contains nonfinite values")
            positions.append(value)

    episode_parts = []
    start_parts = []
    row_parts = []
    displacement_parts = []
    immediate_parts = []
    for episode_index, (offset, length) in enumerate(zip(offsets, lengths)):
        offset, length = int(offset), int(length)
        if length <= goal_offset:
            continue
        if expert_only:
            episode_policy = source_policy[offset : offset + length]
            if not np.all(episode_policy == episode_policy[0]):
                raise ValueError(f"source_policy changes inside episode {episode_index}")
            if int(episode_policy[0]) != 1:
                continue
        valid = length - goal_offset
        distances = np.stack(
            [
                np.linalg.norm(
                    pos[offset + goal_offset : offset + length]
                    - pos[offset : offset + valid],
                    axis=1,
                )
                for pos in positions
            ],
            axis=1,
        )
        episode_parts.append(np.full(valid, episode_index, dtype=np.int64))
        start_parts.append(np.arange(valid, dtype=np.int64))
        row_parts.append(offset + np.arange(valid, dtype=np.int64))
        displacement_parts.append(distances.max(axis=1))
        immediate_parts.append(np.all(distances <= success_threshold, axis=1))

    if not row_parts:
        raise ValueError("goal offset leaves no valid windows")
    episode_indices = np.concatenate(episode_parts)
    start_steps = np.concatenate(start_parts)
    start_rows = np.concatenate(row_parts)
    max_displacements = np.concatenate(displacement_parts)
    start_already_success = np.concatenate(immediate_parts)
    eligible = max_displacements > success_threshold if nontrivial_only else np.ones(
        len(start_rows), dtype=bool
    )
    eligible_indices = np.flatnonzero(eligible)
    if count > len(eligible_indices):
        raise ValueError(
            f"requested {count} windows but only {len(eligible_indices)} are eligible"
        )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible_indices, size=count, replace=False)
    chosen = chosen[np.argsort(start_rows[chosen])]
    specs = []
    for sample_index, candidate in enumerate(chosen):
        episode_index = int(episode_indices[candidate])
        specs.append(
            LocalGoalSpec(
                sample_index=sample_index,
                episode_index=episode_index,
                offset=int(offsets[episode_index]),
                length=int(lengths[episode_index]),
                start_step=int(start_steps[candidate]),
                goal_offset=int(goal_offset),
                max_cube_displacement=float(max_displacements[candidate]),
                start_already_success=bool(start_already_success[candidate]),
                nontrivial=bool(max_displacements[candidate] > success_threshold),
            )
        )
    return specs
