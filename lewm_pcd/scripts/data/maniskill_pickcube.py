"""Shared data-contract helpers for ManiSkill PickCube replay."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def align_actions_to_states(actions: np.ndarray, num_states: int) -> np.ndarray:
    """Align T actions with T+1 rendered states using one terminal NaN row."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"actions must be rank 2, got {actions.shape}")
    if num_states != len(actions) + 1:
        raise ValueError(
            f"expected one more state than actions, got {num_states} states "
            f"and {len(actions)} actions"
        )
    terminal = np.full((1, actions.shape[1]), np.nan, dtype=np.float32)
    return np.concatenate([actions, terminal], axis=0)


def sample_point_cloud(
    xyzw: np.ndarray,
    rgb: np.ndarray,
    num_points: int,
    seed: int,
    bounds: tuple[tuple[float, float], ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a deterministic fixed-size mixed scene/robot point cloud."""
    xyzw = np.asarray(xyzw, dtype=np.float32).reshape(-1, 4)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    if len(xyzw) != len(rgb):
        raise ValueError(f"xyzw/rgb length mismatch: {len(xyzw)} != {len(rgb)}")
    valid = (xyzw[:, 3] > 0.5) & np.isfinite(xyzw[:, :3]).all(axis=1)
    if bounds is not None:
        if len(bounds) != 3:
            raise ValueError(f"bounds must contain xyz intervals, got {bounds}")
        for axis, (lower, upper) in enumerate(bounds):
            valid &= (xyzw[:, axis] >= lower) & (xyzw[:, axis] <= upper)
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise ValueError("point cloud contains no valid finite points")
    rng = np.random.default_rng(seed)
    selected = rng.choice(indices, size=num_points, replace=len(indices) < num_points)
    return xyzw[selected, :3].astype(np.float32), rgb[selected].astype(np.uint8)


def validate_episode_arrays(
    arrays: dict[str, np.ndarray],
    expected_frames: int,
    num_points: int,
) -> None:
    """Validate the columns written by one replayed episode."""
    required = {"rgb", "depth", "pcd_xyz", "pcd_rgb", "proprio", "action"}
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"missing episode columns: {sorted(missing)}")
    for name in arrays:
        if len(arrays[name]) != expected_frames:
            raise ValueError(
                f"{name} has {len(arrays[name])} frames, expected {expected_frames}"
            )
    if arrays["pcd_xyz"].shape != (expected_frames, num_points, 3):
        raise ValueError(f"unexpected pcd_xyz shape: {arrays['pcd_xyz'].shape}")
    if arrays["pcd_rgb"].shape != (expected_frames, num_points, 3):
        raise ValueError(f"unexpected pcd_rgb shape: {arrays['pcd_rgb'].shape}")
    if "pcd_group" in arrays:
        group = np.asarray(arrays["pcd_group"])
        if group.shape != (expected_frames, num_points):
            raise ValueError(f"pcd_group shape {group.shape} does not match {(expected_frames, num_points)}")
        if group.dtype != np.uint8:
            raise ValueError(f"pcd_group dtype {group.dtype} must be uint8")
        if not np.isin(group, np.array([1, 2, 3, 4], dtype=np.uint8)).all():
            raise ValueError("pcd_group labels must be in 1..4")
        cube_min = max(1, int(round(0.25 * num_points)))
        if np.any(np.count_nonzero(group == 1, axis=1) < cube_min):
            raise ValueError(
                f"pcd_group Cube count must be at least {cube_min} for {num_points} points"
            )
    if not np.isfinite(arrays["pcd_xyz"]).all():
        raise ValueError("pcd_xyz contains non-finite values")
    if not np.isfinite(arrays["proprio"]).all():
        raise ValueError("proprio contains non-finite values")
    actions = np.asarray(arrays["action"])
    if not np.isfinite(actions[:-1]).all() or not np.isnan(actions[-1]).all():
        raise ValueError("only the terminal action may be NaN")


def save_episode_shard(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    """Atomically write one independently recoverable compressed episode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp_path, path)


def load_episode_shard(
    path: str | Path,
    expected_frames: int,
    num_points: int,
) -> dict[str, np.ndarray]:
    """Load and validate a completed episode shard."""
    with np.load(path, allow_pickle=False) as src:
        arrays = {key: src[key] for key in src.files}
    validate_episode_arrays(arrays, expected_frames, num_points)
    return arrays
