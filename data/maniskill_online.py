"""Online ManiSkill point-cloud preprocessing compatible with compact data."""

from __future__ import annotations

import numpy as np

from data.maniskill import _sample_points


SCENE_POINT_COUNT = 1024
XYZ_LOW = np.array([-0.75, -0.60, -0.05], dtype=np.float32)
XYZ_HIGH = np.array([0.75, 0.60, 0.80], dtype=np.float32)
_ONLINE_SEED = 3072


def preprocess_online_cloud(
    xyzw: np.ndarray,
    robot_seg: np.ndarray,
    *,
    sample_key: int,
    point_count: int,
    rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Filter one world-space observation and sample XYZ or aligned XYZ+RGB."""
    xyzw = np.asarray(xyzw, dtype=np.float32)
    if xyzw.ndim != 2 or xyzw.shape[1] != 4:
        raise ValueError(f"xyzw must have shape [N,4], got {xyzw.shape}")

    robot_seg = np.asarray(robot_seg, dtype=bool)
    if robot_seg.ndim == 2 and robot_seg.shape[1] == 1:
        robot_seg = robot_seg[:, 0]
    if robot_seg.shape != (len(xyzw),):
        raise ValueError(
            f"robot_seg must have shape [{len(xyzw)}] or [{len(xyzw)},1], "
            f"got {robot_seg.shape}"
        )
    if rgb is not None:
        rgb = np.asarray(rgb)
        if rgb.shape != (len(xyzw), 3) or rgb.dtype != np.uint8:
            raise ValueError(
                f"rgb must have shape [{len(xyzw)},3] and dtype uint8, got "
                f"{rgb.shape} {rgb.dtype}"
            )
    if not 1 <= point_count <= SCENE_POINT_COUNT:
        raise ValueError(f"point_count must be in [1,{SCENE_POINT_COUNT}]")
    if sample_key < 0:
        raise ValueError("sample_key must be non-negative")

    xyz = xyzw[:, :3]
    finite = np.isfinite(xyzw).all(axis=1)
    in_bounds = np.logical_and(xyz >= XYZ_LOW, xyz <= XYZ_HIGH).all(axis=1)
    valid = finite & (xyzw[:, 3] > 0) & in_bounds & ~robot_seg
    scene_xyz = xyz[valid]
    scene_rgb = None if rgb is None else rgb[valid]
    if len(scene_xyz) < SCENE_POINT_COUNT:
        raise ValueError(
            f"frame has {len(scene_xyz)} valid non-robot points; "
            f"at least {SCENE_POINT_COUNT} are required"
        )

    rng = np.random.default_rng(_ONLINE_SEED + int(sample_key))
    indices = np.sort(
        rng.choice(len(scene_xyz), size=SCENE_POINT_COUNT, replace=False)
    )
    compact_xyz = np.ascontiguousarray(scene_xyz[indices], dtype=np.float32)
    if scene_rgb is None:
        return _sample_points(
            compact_xyz,
            point_count,
            episode_id=int(sample_key),
            frame_id=0,
        )
    compact_rgb = np.ascontiguousarray(scene_rgb[indices], dtype=np.float32) / 255.0
    return _sample_points(
        np.concatenate([compact_xyz, compact_rgb], axis=-1),
        point_count,
        episode_id=int(sample_key),
        frame_id=0,
    )
