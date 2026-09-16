"""Semantic, deterministic point-cloud sampling for ManiSkill PickCube."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ObjectAwareSamplingError(ValueError):
    """Raised when a frame cannot satisfy the object-aware point contract."""

    def __init__(self, message: str, *, episode=None, frame=None, pools=None):
        self.episode = episode
        self.frame = frame
        self.pools = dict(pools or {})
        context = []
        if episode is not None:
            context.append(f"episode={episode}")
        if frame is not None:
            context.append(f"frame={frame}")
        if self.pools:
            context.append(f"pools={self.pools}")
        suffix = f" ({', '.join(context)})" if context else ""
        super().__init__(message + suffix)


@dataclass(frozen=True)
class SamplingProtocol:
    """Parameters shared by replay and live CEM sampling."""

    num_points: int = 1200
    cube_min: int = 300
    near_tcp_points: int = 300
    table_max: int = 120
    voxel_size: float = 0.0025


def _as_robot_id_set(robot_ids) -> set[int]:
    if np.isscalar(robot_ids):
        return {int(robot_ids)}
    return {int(value) for value in robot_ids}


def _deduplicate_by_voxel(
    points: np.ndarray,
    colors: np.ndarray,
    group_ids: np.ndarray,
    voxel_size: float,
    semantic_priority: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep one deterministic representative per voxel by semantic priority."""
    voxel_keys = np.floor(points / voxel_size).astype(np.int64)
    source_order = np.arange(len(points), dtype=np.int64)
    order = np.lexsort(
        (source_order, semantic_priority, voxel_keys[:, 2], voxel_keys[:, 1], voxel_keys[:, 0])
    )
    ordered_keys = voxel_keys[order]
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = np.any(ordered_keys[1:] != ordered_keys[:-1], axis=1)
    selected = order[keep]
    selected.sort()
    return points[selected], colors[selected], group_ids[selected]


def sample_object_aware_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    group_ids: np.ndarray,
    tcp_xyz: np.ndarray,
    cube_id: int,
    robot_ids,
    table_id: int,
    seed: int,
    num_points: int = 1200,
    cube_min: int = 300,
    near_tcp_points: int = 300,
    table_max: int = 120,
    voxel_size: float = 0.0025,
    bounds: tuple[tuple[float, float], ...] | None = None,
    episode=None,
    frame=None,
    allow_cube_replacement: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a fixed-size semantic cloud without replacement.

    The mandatory Cube and near-TCP robot quotas are lower-bound/exact
    constraints. Table points are capped at ``table_max``; any residual slots
    are filled by remaining robot points and then unused Cube points.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    group_ids = np.asarray(group_ids).reshape(-1)
    tcp_xyz = np.asarray(tcp_xyz, dtype=np.float32).reshape(3)
    if len(points) != len(colors) or len(points) != len(group_ids):
        raise ValueError("points, colors, and group_ids must have equal length")
    if num_points <= 0 or cube_min < 0 or near_tcp_points < 0 or table_max < 0:
        raise ValueError("sampling quotas must be non-negative and num_points > 0")
    if cube_min + near_tcp_points > num_points:
        raise ValueError("mandatory quotas exceed num_points")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    valid = np.isfinite(points).all(axis=1)
    if bounds is not None:
        if len(bounds) != 3:
            raise ValueError("bounds must contain three xyz intervals")
        for axis, (lower, upper) in enumerate(bounds):
            valid &= (points[:, axis] >= lower) & (points[:, axis] <= upper)
    points, colors, group_ids = points[valid], colors[valid], group_ids[valid]
    if not len(points):
        raise ObjectAwareSamplingError(
            "point cloud contains no valid points", episode=episode, frame=frame
        )

    robot_id_set = _as_robot_id_set(robot_ids)
    priority = np.full(len(group_ids), 3, dtype=np.uint8)
    priority[np.isin(group_ids, list(robot_id_set))] = 1
    priority[group_ids == table_id] = 2
    priority[group_ids == cube_id] = 0
    points, colors, group_ids = _deduplicate_by_voxel(
        points, colors, group_ids, voxel_size, priority
    )
    cube_idx = np.flatnonzero(group_ids == cube_id)
    robot_idx = np.flatnonzero(np.isin(group_ids, list(robot_id_set)))
    table_idx = np.flatnonzero(group_ids == table_id)
    pools = {
        "cube": int(cube_idx.size),
        "robot": int(robot_idx.size),
        "table": int(table_idx.size),
        "unique": int(len(points)),
    }
    if cube_idx.size < cube_min and not allow_cube_replacement:
        raise ObjectAwareSamplingError(
            f"Cube pool has {cube_idx.size} unique points, requires at least {cube_min}",
            episode=episode,
            frame=frame,
            pools=pools,
        )
    if robot_idx.size < near_tcp_points:
        raise ObjectAwareSamplingError(
            f"near-TCP robot pool has {robot_idx.size} unique points, requires {near_tcp_points}",
            episode=episode,
            frame=frame,
            pools=pools,
        )

    rng = np.random.default_rng(seed)
    mandatory_cube = rng.choice(
        cube_idx, size=cube_min, replace=allow_cube_replacement and cube_idx.size < cube_min
    )
    mandatory_cube_set = set(int(index) for index in mandatory_cube)

    distances = np.linalg.norm(points[robot_idx] - tcp_xyz, axis=1)
    order = np.lexsort((robot_idx, distances))
    near_robot = robot_idx[order[:near_tcp_points]]
    near_robot_set = set(int(index) for index in near_robot)
    remaining_robot = np.asarray(
        [index for index in robot_idx if int(index) not in near_robot_set], dtype=np.int64
    )

    table_count = min(table_max, int(table_idx.size), num_points - cube_min - near_tcp_points)
    table_selected = (
        rng.choice(table_idx, size=table_count, replace=False)
        if table_count
        else np.empty(0, dtype=np.int64)
    )
    remaining_slots = num_points - cube_min - near_tcp_points - table_count
    body_count = min(remaining_slots, int(remaining_robot.size))
    body_selected = (
        rng.choice(remaining_robot, size=body_count, replace=False)
        if body_count
        else np.empty(0, dtype=np.int64)
    )
    remaining_slots -= body_count

    unused_cube = np.asarray(
        [index for index in cube_idx if int(index) not in mandatory_cube_set], dtype=np.int64
    )
    extra_cube_count = min(remaining_slots, int(unused_cube.size))
    extra_cube = (
        rng.choice(unused_cube, size=extra_cube_count, replace=False)
        if extra_cube_count
        else np.empty(0, dtype=np.int64)
    )
    remaining_slots -= extra_cube_count
    if remaining_slots:
        raise ObjectAwareSamplingError(
            f"cannot produce {num_points} unique points; {remaining_slots} slots remain",
            episode=episode,
            frame=frame,
            pools=pools,
        )

    selected = np.concatenate([mandatory_cube, near_robot, body_selected, table_selected, extra_cube])
    labels = np.concatenate(
        [
            np.full(len(mandatory_cube), 1, dtype=np.uint8),
            np.full(len(near_robot), 2, dtype=np.uint8),
            np.full(len(body_selected), 3, dtype=np.uint8),
            np.full(len(table_selected), 4, dtype=np.uint8),
            np.full(len(extra_cube), 1, dtype=np.uint8),
        ]
    )
    permutation = rng.permutation(len(selected))
    selected = selected[permutation]
    labels = labels[permutation]
    return points[selected].astype(np.float32), colors[selected].astype(np.uint8), labels
