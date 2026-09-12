"""Recover point-aligned RGB for the existing compact LiftCube trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np


XYZ_LOW = np.array([-0.75, -0.60, -0.05], dtype=np.float32)
XYZ_HIGH = np.array([0.75, 0.60, 0.80], dtype=np.float32)
SAMPLING_SEED = 3072
SOURCE_FIELDS = ("xyz", "action", "object_pos", "goal_pos", "success")


def aligned_sample_indices(
    xyzw: np.ndarray,
    robot_seg: np.ndarray,
    episode_id: int,
    frame_id: int,
    point_count: int = 1024,
) -> np.ndarray:
    """Return raw point indices using the historical compact-data contract."""
    xyzw = np.asarray(xyzw, dtype=np.float32)
    robot = np.asarray(robot_seg, dtype=bool)
    if xyzw.ndim != 2 or xyzw.shape[1] != 4:
        raise ValueError(f"xyzw must be [N,4], got {xyzw.shape}")
    if robot.ndim == 2 and robot.shape[1] == 1:
        robot = robot[:, 0]
    if robot.shape != (len(xyzw),):
        raise ValueError(f"robot_seg must be [{len(xyzw)}], got {robot.shape}")
    xyz = xyzw[:, :3]
    valid = (
        np.isfinite(xyzw).all(axis=1)
        & (xyzw[:, 3] > 0)
        & (xyz >= XYZ_LOW).all(axis=1)
        & (xyz <= XYZ_HIGH).all(axis=1)
        & ~robot
    )
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < point_count:
        raise ValueError(
            f"episode {episode_id} frame {frame_id} has {len(valid_indices)} "
            f"valid scene points; {point_count} required"
        )
    rng = np.random.default_rng(
        SAMPLING_SEED + 1_000_003 * int(episode_id) + int(frame_id)
    )
    local = np.sort(rng.choice(len(valid_indices), point_count, replace=False))
    return valid_indices[local]


def extract_aligned_frame(
    xyzw: np.ndarray,
    rgb: np.ndarray,
    robot_seg: np.ndarray,
    episode_id: int,
    frame_id: int,
    point_count: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather XYZ and RGB with one shared raw-point index array."""
    rgb = np.asarray(rgb)
    if rgb.shape != (len(xyzw), 3) or rgb.dtype != np.uint8:
        raise ValueError(
            f"rgb must be uint8 [{len(xyzw)},3], got {rgb.shape} {rgb.dtype}"
        )
    indices = aligned_sample_indices(
        xyzw, robot_seg, episode_id, frame_id, point_count
    )
    return (
        np.ascontiguousarray(np.asarray(xyzw, dtype=np.float32)[indices, :3]),
        np.ascontiguousarray(rgb[indices], dtype=np.uint8),
    )


def _load_episode(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing compact episode: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def prepare_episode(
    replay_group: h5py.Group,
    source: Mapping[str, np.ndarray],
    *,
    episode_id: int,
    point_count: int = 1024,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Recover one episode and prove that its regenerated XYZ is unchanged."""
    missing = [key for key in SOURCE_FIELDS if key not in source]
    if missing:
        raise ValueError(f"source compact episode is missing fields: {missing}")
    required = (
        "actions",
        "obs/pointcloud/xyzw",
        "obs/pointcloud/rgb",
        "obs/pointcloud/robot_seg",
    )
    absent = [key for key in required if key not in replay_group]
    if absent:
        raise ValueError(f"replay episode is missing datasets: {absent}")

    clouds = replay_group["obs/pointcloud/xyzw"]
    colors = replay_group["obs/pointcloud/rgb"]
    robot_masks = replay_group["obs/pointcloud/robot_seg"]
    frame_count = int(np.asarray(source["xyz"]).shape[0])
    if len(clouds) != frame_count or len(colors) != frame_count:
        raise ValueError(f"episode {episode_id} replay frame count differs from compact")
    if len(robot_masks) != frame_count:
        raise ValueError(f"episode {episode_id} robot mask frame count differs")
    replay_actions = np.asarray(replay_group["actions"], dtype=np.float32)
    if not np.array_equal(replay_actions, np.asarray(source["action"])):
        raise ValueError(f"episode {episode_id} replay actions differ from compact")

    xyz_frames: list[np.ndarray] = []
    rgb_frames: list[np.ndarray] = []
    for frame_id in range(frame_count):
        xyz, rgb = extract_aligned_frame(
            clouds[frame_id],
            colors[frame_id],
            robot_masks[frame_id],
            episode_id,
            frame_id,
            point_count,
        )
        xyz_frames.append(xyz)
        rgb_frames.append(rgb)
    regenerated_xyz = np.stack(xyz_frames)
    if not np.array_equal(regenerated_xyz, np.asarray(source["xyz"])):
        delta = float(
            np.max(np.abs(regenerated_xyz - np.asarray(source["xyz"], np.float32)))
        )
        raise ValueError(
            f"episode {episode_id} regenerated XYZ does not exactly match "
            f"the existing compact episode; max_abs_diff={delta}"
        )
    aligned_rgb = np.stack(rgb_frames).astype(np.uint8, copy=False)
    result = {key: np.asarray(source[key]) for key in SOURCE_FIELDS}
    result["rgb"] = aligned_rgb
    record = {
        "episode_id": int(episode_id),
        "frames": frame_count,
        "points_per_frame": point_count,
        "rgb_min": int(aligned_rgb.min()),
        "rgb_max": int(aligned_rgb.max()),
        "unique_rgb": int(np.unique(aligned_rgb.reshape(-1, 3), axis=0).shape[0]),
    }
    return result, record


def atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write, reopen, compare, and atomically publish one compressed episode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as file:
            np.savez_compressed(file, **arrays)
        with np.load(temporary, allow_pickle=False) as check:
            if set(check.files) != set(arrays):
                raise RuntimeError(f"field mismatch while verifying {path.name}")
            for key, expected in arrays.items():
                if not np.array_equal(check[key], expected):
                    raise RuntimeError(
                        f"field {key!r} differs while verifying {path.name}"
                    )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_root(
    replay_h5: str | Path,
    source_compact: str | Path,
    output_root: str | Path,
    *,
    verify_only: bool = False,
) -> dict[str, object]:
    """Verify or publish all episodes named by the existing compact manifest."""
    replay_h5 = Path(replay_h5)
    source_compact = Path(source_compact)
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not verify_only:
        raise FileExistsError(f"output root is not empty: {output_root}")
    manifest_path = source_compact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != "LiftCube-v0":
        raise ValueError(f"expected LiftCube-v0 manifest, got {manifest.get('task')!r}")
    episode_ids = [int(value) for value in manifest["episode_ids"]]
    point_count = int(manifest.get("point_count", 1024))
    records: list[dict[str, int]] = []
    if not verify_only:
        output_root.mkdir(parents=True, exist_ok=True)

    with h5py.File(replay_h5, "r") as replay:
        for episode_id in episode_ids:
            group_name = f"traj_{episode_id}"
            if group_name not in replay:
                raise ValueError(f"replay is missing {group_name}")
            source = _load_episode(
                source_compact / f"episode_{episode_id:04d}.npz"
            )
            arrays, record = prepare_episode(
                replay[group_name],
                source,
                episode_id=episode_id,
                point_count=point_count,
            )
            records.append(record)
            if not verify_only:
                atomic_save_npz(
                    output_root / f"episode_{episode_id:04d}.npz", arrays
                )

    replay_stat = replay_h5.stat()
    result = dict(manifest)
    result.update(
        {
            "point_features": ["xyz", "rgb"],
            "rgb_dtype": "uint8",
            "rgb_normalization": "float32(rgb) / 255",
            "rgb_alignment": "same validity mask and sample indices as xyz",
            "source_compact": str(source_compact),
            "source_manifest_sha256": _sha256(manifest_path),
            "source_replay_h5": str(replay_h5),
            "source_replay_size": int(replay_stat.st_size),
            "source_replay_mtime_ns": int(replay_stat.st_mtime_ns),
            "episodes": records,
        }
    )
    if not verify_only:
        _atomic_write_json(output_root / "manifest.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-h5", type=Path, required=True)
    parser.add_argument("--source-compact", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_root(
        args.replay_h5,
        args.source_compact,
        args.output_root,
        verify_only=args.verify_only,
    )
    print(
        json.dumps(
            {
                "verified": len(result["episode_ids"]),
                "output_root": str(args.output_root),
                "published": not args.verify_only,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
