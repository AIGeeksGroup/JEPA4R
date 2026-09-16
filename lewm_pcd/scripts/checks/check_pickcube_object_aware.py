#!/usr/bin/env python3
"""Validate object-aware PickCube shards before merge, training, or CEM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


DATA_SCRIPTS = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(DATA_SCRIPTS))

from maniskill_object_aware_render import default_camera_specs  # noqa: E402
from maniskill_pickcube import load_episode_shard  # noqa: E402


WORKSPACE_BOUNDS = ((-0.8, 0.5), (-0.6, 0.6), (-0.05, 0.8))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def validate_object_aware_shard(
    arrays: dict[str, np.ndarray],
    episode_id: int,
    num_points: int,
    voxel_size: float,
) -> dict:
    if "pcd_group" not in arrays:
        raise ValueError(f"episode {episode_id}: missing pcd_group")
    xyz = arrays["pcd_xyz"]
    group = arrays["pcd_group"]
    cube_counts = np.count_nonzero(group == 1, axis=1)
    near_counts = np.count_nonzero(group == 2, axis=1)
    body_counts = np.count_nonzero(group == 3, axis=1)
    table_counts = np.count_nonzero(group == 4, axis=1)
    if np.any(cube_counts < round(0.25 * num_points)):
        frame = int(np.flatnonzero(cube_counts < round(0.25 * num_points))[0])
        raise ValueError(f"episode {episode_id} frame {frame}: Cube below 25%")
    if np.any(near_counts != 300):
        frame = int(np.flatnonzero(near_counts != 300)[0])
        raise ValueError(f"episode {episode_id} frame {frame}: near-TCP count {near_counts[frame]}")
    if np.any(table_counts > 120):
        frame = int(np.flatnonzero(table_counts > 120)[0])
        raise ValueError(f"episode {episode_id} frame {frame}: table count {table_counts[frame]}")
    if not np.all(cube_counts + near_counts + body_counts + table_counts == num_points):
        raise ValueError(f"episode {episode_id}: semantic counts do not sum to {num_points}")

    tcp_xyz = arrays["proprio"][:, 18:21]
    for frame in range(len(xyz)):
        voxel_keys = np.floor(xyz[frame] / voxel_size).astype(np.int64)
        unique_count = np.unique(voxel_keys, axis=0).shape[0]
        if unique_count != num_points:
            raise ValueError(
                f"episode {episode_id} frame {frame}: only {unique_count} unique voxels"
            )
        near_distance = np.linalg.norm(xyz[frame, group[frame] == 2] - tcp_xyz[frame], axis=1)
        body_distance = np.linalg.norm(xyz[frame, group[frame] == 3] - tcp_xyz[frame], axis=1)
        if len(body_distance) and near_distance.max() > body_distance.min() + 1e-5:
            raise ValueError(f"episode {episode_id} frame {frame}: near-TCP ordering failed")
        for axis, (lower, upper) in enumerate(WORKSPACE_BOUNDS):
            if xyz[frame, :, axis].min() < lower - 1e-5 or xyz[frame, :, axis].max() > upper + 1e-5:
                raise ValueError(f"episode {episode_id} frame {frame}: workspace bounds exceeded")
    return {
        "frames": int(len(xyz)),
        "cube_min": int(cube_counts.min()),
        "cube_max": int(cube_counts.max()),
        "cube_mean": float(cube_counts.mean()),
        "near_tcp_min": int(near_counts.min()),
        "near_tcp_max": int(near_counts.max()),
        "body_min": int(body_counts.min()),
        "body_max": int(body_counts.max()),
        "table_min": int(table_counts.min()),
        "table_max": int(table_counts.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-json", required=True, type=Path)
    parser.add_argument("--trajectory-h5", required=True, type=Path)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--expected-episodes", required=True, type=int)
    parser.add_argument("--num-points", type=int, default=1200)
    parser.add_argument("--extra-resolution", type=int, default=384)
    parser.add_argument("--voxel-size", type=float, default=0.0025)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--manifest-json", type=Path)
    args = parser.parse_args()

    metadata = json.loads(args.metadata_json.read_text())
    selected = metadata["episodes"][args.start_episode : args.start_episode + args.expected_episodes]
    if len(selected) != args.expected_episodes:
        raise ValueError(
            f"requested {args.expected_episodes} episodes from {args.start_episode}, got {len(selected)}"
        )
    files, episode_stats = [], []
    total_frames = 0
    total_bytes = 0
    for episode in selected:
        episode_id = int(episode["episode_id"])
        expected_frames = int(episode["elapsed_steps"]) + 1
        path = args.shard_dir / f"episode_{episode_id:04d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing episode shard: {path}")
        arrays = load_episode_shard(path, expected_frames, args.num_points)
        stats = validate_object_aware_shard(
            arrays, episode_id, args.num_points, args.voxel_size
        )
        if bool(arrays["source_success"][-1]) != bool(episode["success"]):
            raise ValueError(f"episode {episode_id}: source terminal success mismatch")
        if bool(arrays["success"][-1]) != bool(arrays["source_success"][-1]):
            raise ValueError(f"episode {episode_id}: replay terminal success mismatch")
        digest = sha256_file(path)
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": digest})
        episode_stats.append({"episode": episode_id, **stats})
        total_frames += expected_frames
        total_bytes += path.stat().st_size

    summary = {
        "status": "PASS",
        "sampling_protocol": "object_aware_v2",
        "episodes": len(selected),
        "frames": total_frames,
        "bytes": total_bytes,
        "num_points": args.num_points,
        "voxel_size": args.voxel_size,
        "cube_min_points": 300,
        "near_tcp_points": 300,
        "local_table_max_points": 120,
        "extra_resolution": args.extra_resolution,
        "episode_stats": episode_stats,
        "files": files,
    }
    summary_path = args.summary_json or args.shard_dir.parent / "validation_summary.json"
    _write_json(summary_path, summary)

    if args.manifest_json:
        repo = Path(__file__).resolve().parents[3]
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        cameras = [spec.__dict__ for spec in default_camera_specs(args.extra_resolution)]
        manifest = {
            "status": "PASS",
            "schema_version": 2,
            "sampling_protocol": "object_aware_v2",
            "seed_formula": "3072 + episode_id * 1000003 + frame_idx",
            "workspace_bounds": WORKSPACE_BOUNDS,
            "voxel_size": args.voxel_size,
            "nominal_group_counts": [300, 300, 480, 120],
            "cube_min_points": 300,
            "camera_protocol": "six_view_pickcube_v1",
            "cameras": cameras,
            "source_metadata": {
                "path": str(args.metadata_json),
                "sha256": sha256_file(args.metadata_json),
            },
            "source_trajectory": {
                "path": str(args.trajectory_h5),
                "sha256": sha256_file(args.trajectory_h5),
            },
            "code_commit": commit,
            "worktree_dirty": dirty,
            "summary": str(summary_path),
            "files": files,
        }
        _write_json(args.manifest_json, manifest)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
