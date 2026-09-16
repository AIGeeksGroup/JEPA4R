#!/usr/bin/env python3
"""Validate replayed PickCube episode shards before merge or training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


DATA_SCRIPTS = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(DATA_SCRIPTS))

from maniskill_pickcube import load_episode_shard  # noqa: E402


WORKSPACE_BOUNDS = ((-0.8, 0.5), (-0.6, 0.6), (-0.05, 0.8))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-json", required=True, type=Path)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--expected-episodes", required=True, type=int)
    parser.add_argument("--num-points", type=int, default=1200)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    metadata = json.loads(args.metadata_json.read_text())
    selected = metadata["episodes"][
        args.start_episode : args.start_episode + args.expected_episodes
    ]
    if len(selected) != args.expected_episodes:
        raise ValueError(
            f"requested {args.expected_episodes} episodes from {args.start_episode}, "
            f"but metadata only provided {len(selected)}"
        )

    total_frames = 0
    total_bytes = 0
    pcd_min = np.full(3, np.inf, dtype=np.float64)
    pcd_max = np.full(3, -np.inf, dtype=np.float64)
    rgb_range = [255, 0]
    depth_range = [np.iinfo(np.int16).max, np.iinfo(np.int16).min]
    files = []

    for episode in selected:
        episode_id = int(episode["episode_id"])
        expected_frames = int(episode["elapsed_steps"]) + 1
        path = args.shard_dir / f"episode_{episode_id:04d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing episode shard: {path}")
        arrays = load_episode_shard(path, expected_frames, args.num_points)

        expected_shapes = {
            "rgb": (expected_frames, 128, 128, 3),
            "depth": (expected_frames, 128, 128, 1),
            "pcd_xyz": (expected_frames, args.num_points, 3),
            "pcd_rgb": (expected_frames, args.num_points, 3),
            "proprio": (expected_frames, 25),
            "action": (expected_frames, 8),
            "cube_pose": (expected_frames, 7),
            "goal_pos": (expected_frames, 3),
        }
        for key, shape in expected_shapes.items():
            if key not in arrays or arrays[key].shape != shape:
                got = None if key not in arrays else arrays[key].shape
                raise ValueError(f"episode {episode_id}: {key} shape {got}, expected {shape}")

        expected_dtypes = {
            "rgb": np.uint8,
            "depth": np.int16,
            "pcd_xyz": np.float32,
            "pcd_rgb": np.uint8,
            "proprio": np.float32,
            "action": np.float32,
            "cube_pose": np.float32,
            "goal_pos": np.float32,
        }
        for key, dtype in expected_dtypes.items():
            if arrays[key].dtype != dtype:
                raise ValueError(
                    f"episode {episode_id}: {key} dtype {arrays[key].dtype}, expected {dtype}"
                )

        for key in ("pcd_xyz", "proprio", "cube_pose", "goal_pos"):
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"episode {episode_id}: non-finite {key}")
        if not np.array_equal(arrays["episode_idx"], np.full(expected_frames, episode_id)):
            raise ValueError(f"episode {episode_id}: incorrect episode_idx")
        if not np.array_equal(arrays["step_idx"], np.arange(expected_frames)):
            raise ValueError(f"episode {episode_id}: incorrect step_idx")
        if bool(arrays["source_success"][-1]) != bool(episode["success"]):
            raise ValueError(f"episode {episode_id}: source terminal success mismatch")
        if bool(arrays["success"][-1]) != bool(arrays["source_success"][-1]):
            raise ValueError(f"episode {episode_id}: replay terminal success mismatch")

        xyz = arrays["pcd_xyz"]
        for axis, (lower, upper) in enumerate(WORKSPACE_BOUNDS):
            if xyz[..., axis].min() < lower - 1e-5 or xyz[..., axis].max() > upper + 1e-5:
                raise ValueError(f"episode {episode_id}: points exceed workspace bounds")
        pcd_min = np.minimum(pcd_min, xyz.min(axis=(0, 1)))
        pcd_max = np.maximum(pcd_max, xyz.max(axis=(0, 1)))
        rgb_range[0] = min(rgb_range[0], int(arrays["rgb"].min()))
        rgb_range[1] = max(rgb_range[1], int(arrays["rgb"].max()))
        depth_range[0] = min(depth_range[0], int(arrays["depth"].min()))
        depth_range[1] = max(depth_range[1], int(arrays["depth"].max()))
        total_frames += expected_frames
        total_bytes += path.stat().st_size

        file_info = {"path": path.name, "bytes": path.stat().st_size}
        if args.write_manifest:
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            file_info["sha256"] = digest.hexdigest()
        files.append(file_info)

    summary = {
        "status": "PASS",
        "episodes": len(selected),
        "frames": total_frames,
        "bytes": total_bytes,
        "pcd_shape_per_frame": [args.num_points, 3],
        "pcd_min": pcd_min.tolist(),
        "pcd_max": pcd_max.tolist(),
        "rgb_range": rgb_range,
        "depth_mm_range": depth_range,
        "files": files,
    }
    summary_path = args.summary_json or args.shard_dir / "validation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(summary, indent=2))
    tmp_path.replace(summary_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
