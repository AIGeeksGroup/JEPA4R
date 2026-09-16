#!/usr/bin/env python3
"""Merge resumable PickCube NPZ episodes into one training HDF5 file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from maniskill_pickcube import load_episode_shard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-json", required=True, type=Path)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--output-h5", required=True, type=Path)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--num-episodes", required=True, type=int)
    parser.add_argument("--num-points", type=int, default=1200)
    parser.add_argument(
        "--sampling-protocol",
        choices=("uniform_v1", "object_aware_v2"),
        default="uniform_v1",
    )
    parser.add_argument("--extra-resolution", type=int, default=None)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_h5.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {args.output_h5}; pass --overwrite")
    metadata = json.loads(args.metadata_json.read_text())
    selected = metadata["episodes"][
        args.start_episode : args.start_episode + args.num_episodes
    ]
    if len(selected) != args.num_episodes:
        raise ValueError(
            f"requested {args.num_episodes} episodes from {args.start_episode}, "
            f"but metadata only provided {len(selected)}"
        )

    lengths = np.asarray(
        [int(episode["elapsed_steps"]) + 1 for episode in selected], dtype=np.int32
    )
    offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64)]
    )
    total_frames = int(lengths.sum())
    first_id = int(selected[0]["episode_id"])
    first = load_episode_shard(
        args.shard_dir / f"episode_{first_id:04d}.npz",
        int(lengths[0]),
        args.num_points,
    )
    if args.sampling_protocol == "object_aware_v2":
        if "pcd_group" not in first:
            raise ValueError("object_aware_v2 shards must contain pcd_group")
        if args.extra_resolution is None or args.voxel_size is None:
            raise ValueError(
                "object_aware_v2 requires --extra-resolution and --voxel-size"
            )
    elif args.extra_resolution is not None or args.voxel_size is not None:
        raise ValueError(
            "--extra-resolution and --voxel-size are only valid for object_aware_v2"
        )

    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output_h5.with_suffix(args.output_h5.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        with h5py.File(tmp_path, "w", libver="latest") as output:
            datasets = {}
            for key, values in first.items():
                row_bytes = max(1, int(np.prod(values.shape[1:])) * values.dtype.itemsize)
                chunk_rows = min(total_frames, max(1, min(64, (2 * 1024 * 1024) // row_bytes)))
                datasets[key] = output.create_dataset(
                    key,
                    shape=(total_frames, *values.shape[1:]),
                    dtype=values.dtype,
                    chunks=(chunk_rows, *values.shape[1:]),
                    compression="lzf",
                )
            output.create_dataset("ep_len", data=lengths)
            output.create_dataset("ep_offset", data=offsets)
            output.attrs["env_id"] = metadata["env_info"]["env_id"]
            output.attrs["control_mode"] = metadata["env_info"]["env_kwargs"][
                "control_mode"
            ]
            output.attrs["start_episode"] = args.start_episode
            output.attrs["num_episodes"] = args.num_episodes
            output.attrs["num_points"] = args.num_points
            output.attrs["schema_version"] = (
                2 if args.sampling_protocol == "object_aware_v2" else 1
            )
            output.attrs["sampling_protocol"] = args.sampling_protocol
            if args.sampling_protocol == "object_aware_v2":
                output.attrs["cube_min_points"] = max(
                    1, int(round(0.25 * args.num_points))
                )
                output.attrs["extra_resolution"] = args.extra_resolution
                output.attrs["voxel_size"] = args.voxel_size
                output.attrs["nominal_group_counts"] = json.dumps(
                    {"cube": 300, "near_tcp_robot": 300, "robot": 480, "table": 120},
                    sort_keys=True,
                )

            for index, episode in enumerate(selected):
                episode_id = int(episode["episode_id"])
                arrays = load_episode_shard(
                    args.shard_dir / f"episode_{episode_id:04d}.npz",
                    int(lengths[index]),
                    args.num_points,
                )
                if args.sampling_protocol == "object_aware_v2":
                    cube_count = np.count_nonzero(arrays["pcd_group"] == 1, axis=1)
                    cube_min = max(1, int(round(0.25 * args.num_points)))
                    if np.any(cube_count < cube_min):
                        raise ValueError(
                            f"episode {episode_id}: Cube count below {cube_min}"
                        )
                if arrays.keys() != first.keys():
                    raise ValueError(
                        f"episode {episode_id}: schema differs from episode {first_id}"
                    )
                start = int(offsets[index])
                end = start + int(lengths[index])
                for key, values in arrays.items():
                    if values.shape[1:] != datasets[key].shape[1:]:
                        raise ValueError(
                            f"episode {episode_id}: {key} shape {values.shape[1:]} "
                            f"differs from {datasets[key].shape[1:]}"
                        )
                    datasets[key][start:end] = values
            output.flush()

        os.replace(tmp_path, args.output_h5)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    with h5py.File(args.output_h5, "r") as output:
        if not np.array_equal(output["ep_len"][:], lengths):
            raise ValueError("written ep_len does not match source episodes")
        if not np.array_equal(output["ep_offset"][:], offsets):
            raise ValueError("written ep_offset does not match source episodes")
        for key in first:
            if len(output[key]) != total_frames:
                raise ValueError(f"written {key} has {len(output[key])} rows")

    digest = hashlib.sha256()
    with args.output_h5.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output_h5),
                "episodes": args.num_episodes,
                "frames": total_frames,
                "bytes": args.output_h5.stat().st_size,
                "sha256": digest.hexdigest(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
