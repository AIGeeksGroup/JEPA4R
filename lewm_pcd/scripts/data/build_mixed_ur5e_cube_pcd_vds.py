"""Build no-copy mixed UR5e 1cube/2cube point-cloud HDF5 datasets."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


COMMON_KEYS = ("pcd_xyz", "pcd_rgb", "action")


@dataclass(frozen=True)
class SourceInfo:
    path: Path
    lengths: np.ndarray
    offsets: np.ndarray
    shapes: dict[str, tuple[int, ...]]
    dtypes: dict[str, np.dtype]
    robot_shapes: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class Segment:
    source: SourceInfo
    episode_start: int
    episode_stop: int
    source_id: int


def _inspect(path: Path) -> SourceInfo:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as h5:
        robot_keys = ("qpos", "qvel", "control")
        missing = [
            key
            for key in ("ep_len", "ep_offset", *COMMON_KEYS, *robot_keys)
            if key not in h5
        ]
        if missing:
            raise ValueError(f"{path} missing datasets: {missing}")
        lengths = np.asarray(h5["ep_len"][:], dtype=np.int64)
        offsets = np.asarray(h5["ep_offset"][:], dtype=np.int64)
        expected = np.concatenate(([0], np.cumsum(lengths[:-1]))).astype(np.int64)
        if lengths.ndim != 1 or len(lengths) == 0 or np.any(lengths <= 0):
            raise ValueError(f"{path} has invalid episode lengths")
        if not np.array_equal(offsets, expected):
            raise ValueError(f"{path} has non-contiguous episode offsets")
        total = int(lengths.sum())
        shapes = {key: tuple(h5[key].shape) for key in COMMON_KEYS}
        dtypes = {key: h5[key].dtype for key in COMMON_KEYS}
        robot_shapes = {key: tuple(h5[key].shape) for key in robot_keys}
        for key in COMMON_KEYS:
            if shapes[key][0] != total:
                raise ValueError(
                    f"{path} {key} frame count {shapes[key][0]} != {total}"
                )
        for key in robot_keys:
            if robot_shapes[key][0] != total:
                raise ValueError(
                    f"{path} {key} frame count {robot_shapes[key][0]} != {total}"
                )
    return SourceInfo(path, lengths, offsets, shapes, dtypes, robot_shapes)


def _validate_ur5e_contract(source: SourceInfo, cube_count: int) -> None:
    expected = {
        "qpos": (14 + 7 * cube_count,),
        "qvel": (14 + 6 * cube_count,),
        "control": (7,),
    }
    for key, trailing_shape in expected.items():
        actual = source.robot_shapes[key][1:]
        if actual != trailing_shape:
            raise ValueError(
                f"{key} trailing shape {actual} does not match the OGBench UR5e "
                f"contract {trailing_shape} for {cube_count} cube(s): {source.path}"
            )


def _validate_compatible(sources: list[SourceInfo]) -> None:
    reference = sources[0]
    for source in sources[1:]:
        for key in COMMON_KEYS:
            if source.shapes[key][1:] != reference.shapes[key][1:]:
                raise ValueError(
                    f"{key} trailing shape mismatch: "
                    f"{reference.shapes[key][1:]} vs {source.shapes[key][1:]}"
                )
            if source.dtypes[key] != reference.dtypes[key]:
                raise ValueError(
                    f"{key} dtype mismatch: {reference.dtypes[key]} vs {source.dtypes[key]}"
                )


def _segment_bounds(segment: Segment) -> tuple[int, int]:
    if not 0 <= segment.episode_start < segment.episode_stop <= len(segment.source.lengths):
        raise ValueError(f"invalid episode range for {segment.source.path}")
    start = int(segment.source.offsets[segment.episode_start])
    last = segment.episode_stop - 1
    stop = int(segment.source.offsets[last] + segment.source.lengths[last])
    return start, stop


def _write_vds(path: Path, segments: list[Segment]) -> dict:
    selected_lengths = [
        segment.source.lengths[segment.episode_start : segment.episode_stop]
        for segment in segments
    ]
    lengths = np.concatenate(selected_lengths).astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1]))).astype(np.int64)
    episode_source = np.concatenate(
        [
            np.full(segment.episode_stop - segment.episode_start, segment.source_id, dtype=np.uint8)
            for segment in segments
        ]
    )
    source_episode_index = np.concatenate(
        [
            np.arange(segment.episode_start, segment.episode_stop, dtype=np.int64)
            for segment in segments
        ]
    )
    total_frames = int(lengths.sum())

    partial = path.with_suffix(path.suffix + ".partial")
    with h5py.File(partial, "w", libver="latest") as h5:
        h5.attrs["format"] = "ur5e_mixed_cube_pcd_vds_v1"
        h5.attrs["virtual"] = True
        h5.create_dataset("ep_len", data=lengths)
        h5.create_dataset("ep_offset", data=offsets)
        h5.create_dataset("episode_source", data=episode_source)
        h5.create_dataset("source_episode_index", data=source_episode_index)

        for key in COMMON_KEYS:
            tail = segments[0].source.shapes[key][1:]
            dtype = segments[0].source.dtypes[key]
            layout = h5py.VirtualLayout(shape=(total_frames, *tail), dtype=dtype)
            output_start = 0
            for segment in segments:
                source_start, source_stop = _segment_bounds(segment)
                count = source_stop - source_start
                virtual_source = h5py.VirtualSource(
                    str(segment.source.path), key, shape=segment.source.shapes[key]
                )
                layout[output_start : output_start + count] = virtual_source[
                    source_start:source_stop
                ]
                output_start += count
            h5.create_virtual_dataset(key, layout)
        h5.flush()
    os.replace(partial, path)
    counts = {
        str(segment.source_id): segment.episode_stop - segment.episode_start
        for segment in segments
    }
    return {"episodes": int(len(lengths)), "frames": total_frames, "source_episode_counts": counts}


def build_mixed_datasets(
    *,
    one_cube: Path,
    two_cube_train: Path,
    two_cube_holdout: Path,
    output_dir: Path,
    one_cube_validation_episodes: int = 100,
) -> dict:
    """Create episode-disjoint train/validation VDS files and a manifest."""

    one = _inspect(Path(one_cube))
    two_train = _inspect(Path(two_cube_train))
    two_holdout = _inspect(Path(two_cube_holdout))
    _validate_ur5e_contract(one, cube_count=1)
    _validate_ur5e_contract(two_train, cube_count=2)
    _validate_ur5e_contract(two_holdout, cube_count=2)
    _validate_compatible([one, two_train, two_holdout])

    val_count = int(one_cube_validation_episodes)
    if not 0 < val_count < len(one.lengths):
        raise ValueError("one_cube_validation_episodes must leave non-empty train and validation splits")
    split = len(one.lengths) - val_count

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / "train.h5", output_dir / "validation.h5", output_dir / "manifest.json"]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")

    train_segments = [
        Segment(one, 0, split, 0),
        Segment(two_train, 0, len(two_train.lengths), 1),
    ]
    validation_segments = [
        Segment(one, split, len(one.lengths), 0),
        Segment(two_holdout, 0, len(two_holdout.lengths), 2),
    ]
    train_summary = _write_vds(outputs[0], train_segments)
    validation_summary = _write_vds(outputs[1], validation_segments)
    manifest = {
        "format": "ur5e_mixed_cube_pcd_vds_v1",
        "robot": "UR5e + Robotiq 2F-85",
        "common_keys": list(COMMON_KEYS),
        "sources": {
            "0": str(one.path),
            "1": str(two_train.path),
            "2": str(two_holdout.path),
        },
        "one_cube_train_episode_range": [0, split],
        "one_cube_validation_episode_range": [split, len(one.lengths)],
        "train": train_summary,
        "validation": validation_summary,
    }
    outputs[2].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one-cube", required=True, type=Path)
    parser.add_argument("--two-cube-train", required=True, type=Path)
    parser.add_argument("--two-cube-holdout", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--one-cube-validation-episodes", type=int, default=100)
    args = parser.parse_args()
    manifest = build_mixed_datasets(**vars(args))
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
