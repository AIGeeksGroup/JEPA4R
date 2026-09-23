"""Acceptance checks for mixed UR5e Cube point-cloud virtual datasets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np


COMMON_KEYS = ("pcd_xyz", "pcd_rgb", "action")


def _inspect(path: Path, sample_count: int, rng: np.random.Generator) -> tuple[dict, set[int]]:
    with h5py.File(path, "r") as h5:
        lengths = np.asarray(h5["ep_len"][:], dtype=np.int64)
        offsets = np.asarray(h5["ep_offset"][:], dtype=np.int64)
        expected_offsets = np.concatenate(([0], np.cumsum(lengths[:-1]))).astype(np.int64)
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError(f"{path}: non-contiguous episode offsets")
        frames = int(lengths.sum())
        for key in COMMON_KEYS:
            if key not in h5 or h5[key].shape[0] != frames:
                raise ValueError(f"{path}: invalid {key} frame count")
        if h5["action"].shape[1:] != (5,):
            raise ValueError(f"{path}: action shape must end in (5,)")

        actions = np.asarray(h5["action"][:])
        any_nan = np.isnan(actions).any(axis=1)
        all_nan = np.isnan(actions).all(axis=1)
        if not np.array_equal(any_nan, all_nan):
            raise ValueError(f"{path}: partially-NaN action row")
        terminal_rows = set((offsets + lengths - 1).tolist())
        nan_rows = set(np.flatnonzero(all_nan).tolist())
        if not nan_rows.issubset(terminal_rows):
            raise ValueError(f"{path}: NaN action outside episode terminal row")
        if not np.isfinite(actions[~all_nan]).all():
            raise ValueError(f"{path}: non-finite non-terminal action")

        count = min(int(sample_count), frames)
        sample_rows = np.sort(rng.choice(frames, size=count, replace=False))
        if not np.isfinite(h5["pcd_xyz"][sample_rows]).all():
            raise ValueError(f"{path}: non-finite sampled point cloud")
        if h5["pcd_xyz"].shape[1:] != (1200, 3) and frames > 20:
            raise ValueError(f"{path}: real point cloud shape must end in (1200, 3)")

        episode_source = np.asarray(h5["episode_source"][:], dtype=np.int64)
        source_episode_index = np.asarray(h5["source_episode_index"][:], dtype=np.int64)
        if episode_source.shape != lengths.shape or source_episode_index.shape != lengths.shape:
            raise ValueError(f"{path}: invalid episode provenance shape")
        source_counts = {
            str(int(source)): int((episode_source == source).sum())
            for source in np.unique(episode_source)
        }
        one_cube_episodes = set(source_episode_index[episode_source == 0].tolist())
        report = {
            "path": str(path),
            "episodes": int(len(lengths)),
            "frames": frames,
            "source_episode_counts": source_counts,
            "all_common_fields_virtual": all(h5[key].is_virtual for key in COMMON_KEYS),
            "pcd_xyz_shape": list(h5["pcd_xyz"].shape),
            "pcd_rgb_shape": list(h5["pcd_rgb"].shape),
            "action_shape": list(h5["action"].shape),
            "nan_action_rows": int(len(nan_rows)),
            "sampled_rows": sample_rows.tolist(),
        }
    return report, one_cube_episodes


def inspect_mixed_datasets(
    *, train_path: Path, validation_path: Path, sample_count: int = 200, seed: int = 3072
) -> dict:
    rng = np.random.default_rng(seed)
    train, train_one = _inspect(Path(train_path).resolve(), sample_count, rng)
    validation, validation_one = _inspect(Path(validation_path).resolve(), sample_count, rng)
    overlap = train_one & validation_one
    if overlap:
        raise ValueError(f"1cube train/validation episode overlap: {sorted(overlap)[:10]}")
    return {
        "status": "PASS",
        "seed": int(seed),
        "sample_count_requested": int(sample_count),
        "one_cube_episode_overlap": 0,
        "train": train,
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_mixed_datasets(
        train_path=args.train,
        validation_path=args.validation,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_suffix(args.output.suffix + ".tmp")
        partial.write_text(text)
        os.replace(partial, args.output)
    print(text, end="", flush=True)


if __name__ == "__main__":
    main()
