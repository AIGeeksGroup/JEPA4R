"""Select and validate simulator-confirmed fixed-task episodes in HDF5 files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from envs.ur5e_cube_tasks import get_task

_PROVENANCE_COLUMNS = {
    "source_policy",
    "accepted_success",
    "fixed_task_id",
    "cube_count",
}


def _dataset_paths(group: h5py.Group, prefix: str = "") -> list[str]:
    paths = []
    for name, value in group.items():
        path = f"{prefix}{name}"
        if isinstance(value, h5py.Dataset):
            paths.append(path)
        else:
            paths.extend(_dataset_paths(value, f"{path}/"))
    return paths


def _episode_layout(handle: h5py.File):
    if "ep_len" not in handle or "ep_offset" not in handle:
        raise ValueError("missing ep_len or ep_offset")
    lengths = np.asarray(handle["ep_len"][:])
    offsets = np.asarray(handle["ep_offset"][:])
    if lengths.ndim != 1 or offsets.ndim != 1 or len(lengths) != len(offsets):
        raise ValueError("invalid episode arrays")
    if not np.issubdtype(lengths.dtype, np.integer) or np.any(lengths <= 0):
        raise ValueError("ep_len must contain positive integers")
    expected_offsets = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(offsets.dtype)
    if not np.array_equal(offsets, expected_offsets):
        raise ValueError("non-contiguous ep_offset")
    return lengths.astype(np.int64), offsets.astype(np.int64)


def classify_episodes(handle: h5py.File) -> list[bool]:
    """Classify by any post-reset success; reset-only success is ignored."""

    lengths, offsets = _episode_layout(handle)
    if "success" not in handle:
        raise ValueError("missing success column")
    success = handle["success"]
    result = []
    for length, offset in zip(lengths, offsets):
        values = np.asarray(success[offset : offset + length]).reshape(-1)
        if values.dtype.kind not in "bifu" or not np.isfinite(values).all():
            raise ValueError("success must be finite boolean or numeric data")
        if not np.isin(values, [0, 1]).all():
            raise ValueError("success must contain only 0/1 values")
        result.append(bool(values[1:].astype(bool).any()))
    return result


def _create_like(destination: h5py.File, path: str, source: h5py.Dataset, rows: int):
    if source.ndim == 0:
        raise ValueError(f"scalar dataset is unsupported: {path}")
    parent, _, name = path.rpartition("/")
    group = destination.require_group(parent) if parent else destination
    shape = (rows, *source.shape[1:])
    chunk_rows = max(1, min(20, rows))
    chunks = (chunk_rows, *source.shape[1:])
    return group.create_dataset(
        path.rsplit("/", 1)[-1],
        shape=shape,
        maxshape=(None, *source.shape[1:]),
        dtype=source.dtype,
        chunks=chunks,
    )


def copy_accepted_episodes(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    outcome: str,
    limit: int,
    family: str,
    task_id: int,
    policy_kind: str,
) -> int:
    """Copy up to ``limit`` matching episodes and attach row-level provenance."""

    source_path = Path(source_path)
    destination_path = Path(destination_path)
    if destination_path.exists():
        raise FileExistsError(destination_path)
    if outcome not in {"success", "failure"}:
        raise ValueError("outcome must be success or failure")
    if policy_kind not in {"expert", "random"}:
        raise ValueError("policy_kind must be expert or random")
    if limit <= 0:
        raise ValueError("limit must be positive")
    task = get_task(family, task_id)

    with h5py.File(source_path, "r") as source:
        lengths, offsets = _episode_layout(source)
        outcomes = classify_episodes(source)
        desired = outcome == "success"
        selected = [index for index, value in enumerate(outcomes) if value == desired][:limit]
        if not selected:
            return 0
        selected_lengths = lengths[selected]
        total_rows = int(selected_lengths.sum())
        paths = [
            path
            for path in _dataset_paths(source)
            if path not in {"ep_len", "ep_offset"} | _PROVENANCE_COLUMNS
        ]

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with h5py.File(destination_path, "x") as destination:
                for key, value in source.attrs.items():
                    destination.attrs[key] = value
                destination.attrs["fixed_task_family"] = task.family
                destination.attrs["fixed_task_id"] = task.task_id
                destination.attrs["fixed_task_name"] = task.name
                destination.attrs["source_policy"] = policy_kind
                destination.attrs["accepted_outcome"] = outcome
                destination.create_dataset(
                    "ep_len", data=selected_lengths, maxshape=(None,), chunks=True
                )
                selected_offsets = np.concatenate(
                    ([0], np.cumsum(selected_lengths)[:-1])
                ).astype(np.int64)
                destination.create_dataset(
                    "ep_offset", data=selected_offsets, maxshape=(None,), chunks=True
                )
                outputs = {
                    path: _create_like(destination, path, source[path], total_rows)
                    for path in paths
                }
                row = 0
                for episode_index in selected:
                    start = int(offsets[episode_index])
                    length = int(lengths[episode_index])
                    stop = start + length
                    for path, output in outputs.items():
                        for chunk_start in range(start, stop, 64):
                            chunk_stop = min(chunk_start + 64, stop)
                            width = chunk_stop - chunk_start
                            dst_start = row + (chunk_start - start)
                            output[dst_start : dst_start + width] = source[path][chunk_start:chunk_stop]
                    row += length

                provenance = {
                    "source_policy": 1 if policy_kind == "expert" else 0,
                    "accepted_success": 1 if desired else 0,
                    "fixed_task_id": task.task_id,
                    "cube_count": task.num_cubes,
                }
                for name, value in provenance.items():
                    destination.create_dataset(
                        name,
                        data=np.full(total_rows, value, dtype=np.uint8),
                        maxshape=(None,),
                        chunks=(max(1, min(4096, total_rows)),),
                    )
                destination.flush()
        except BaseException:
            destination_path.unlink(missing_ok=True)
            raise
    return len(selected)


def _check_finite(dataset: h5py.Dataset, name: str):
    if dataset.dtype.kind not in "biufc":
        return
    for start in range(0, len(dataset), 256):
        if not np.isfinite(dataset[start : start + 256]).all():
            raise ValueError(f"{name} contains non-finite values")


def _check_action_boundaries(dataset: h5py.Dataset, lengths, offsets):
    terminal_rows = set((offsets + lengths - 1).tolist())
    for start in range(0, len(dataset), 256):
        values = np.asarray(dataset[start : start + 256])
        for local_index, row in enumerate(values):
            if np.isfinite(row).all():
                continue
            absolute_index = start + local_index
            if absolute_index in terminal_rows and np.isnan(row).all():
                continue
            raise ValueError(
                "action contains non-finite values outside a full-NaN episode terminal row"
            )


def validate_labeled_dataset(
    path: str | Path,
    *,
    family: str,
    task_id: int,
    expected_episodes: int,
    expected_policy_counts: dict[str, int],
) -> dict:
    task = get_task(family, task_id)
    with h5py.File(path, "r") as handle:
        lengths, offsets = _episode_layout(handle)
        if len(lengths) != expected_episodes:
            raise ValueError(f"expected {expected_episodes} episodes, found {len(lengths)}")
        rows = int(lengths.sum())
        required_shapes = {
            "qpos": (rows, task.qpos_dim),
            "qvel": (rows, task.qvel_dim),
            "control": (rows, task.control_dim),
            "action": (rows, task.action_dim),
            "observation": (rows, task.observation_dim),
            "pcd_xyz": (rows, 1200, 3),
            "pcd_rgb": (rows, 1200, 3),
        }
        for name, shape in required_shapes.items():
            if name not in handle or handle[name].shape != shape:
                actual = None if name not in handle else handle[name].shape
                raise ValueError(f"{name} shape {actual}, expected {shape}")
            if name != "action":
                _check_finite(handle[name], name)
        _check_action_boundaries(handle["action"], lengths, offsets)
        if handle["pcd_rgb"].dtype != np.uint8:
            raise ValueError(f"pcd_rgb dtype must be uint8, got {handle['pcd_rgb'].dtype}")

        for name in _PROVENANCE_COLUMNS:
            if name not in handle or handle[name].shape != (rows,):
                raise ValueError(f"invalid provenance column: {name}")
        if not np.all(handle["fixed_task_id"][:] == task.task_id):
            raise ValueError("fixed_task_id mismatch")
        if not np.all(handle["cube_count"][:] == task.num_cubes):
            raise ValueError("cube_count mismatch")

        outcomes = classify_episodes(handle)
        policy_codes = []
        accepted_labels = []
        for offset in offsets:
            policy_codes.append(int(handle["source_policy"][offset]))
            accepted_labels.append(bool(handle["accepted_success"][offset]))
        for index, (policy, outcome, label) in enumerate(
            zip(policy_codes, outcomes, accepted_labels)
        ):
            if outcome != label:
                raise ValueError(f"episode {index} accepted_success mismatch")
            if policy == 1 and not outcome:
                raise ValueError(f"expert episode {index} did not succeed")
            if policy == 0 and outcome:
                raise ValueError(f"random episode {index} is not a failure")
            if policy not in {0, 1}:
                raise ValueError(f"episode {index} has invalid source_policy")
        counts = {
            "expert": int(np.sum(np.asarray(policy_codes) == 1)),
            "random": int(np.sum(np.asarray(policy_codes) == 0)),
        }
        if counts != expected_policy_counts:
            raise ValueError(f"policy counts {counts}, expected {expected_policy_counts}")
        return {
            "episodes": len(lengths),
            "rows": rows,
            "successes": int(sum(outcomes)),
            "failures": int(len(outcomes) - sum(outcomes)),
            "policy_counts": counts,
            "family": task.family,
            "task_id": task.task_id,
            "task_name": task.name,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--family", required=True, choices=["double", "triple"])
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--episodes", required=True, type=int)
    parser.add_argument("--expert", required=True, type=int)
    parser.add_argument("--random", required=True, type=int)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_labeled_dataset(
        args.dataset,
        family=args.family,
        task_id=args.task_id,
        expected_episodes=args.episodes,
        expected_policy_counts={"expert": args.expert, "random": args.random},
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
