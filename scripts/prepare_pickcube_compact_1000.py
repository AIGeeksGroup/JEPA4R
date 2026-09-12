"""Replay and compact 1,000 PickCube demonstrations in recoverable batches."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np

from scripts.prepare_liftcube_proprio_compact import PROPRIO_STD_FLOOR, proprio_array
from scripts.prepare_maniskill_compact import (
    SAMPLING_SEED,
    _atomic_write_json,
    convert_replay_root,
    split_episode_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK = "PickCube-v0"
REPLAY_STEM = "trajectory.pointcloud_robot_seg.pd_ee_delta_pose"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_source_batch(
    metadata: Mapping[str, object], start: int, count: int
) -> tuple[dict[str, object], dict[int, int]]:
    """Copy one source slice while preserving original IDs through seed mapping."""
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("source metadata is missing episodes")
    selected_episodes = episodes[start : start + count]
    if len(selected_episodes) != count:
        raise ValueError(f"requested {count} episodes at {start}, found {len(selected_episodes)}")
    selected = copy.deepcopy(dict(metadata))
    selected["episodes"] = copy.deepcopy(selected_episodes)
    source_ids_by_seed = {
        int(episode["episode_seed"]): int(episode["episode_id"])
        for episode in selected_episodes
    }
    if len(source_ids_by_seed) != count:
        raise ValueError("episode seeds must be unique within a replay batch")
    return selected, source_ids_by_seed


def remap_replay_episode_ids(
    h5_path: Path,
    json_path: Path,
    source_ids_by_seed: Mapping[int, int],
) -> None:
    """Replace RecordEpisode's local IDs with immutable source episode IDs."""
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(source_ids_by_seed):
        raise RuntimeError("replayed episode count differs from the selected source batch")
    mappings: list[tuple[int, int]] = []
    for episode in episodes:
        local_id = int(episode["episode_id"])
        seed = int(episode["episode_seed"])
        if seed not in source_ids_by_seed:
            raise RuntimeError(f"unexpected replay episode seed: {seed}")
        source_id = int(source_ids_by_seed[seed])
        mappings.append((local_id, source_id))
        episode["episode_id"] = source_id
    if len({source_id for _, source_id in mappings}) != len(mappings):
        raise RuntimeError("source episode IDs are not unique")

    with h5py.File(h5_path, "r+") as handle:
        if set(handle) != {f"traj_{local_id}" for local_id, _ in mappings}:
            raise RuntimeError("replayed H5 groups differ from replay metadata")
        for local_id, _ in mappings:
            handle.move(f"traj_{local_id}", f"__remap_{local_id}")
        for local_id, source_id in mappings:
            handle.move(f"__remap_{local_id}", f"traj_{source_id}")
    _atomic_write_json(json_path, metadata)


def copy_verified(source: Path, destination: Path) -> str:
    """Copy through a temporary file and return the verified SHA-256."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    source_hash = sha256(source)
    if sha256(temporary) != source_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"copy verification failed: {destination}")
    os.replace(temporary, destination)
    return source_hash


def _safe_remove_batch(path: Path, temp_root: Path) -> None:
    resolved = path.resolve()
    root = temp_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"refusing to remove path outside temporary root: {path}")
    shutil.rmtree(resolved, ignore_errors=True)


def _batch_complete(marker: Path, staging: Path, episode_ids: Sequence[int]) -> bool:
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if [int(value) for value in payload["episode_ids"]] != list(episode_ids):
            return False
        hashes = payload["sha256"]
        return all(
            sha256(staging / f"episode_{episode_id:04d}.npz")
            == hashes[f"episode_{episode_id:04d}.npz"]
            for episode_id in episode_ids
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False


def build_final_manifest(
    compact_root: Path,
    batch_manifests: Sequence[Mapping[str, object]],
    episode_ids: Sequence[int],
    source_h5: Path,
    source_json: Path,
    *,
    seed: int,
) -> dict[str, object]:
    """Combine batch metadata and recompute normalization over the global split."""
    if not batch_manifests:
        raise ValueError("at least one batch manifest is required")
    all_ids = [int(value) for value in episode_ids]
    records = [record for manifest in batch_manifests for record in manifest["episodes"]]
    records_by_id = {int(record["episode_id"]): record for record in records}
    if len(records_by_id) != len(records) or set(records_by_id) != set(all_ids):
        raise ValueError("batch records do not exactly cover source episode IDs")
    train_ids, val_ids = split_episode_ids(all_ids, seed=seed)

    total = np.zeros(25, dtype=np.float64)
    total_square = np.zeros(25, dtype=np.float64)
    sample_count = 0
    for episode_id in train_ids:
        path = compact_root / f"episode_{episode_id:04d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            values = proprio_array(
                {name: np.asarray(archive[name]) for name in ("qpos", "qvel", "tcp_pose")}
            ).astype(np.float64, copy=False)
        total += values.sum(axis=0, dtype=np.float64)
        total_square += np.square(values).sum(axis=0, dtype=np.float64)
        sample_count += len(values)
    mean = total / sample_count
    variance = np.maximum(total_square / sample_count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), PROPRIO_STD_FLOOR)

    manifest = copy.deepcopy(dict(batch_manifests[0]))
    for key in (
        "source_replay_h5",
        "source_replay_h5_sha256",
        "source_replay_json",
        "source_replay_json_sha256",
    ):
        manifest.pop(key, None)
    manifest.update(
        {
            "episode_ids": all_ids,
            "train_episode_ids": train_ids,
            "val_episode_ids": val_ids,
            "episodes": [records_by_id[value] for value in all_ids],
            "proprio_normalization": {
                "source_split": "train",
                "episode_ids": train_ids,
                "sample_count": int(sample_count),
                "mean": mean.astype(np.float32).tolist(),
                "std": std.astype(np.float32).tolist(),
                "std_floor": PROPRIO_STD_FLOOR,
            },
            "source_raw_h5": str(source_h5.resolve()),
            "source_raw_h5_sha256": sha256(source_h5),
            "source_raw_json": str(source_json.resolve()),
            "source_raw_json_sha256": sha256(source_json),
            "replay_allow_failure": True,
            "preparation": "bounded /dev/shm replay batches with verified compact copies",
        }
    )
    return manifest


def _verify_compact_episode(path: Path) -> None:
    required = {
        "xyz",
        "rgb",
        "action",
        "object_pos",
        "goal_pos",
        "success",
        "qpos",
        "qvel",
        "tcp_pose",
        "base_pose",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError(f"{path}: compact fields differ from the required contract")
        frames = len(archive["xyz"])
        expected = {
            "xyz": (frames, 1024, 3),
            "rgb": (frames, 1024, 3),
            "action": (frames - 1, 7),
            "object_pos": (frames, 3),
            "goal_pos": (3,),
            "success": (frames - 1,),
            "qpos": (frames, 9),
            "qvel": (frames, 9),
            "tcp_pose": (frames, 7),
            "base_pose": (frames, 7),
        }
        for name, shape in expected.items():
            if archive[name].shape != shape:
                raise ValueError(f"{path}: {name} must be {shape}, got {archive[name].shape}")
        if archive["rgb"].dtype != np.uint8 or archive["success"].dtype != np.bool_:
            raise ValueError(f"{path}: rgb/success dtypes are invalid")
        for name in required - {"rgb", "success"}:
            if not np.isfinite(archive[name]).all():
                raise ValueError(f"{path}: {name} contains non-finite values")


def verify_compact_root(root: Path, episode_ids: Sequence[int]) -> None:
    files = sorted(root.glob("episode_*.npz"))
    expected_ids = {int(value) for value in episode_ids}
    actual_ids = {int(path.stem.split("_")[1]) for path in files}
    if actual_ids != expected_ids or len(files) != len(expected_ids):
        raise ValueError("compact files do not exactly cover expected episode IDs")
    for path in files:
        _verify_compact_episode(path)


def _replay_batch(
    source_h5: Path,
    selected_metadata: Mapping[str, object],
    source_ids_by_seed: Mapping[int, int],
    batch_root: Path,
    *,
    point_count: int,
    seed: int,
) -> tuple[Path, dict[str, object]]:
    batch_root.mkdir(parents=True)
    (batch_root / "trajectory.h5").symlink_to(source_h5)
    _atomic_write_json(batch_root / "trajectory.json", selected_metadata)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.replay_maniskill_trajectory",
            "--traj-path",
            str(batch_root / "trajectory.h5"),
            "--obs-mode",
            "pointcloud_robot_seg",
            "--target-control-mode",
            "pd_ee_delta_pose",
            "--save-traj",
            "--allow-failure",
            "--num-procs",
            "1",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    replay_h5 = batch_root / f"{REPLAY_STEM}.h5"
    replay_json = batch_root / f"{REPLAY_STEM}.json"
    remap_replay_episode_ids(replay_h5, replay_json, source_ids_by_seed)
    compact_root = batch_root / "compact"
    manifest = convert_replay_root(
        replay_h5,
        replay_json,
        compact_root,
        point_count=point_count,
        seed=seed,
    )
    return compact_root, manifest


def prepare_dataset(
    source_h5: Path,
    source_json: Path,
    output_root: Path,
    temp_root: Path,
    *,
    total_episodes: int,
    batch_size: int,
    point_count: int,
    seed: int,
) -> None:
    metadata = json.loads(source_json.read_text(encoding="utf-8"))
    if metadata.get("env_info", {}).get("env_id") != TASK:
        raise ValueError(f"source metadata is not {TASK}")
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != total_episodes:
        raise ValueError(f"source must contain exactly {total_episodes} episodes")
    episode_ids = [int(episode["episode_id"]) for episode in episodes]
    if len(set(episode_ids)) != total_episodes:
        raise ValueError("source episode IDs must be unique")
    if output_root.exists():
        verify_compact_root(output_root, episode_ids)
        print(f"DATASET_ALREADY_COMPLETE output={output_root}", flush=True)
        return

    staging = output_root.with_name(f".{output_root.name}.tmp")
    staging.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    for start in range(0, total_episodes, batch_size):
        count = min(batch_size, total_episodes - start)
        selected, source_ids_by_seed = select_source_batch(metadata, start, count)
        selected_ids = [int(episode["episode_id"]) for episode in selected["episodes"]]
        stop = start + count
        marker = staging / f".batch_{start:04d}_{stop - 1:04d}.json"
        if _batch_complete(marker, staging, selected_ids):
            print(f"BATCH_SKIP start={start} stop={stop}", flush=True)
            continue
        batch_root = temp_root / f"batch_{start:04d}_{stop - 1:04d}"
        _safe_remove_batch(batch_root, temp_root)
        print(f"BATCH_START start={start} stop={stop}", flush=True)
        compact_batch, batch_manifest = _replay_batch(
            source_h5,
            selected,
            source_ids_by_seed,
            batch_root,
            point_count=point_count,
            seed=seed,
        )
        if [int(value) for value in batch_manifest["episode_ids"]] != selected_ids:
            raise RuntimeError("batch compact manifest changed source episode ordering")
        hashes = {}
        for episode_id in selected_ids:
            name = f"episode_{episode_id:04d}.npz"
            hashes[name] = copy_verified(compact_batch / name, staging / name)
        _atomic_write_json(
            marker,
            {
                "episode_ids": selected_ids,
                "sha256": hashes,
                "manifest": batch_manifest,
            },
        )
        _safe_remove_batch(batch_root, temp_root)
        print(f"BATCH_DONE start={start} stop={stop}", flush=True)

    marker_paths = sorted(staging.glob(".batch_*.json"))
    expected_batches = (total_episodes + batch_size - 1) // batch_size
    if len(marker_paths) != expected_batches:
        raise RuntimeError("batch markers do not cover the complete source dataset")
    batch_manifests = [
        json.loads(path.read_text(encoding="utf-8"))["manifest"]
        for path in marker_paths
    ]
    manifest = build_final_manifest(
        staging,
        batch_manifests,
        episode_ids,
        source_h5,
        source_json,
        seed=seed,
    )
    manifest["batch_size"] = int(batch_size)
    manifest["batch_count"] = expected_batches
    _atomic_write_json(staging / "manifest.json", manifest)
    verify_compact_root(staging, episode_ids)
    for marker in marker_paths:
        marker.unlink()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, output_root)
    verify_compact_root(output_root, episode_ids)
    print(f"DATASET_DONE episodes={total_episodes} output={output_root}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--total-episodes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--point-count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=SAMPLING_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(
        args.source_h5,
        args.source_json,
        args.output_root,
        args.temp_root,
        total_episodes=args.total_episodes,
        batch_size=args.batch_size,
        point_count=args.point_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
