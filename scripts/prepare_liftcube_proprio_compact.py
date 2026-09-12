"""Add aligned Panda proprioception to compact LiftCube trajectories.

This script treats the existing compact RGB episodes as immutable source data.
It copies their arrays exactly, joins replay state at the same observation index,
and computes normalization statistics from the manifest's training split only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np


SOURCE_FIELDS = ("xyz", "rgb", "action", "object_pos", "goal_pos", "success")
STATE_FIELDS = {
    "qpos": ("obs/agent/qpos", 9),
    "qvel": ("obs/agent/qvel", 9),
    "tcp_pose": ("obs/extra/tcp_pose", 7),
    "base_pose": ("obs/agent/base_pose", 7),
}
PROPRIO_LAYOUT = [
    {"name": "qpos", "start": 0, "stop": 9},
    {"name": "qvel", "start": 9, "stop": 18},
    {"name": "tcp_pose", "start": 18, "stop": 25},
]
PROPRIO_STD_FLOOR = 1e-6
EXPECTED_ROBOT_UID = "panda"
EXPECTED_AGENT_CLASS = "Panda"
EXPECTED_ROBOT_NAME = "panda_v2"
EXPECTED_CONTROL_MODE = "pd_ee_delta_pose"


def _load_episode(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing compact episode: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _require_finite(name: str, value: np.ndarray, episode_id: int) -> None:
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError(
            f"episode {episode_id} {name} must be numeric, got {value.dtype}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"episode {episode_id} {name} must be finite")


def _validate_source(source: Mapping[str, np.ndarray], episode_id: int) -> int:
    missing = [name for name in SOURCE_FIELDS if name not in source]
    if missing:
        raise ValueError(f"source compact episode is missing fields: {missing}")

    xyz = np.asarray(source["xyz"])
    rgb = np.asarray(source["rgb"])
    action = np.asarray(source["action"])
    object_pos = np.asarray(source["object_pos"])
    goal_pos = np.asarray(source["goal_pos"])
    success = np.asarray(source["success"])
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError(f"episode {episode_id} xyz must be [F,N,3], got {xyz.shape}")
    frames = int(xyz.shape[0])
    if frames < 2:
        raise ValueError(f"episode {episode_id} must contain at least two frames")
    if rgb.shape != xyz.shape or rgb.dtype != np.uint8:
        raise ValueError(
            f"episode {episode_id} rgb must be uint8 {xyz.shape}, "
            f"got {rgb.shape} {rgb.dtype}"
        )
    expected_actions = (frames - 1, 7)
    if action.shape != expected_actions:
        raise ValueError(
            f"episode {episode_id} action must be {expected_actions}, "
            f"got {action.shape}"
        )
    if object_pos.shape != (frames, 3):
        raise ValueError(
            f"episode {episode_id} object_pos must be {(frames, 3)}, "
            f"got {object_pos.shape}"
        )
    if goal_pos.shape != (3,):
        raise ValueError(
            f"episode {episode_id} goal_pos must be static [3], got {goal_pos.shape}"
        )
    if success.shape != (frames - 1,):
        raise ValueError(
            f"episode {episode_id} success must be [{frames - 1}], "
            f"got {success.shape}"
        )
    for name in ("xyz", "action", "object_pos", "goal_pos"):
        _require_finite(name, np.asarray(source[name]), episode_id)
    if success.dtype != np.bool_:
        raise ValueError(
            f"episode {episode_id} success must be bool, got {success.dtype}"
        )
    return frames


def prepare_episode(
    replay_group: h5py.Group,
    source: Mapping[str, np.ndarray],
    *,
    episode_id: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Join one compact episode with frame-aligned replay state."""
    frames = _validate_source(source, episode_id)
    required = ["actions", *(path for path, _ in STATE_FIELDS.values())]
    missing = [path for path in required if path not in replay_group]
    if missing:
        raise ValueError(f"replay episode is missing datasets: {missing}")

    replay_actions = np.asarray(replay_group["actions"])
    source_actions = np.asarray(source["action"])
    if replay_actions.shape != source_actions.shape or not np.array_equal(
        replay_actions, source_actions
    ):
        raise ValueError(f"episode {episode_id} replay actions differ from compact")

    states: dict[str, np.ndarray] = {}
    for name, (path, width) in STATE_FIELDS.items():
        raw = np.asarray(replay_group[path])
        if raw.shape != (frames, width):
            raise ValueError(
                f"episode {episode_id} {name} must be {(frames, width)}, "
                f"got {raw.shape}"
            )
        _require_finite(name, raw, episode_id)
        states[name] = np.ascontiguousarray(raw, dtype=np.float32)

    result = {name: np.asarray(source[name]) for name in SOURCE_FIELDS}
    result.update(states)
    record = {
        "episode_id": int(episode_id),
        "frames": frames,
        "actions": frames - 1,
    }
    return result, record


def proprio_array(episode: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return qpos, qvel and TCP pose in the documented 25-D layout."""
    arrays = [np.asarray(episode[item["name"]]) for item in PROPRIO_LAYOUT]
    frames = int(arrays[0].shape[0])
    for item, array in zip(PROPRIO_LAYOUT, arrays, strict=True):
        width = int(item["stop"]) - int(item["start"])
        if array.shape != (frames, width):
            raise ValueError(
                f"{item['name']} must be {(frames, width)}, got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{item['name']} must be finite")
    return np.ascontiguousarray(np.concatenate(arrays, axis=-1), dtype=np.float32)


def compute_proprio_stats(
    episodes: Mapping[int, Mapping[str, np.ndarray]],
    train_episode_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute population statistics without touching validation episodes."""
    total = np.zeros(25, dtype=np.float64)
    total_square = np.zeros(25, dtype=np.float64)
    count = 0
    for episode_id in train_episode_ids:
        values = proprio_array(episodes[int(episode_id)]).astype(
            np.float64, copy=False
        )
        total += values.sum(axis=0, dtype=np.float64)
        total_square += np.square(values).sum(axis=0, dtype=np.float64)
        count += int(values.shape[0])
    if count == 0:
        raise ValueError("training split contains no proprio samples")
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), PROPRIO_STD_FLOOR)
    return mean.astype(np.float32), std.astype(np.float32), count


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
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_manifest(manifest: Mapping[str, object]) -> tuple[list[int], list[int], list[int]]:
    if manifest.get("task") != "LiftCube-v0":
        raise ValueError(f"expected LiftCube-v0 manifest, got {manifest.get('task')!r}")
    episode_ids = [int(value) for value in manifest["episode_ids"]]
    train_ids = [int(value) for value in manifest["train_episode_ids"]]
    val_ids = [int(value) for value in manifest["val_episode_ids"]]
    if len(episode_ids) != 100 or len(set(episode_ids)) != 100:
        raise ValueError("LiftCube manifest must contain 100 unique episode IDs")
    if len(train_ids) != 90 or len(set(train_ids)) != 90:
        raise ValueError("LiftCube training split must contain 90 unique episode IDs")
    if len(val_ids) != 10 or len(set(val_ids)) != 10:
        raise ValueError("LiftCube validation split must contain 10 unique episode IDs")
    if set(train_ids) & set(val_ids):
        raise ValueError("LiftCube train and validation splits overlap")
    if set(train_ids) | set(val_ids) != set(episode_ids):
        raise ValueError("LiftCube splits do not exactly cover episode_ids")
    return episode_ids, train_ids, val_ids


def _validate_replay_metadata(
    metadata: Mapping[str, object], episode_ids: Sequence[int]
) -> None:
    env_info = metadata.get("env_info")
    if not isinstance(env_info, Mapping):
        raise ValueError("replay metadata is missing env_info")
    if env_info.get("env_id") != "LiftCube-v0":
        raise ValueError(f"expected LiftCube-v0 replay, got {env_info.get('env_id')!r}")
    env_kwargs = env_info.get("env_kwargs")
    if not isinstance(env_kwargs, Mapping):
        raise ValueError("replay metadata is missing env_info.env_kwargs")
    if env_kwargs.get("control_mode") != EXPECTED_CONTROL_MODE:
        raise ValueError(
            f"expected control mode {EXPECTED_CONTROL_MODE!r}, "
            f"got {env_kwargs.get('control_mode')!r}"
        )
    metadata_episodes = metadata.get("episodes")
    if not isinstance(metadata_episodes, list):
        raise ValueError("replay metadata is missing episodes")
    replay_ids: list[int] = []
    for episode in metadata_episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("replay episode metadata must be an object")
        replay_ids.append(int(episode["episode_id"]))
        if episode.get("control_mode") != EXPECTED_CONTROL_MODE:
            raise ValueError(
                f"episode {episode.get('episode_id')} has unexpected control mode"
            )
    if len(replay_ids) != len(set(replay_ids)) or set(replay_ids) != set(episode_ids):
        raise ValueError("replay metadata episode IDs differ from compact manifest")


def _verify_npz(path: Path, expected: Mapping[str, np.ndarray]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing prepared episode: {path}")
    with np.load(path, allow_pickle=False) as actual:
        if set(actual.files) != set(expected):
            raise ValueError(f"prepared field mismatch: {path.name}")
        for key, value in expected.items():
            if not np.array_equal(actual[key], value):
                raise ValueError(f"prepared {path.name} field {key!r} differs")


def prepare_root(
    replay_h5: str | Path,
    replay_json: str | Path,
    source_compact: str | Path,
    output_root: str | Path,
    *,
    verified_robot_uid: str,
    verified_agent_class: str,
    verified_robot_name: str,
    verify_only: bool = False,
) -> dict[str, object]:
    """Verify or publish all 100 LiftCube episodes and train-only stats."""
    replay_h5 = Path(replay_h5)
    replay_json = Path(replay_json)
    source_compact = Path(source_compact)
    output_root = Path(output_root)
    output_was_created = False
    created_files: list[Path] = []

    if output_root.exists() and any(output_root.iterdir()) and not verify_only:
        raise FileExistsError(f"output root is not empty: {output_root}")
    if verify_only and (not output_root.is_dir() or not any(output_root.iterdir())):
        raise FileNotFoundError(f"prepared output is missing or empty: {output_root}")
    if verified_robot_uid != EXPECTED_ROBOT_UID:
        raise ValueError(
            f"verified_robot_uid must be {EXPECTED_ROBOT_UID!r}, "
            f"got {verified_robot_uid!r}"
        )
    if verified_agent_class != EXPECTED_AGENT_CLASS:
        raise ValueError(
            f"verified_agent_class must be {EXPECTED_AGENT_CLASS!r}, "
            f"got {verified_agent_class!r}"
        )
    if verified_robot_name != EXPECTED_ROBOT_NAME:
        raise ValueError(
            f"verified_robot_name must be {EXPECTED_ROBOT_NAME!r}, "
            f"got {verified_robot_name!r}"
        )

    manifest_path = source_compact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episode_ids, train_ids, val_ids = _validate_manifest(manifest)
    replay_metadata = json.loads(replay_json.read_text(encoding="utf-8"))
    _validate_replay_metadata(replay_metadata, episode_ids)
    replay_stat = replay_h5.stat()

    if not verify_only and not output_root.exists():
        output_root.mkdir(parents=True)
        output_was_created = True

    records: list[dict[str, int]] = []
    states: dict[int, dict[str, np.ndarray]] = {}
    try:
        with h5py.File(replay_h5, "r") as replay:
            replay_group_ids = {
                int(name.removeprefix("traj_"))
                for name in replay.keys()
                if name.startswith("traj_")
            }
            if replay_group_ids != set(episode_ids):
                raise ValueError("replay HDF5 trajectory IDs differ from compact manifest")
            for episode_id in episode_ids:
                source = _load_episode(
                    source_compact / f"episode_{episode_id:04d}.npz"
                )
                arrays, record = prepare_episode(
                    replay[f"traj_{episode_id}"], source, episode_id=episode_id
                )
                records.append(record)
                states[episode_id] = {
                    name: arrays[name] for name in ("qpos", "qvel", "tcp_pose")
                }
                destination = output_root / f"episode_{episode_id:04d}.npz"
                if verify_only:
                    _verify_npz(destination, arrays)
                else:
                    atomic_save_npz(destination, arrays)
                    created_files.append(destination)

        mean, std, sample_count = compute_proprio_stats(states, train_ids)
        result = dict(manifest)
        result.update(
            {
                "robot_uid": EXPECTED_ROBOT_UID,
                "robot_identity_verification": {
                    "agent_class": verified_agent_class,
                    "robot_name": verified_robot_name,
                    "method": "runtime_environment_probe",
                },
                "control_mode": EXPECTED_CONTROL_MODE,
                "action_dim": 7,
                "qpos_dim": 9,
                "qvel_dim": 9,
                "tcp_pose_dim": 7,
                "base_pose_dim": 7,
                "proprio_dim": 25,
                "proprio_layout": PROPRIO_LAYOUT,
                "proprio_normalization": {
                    "source_split": "train",
                    "episode_ids": train_ids,
                    "sample_count": sample_count,
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "std_floor": PROPRIO_STD_FLOOR,
                },
                "source_compact": str(source_compact),
                "source_manifest_sha256": _sha256(manifest_path),
                "source_replay_h5": str(replay_h5),
                "source_replay_size": int(replay_stat.st_size),
                "source_replay_mtime_ns": int(replay_stat.st_mtime_ns),
                "source_replay_json": str(replay_json),
                "source_replay_json_sha256": _sha256(replay_json),
                "episodes": records,
            }
        )
        output_manifest = output_root / "manifest.json"
        if verify_only:
            published = json.loads(output_manifest.read_text(encoding="utf-8"))
            if published != result:
                raise ValueError("prepared manifest differs from recomputed manifest")
            expected_names = {
                "manifest.json",
                *(f"episode_{episode_id:04d}.npz" for episode_id in episode_ids),
            }
            actual_names = {path.name for path in output_root.iterdir()}
            if actual_names != expected_names:
                raise ValueError("prepared output contains missing or unexpected files")
        else:
            _atomic_write_json(output_manifest, result)
            created_files.append(output_manifest)
        return result
    except BaseException:
        if not verify_only:
            for path in reversed(created_files):
                path.unlink(missing_ok=True)
            for temporary in output_root.glob("*.tmp") if output_root.exists() else ():
                temporary.unlink(missing_ok=True)
            if output_was_created:
                output_root.rmdir()
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-h5", type=Path, required=True)
    parser.add_argument("--replay-json", type=Path, required=True)
    parser.add_argument("--source-compact", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verified-robot-uid", required=True)
    parser.add_argument("--verified-agent-class", required=True)
    parser.add_argument("--verified-robot-name", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_root(
        args.replay_h5,
        args.replay_json,
        args.source_compact,
        args.output_root,
        verified_robot_uid=args.verified_robot_uid,
        verified_agent_class=args.verified_agent_class,
        verified_robot_name=args.verified_robot_name,
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
