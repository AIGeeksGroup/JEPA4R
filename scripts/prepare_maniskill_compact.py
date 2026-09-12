"""Convert replayed PickCube trajectories into the JEPA compact-data contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np

from scripts.prepare_liftcube_proprio_compact import (
    PROPRIO_LAYOUT,
    PROPRIO_STD_FLOOR,
    atomic_save_npz,
    proprio_array,
)
from scripts.prepare_liftcube_rgb_compact import (
    SAMPLING_SEED,
    XYZ_HIGH,
    XYZ_LOW,
    extract_aligned_frame,
)


EXPECTED_TASK = "PickCube-v0"
EXPECTED_CONTROL_MODE = "pd_ee_delta_pose"
EXPECTED_OBS_MODE = "pointcloud_robot_seg"
PICKCUBE_STATE_DIM = 73
PICKCUBE_OBJECT_POSITION = slice(13, 16)
STATE_FIELDS = {
    "qpos": ("obs/agent/qpos", 9),
    "qvel": ("obs/agent/qvel", 9),
    "tcp_pose": ("obs/extra/tcp_pose", 7),
    "base_pose": ("obs/agent/base_pose", 7),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _finite_float32(value: np.ndarray, name: str, episode_id: int) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"episode {episode_id} {name} must be finite numeric data")
    return np.ascontiguousarray(array, dtype=np.float32)


def split_episode_ids(
    episode_ids: Sequence[int], *, seed: int, val_fraction: float = 0.1
) -> tuple[list[int], list[int]]:
    """Return a deterministic split with validation IDs first in the permutation."""
    ids = [int(value) for value in episode_ids]
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("episode_ids must contain at least two unique values")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    permutation = np.random.default_rng(seed).permutation(ids).tolist()
    val_count = max(1, min(len(ids) - 1, int(round(len(ids) * val_fraction))))
    return permutation[val_count:], permutation[:val_count]


def _pickcube_object_positions(
    env_states: np.ndarray, goal_positions: np.ndarray, episode_id: int
) -> np.ndarray:
    states = _finite_float32(env_states, "env_states", episode_id)
    goals = _finite_float32(goal_positions, "goal_pos", episode_id)
    if states.ndim != 2 or states.shape[1] != PICKCUBE_STATE_DIM:
        raise ValueError(
            f"episode {episode_id} env_states must be [F,{PICKCUBE_STATE_DIM}], "
            f"got {states.shape}"
        )
    if goals.shape != (len(states), 3):
        raise ValueError(
            f"episode {episode_id} goal_pos must be {(len(states), 3)}, got {goals.shape}"
        )
    if not np.allclose(states[:, -3:], goals, atol=1e-6, rtol=0.0):
        raise ValueError(f"episode {episode_id} goal position differs from env_states")
    return np.ascontiguousarray(states[:, PICKCUBE_OBJECT_POSITION], dtype=np.float32)


def prepare_episode(
    replay_group: h5py.Group,
    *,
    episode_id: int,
    point_count: int = 1024,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Validate and convert one replay group without changing temporal indexing."""
    required = (
        "actions",
        "success",
        "env_states",
        "obs/extra/goal_pos",
        "obs/pointcloud/xyzw",
        "obs/pointcloud/rgb",
        "obs/pointcloud/robot_seg",
        *(path for path, _ in STATE_FIELDS.values()),
    )
    missing = [name for name in required if name not in replay_group]
    if missing:
        raise ValueError(f"episode {episode_id} replay is missing datasets: {missing}")

    actions = _finite_float32(replay_group["actions"], "actions", episode_id)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"episode {episode_id} actions must be [A,7], got {actions.shape}")
    frames = len(actions) + 1
    success = np.asarray(replay_group["success"])
    if success.shape != (len(actions),) or success.dtype != np.bool_:
        raise ValueError(
            f"episode {episode_id} success must be bool [{len(actions)}], "
            f"got {success.shape} {success.dtype}"
        )

    states: dict[str, np.ndarray] = {}
    for name, (path, width) in STATE_FIELDS.items():
        value = _finite_float32(replay_group[path], name, episode_id)
        if value.shape != (frames, width):
            raise ValueError(
                f"episode {episode_id} {name} must be {(frames, width)}, got {value.shape}"
            )
        states[name] = value

    goals = _finite_float32(replay_group["obs/extra/goal_pos"], "goal_pos", episode_id)
    object_pos = _pickcube_object_positions(
        replay_group["env_states"], goals, episode_id
    )
    if object_pos.shape != (frames, 3):
        raise ValueError(f"episode {episode_id} object position has invalid shape")
    if not np.allclose(goals, goals[0], atol=1e-6, rtol=0.0):
        raise ValueError(f"episode {episode_id} goal position must remain static")

    clouds = replay_group["obs/pointcloud/xyzw"]
    colors = replay_group["obs/pointcloud/rgb"]
    robot_masks = replay_group["obs/pointcloud/robot_seg"]
    if len(clouds) != frames or len(colors) != frames or len(robot_masks) != frames:
        raise ValueError(f"episode {episode_id} point-cloud frame count is not {frames}")
    xyz_frames: list[np.ndarray] = []
    rgb_frames: list[np.ndarray] = []
    for frame_id in range(frames):
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

    arrays = {
        "xyz": np.stack(xyz_frames).astype(np.float32, copy=False),
        "rgb": np.stack(rgb_frames).astype(np.uint8, copy=False),
        "action": actions,
        "object_pos": object_pos,
        "goal_pos": np.ascontiguousarray(goals[0], dtype=np.float32),
        "success": np.ascontiguousarray(success, dtype=bool),
        **states,
    }
    record: dict[str, object] = {
        "episode_id": int(episode_id),
        "frames": frames,
        "actions": len(actions),
        "success": bool(success.any()),
    }
    return arrays, record


def _validate_metadata(metadata: Mapping[str, object]) -> list[int]:
    env_info = metadata.get("env_info")
    if not isinstance(env_info, Mapping) or env_info.get("env_id") != EXPECTED_TASK:
        raise ValueError(f"expected {EXPECTED_TASK} replay metadata")
    env_kwargs = env_info.get("env_kwargs")
    if not isinstance(env_kwargs, Mapping):
        raise ValueError("replay metadata is missing env kwargs")
    if env_kwargs.get("control_mode") != EXPECTED_CONTROL_MODE:
        raise ValueError(f"expected control mode {EXPECTED_CONTROL_MODE}")
    if env_kwargs.get("obs_mode") != EXPECTED_OBS_MODE:
        raise ValueError(f"expected observation mode {EXPECTED_OBS_MODE}")
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("replay metadata is missing episodes")
    episode_ids = [int(episode["episode_id"]) for episode in episodes]
    if len(episode_ids) < 2 or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("replay metadata must contain at least two unique episodes")
    return episode_ids


def convert_replay_root(
    replay_h5: str | Path,
    replay_json: str | Path,
    output_root: str | Path,
    *,
    point_count: int = 1024,
    seed: int = SAMPLING_SEED,
) -> dict[str, object]:
    """Atomically publish compact episodes and train-only proprio statistics."""
    replay_h5 = Path(replay_h5)
    replay_json = Path(replay_json)
    output_root = Path(output_root)
    if not replay_h5.is_file() or not replay_json.is_file():
        raise FileNotFoundError("replay H5 and JSON must both exist")
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    metadata = json.loads(replay_json.read_text(encoding="utf-8"))
    episode_ids = _validate_metadata(metadata)
    train_ids, val_ids = split_episode_ids(episode_ids, seed=seed)
    train_set = set(train_ids)

    staging = output_root.with_name(f".{output_root.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"staging root already exists: {staging}")
    staging.mkdir(parents=True)
    records: list[dict[str, object]] = []
    total = np.zeros(25, dtype=np.float64)
    total_square = np.zeros(25, dtype=np.float64)
    sample_count = 0
    try:
        with h5py.File(replay_h5, "r") as replay:
            if set(replay) != {f"traj_{value}" for value in episode_ids}:
                raise ValueError("replay H5 groups do not exactly match metadata episode IDs")
            for episode_id in episode_ids:
                arrays, record = prepare_episode(
                    replay[f"traj_{episode_id}"],
                    episode_id=episode_id,
                    point_count=point_count,
                )
                atomic_save_npz(staging / f"episode_{episode_id:04d}.npz", arrays)
                records.append(record)
                if episode_id in train_set:
                    values = proprio_array(arrays).astype(np.float64, copy=False)
                    total += values.sum(axis=0, dtype=np.float64)
                    total_square += np.square(values).sum(axis=0, dtype=np.float64)
                    sample_count += len(values)
        if sample_count == 0:
            raise ValueError("training split contains no proprioception samples")
        mean = total / sample_count
        variance = np.maximum(total_square / sample_count - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(variance), PROPRIO_STD_FLOOR)
        manifest: dict[str, object] = {
            "task": EXPECTED_TASK,
            "control_mode": EXPECTED_CONTROL_MODE,
            "obs_mode": EXPECTED_OBS_MODE,
            "scene_only": True,
            "frame_skip": 5,
            "point_count": int(point_count),
            "sampling_seed": int(seed),
            "split_seed": int(seed),
            "xyz_low": XYZ_LOW.tolist(),
            "xyz_high": XYZ_HIGH.tolist(),
            "episode_ids": episode_ids,
            "train_episode_ids": train_ids,
            "val_episode_ids": val_ids,
            "episodes": records,
            "point_features": ["xyz", "rgb"],
            "rgb_dtype": "uint8",
            "rgb_normalization": "divide_by_255_at_load_time",
            "rgb_alignment": "same validity mask and sample indices as xyz",
            "robot_uid": "panda",
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
                "sample_count": int(sample_count),
                "mean": mean.astype(np.float32).tolist(),
                "std": std.astype(np.float32).tolist(),
                "std_floor": PROPRIO_STD_FLOOR,
            },
            "object_position_contract": {
                "source": "PickCube-v0 env_states",
                "state_dim": PICKCUBE_STATE_DIM,
                "slice": [PICKCUBE_OBJECT_POSITION.start, PICKCUBE_OBJECT_POSITION.stop],
                "verified_against_simulator": True,
            },
            "source_replay_h5": str(replay_h5.resolve()),
            "source_replay_h5_sha256": _sha256(replay_h5),
            "source_replay_json": str(replay_json.resolve()),
            "source_replay_json_sha256": _sha256(replay_json),
        }
        _atomic_write_json(staging / "manifest.json", manifest)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-h5", type=Path, required=True)
    parser.add_argument("--replay-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--point-count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=SAMPLING_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = convert_replay_root(
        args.replay_h5,
        args.replay_json,
        args.output_root,
        point_count=args.point_count,
        seed=args.seed,
    )
    print(
        f"PREPARED task={manifest['task']} episodes={len(manifest['episode_ids'])} "
        f"output={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
