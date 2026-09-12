import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.prepare_maniskill_compact import (
    convert_replay_root,
    prepare_episode,
    split_episode_ids,
)


def _write_trajectory(handle: h5py.File, episode_id: int, *, goal_mismatch: bool = False) -> None:
    frames = 3
    points = 6
    group = handle.create_group(f"traj_{episode_id}")
    group.create_dataset("actions", data=np.full((frames - 1, 7), episode_id, np.float32))
    group.create_dataset("success", data=np.array([False, True], dtype=bool))

    observations = group.create_group("obs")
    agent = observations.create_group("agent")
    agent.create_dataset("qpos", data=np.full((frames, 9), episode_id + 1, np.float32))
    agent.create_dataset("qvel", data=np.full((frames, 9), episode_id + 2, np.float32))
    agent.create_dataset("base_pose", data=np.zeros((frames, 7), np.float32))
    extra = observations.create_group("extra")
    extra.create_dataset("tcp_pose", data=np.full((frames, 7), episode_id + 3, np.float32))
    goal = np.array([0.08, -0.06, 0.24], dtype=np.float32)
    extra.create_dataset("goal_pos", data=np.repeat(goal[None], frames, axis=0))

    xyzw = np.zeros((frames, points, 4), dtype=np.float32)
    rgb = np.zeros((frames, points, 3), dtype=np.uint8)
    for frame_id in range(frames):
        xyzw[frame_id, :, 0] = np.arange(points, dtype=np.float32) * 0.01
        xyzw[frame_id, :, 1] = frame_id * 0.01
        xyzw[frame_id, :, 2] = 0.02
        xyzw[frame_id, :, 3] = 1.0
        rgb[frame_id, :, 0] = np.arange(points, dtype=np.uint8)
    pointcloud = observations.create_group("pointcloud")
    pointcloud.create_dataset("xyzw", data=xyzw)
    pointcloud.create_dataset("rgb", data=rgb)
    robot_seg = np.zeros((frames, points, 1), dtype=bool)
    robot_seg[:, 0] = True
    pointcloud.create_dataset("robot_seg", data=robot_seg)

    env_states = np.zeros((frames, 73), dtype=np.float32)
    object_pos = np.array(
        [[0.01, 0.02, 0.02], [0.02, 0.02, 0.08], [0.03, 0.02, 0.15]],
        dtype=np.float32,
    )
    env_states[:, 13:16] = object_pos + episode_id * 0.001
    env_states[:, -3:] = goal
    if goal_mismatch:
        env_states[1, -1] += 0.1
    group.create_dataset("env_states", data=env_states)


def _write_replay(tmp_path: Path, count: int = 2) -> tuple[Path, Path]:
    h5_path = tmp_path / "trajectory.h5"
    json_path = tmp_path / "trajectory.json"
    with h5py.File(h5_path, "w") as handle:
        for episode_id in range(count):
            _write_trajectory(handle, episode_id)
    json_path.write_text(
        json.dumps(
            {
                "env_info": {
                    "env_id": "PickCube-v0",
                    "env_kwargs": {
                        "obs_mode": "pointcloud_robot_seg",
                        "control_mode": "pd_ee_delta_pose",
                    },
                },
                "episodes": [{"episode_id": value} for value in range(count)],
            }
        ),
        encoding="utf-8",
    )
    return h5_path, json_path


def test_prepare_episode_aligns_rgb_proprio_and_pickcube_state(tmp_path: Path) -> None:
    h5_path, _ = _write_replay(tmp_path, count=1)
    with h5py.File(h5_path, "r") as handle:
        arrays, record = prepare_episode(handle["traj_0"], episode_id=0, point_count=3)

    assert arrays["xyz"].shape == (3, 3, 3)
    assert arrays["rgb"].shape == arrays["xyz"].shape
    np.testing.assert_array_equal(
        np.rint(arrays["xyz"][..., 0] * 100).astype(np.uint8), arrays["rgb"][..., 0]
    )
    np.testing.assert_allclose(
        arrays["object_pos"],
        [[0.01, 0.02, 0.02], [0.02, 0.02, 0.08], [0.03, 0.02, 0.15]],
    )
    np.testing.assert_allclose(arrays["goal_pos"], [0.08, -0.06, 0.24])
    assert arrays["qpos"].shape == (3, 9)
    assert arrays["qvel"].shape == (3, 9)
    assert arrays["tcp_pose"].shape == (3, 7)
    assert arrays["base_pose"].shape == (3, 7)
    assert arrays["action"].shape == (2, 7)
    assert arrays["success"].tolist() == [False, True]
    assert record == {"episode_id": 0, "frames": 3, "actions": 2, "success": True}


def test_prepare_episode_rejects_goal_state_mismatch(tmp_path: Path) -> None:
    h5_path = tmp_path / "trajectory.h5"
    with h5py.File(h5_path, "w") as handle:
        _write_trajectory(handle, 0, goal_mismatch=True)
    with h5py.File(h5_path, "r") as handle:
        with pytest.raises(ValueError, match="goal position differs from env_states"):
            prepare_episode(handle["traj_0"], episode_id=0, point_count=3)


def test_split_episode_ids_is_seeded_disjoint_and_ten_percent() -> None:
    episode_ids = list(range(20))
    train, val = split_episode_ids(episode_ids, seed=3072)

    assert len(train) == 18
    assert len(val) == 2
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(episode_ids)
    assert (train, val) == split_episode_ids(episode_ids, seed=3072)


def test_convert_replay_root_publishes_manifest_and_train_only_stats(tmp_path: Path) -> None:
    replay_h5, replay_json = _write_replay(tmp_path)
    output_root = tmp_path / "compact"

    manifest = convert_replay_root(
        replay_h5,
        replay_json,
        output_root,
        point_count=3,
        seed=3072,
    )

    assert sorted(path.name for path in output_root.glob("episode_*.npz")) == [
        "episode_0000.npz",
        "episode_0001.npz",
    ]
    assert json.loads((output_root / "manifest.json").read_text()) == manifest
    assert manifest["task"] == "PickCube-v0"
    assert len(manifest["train_episode_ids"]) == 1
    assert len(manifest["val_episode_ids"]) == 1
    assert set(manifest["train_episode_ids"]).isdisjoint(manifest["val_episode_ids"])
    assert set(manifest["train_episode_ids"]) | set(manifest["val_episode_ids"]) == {0, 1}
    train_id = manifest["train_episode_ids"][0]
    with np.load(output_root / f"episode_{train_id:04d}.npz") as episode:
        proprio = np.concatenate(
            [episode["qpos"], episode["qvel"], episode["tcp_pose"]], axis=-1
        )
    np.testing.assert_allclose(manifest["proprio_normalization"]["mean"], proprio.mean(0))
    assert manifest["proprio_normalization"]["sample_count"] == len(proprio)
    assert manifest["source_replay_h5_sha256"]
    assert manifest["source_replay_json_sha256"]
