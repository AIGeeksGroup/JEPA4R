import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.prepare_liftcube_proprio_compact import (
    PROPRIO_LAYOUT,
    atomic_save_npz,
    compute_proprio_stats,
    prepare_episode,
    prepare_root,
    proprio_array,
)


SOURCE_FIELDS = ("xyz", "rgb", "action", "object_pos", "goal_pos", "success")


def _source_episode(frames: int = 4, points: int = 3) -> dict[str, np.ndarray]:
    return {
        "xyz": np.arange(frames * points * 3, dtype=np.float32).reshape(
            frames, points, 3
        ),
        "rgb": np.arange(frames * points * 3, dtype=np.uint8).reshape(
            frames, points, 3
        ),
        "action": np.arange((frames - 1) * 7, dtype=np.float32).reshape(
            frames - 1, 7
        ),
        "object_pos": np.arange(frames * 3, dtype=np.float32).reshape(frames, 3),
        "goal_pos": np.array([0.0, 0.0, 0.22], dtype=np.float32),
        "success": np.zeros(frames - 1, dtype=bool),
    }


def _state_arrays(frames: int, value: float = 1.0) -> dict[str, np.ndarray]:
    return {
        "qpos": np.full((frames, 9), value, dtype=np.float32),
        "qvel": np.full((frames, 9), value, dtype=np.float32),
        "tcp_pose": np.full((frames, 7), value, dtype=np.float32),
        "base_pose": np.full((frames, 7), value, dtype=np.float32),
    }


def _write_replay_group(
    group: h5py.Group,
    source: dict[str, np.ndarray],
    *,
    value: float = 1.0,
    shorten: str | None = None,
    nonfinite: str | None = None,
) -> None:
    group.create_dataset("actions", data=source["action"])
    arrays = _state_arrays(len(source["xyz"]), value)
    paths = {
        "qpos": "obs/agent/qpos",
        "qvel": "obs/agent/qvel",
        "tcp_pose": "obs/extra/tcp_pose",
        "base_pose": "obs/agent/base_pose",
    }
    for name, path in paths.items():
        array = arrays[name]
        if shorten == name:
            array = array[:-1]
        if nonfinite == name:
            array = array.copy()
            array[0, 0] = np.nan
        group.create_dataset(path, data=array)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_prepare_episode_preserves_source_and_adds_aligned_proprio(tmp_path):
    source = _source_episode()
    replay = tmp_path / "replay.h5"
    with h5py.File(replay, "w") as file:
        _write_replay_group(file.create_group("traj_7"), source, value=2.0)
    with h5py.File(replay, "r") as file:
        result, record = prepare_episode(file["traj_7"], source, episode_id=7)

    for key in SOURCE_FIELDS:
        np.testing.assert_array_equal(result[key], source[key])
    assert result["qpos"].shape == (4, 9)
    assert result["qvel"].shape == (4, 9)
    assert result["tcp_pose"].shape == (4, 7)
    assert result["base_pose"].shape == (4, 7)
    assert result["qpos"].dtype == np.float32
    assert record["episode_id"] == 7
    assert record["frames"] == 4
    assert record["actions"] == 3
    assert PROPRIO_LAYOUT == [
        {"name": "qpos", "start": 0, "stop": 9},
        {"name": "qvel", "start": 9, "stop": 18},
        {"name": "tcp_pose", "start": 18, "stop": 25},
    ]


@pytest.mark.parametrize("field", ["qpos", "qvel", "tcp_pose", "base_pose"])
def test_prepare_episode_rejects_proprio_length_mismatch(tmp_path, field):
    source = _source_episode()
    replay = tmp_path / "replay.h5"
    with h5py.File(replay, "w") as file:
        _write_replay_group(file.create_group("traj_7"), source, shorten=field)
    with h5py.File(replay, "r") as file:
        with pytest.raises(ValueError, match=field):
            prepare_episode(file["traj_7"], source, episode_id=7)


def test_prepare_episode_rejects_nonfinite_proprio(tmp_path):
    source = _source_episode()
    replay = tmp_path / "replay.h5"
    with h5py.File(replay, "w") as file:
        _write_replay_group(file.create_group("traj_7"), source, nonfinite="qvel")
    with h5py.File(replay, "r") as file:
        with pytest.raises(ValueError, match="qvel.*finite"):
            prepare_episode(file["traj_7"], source, episode_id=7)


def test_prepare_episode_rejects_static_and_action_alignment(tmp_path):
    source = _source_episode()
    source["goal_pos"] = np.zeros((4, 3), dtype=np.float32)
    replay = tmp_path / "replay.h5"
    with h5py.File(replay, "w") as file:
        _write_replay_group(file.create_group("traj_7"), source)
    with h5py.File(replay, "r") as file:
        with pytest.raises(ValueError, match="goal_pos"):
            prepare_episode(file["traj_7"], source, episode_id=7)


class RecordingEpisodes(dict):
    def __init__(self, values):
        super().__init__(values)
        self.opened: list[int] = []

    def __getitem__(self, episode_id):
        self.opened.append(int(episode_id))
        return super().__getitem__(episode_id)


def test_proprio_stats_never_access_validation_episode():
    episodes = RecordingEpisodes(
        {
            0: {**_state_arrays(4, 1.0)},
            1: {**_state_arrays(4, 3.0)},
            9: {**_state_arrays(4, 999.0)},
        }
    )

    mean, std, count = compute_proprio_stats(episodes, [0, 1])

    assert episodes.opened == [0, 1]
    np.testing.assert_allclose(mean, np.full(25, 2.0, dtype=np.float32))
    np.testing.assert_allclose(std, np.full(25, 1.0, dtype=np.float32))
    assert count == 8
    assert proprio_array(episodes[0]).shape == (4, 25)


def _write_full_fixture(tmp_path: Path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    replay_h5 = tmp_path / "replay.h5"
    replay_json = tmp_path / "replay.json"
    train_ids = list(range(99, 9, -1))
    val_ids = list(range(10))
    episode_ids = train_ids + val_ids
    manifest = {
        "task": "LiftCube-v0",
        "episode_ids": episode_ids,
        "train_episode_ids": train_ids,
        "val_episode_ids": val_ids,
        "point_count": 1,
        "point_features": ["xyz", "rgb"],
    }
    (source_root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    replay_metadata = {
        "env_info": {
            "env_id": "LiftCube-v0",
            "env_kwargs": {"control_mode": "pd_ee_delta_pose"},
        },
        "episodes": [
            {"episode_id": episode_id, "control_mode": "pd_ee_delta_pose"}
            for episode_id in episode_ids
        ],
    }
    replay_json.write_text(json.dumps(replay_metadata), encoding="utf-8")
    with h5py.File(replay_h5, "w") as replay:
        for episode_id in episode_ids:
            source = _source_episode(frames=2, points=1)
            np.savez_compressed(
                source_root / f"episode_{episode_id:04d}.npz", **source
            )
            _write_replay_group(
                replay.create_group(f"traj_{episode_id}"),
                source,
                value=float(episode_id),
            )
    return replay_h5, replay_json, source_root, manifest


def test_prepare_root_preserves_split_and_writes_train_only_stats(tmp_path):
    replay_h5, replay_json, source_root, source_manifest = _write_full_fixture(
        tmp_path
    )
    output = tmp_path / "output"

    result = prepare_root(
        replay_h5,
        replay_json,
        source_root,
        output,
        verified_robot_uid="panda",
        verified_agent_class="Panda",
        verified_robot_name="panda_v2",
    )

    assert result["episode_ids"] == source_manifest["episode_ids"]
    assert result["train_episode_ids"] == source_manifest["train_episode_ids"]
    assert result["val_episode_ids"] == source_manifest["val_episode_ids"]
    assert result["robot_uid"] == "panda"
    assert result["robot_identity_verification"] == {
        "agent_class": "Panda",
        "robot_name": "panda_v2",
        "method": "runtime_environment_probe",
    }
    assert result["control_mode"] == "pd_ee_delta_pose"
    assert result["action_dim"] == 7
    assert result["qpos_dim"] == 9
    assert result["proprio_dim"] == 25
    normalization = result["proprio_normalization"]
    assert normalization["source_split"] == "train"
    assert normalization["episode_ids"] == source_manifest["train_episode_ids"]
    assert normalization["sample_count"] == 180
    assert np.isclose(normalization["mean"][0], np.mean(np.arange(10, 100)))
    assert len(result["episodes"]) == 100
    with np.load(output / "episode_0000.npz", allow_pickle=False) as episode:
        assert set(episode.files) == set(SOURCE_FIELDS) | {
            "qpos",
            "qvel",
            "tcp_pose",
            "base_pose",
        }


def test_prepare_root_verify_only_does_not_change_published_files(tmp_path):
    replay_h5, replay_json, source_root, _ = _write_full_fixture(tmp_path)
    output = tmp_path / "output"
    prepare_root(
        replay_h5,
        replay_json,
        source_root,
        output,
        verified_robot_uid="panda",
        verified_agent_class="Panda",
        verified_robot_name="panda_v2",
    )
    before = {
        path.name: _sha256(path) for path in sorted(output.iterdir()) if path.is_file()
    }

    verified = prepare_root(
        replay_h5,
        replay_json,
        source_root,
        output,
        verified_robot_uid="panda",
        verified_agent_class="Panda",
        verified_robot_name="panda_v2",
        verify_only=True,
    )

    after = {
        path.name: _sha256(path) for path in sorted(output.iterdir()) if path.is_file()
    }
    assert before == after
    assert len(verified["episodes"]) == 100


def test_prepare_root_refuses_nonempty_unverified_output(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "owned.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        prepare_root(
            tmp_path / "missing.h5",
            tmp_path / "missing.json",
            tmp_path / "source",
            output,
            verified_robot_uid="panda",
            verified_agent_class="Panda",
            verified_robot_name="panda_v2",
        )
    assert (output / "owned.txt").read_text(encoding="utf-8") == "keep"


def test_atomic_npz_removes_temporary_file(tmp_path):
    output = tmp_path / "episode_0000.npz"
    atomic_save_npz(output, {"qpos": np.ones((2, 9), dtype=np.float32)})
    with np.load(output, allow_pickle=False) as episode:
        assert episode["qpos"].shape == (2, 9)
    assert not output.with_suffix(".npz.tmp").exists()
