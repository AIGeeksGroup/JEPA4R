from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import h5py


DATA_SCRIPTS = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(DATA_SCRIPTS))

from maniskill_pickcube import (  # noqa: E402
    align_actions_to_states,
    load_episode_shard,
    sample_point_cloud,
    save_episode_shard,
    validate_episode_arrays,
)


def test_align_actions_to_states_pads_only_the_terminal_state():
    actions = np.arange(24, dtype=np.float32).reshape(3, 8)

    aligned = align_actions_to_states(actions, num_states=4)

    np.testing.assert_array_equal(aligned[:3], actions)
    assert aligned.shape == (4, 8)
    assert np.isnan(aligned[-1]).all()


def test_align_actions_to_states_rejects_wrong_lengths():
    with pytest.raises(ValueError, match="one more state"):
        align_actions_to_states(np.zeros((3, 8), np.float32), num_states=3)


def test_sample_point_cloud_is_deterministic_and_filters_invalid_points():
    xyzw = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0, 1.0],
            [3.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rgb = np.arange(15, dtype=np.uint8).reshape(5, 3)

    xyz_a, rgb_a = sample_point_cloud(xyzw, rgb, num_points=5, seed=17)
    xyz_b, rgb_b = sample_point_cloud(xyzw, rgb, num_points=5, seed=17)

    np.testing.assert_array_equal(xyz_a, xyz_b)
    np.testing.assert_array_equal(rgb_a, rgb_b)
    assert xyz_a.shape == (5, 3)
    assert rgb_a.shape == (5, 3)
    assert np.isfinite(xyz_a).all()
    assert set(xyz_a[:, 0]).issubset({0.0, 1.0, 3.0})


def test_sample_point_cloud_applies_workspace_bounds():
    xyzw = np.array(
        [[-0.2, 0.0, 0.1, 1.0], [2.0, 0.0, 0.1, 1.0]], dtype=np.float32
    )
    rgb = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)

    xyz, colors = sample_point_cloud(
        xyzw,
        rgb,
        num_points=3,
        seed=1,
        bounds=((-0.8, 0.5), (-0.6, 0.6), (-0.05, 0.8)),
    )

    expected_xyz = np.tile(np.array([[-0.2, 0.0, 0.1]], dtype=np.float32), (3, 1))
    np.testing.assert_array_equal(xyz, expected_xyz)
    np.testing.assert_array_equal(colors, np.tile([[1, 2, 3]], (3, 1)))


def test_validate_episode_arrays_accepts_one_terminal_nan_action():
    arrays = {
        "rgb": np.zeros((4, 8, 8, 3), dtype=np.uint8),
        "depth": np.zeros((4, 8, 8, 1), dtype=np.int16),
        "pcd_xyz": np.zeros((4, 12, 3), dtype=np.float32),
        "pcd_rgb": np.zeros((4, 12, 3), dtype=np.uint8),
        "proprio": np.zeros((4, 25), dtype=np.float32),
        "action": np.vstack(
            [np.zeros((3, 8), dtype=np.float32), np.full((1, 8), np.nan, dtype=np.float32)]
        ),
    }

    validate_episode_arrays(arrays, expected_frames=4, num_points=12)


def test_validate_episode_arrays_rejects_nonterminal_nan_action():
    arrays = {
        "rgb": np.zeros((4, 8, 8, 3), dtype=np.uint8),
        "depth": np.zeros((4, 8, 8, 1), dtype=np.int16),
        "pcd_xyz": np.zeros((4, 12, 3), dtype=np.float32),
        "pcd_rgb": np.zeros((4, 12, 3), dtype=np.uint8),
        "proprio": np.zeros((4, 25), dtype=np.float32),
        "action": np.vstack(
            [np.zeros((2, 8), dtype=np.float32), np.full((2, 8), np.nan, dtype=np.float32)]
        ),
    }

    with pytest.raises(ValueError, match="terminal action"):
        validate_episode_arrays(arrays, expected_frames=4, num_points=12)


def test_validate_episode_arrays_rejects_any_misaligned_extra_column():
    arrays = {
        "rgb": np.zeros((4, 8, 8, 3), dtype=np.uint8),
        "depth": np.zeros((4, 8, 8, 1), dtype=np.int16),
        "pcd_xyz": np.zeros((4, 12, 3), dtype=np.float32),
        "pcd_rgb": np.zeros((4, 12, 3), dtype=np.uint8),
        "proprio": np.zeros((4, 25), dtype=np.float32),
        "action": np.vstack(
            [np.zeros((3, 8), dtype=np.float32), np.full((1, 8), np.nan, dtype=np.float32)]
        ),
        "goal_pos": np.zeros((3, 3), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="goal_pos has 3 frames"):
        validate_episode_arrays(arrays, expected_frames=4, num_points=12)


def test_episode_shard_round_trip_is_atomic_and_valid(tmp_path):
    arrays = {
        "rgb": np.zeros((2, 8, 8, 3), dtype=np.uint8),
        "depth": np.zeros((2, 8, 8, 1), dtype=np.int16),
        "pcd_xyz": np.zeros((2, 12, 3), dtype=np.float32),
        "pcd_rgb": np.zeros((2, 12, 3), dtype=np.uint8),
        "proprio": np.zeros((2, 25), dtype=np.float32),
        "action": np.vstack(
            [np.zeros((1, 8), dtype=np.float32), np.full((1, 8), np.nan, dtype=np.float32)]
        ),
    }
    path = tmp_path / "episode_0000.npz"

    save_episode_shard(path, arrays)
    loaded = load_episode_shard(path, expected_frames=2, num_points=12)

    assert path.exists()
    assert not (tmp_path / "episode_0000.npz.tmp").exists()
    assert loaded.keys() == arrays.keys()
    for key in arrays:
        np.testing.assert_array_equal(loaded[key], arrays[key])


def test_collection_cli_exposes_resumable_episode_range():
    script = DATA_SCRIPTS / "collect_pickcube_maniskill.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--start-episode" in result.stdout
    assert "--num-episodes" in result.stdout
    assert "--output-dir" in result.stdout


def test_check_cli_exposes_full_dataset_gate():
    script = Path(__file__).resolve().parent / "check_pickcube_maniskill.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--expected-episodes" in result.stdout
    assert "--write-manifest" in result.stdout


def test_merge_cli_writes_stable_worldmodel_episode_index(tmp_path):
    script = DATA_SCRIPTS / "merge_pickcube_maniskill.py"
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    episode_lengths = [2, 3]
    episodes = []
    for episode_id, length in enumerate(episode_lengths):
        arrays = {
            "rgb": np.zeros((length, 8, 8, 3), dtype=np.uint8),
            "depth": np.zeros((length, 8, 8, 1), dtype=np.int16),
            "pcd_xyz": np.full((length, 12, 3), episode_id, dtype=np.float32),
            "pcd_rgb": np.zeros((length, 12, 3), dtype=np.uint8),
            "proprio": np.zeros((length, 25), dtype=np.float32),
            "action": np.vstack(
                [
                    np.zeros((length - 1, 8), dtype=np.float32),
                    np.full((1, 8), np.nan, dtype=np.float32),
                ]
            ),
            "episode_idx": np.full(length, episode_id, dtype=np.int32),
            "step_idx": np.arange(length, dtype=np.int32),
        }
        save_episode_shard(shard_dir / f"episode_{episode_id:04d}.npz", arrays)
        episodes.append(
            {
                "episode_id": episode_id,
                "elapsed_steps": length - 1,
                "success": True,
            }
        )
    metadata_path = tmp_path / "trajectory.json"
    metadata_path.write_text(
        __import__("json").dumps(
            {
                "env_info": {
                    "env_id": "PickCube-v1",
                    "env_kwargs": {"control_mode": "pd_joint_pos"},
                },
                "episodes": episodes,
            }
        )
    )
    output_h5 = tmp_path / "merged.h5"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--metadata-json",
            str(metadata_path),
            "--shard-dir",
            str(shard_dir),
            "--output-h5",
            str(output_h5),
            "--num-episodes",
            "2",
            "--num-points",
            "12",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    with h5py.File(output_h5, "r") as h5:
        np.testing.assert_array_equal(h5["ep_len"][:], episode_lengths)
        np.testing.assert_array_equal(h5["ep_offset"][:], [0, 2])
        assert h5["pcd_xyz"].shape == (5, 12, 3)
        assert h5["action"].shape == (5, 8)
        assert h5.attrs["env_id"] == "PickCube-v1"
