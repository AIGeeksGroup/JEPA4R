import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.prepare_liftcube_rgb_compact import (
    aligned_sample_indices,
    atomic_save_npz,
    extract_aligned_frame,
    prepare_episode,
    prepare_root,
)


def _raw_frame():
    xyzw = np.array(
        [
            [-0.4, 0.0, 0.1, 1.0],
            [-0.3, 0.0, 0.1, 1.0],
            [-0.2, 0.0, 0.1, 0.0],
            [-0.1, 0.0, 0.1, 1.0],
            [0.0, 0.0, 0.1, 1.0],
            [2.0, 0.0, 0.1, 1.0],
        ],
        dtype=np.float32,
    )
    rgb = np.arange(18, dtype=np.uint8).reshape(6, 3)
    robot = np.array([[False], [True], [False], [False], [False], [False]])
    return xyzw, rgb, robot


def _source_episode(xyz):
    frames = xyz.shape[0]
    return {
        "xyz": xyz,
        "action": np.zeros((frames - 1, 7), dtype=np.float32),
        "object_pos": np.zeros((frames, 3), dtype=np.float32),
        "goal_pos": np.ones(3, dtype=np.float32),
        "success": np.zeros(frames - 1, dtype=bool),
    }


def _write_fixture(tmp_path: Path):
    replay = tmp_path / "replay.h5"
    source_root = tmp_path / "source"
    source_root.mkdir()
    xyzw, rgb, robot = _raw_frame()
    clouds = np.stack([xyzw, xyzw])
    colors = np.stack([rgb, rgb + 1])
    masks = np.stack([robot, robot])
    with h5py.File(replay, "w") as file:
        group = file.create_group("traj_0")
        group.create_dataset("actions", data=np.zeros((1, 7), dtype=np.float32))
        group.create_dataset("obs/pointcloud/xyzw", data=clouds)
        group.create_dataset("obs/pointcloud/rgb", data=colors)
        group.create_dataset("obs/pointcloud/robot_seg", data=masks)
    expected = np.stack(
        [
            extract_aligned_frame(xyzw, colors[i], robot, 0, i, point_count=2)[0]
            for i in range(2)
        ]
    )
    np.savez_compressed(source_root / "episode_0000.npz", **_source_episode(expected))
    manifest = {
        "task": "LiftCube-v0",
        "episode_ids": [0],
        "train_episode_ids": [0],
        "val_episode_ids": [],
        "point_count": 2,
    }
    (source_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return replay, source_root, expected


def test_extract_frame_uses_one_mask_and_one_sample_for_xyz_and_rgb():
    xyzw, rgb, robot = _raw_frame()
    indices = aligned_sample_indices(xyzw, robot, 7, 3, point_count=2)

    xyz_out, rgb_out = extract_aligned_frame(
        xyzw, rgb, robot, episode_id=7, frame_id=3, point_count=2
    )

    assert np.array_equal(xyz_out, xyzw[indices, :3])
    assert np.array_equal(rgb_out, rgb[indices])
    assert rgb_out.dtype == np.uint8
    assert not np.isin(indices, [1, 2, 5]).any()


def test_prepare_episode_requires_exact_reference_xyz(tmp_path):
    replay, source_root, _ = _write_fixture(tmp_path)
    with np.load(source_root / "episode_0000.npz", allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}
    source["xyz"] = source["xyz"].copy()
    source["xyz"][0, 0, 0] += 1e-6

    with h5py.File(replay, "r") as file:
        with pytest.raises(ValueError, match="does not exactly match"):
            prepare_episode(file["traj_0"], source, episode_id=0, point_count=2)


def test_prepare_episode_keeps_source_fields_and_adds_uint8_rgb(tmp_path):
    replay, source_root, expected_xyz = _write_fixture(tmp_path)
    with np.load(source_root / "episode_0000.npz", allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}

    with h5py.File(replay, "r") as file:
        result, record = prepare_episode(
            file["traj_0"], source, episode_id=0, point_count=2
        )

    assert np.array_equal(result["xyz"], expected_xyz)
    for key in ("action", "object_pos", "goal_pos", "success"):
        assert np.array_equal(result[key], source[key])
    assert result["rgb"].shape == (2, 2, 3)
    assert result["rgb"].dtype == np.uint8
    assert record["episode_id"] == 0
    assert record["frames"] == 2


def test_prepare_root_refuses_nonempty_output(tmp_path):
    output = tmp_path / "already-there"
    output.mkdir()
    (output / "keep.txt").write_text("owned by user", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        prepare_root(
            tmp_path / "missing.h5",
            tmp_path / "missing-source",
            output,
        )

    assert (output / "keep.txt").read_text(encoding="utf-8") == "owned by user"


def test_atomic_npz_leaves_no_temporary_file(tmp_path):
    output = tmp_path / "episode_0000.npz"
    atomic_save_npz(output, {"xyz": np.zeros((1, 2, 3), dtype=np.float32)})

    with np.load(output, allow_pickle=False) as archive:
        assert archive["xyz"].shape == (1, 2, 3)
    assert not output.with_suffix(".npz.tmp").exists()
