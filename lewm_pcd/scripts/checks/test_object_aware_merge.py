from pathlib import Path
import json
import subprocess
import sys

import h5py
import numpy as np


DATA_SCRIPTS = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(DATA_SCRIPTS))

from maniskill_pickcube import save_episode_shard  # noqa: E402


def test_object_aware_merge_preserves_audit_groups_and_protocol_attributes(tmp_path):
    shard_dir = tmp_path / "episodes"
    shard_dir.mkdir()
    frames, points = 2, 12
    groups = np.tile(np.array([1, 2, 3, 4], dtype=np.uint8), (frames, 3))
    arrays = {
        "rgb": np.zeros((frames, 8, 8, 3), dtype=np.uint8),
        "depth": np.zeros((frames, 8, 8, 1), dtype=np.int16),
        "pcd_xyz": np.zeros((frames, points, 3), dtype=np.float32),
        "pcd_rgb": np.zeros((frames, points, 3), dtype=np.uint8),
        "pcd_group": groups,
        "proprio": np.zeros((frames, 25), dtype=np.float32),
        "action": np.vstack(
            [np.zeros((1, 8), dtype=np.float32), np.full((1, 8), np.nan, dtype=np.float32)]
        ),
    }
    save_episode_shard(shard_dir / "episode_0000.npz", arrays)
    metadata = {
        "env_info": {
            "env_id": "PickCube-v1",
            "env_kwargs": {"control_mode": "pd_joint_pos"},
        },
        "episodes": [{"episode_id": 0, "elapsed_steps": 1, "success": True}],
    }
    metadata_path = tmp_path / "trajectory.json"
    metadata_path.write_text(json.dumps(metadata))
    output = tmp_path / "object_aware.h5"

    result = subprocess.run(
        [
            sys.executable,
            str(DATA_SCRIPTS / "merge_pickcube_maniskill.py"),
            "--metadata-json",
            str(metadata_path),
            "--shard-dir",
            str(shard_dir),
            "--output-h5",
            str(output),
            "--num-episodes",
            "1",
            "--num-points",
            str(points),
            "--sampling-protocol",
            "object_aware_v2",
            "--extra-resolution",
            "320",
            "--voxel-size",
            "0.0025",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    with h5py.File(output, "r") as h5:
        np.testing.assert_array_equal(h5["pcd_group"][:], groups)
        assert h5.attrs["schema_version"] == 2
        assert h5.attrs["sampling_protocol"] == "object_aware_v2"
        assert h5.attrs["cube_min_points"] == 3
        assert h5.attrs["extra_resolution"] == 320
        assert h5.attrs["voxel_size"] == 0.0025
