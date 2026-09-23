import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.checks.check_mixed_ur5e_cube_pcd import inspect_mixed_datasets
from scripts.data.build_mixed_ur5e_cube_pcd_vds import build_mixed_datasets


COMMON_KEYS = ("pcd_xyz", "pcd_rgb", "action")


def _write_source(
    path: Path,
    lengths,
    marker: int,
    action_dim: int = 5,
    robot_qpos_dim: int | None = None,
):
    lengths = np.asarray(lengths, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1]))).astype(np.int64)
    frames = int(lengths.sum())
    cube_count = 1 if marker == 1 else 2
    qpos_dim = robot_qpos_dim or 14 + 7 * cube_count
    with h5py.File(path, "w") as h5:
        h5.create_dataset("ep_len", data=lengths)
        h5.create_dataset("ep_offset", data=offsets)
        h5.create_dataset(
            "pcd_xyz", data=np.full((frames, 4, 3), marker, dtype=np.float32)
        )
        h5.create_dataset(
            "pcd_rgb", data=np.full((frames, 4, 3), marker, dtype=np.uint8)
        )
        h5.create_dataset(
            "action", data=np.full((frames, action_dim), marker, dtype=np.float32)
        )
        h5.create_dataset("qpos", data=np.zeros((frames, qpos_dim), dtype=np.float64))
        h5.create_dataset(
            "qvel", data=np.zeros((frames, 14 + 6 * cube_count), dtype=np.float64)
        )
        h5.create_dataset("control", data=np.zeros((frames, 7), dtype=np.float64))
        h5.create_dataset(
            "observation", data=np.zeros((frames, marker + 2), dtype=np.float32)
        )


def test_builds_disjoint_train_and_validation_virtual_datasets(tmp_path):
    one = tmp_path / "one.h5"
    two_train = tmp_path / "two_train.h5"
    two_holdout = tmp_path / "two_holdout.h5"
    _write_source(one, [3, 4, 5], 1)
    _write_source(two_train, [2, 3], 2)
    _write_source(two_holdout, [4], 3)

    output = tmp_path / "mixed"
    manifest = build_mixed_datasets(
        one_cube=one,
        two_cube_train=two_train,
        two_cube_holdout=two_holdout,
        output_dir=output,
        one_cube_validation_episodes=1,
    )

    assert manifest["train"]["episodes"] == 4
    assert manifest["validation"]["episodes"] == 2
    assert manifest["one_cube_train_episode_range"] == [0, 2]
    assert manifest["one_cube_validation_episode_range"] == [2, 3]
    assert json.loads((output / "manifest.json").read_text()) == manifest

    with h5py.File(output / "train.h5", "r") as h5:
        assert np.array_equal(h5["ep_len"][:], [3, 4, 2, 3])
        assert np.array_equal(h5["ep_offset"][:], [0, 3, 7, 9])
        assert np.array_equal(h5["episode_source"][:], [0, 0, 1, 1])
        assert np.array_equal(h5["source_episode_index"][:], [0, 1, 0, 1])
        assert all(h5[key].is_virtual for key in COMMON_KEYS)
        assert np.all(h5["action"][:7] == 1)
        assert np.all(h5["action"][7:] == 2)

    with h5py.File(output / "validation.h5", "r") as h5:
        assert np.array_equal(h5["ep_len"][:], [5, 4])
        assert np.array_equal(h5["episode_source"][:], [0, 2])
        assert np.all(h5["pcd_xyz"][:5] == 1)
        assert np.all(h5["pcd_xyz"][5:] == 3)


def test_rejects_incompatible_common_field_shape(tmp_path):
    one = tmp_path / "one.h5"
    two_train = tmp_path / "two_train.h5"
    two_holdout = tmp_path / "two_holdout.h5"
    _write_source(one, [3, 4], 1)
    _write_source(two_train, [2], 2, action_dim=4)
    _write_source(two_holdout, [2], 3)

    with pytest.raises(ValueError, match="action.*shape"):
        build_mixed_datasets(
            one_cube=one,
            two_cube_train=two_train,
            two_cube_holdout=two_holdout,
            output_dir=tmp_path / "mixed",
            one_cube_validation_episodes=1,
        )


def test_rejects_non_ur5e_robot_state_contract(tmp_path):
    one = tmp_path / "one.h5"
    two_train = tmp_path / "two_train.h5"
    two_holdout = tmp_path / "two_holdout.h5"
    _write_source(one, [3, 4], 1, robot_qpos_dim=22)
    _write_source(two_train, [2], 2)
    _write_source(two_holdout, [2], 3)

    with pytest.raises(ValueError, match="qpos.*UR5e"):
        build_mixed_datasets(
            one_cube=one,
            two_cube_train=two_train,
            two_cube_holdout=two_holdout,
            output_dir=tmp_path / "mixed",
            one_cube_validation_episodes=1,
        )


def test_acceptance_report_checks_sources_actions_and_samples(tmp_path):
    one = tmp_path / "one.h5"
    two_train = tmp_path / "two_train.h5"
    two_holdout = tmp_path / "two_holdout.h5"
    _write_source(one, [3, 4, 5], 1)
    _write_source(two_train, [2, 3], 2)
    _write_source(two_holdout, [4], 3)
    output = tmp_path / "mixed"
    build_mixed_datasets(
        one_cube=one,
        two_cube_train=two_train,
        two_cube_holdout=two_holdout,
        output_dir=output,
        one_cube_validation_episodes=1,
    )

    report = inspect_mixed_datasets(
        train_path=output / "train.h5",
        validation_path=output / "validation.h5",
        sample_count=4,
        seed=7,
    )

    assert report["status"] == "PASS"
    assert report["train"]["episodes"] == 4
    assert report["validation"]["episodes"] == 2
    assert report["one_cube_episode_overlap"] == 0
    assert report["train"]["all_common_fields_virtual"] is True


def test_mixed_training_config_loads_only_common_fields():
    config_path = (
        Path(__file__).parents[2] / "config/train/data/ogb_ur5e_mixed_pcd.yaml"
    )
    text = config_path.read_text()
    assert "UR5E_MIXED_TRAIN_DATA" in text
    assert "UR5E_MIXED_VALIDATION_DATA" in text
    keys_block = text.split("keys_to_load:", 1)[1].split("keys_to_cache:", 1)[0]
    assert "pcd_xyz" in keys_block
    assert "pcd_rgb" in keys_block
    assert "action" in keys_block
    assert "observation" not in keys_block


def test_mixed_training_runner_has_persistent_smoke_and_formal_gates():
    runner = Path(__file__).parents[1] / "train/run_ur5e_mixed_cube_training.sh"
    text = runner.read_text()
    assert "set -euo pipefail" in text
    assert "ogbench-ur5e-mixed-1cube-2cube-pcd-v2" in text
    assert "state.json" in text
    assert "exit_code" in text
    assert "flock" in text
    assert "SMOKE_TRAINING" in text
    assert "FORMAL_TRAINING" in text
    assert "check_ur5e_holdout_checkpoint.py" in text
    assert 'run_training "$RUN" 10 128 no' in text
    assert "workers=6" in text
    assert 'loader.num_workers="$workers"' in text
    assert 'loader.persistent_workers="$persistent"' in text
    assert "model/encoder=dexart" in text
