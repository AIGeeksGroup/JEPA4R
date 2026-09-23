"""A mixed checkpoint must be paired with the data that fitted its actions."""

import sys
from pathlib import Path

import pytest


def test_transfer_cli_accepts_explicit_mixed_model_and_action_source(tmp_path):
    from scripts.eval.evaluate_ur5e_2cube_to_1cube_local25 import build_parser

    args = build_parser().parse_args(
        [
            "--policy", "cem", "--selection", "all",
            "--checkpoint", str(tmp_path / "weights_epoch_2.pt"),
            "--scaler-train", str(tmp_path / "train.h5"),
        ]
    )
    assert args.checkpoint == tmp_path / "weights_epoch_2.pt"
    assert args.scaler_train == tmp_path / "train.h5"


def test_two_cube_cli_accepts_explicit_mixed_model_and_action_source(
    tmp_path, monkeypatch
):
    from scripts.eval.evaluate_ur5e_local_goal_pcd import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate", "--task", "cube_double_task1_single_pnp",
            "--checkpoint", str(tmp_path / "weights_epoch_2.pt"),
            "--scaler-train", str(tmp_path / "train.h5"),
            "--dataset-path", str(tmp_path / "holdout.h5"),
        ],
    )
    args = parse_args()
    assert args.checkpoint == tmp_path / "weights_epoch_2.pt"
    assert args.scaler_train == tmp_path / "train.h5"
    assert args.dataset_path == tmp_path / "holdout.h5"


def _source(tmp_path: Path):
    from scripts.eval.evaluate_ur5e_fixed_pcd import validate_external_checkpoint_source

    run = tmp_path / "checkpoints" / "mixed"
    run.mkdir(parents=True)
    checkpoint = run / "weights_epoch_2.pt"
    checkpoint.write_bytes(b"placeholder")
    train = tmp_path / "train.h5"
    train.write_bytes(b"placeholder")
    (run / "config.yaml").write_text(
        "model:\n  encoder:\n    _target_: encoders.DexArtPointNet\n"
        "data:\n  dataset:\n    name: " + str(train) + "\n"
        "  preprocessor:\n    center: [0.425, 0.0, 0.2]\n"
        "    scale: 4.0\n    use_rgb: true\n",
        encoding="utf-8",
    )
    return validate_external_checkpoint_source, checkpoint, train


def test_external_checkpoint_requires_its_own_training_action_source(tmp_path):
    validate, checkpoint, train = _source(tmp_path)

    assert validate(checkpoint, train) == (checkpoint, train)
    other = tmp_path / "other.h5"
    other.write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="training dataset"):
        validate(checkpoint, other)


def test_external_checkpoint_rejects_wrong_point_cloud_preprocessing(tmp_path):
    validate, checkpoint, train = _source(tmp_path)
    config = checkpoint.parent / "config.yaml"
    config.write_text(config.read_text().replace("scale: 4.0", "scale: 2.0"))

    with pytest.raises(ValueError, match="preprocessor mismatch"):
        validate(checkpoint, train)


def test_single_cube_mixed_eval_samples_only_unseen_episode_range(tmp_path):
    from scripts.eval.evaluate_ur5e_2cube_to_1cube_local25 import (
        sample_single_cube_specs,
    )

    import h5py
    import numpy as np

    path = tmp_path / "single.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("ep_len", data=np.array([31, 31]))
        h5.create_dataset("ep_offset", data=np.array([0, 31]))
        h5.create_dataset("qpos", data=np.zeros((62, 21)))
        h5.create_dataset("qvel", data=np.zeros((62, 20)))
        h5.create_dataset("pcd_xyz", data=np.zeros((62, 1200, 3)))
        h5.create_dataset("pcd_rgb", data=np.zeros((62, 1200, 3)))
        h5.create_dataset("action", data=np.zeros((62, 5)))
        h5.create_dataset("privileged_block_0_pos", data=np.zeros((62, 3)))
        h5.create_dataset("privileged_block_0_quat", data=np.zeros((62, 4)))

    specs = sample_single_cube_specs(
        path, count=2, seed=42, goal_offset=25, episode_start=1, episode_stop=2
    )
    assert {spec.episode_index for spec in specs} == {1}
