import json
from pathlib import Path

import pytest

from scripts.train_liftcube import _wrong_actions, load_liftcube_config, validate_liftcube_manifest


def _write_manifest(root: Path, *, task="LiftCube-v0", train=None, val=None):
    train = list(range(90)) if train is None else train
    val = list(range(90, 100)) if val is None else val
    manifest = {
        "task": task,
        "episode_ids": list(range(100)),
        "train_episode_ids": train,
        "val_episode_ids": val,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_liftcube_manifest_requires_90_10_split(tmp_path):
    root = tmp_path / "LiftCube-v0"
    _write_manifest(root)
    result = validate_liftcube_manifest(root)
    assert result["task"] == "LiftCube-v0"
    assert len(result["train_episode_ids"]) == 90
    assert len(result["val_episode_ids"]) == 10

    _write_manifest(root, task="PickCube-v0")
    with pytest.raises(ValueError, match="LiftCube-v0"):
        validate_liftcube_manifest(root)


def test_liftcube_config_has_baseline_training_defaults():
    repo_root = Path(__file__).resolve().parents[1]
    model_cfg, train_cfg = load_liftcube_config(repo_root / "configs" / "liftcube_baseline.yaml", [])
    assert train_cfg.task == "LiftCube-v0"
    assert train_cfg.env == "maniskill2"
    assert train_cfg.steps > 0
    assert model_cfg.T == 3
    assert model_cfg.C == 3
    assert model_cfg.action.envs["maniskill2"].dim == 7


def test_wrong_actions_preserve_chunks_but_break_batch_alignment():
    import torch

    actions = torch.arange(3 * 6, dtype=torch.float32).view(3, 6, 1)
    wrong = _wrong_actions(actions, chunk_size=2)
    assert torch.equal(wrong, actions.roll(1, dims=0))
    assert not torch.equal(wrong, actions)


def test_wrong_actions_roll_transition_chunks_for_batch_size_one():
    import torch

    actions = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    wrong = _wrong_actions(actions, chunk_size=2)
    expected = actions.view(1, 3, 2, 1).roll(1, dims=1).reshape_as(actions)
    assert torch.equal(wrong, expected)
    assert not torch.equal(wrong, actions)


def test_skip10_action_tokens_config_disables_motion_weighting():
    repo_root = Path(__file__).resolve().parents[1]
    model_cfg, train_cfg = load_liftcube_config(
        repo_root / "configs" / "liftcube_skip10_action_tokens.yaml", []
    )

    assert model_cfg.skip_frame == 10
    assert model_cfg.action.output_scale == 1.0
    assert model_cfg.action.token_encoder_layers == 1
    assert model_cfg.action.token_encoder_heads == 4
    assert model_cfg.action.token_mlp_ratio == 2.0
    assert model_cfg.action.token_position_encoding == "sinusoidal"
    assert model_cfg.action.token_position_scale == 1.0
    assert model_cfg.predictor.fusion == "cross_attention"
    assert model_cfg.loss.flow_weight == 0.0
    assert model_cfg.loss.ranking_weight == 1.0
    assert model_cfg.loss.ranking_motion_weighted is False
    assert train_cfg.batch_size == 8
    assert train_cfg.steps == 2500
    assert train_cfg.run_name == "liftcube_skip10_action_tokens_2500"
