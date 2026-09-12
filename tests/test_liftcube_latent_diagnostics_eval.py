import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.eval_liftcube_latent_diagnostics as latent_eval
from scripts.eval_liftcube_latent_diagnostics import (
    annotate_action_judgments,
    atomic_write_json,
    evaluate_representation_stages,
)


class _TinyDataset(torch.utils.data.Dataset):
    horizon = 1
    skip_frame = 1
    include_rgb = False
    episode_ids = [90, 91]
    T = 1
    point_count = 3

    def __len__(self):
        return 2

    def context_actions_for_index(self, index):
        return torch.empty(0, 1)

    def __getitem__(self, index):
        context = torch.tensor(
            [[[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]]
        )
        future = torch.tensor(
            [[[2.01, 0, 0], [0.01, 0, 0], [1.01, 0, 0]]]
        )
        actions = torch.ones(1, 1) * (index + 1)
        return index, context, future, actions


class _TinyAggregator(torch.nn.Module):
    def forward(self, points):
        return points[..., :2], points[..., :3]


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(T=1, K=3, D=2, C=3, skip_frame=1)
        self.aggregator = _TinyAggregator()

    def _aggregate(self, points, return_positions=False):
        projected = points[..., :2]
        positions = points[..., :3]
        return (projected, positions) if return_positions else projected

    def temporal(self, latent, causal=True):
        return latent[:, -1]

    def rollout(self, points, actions, env=None, context_actions=None):
        value = actions.sum(dim=(1, 2))[:, None, None, None]
        return points[:, -1:, :, :2] + value


class _IdentityProprioEncoder:
    def normalize(self, value):
        return value


class _TinyProprioDataset(_TinyDataset):
    def proprio_for_index(self, index):
        value = torch.zeros(1, 25)
        value[:, 0] = index + 1
        return value


class _TinyProprioModel(_TinyModel):
    def __init__(self):
        super().__init__()
        self.cfg.proprio = SimpleNamespace(enabled=True, dim=25)
        self.proprio_encoder = _IdentityProprioEncoder()

    def rollout(self, *args, **kwargs):
        raise AssertionError("legacy rollout bypasses proprio conditioning")

    def predict_transition(
        self, latent_history, proprio_history_normalized, actions, env=None
    ):
        increment = actions.sum(dim=(1, 2))
        increment = increment + proprio_history_normalized[:, -1, 0]
        next_scene = latent_history[:, -1] + increment[:, None, None]
        next_proprio = proprio_history_normalized[:, -1]
        return SimpleNamespace(
            scene_latent=next_scene,
            next_proprio_normalized=next_proprio,
            next_proprio=next_proprio,
            object_pos=next_scene.new_zeros((next_scene.shape[0], 3)),
        )


def test_representation_evaluator_reports_required_stages_and_matches():
    result = evaluate_representation_stages(
        _TinyModel(),
        _TinyDataset(),
        batch_size=2,
        device=torch.device("cpu"),
        seed=3072,
        max_batches=1,
    )

    assert result["raw_spatial"] is None
    for key in ("projected_spatial", "temporal_context", "predicted"):
        assert result[key]["finite"] is True
    assert result["matched_temporal_signal"]["match_count"] == 6
    assert result["action_gradient_norm"] > 0
    assert result["window_indices"] == [0, 1]


def test_representation_evaluator_uses_proprio_conditioned_rollout():
    result = evaluate_representation_stages(
        _TinyProprioModel(),
        _TinyProprioDataset(),
        batch_size=2,
        device=torch.device("cpu"),
        seed=3072,
        max_batches=1,
    )

    assert result["predicted"]["finite"] is True
    assert result["action_gradient_norm"] > 0


def test_action_judgment_requires_shuffle_gap_and_persistence_gain():
    controls = {
        "per_horizon": {
            "1": {
                "model_correct_action": {"mse": 2.0},
                "model_shuffled_action": {"mse": 3.0},
                "persistence": {"mse": 1.5},
            },
            "2": {
                "model_correct_action": {"mse": 1.0},
                "model_shuffled_action": {"mse": 1.2},
                "persistence": {"mse": 1.5},
            },
        }
    }

    result = annotate_action_judgments(controls)

    assert result["per_horizon"]["1"]["action_valid"] is False
    assert result["per_horizon"]["2"]["action_valid"] is True


def test_atomic_json_replaces_complete_document(tmp_path):
    output = tmp_path / "diagnostics.json"
    atomic_write_json(output, {"value": 1})
    atomic_write_json(output, {"value": 2})

    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 2}
    assert not output.with_suffix(".json.tmp").exists()


def test_main_writes_checkpoint_and_data_provenance(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    data_root = tmp_path / "LiftCube-v0"
    data_root.mkdir()
    (data_root / "manifest.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "result.json"
    state = {"step": 500, "model_config": {"C": 3}, "train_config": {}}
    dataset = _TinyDataset()
    monkeypatch.setattr(latent_eval, "load_checkpoint_model", lambda *args: (_TinyModel(), state))
    monkeypatch.setattr(latent_eval, "LiftCubeRolloutDataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(
        latent_eval,
        "evaluate_rollouts",
        lambda *args, **kwargs: {
            "window_count": 2,
            "per_horizon": {
                "1": {
                    "model_correct_action": {"mse": 1.0},
                    "model_shuffled_action": {"mse": 1.2},
                    "persistence": {"mse": 2.0},
                }
            },
        },
    )
    monkeypatch.setattr(
        latent_eval,
        "evaluate_representation_stages",
        lambda *args, **kwargs: {"projected_spatial": {"finite": True}},
    )

    result = latent_eval.main(
        [
            "--checkpoint", str(checkpoint),
            "--data-root", str(data_root),
            "--output", str(output),
            "--horizons", "1",
            "--device", "cpu",
        ]
    )

    assert result == json.loads(output.read_text(encoding="utf-8"))
    assert result["checkpoint"]["sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert result["checkpoint"]["step"] == 500
    assert result["data"]["manifest_sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert result["action_controls"]["per_horizon"]["1"]["action_valid"] is True
