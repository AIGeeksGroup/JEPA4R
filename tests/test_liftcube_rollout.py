import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.eval_liftcube_rollout as rollout_eval
from scripts.eval_liftcube_rollout import (
    ErrorTotals,
    LiftCubeRolloutDataset,
    build_action_shuffle_indices,
    evaluate_rollouts,
)

def _write_eval_root(root: Path, *, frames: int = 8, raw_points: int = 4) -> Path:
    root.mkdir(parents=True)
    train_ids = list(range(90))
    val_ids = list(range(90, 100))
    manifest = {
        "task": "LiftCube-v0",
        "episode_ids": train_ids + val_ids,
        "train_episode_ids": train_ids,
        "val_episode_ids": val_ids,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for episode_id in val_ids:
        frame_values = np.arange(frames, dtype=np.float32) + episode_id * 100
        xyz = np.broadcast_to(
            frame_values[:, None, None],
            (frames, raw_points, 3),
        ).copy()
        action = np.stack(
            [
                np.arange(frames - 1, dtype=np.float32) + episode_id,
                np.zeros(frames - 1, dtype=np.float32),
                np.ones(frames - 1, dtype=np.float32),
                np.full(frames - 1, 2.0, dtype=np.float32),
                np.full(frames - 1, 3.0, dtype=np.float32),
                np.full(frames - 1, 4.0, dtype=np.float32),
                np.full(frames - 1, 5.0, dtype=np.float32),
            ],
            axis=1,
        )
        np.savez(root / f"episode_{episode_id:04d}.npz", xyz=xyz, action=action)
    return root


def test_rollout_dataset_aligns_context_actions_and_targets(tmp_path):
    root = _write_eval_root(tmp_path / "LiftCube-v0")
    dataset = LiftCubeRolloutDataset(root, T=2, point_count=4, horizon=3)

    index, context, future, actions = dataset[0]

    assert index == 0
    assert len(dataset) == 10 * (8 - 2 - 3 + 1)
    assert dataset.episode_ids == list(range(90, 100))
    assert context.shape == (2, 4, 3)
    assert future.shape == (3, 4, 3)
    assert actions.shape == (3, 7)
    assert torch.all(context[0] == 9000)
    assert torch.all(context[1] == 9001)
    assert torch.all(future[0] == 9002)
    assert actions[0, 0].item() == pytest.approx(91.0)
    assert torch.equal(actions, dataset.actions_for_index(0))

def test_rollout_dataset_exposes_context_action_history(tmp_path):
    root = _write_eval_root(tmp_path / "LiftCube-v0")
    dataset = LiftCubeRolloutDataset(root, T=2, point_count=4, horizon=3)
    assert dataset.context_actions_for_index(0).shape == (1, 7)
    assert dataset.context_actions_for_index(0)[0, 0].item() == pytest.approx(90.0)


def test_rollout_dataset_skip_frame_keeps_atomic_action_chunks(tmp_path):
    root = _write_eval_root(tmp_path / "LiftCube-v0", frames=9)
    dataset = LiftCubeRolloutDataset(root, T=2, point_count=4, horizon=2, skip_frame=2)
    _, context, future, actions = dataset[0]
    assert len(dataset) == 30
    assert torch.all(context[0] == 9000)
    assert torch.all(context[1] == 9002)
    assert torch.all(future[0] == 9004)
    assert actions.shape == (4, 7)
    assert actions[:, 0].tolist() == pytest.approx([92.0, 93.0, 94.0, 95.0])
    assert dataset.context_actions_for_index(0)[:, 0].tolist() == pytest.approx([90.0, 91.0])


def test_action_shuffle_is_seeded_permutation_without_fixed_points():
    first = build_action_shuffle_indices(11, seed=3072)
    second = build_action_shuffle_indices(11, seed=3072)

    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(11))
    assert np.all(first != np.arange(11))


def test_action_shuffle_rejects_single_window():
    with pytest.raises(ValueError, match="at least two"):
        build_action_shuffle_indices(1, seed=3072)


def test_error_totals_weights_elements_and_feature_vectors():
    totals = ErrorTotals()
    prediction = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

    totals.update(prediction, target)
    result = totals.result()

    assert result["mse"] == pytest.approx(0.5)
    assert result["cosine_distance"] == pytest.approx(0.5)


def test_action_chunk_variants_preserve_chunk_boundaries():
    actions = torch.arange(1, 7, dtype=torch.float32).view(1, 6, 1)

    reversed_actions = rollout_eval.reverse_action_chunks(actions, chunk_size=3)
    repeated_actions = rollout_eval.repeat_last_action_chunks(actions, chunk_size=3)

    assert reversed_actions.flatten().tolist() == [3, 2, 1, 6, 5, 4]
    assert repeated_actions.flatten().tolist() == [3, 3, 3, 6, 6, 6]


class _TinyRolloutDataset(torch.utils.data.Dataset):
    horizon = 2
    skip_frame = 2
    episode_ids = [90, 91]

    def __init__(self):
        self._actions = torch.tensor(
            [
                [
                    [1.0, 0, 0, 0, 0, 0, 0],
                    [2.0, 0, 0, 0, 0, 0, 0],
                    [3.0, 0, 0, 0, 0, 0, 0],
                    [4.0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [2.0, 0, 0, 0, 0, 0, 0],
                    [4.0, 0, 0, 0, 0, 0, 0],
                    [5.0, 0, 0, 0, 0, 0, 0],
                    [8.0, 0, 0, 0, 0, 0, 0],
                ],
            ]
        )

    def __len__(self):
        return 2

    def context_actions_for_index(self, index):
        return torch.empty(
            (0, 7),
            dtype=torch.float32,
        )

    def actions_for_index(self, index):
        return self._actions[index]

    def __getitem__(self, index):
        actions = self._actions[index]
        context = torch.zeros(1, 1, 3)
        chunks = actions.view(self.horizon, self.skip_frame, 7)
        increments = chunks[:, 0, 0] + 2.0 * chunks[:, 1, 0]
        values = increments.cumsum(0)
        future = values[:, None, None].expand(2, 1, 3).clone()
        return index, context, future, actions


class _ActionDrivenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(T=1, K=1, D=1, skip_frame=2)

    def _aggregate(self, points):
        return points[..., :1]

    def rollout(self, points, actions, env=None, context_actions=None):
        assert context_actions is not None
        assert context_actions.shape == (points.shape[0], 0, 7)
        start = points[:, -1:, :, :1]
        chunks = actions.view(actions.shape[0], -1, self.cfg.skip_frame, 7)
        increments = chunks[:, :, 0, 0] + 2.0 * chunks[:, :, 1, 0]
        return start + increments.cumsum(1)[:, :, None, None]


def test_evaluate_rollouts_reports_horizons_and_action_controls():
    result = evaluate_rollouts(
        _ActionDrivenModel(),
        _TinyRolloutDataset(),
        batch_size=2,
        device=torch.device("cpu"),
        horizons=(1, 2),
        seed=3072,
    )

    h2 = result["per_horizon"]["2"]
    assert result["window_count"] == 2
    assert h2["model_correct_action"]["mse"] == pytest.approx(0.0)
    assert h2["model_zero_action"]["mse"] > 0
    assert h2["model_shuffled_action"]["mse"] > 0
    assert h2["model_reversed_action"]["mse"] > 0
    assert h2["model_repeat_last_action"]["mse"] > 0
    assert h2["shuffle_degradation_percent"] > 10.0
    assert h2["reverse_degradation_percent"] > 10.0
    assert h2["repeat_last_degradation_percent"] > 10.0
    assert h2["action_control_success_10pct"] is True
    assert h2["gain_vs_persistence"] > 1
    assert h2["gain_vs_shuffled_action"] > 1
    assert h2["gain_vs_reversed_action"] > 1
    assert h2["gain_vs_repeat_last_action"] > 1
    assert h2["prediction_delta_shuffled_mse"] > 0
    assert h2["prediction_delta_reversed_mse"] > 0
    assert h2["prediction_delta_repeat_last_mse"] > 0


def test_atomic_write_json_replaces_complete_document(tmp_path):
    output = tmp_path / "rollout_metrics.json"
    rollout_eval.atomic_write_json(output, {"value": 1})
    rollout_eval.atomic_write_json(output, {"value": 2})

    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 2}
    assert not output.with_suffix(".json.tmp").exists()


def test_main_writes_provenance_without_loading_simulator(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    data_root = _write_eval_root(tmp_path / "LiftCube-v0")
    output = tmp_path / "rollout_metrics.json"
    fake_model = _ActionDrivenModel()
    state = {
        "step": 8500,
        "model_config": {"T": 1, "K": 1, "D": 1, "C": 3},
        "train_config": {"task": "LiftCube-v0"},
    }

    class _FakeDataset:
        episode_ids = list(range(90, 100))

        def __init__(self, root, *, T, point_count, horizon, skip_frame=1):
            self.root = Path(root)
            self.T = T
            self.point_count = point_count
            self.horizon = horizon
            self.skip_frame = skip_frame

        def __len__(self):
            return 7

    monkeypatch.setattr(
        rollout_eval,
        "load_checkpoint_model",
        lambda path, device: (fake_model, state),
    )
    monkeypatch.setattr(rollout_eval, "LiftCubeRolloutDataset", _FakeDataset)
    monkeypatch.setattr(
        rollout_eval,
        "evaluate_rollouts",
        lambda *args, **kwargs: {
            "window_count": 7,
            "per_horizon": {
                str(h): {
                    "model_correct_action": {"mse": 1.0, "cosine_distance": 0.1},
                    "persistence": {"mse": 2.0, "cosine_distance": 0.2},
                    "model_zero_action": {"mse": 3.0, "cosine_distance": 0.3},
                    "model_shuffled_action": {"mse": 4.0, "cosine_distance": 0.4},
                    "model_reversed_action": {"mse": 5.0, "cosine_distance": 0.5},
                    "model_repeat_last_action": {"mse": 6.0, "cosine_distance": 0.6},
                    "gain_vs_persistence": 2.0,
                    "gain_vs_zero_action": 3.0,
                    "gain_vs_shuffled_action": 4.0,
                    "gain_vs_reversed_action": 5.0,
                    "gain_vs_repeat_last_action": 6.0,
                    "prediction_delta_zero_mse": 0.5,
                    "prediction_delta_shuffled_mse": 0.6,
                    "prediction_delta_reversed_mse": 0.7,
                    "prediction_delta_repeat_last_mse": 0.8,
                    "shuffle_degradation_percent": 300.0,
                    "reverse_degradation_percent": 400.0,
                    "repeat_last_degradation_percent": 500.0,
                    "action_control_success_10pct": True,
                }
                for h in (1, 2, 4, 8)
            },
        },
    )

    result = rollout_eval.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--data-root",
            str(data_root),
            "--output",
            str(output),
            "--device",
            "cpu",
        ]
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(b"checkpoint").hexdigest()
    assert result == saved
    assert saved["checkpoint"]["sha256"] == expected_hash
    assert saved["checkpoint"]["step"] == 8500
    assert saved["data"]["episode_count"] == 10
    assert saved["evaluation"]["horizons"] == [1, 2, 4, 8]
    assert saved["evaluation"]["window_count"] == 7
