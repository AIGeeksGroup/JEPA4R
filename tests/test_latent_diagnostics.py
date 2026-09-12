import pytest
import torch

from scripts.latent_diagnostics import (
    action_gradient_norm,
    judge_action_control,
    matched_temporal_signal,
    mutual_nearest_matches,
    representation_health,
)


def test_representation_health_detects_constant_channels():
    tokens = torch.stack(
        [
            torch.tensor([0.0, 2.0, 5.0]),
            torch.tensor([1.0, 2.0, 4.0]),
            torch.tensor([2.0, 2.0, 3.0]),
        ]
    )

    result = representation_health(tokens, pair_samples=4, seed=3)

    assert result["finite"] is True
    assert result["near_constant_channel_fraction"] == pytest.approx(1 / 3)
    assert result["channel_std_min"] == pytest.approx(0.0)
    assert result["channel_std_max"] > 0


def test_representation_health_reports_two_rank_measures():
    tokens = torch.eye(4).repeat(4, 1)

    result = representation_health(tokens, pair_samples=20, seed=7)

    assert 2.9 < result["effective_rank"] <= 4.0
    assert 2.9 < result["participation_ratio"] <= 4.0
    assert result["token_norm_mean"] > 0


def test_off_diagonal_cosine_is_seeded_and_excludes_self_pairs():
    tokens = torch.eye(8)

    first = representation_health(tokens, pair_samples=40, seed=11)
    second = representation_health(tokens, pair_samples=40, seed=11)

    assert first["off_diagonal_cosine_mean"] == second["off_diagonal_cosine_mean"]
    assert first["off_diagonal_cosine_mean"] == pytest.approx(0.0)


def test_mutual_nearest_matches_ignore_array_index():
    left = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]])
    right = torch.tensor([[[2.01, 0, 0], [0.01, 0, 0], [1.01, 0, 0]]])

    matches = mutual_nearest_matches(left, right, max_distance=0.05)

    assert matches["batch"].tolist() == [0, 0, 0]
    assert matches["left"].tolist() == [0, 1, 2]
    assert matches["right"].tolist() == [1, 2, 0]
    assert matches["distance"].tolist() == pytest.approx([0.01, 0.01, 0.01])


def test_matched_temporal_signal_uses_matched_positions():
    left_xyz = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]])
    right_xyz = torch.tensor([[[2.02, 0, 0], [0.0, 0, 0], [1.01, 0, 0]]])
    left_latent = torch.tensor([[[0.0], [1.0], [2.0]]])
    right_latent = torch.tensor([[[4.0], [0.0], [2.0]]])

    result = matched_temporal_signal(
        left_xyz, right_xyz, left_latent, right_latent, max_distance=0.05
    )

    assert result["match_count"] == 3
    assert result["geometry_displacement_mean"] == pytest.approx(0.01)
    assert result["latent_displacement_mean"] == pytest.approx(1.0)
    assert result["geometry_latent_correlation"] > 0.8


def test_action_gate_rejects_amplification_that_loses_to_persistence():
    rejected = judge_action_control(correct=2.0, shuffled=3.0, persistence=1.5)
    accepted = judge_action_control(correct=1.0, shuffled=1.2, persistence=1.5)

    assert rejected["shuffle_degradation"] == pytest.approx(0.5)
    assert rejected["passed"] is False
    assert rejected["beats_persistence"] is False
    assert accepted["passed"] is True


class _ActionDriven(torch.nn.Module):
    def rollout(self, points, actions, env=None):
        return points[:, :1, :, :1] + actions.sum(dim=(1, 2))[:, None, None, None]


def test_action_gradient_norm_is_nonzero_for_action_driven_predictor():
    result = action_gradient_norm(
        _ActionDriven(),
        torch.ones(2, 1, 3, 1),
        torch.ones(2, 4, 2),
        env="test",
    )

    assert result > 0
    assert torch.isfinite(torch.tensor(result))
