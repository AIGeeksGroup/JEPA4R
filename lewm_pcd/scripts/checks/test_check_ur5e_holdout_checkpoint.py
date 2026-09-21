import pytest
import torch
from torch import nn

from scripts.checks.check_ur5e_holdout_checkpoint import (
    LatentStatistics,
    batch_norm_statistics,
    checkpoint_filename,
    episode_diversity,
    fixed_full_batches,
    weighted_mean,
)


def test_checkpoint_filename():
    assert checkpoint_filename(10) == "weights_epoch_10.pt"
    with pytest.raises(ValueError):
        checkpoint_filename(0)


def test_weighted_mean_counts_examples_not_batches():
    assert weighted_mean([(1.0, 2), (3.0, 1)]) == pytest.approx(5 / 3)
    with pytest.raises(ValueError):
        weighted_mean([])


def test_fixed_full_batches_are_seeded_unique_and_drop_the_remainder():
    first = fixed_full_batches(size=10, batch_size=4, seed=3072)
    repeated = fixed_full_batches(size=10, batch_size=4, seed=3072)
    changed = fixed_full_batches(size=10, batch_size=4, seed=3073)

    assert first == repeated
    assert first != changed
    assert [len(batch) for batch in first] == [4, 4]
    retained = [index for batch in first for index in batch]
    assert len(retained) == 8
    assert len(set(retained)) == 8
    assert all(0 <= index < 10 for index in retained)


def test_episode_diversity_counts_unique_episodes_per_batch():
    clip_indices = [(episode, frame) for episode in range(3) for frame in range(4)]
    batches = [[0, 1, 4, 5], [2, 6, 8, 9]]

    assert episode_diversity(clip_indices, batches) == {
        "min_unique_episodes_per_batch": 2,
        "max_unique_episodes_per_batch": 3,
        "mean_unique_episodes_per_batch": pytest.approx(2.5),
    }


def test_latent_statistics_report_centered_variance_and_effective_rank():
    stats = LatentStatistics()
    stats.update(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    stats.update(torch.tensor([[0.0, 1.0], [0.0, -1.0]]))

    report = stats.report()
    assert report["tokens"] == 4
    assert report["dimensions"] == 2
    assert report["feature_mean"] == pytest.approx([0.0, 0.0])
    assert report["feature_variance"] == pytest.approx([0.5, 0.5])
    assert report["mean_abs_feature_mean"] == pytest.approx(0.0)
    assert report["mean_feature_variance"] == pytest.approx(0.5)
    assert report["covariance_effective_rank"] == pytest.approx(2.0)


def test_batch_norm_statistics_reports_running_state():
    model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3))
    batch_norm = model[1]
    batch_norm.running_mean.copy_(torch.tensor([1.0, 2.0, 3.0]))
    batch_norm.running_var.copy_(torch.tensor([0.5, 1.0, 2.0]))
    batch_norm.num_batches_tracked.copy_(torch.tensor(7))

    assert batch_norm_statistics(model) == [
        {
            "name": "1",
            "type": "BatchNorm1d",
            "num_features": 3,
            "num_batches_tracked": 7,
            "finite": True,
            "running_mean_abs_max": pytest.approx(3.0),
            "running_var_min": pytest.approx(0.5),
            "running_var_max": pytest.approx(2.0),
        }
    ]
