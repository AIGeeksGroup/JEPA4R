import json
from types import SimpleNamespace

import numpy as np
import pytest

from data.maniskill import CrossEpisodeBatchSampler, ManiSkillSequenceDataset, action_stats


def _write_task(root, task, train=(0,), val=(1,), frames=7, points=8):
    task_root = root / task
    task_root.mkdir()
    (task_root / "manifest.json").write_text(
        json.dumps({"task": task, "train_episode_ids": list(train), "val_episode_ids": list(val)}),
        encoding="utf-8",
    )
    for episode_id in (*train, *val):
        xyz = np.arange(frames * points * 3, dtype=np.float32).reshape(frames, points, 3) + episode_id
        action = np.arange((frames - 1) * 7, dtype=np.float32).reshape(frames - 1, 7) / 100
        np.savez_compressed(task_root / f"episode_{episode_id:04d}.npz", xyz=xyz, action=action)
    return task_root


def test_dataset_emits_aligned_world_model_window(tmp_path):
    root = _write_task(tmp_path, "LiftCube-v0")
    ds = ManiSkillSequenceDataset(root, T=3, point_count=4)
    points, actions = ds[0]
    assert points.shape == (4, 4, 3)
    assert actions.shape == (3, 7)
    np.testing.assert_array_equal(actions.numpy(), np.arange(21, dtype=np.float32).reshape(3, 7) / 100)


def test_single_frame_context_does_not_replace_target(tmp_path):
    root = _write_task(tmp_path, "LiftCube-v0")
    ds = ManiSkillSequenceDataset(root, T=3, point_count=8, single_frame_context=True)
    points, _ = ds[0]
    assert np.array_equal(points[:3].numpy(), np.repeat(points[2:3].numpy(), 3, axis=0))
    assert not np.array_equal(points[2].numpy(), points[3].numpy())


def test_windows_are_split_by_episode(tmp_path):
    root = _write_task(tmp_path, "LiftCube-v0", frames=7)
    train = ManiSkillSequenceDataset(root, split="train", T=3, point_count=8)
    val = ManiSkillSequenceDataset(root, split="val", T=3, point_count=8)
    assert {episode_id for episode_id, _ in train.index} == {0}
    assert {episode_id for episode_id, _ in val.index} == {1}
    (root / "manifest.json").write_text(json.dumps({"train_episode_ids": [0, 1], "val_episode_ids": [1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="disjoint"):
        ManiSkillSequenceDataset(root, T=3, point_count=8)


def test_sampling_and_action_stats_are_deterministic(tmp_path):
    root = _write_task(tmp_path, "LiftCube-v0", frames=8, points=8)
    first = ManiSkillSequenceDataset(root, T=2, point_count=4)[0][0]
    second = ManiSkillSequenceDataset(root, T=2, point_count=4)[0][0]
    assert np.array_equal(first.numpy(), second.numpy())
    mean, std = action_stats(root)
    assert mean.shape == std.shape == (7,)
    assert np.all(std > 0)


def test_skip_frame_selects_sparse_observations_and_all_intervening_actions(tmp_path):
    root = _write_task(tmp_path, "LiftCube-v0", frames=9, points=8)
    ds = ManiSkillSequenceDataset(root, T=2, point_count=8, skip_frame=2)
    points, actions = ds[0]
    assert points.shape == (3, 8, 3)
    assert actions.shape == (4, 7)
    np.testing.assert_array_equal(points.numpy(), np.stack([
        np.arange(9 * 8 * 3, dtype=np.float32).reshape(9, 8, 3)[i]
        for i in (0, 2, 4)
    ]))
    np.testing.assert_array_equal(actions.numpy(), np.arange(28, dtype=np.float32).reshape(4, 7) / 100)
    assert len(ds) == 5


def test_cross_episode_batch_sampler_uses_each_window_once_without_episode_collisions(tmp_path):
    root = _write_task(tmp_path, "LiftCube-v0", train=tuple(range(5)), val=(5,), frames=7, points=8)
    dataset = ManiSkillSequenceDataset(root, split="train", T=3, point_count=8)
    sampler = CrossEpisodeBatchSampler(dataset, batch_size=3, shuffle=False, drop_last=False, seed=17)

    batches = list(sampler)

    assert sorted(index for batch in batches for index in batch) == list(range(len(dataset)))
    for batch in batches:
        episodes = [dataset.index[index][0] for index in batch]
        assert len(episodes) == len(set(episodes))


def test_cross_episode_batch_sampler_is_seeded_per_epoch(tmp_path):
    root = _write_task(tmp_path, "LiftCube-v0", train=tuple(range(6)), val=(6,), frames=8, points=8)
    dataset = ManiSkillSequenceDataset(root, split="train", T=3, point_count=8)
    first = CrossEpisodeBatchSampler(dataset, batch_size=4, shuffle=True, drop_last=True, seed=23)
    second = CrossEpisodeBatchSampler(dataset, batch_size=4, shuffle=True, drop_last=True, seed=23)

    first_epoch = list(first)
    second_epoch = list(first)

    assert first_epoch == list(second)
    assert second_epoch == list(second)
    assert first_epoch != second_epoch
    assert all(len(batch) == 4 for batch in first_epoch + second_epoch)


def test_cross_episode_batch_sampler_rejects_more_samples_than_episodes(tmp_path):
    root = _write_task(tmp_path, "LiftCube-v0", train=(0, 1), val=(2,), frames=7, points=8)
    dataset = ManiSkillSequenceDataset(root, split="train", T=3, point_count=8)

    with pytest.raises(ValueError, match="distinct episodes"):
        CrossEpisodeBatchSampler(dataset, batch_size=3, shuffle=False, drop_last=False, seed=0)


def test_cross_episode_batch_sampler_balances_uneven_episode_lengths():
    dataset = SimpleNamespace(index=[(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)])
    sampler = CrossEpisodeBatchSampler(dataset, batch_size=3, shuffle=False, drop_last=False, seed=0)

    batches = list(sampler)

    assert len(batches) == len(sampler) == 2
    assert sorted(index for batch in batches for index in batch) == list(range(5))
    for batch in batches:
        episodes = [dataset.index[index][0] for index in batch]
        assert len(episodes) == len(set(episodes))
