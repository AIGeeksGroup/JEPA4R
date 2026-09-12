import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from data.maniskill import (
    CrossEpisodeBatchSampler,
    ManiSkillSequenceDataset,
    action_stats,
    load_proprio_stats,
)


def _write_task(
    root,
    task,
    train=(0,),
    val=(1,),
    frames=7,
    points=8,
    include_rgb=False,
    include_proprio=False,
):
    task_root = root / task
    task_root.mkdir()
    manifest = {
        "task": task,
        "train_episode_ids": list(train),
        "val_episode_ids": list(val),
    }
    if include_proprio:
        manifest["proprio_normalization"] = {
            "source_split": "train",
            "episode_ids": list(train),
            "sample_count": len(train) * frames,
            "mean": [float(index) for index in range(25)],
            "std": [float(index + 1) for index in range(25)],
        }
    (task_root / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    for episode_id in (*train, *val):
        xyz = np.arange(frames * points * 3, dtype=np.float32).reshape(frames, points, 3) + episode_id
        action = np.arange((frames - 1) * 7, dtype=np.float32).reshape(frames - 1, 7) / 100
        arrays = {"xyz": xyz, "action": action}
        if include_rgb:
            arrays["rgb"] = np.asarray(xyz % 251, dtype=np.uint8)
        if include_proprio:
            qpos = np.zeros((frames, 9), dtype=np.float32)
            qpos[:, 0] = np.arange(frames, dtype=np.float32)
            arrays.update(
                {
                    "qpos": qpos,
                    "qvel": np.full((frames, 9), 2.0, dtype=np.float32),
                    "tcp_pose": np.full((frames, 7), 3.0, dtype=np.float32),
                    "object_pos": np.arange(frames * 3, dtype=np.float32).reshape(
                        frames, 3
                    ),
                }
            )
        np.savez_compressed(task_root / f"episode_{episode_id:04d}.npz", **arrays)
    return task_root


def _write_contact_sidecar(path, episode_ids, labels, *, duplicate=False, error=0.0):
    episode = []
    frame = []
    contact = []
    for episode_id in episode_ids:
        values = labels[int(episode_id)]
        episode.extend([int(episode_id)] * len(values))
        frame.extend(range(len(values)))
        contact.extend(values)
    if duplicate:
        episode.append(episode[0])
        frame.append(frame[0])
        contact.append(contact[0])
    count = len(episode)
    np.savez_compressed(
        path,
        episode_id=np.asarray(episode, dtype=np.int64),
        frame_id=np.asarray(frame, dtype=np.int64),
        cube_robot_contact=np.asarray(contact, dtype=np.uint8),
        object_pos_error_m=np.full(count, error, dtype=np.float64),
        tcp_pos_error_m=np.full(count, error, dtype=np.float64),
    )
    return path


def test_dataset_emits_aligned_proprio_mapping(tmp_path):
    root = _write_task(
        tmp_path,
        "LiftCube-v0",
        frames=9,
        points=8,
        include_rgb=True,
        include_proprio=True,
    )
    dataset = ManiSkillSequenceDataset(
        root,
        split="train",
        T=2,
        skip_frame=3,
        point_count=8,
        include_rgb=True,
        include_proprio=True,
    )

    batch = dataset[0]

    assert set(batch) == {"points", "actions", "proprio", "object_pos"}
    assert batch["points"].shape == (3, 8, 6)
    assert batch["actions"].shape == (6, 7)
    assert batch["proprio"].shape == (3, 25)
    assert batch["object_pos"].shape == (3, 3)
    np.testing.assert_array_equal(
        batch["proprio"][:, 0].numpy(), np.array([0.0, 3.0, 6.0])
    )
    np.testing.assert_array_equal(
        batch["object_pos"].numpy(),
        np.arange(27, dtype=np.float32).reshape(9, 3)[[0, 3, 6]],
    )


def test_legacy_dataset_output_remains_tuple(tmp_path):
    root = _write_task(tmp_path, "LiftCube-v0")
    item = ManiSkillSequenceDataset(root, point_count=8)[0]

    assert isinstance(item, tuple)
    assert len(item) == 2


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("qpos", np.zeros((6, 9), dtype=np.float32)),
        ("qvel", np.zeros((7, 8), dtype=np.float32)),
        ("tcp_pose", np.zeros((7, 6), dtype=np.float32)),
        ("object_pos", np.zeros((6, 3), dtype=np.float32)),
    ],
)
def test_proprio_mode_rejects_misaligned_state(tmp_path, field, replacement):
    root = _write_task(
        tmp_path, "LiftCube-v0", include_proprio=True, frames=7, points=8
    )
    path = root / "episode_0000.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    arrays[field] = replacement
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match=field):
        ManiSkillSequenceDataset(
            root, T=2, point_count=8, include_proprio=True
        )


def test_proprio_mode_rejects_nonfinite_state(tmp_path):
    root = _write_task(
        tmp_path, "LiftCube-v0", include_proprio=True, frames=7, points=8
    )
    path = root / "episode_0000.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    arrays["qvel"] = arrays["qvel"].copy()
    arrays["qvel"][0, 0] = np.nan
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="qvel.*finite"):
        ManiSkillSequenceDataset(
            root, T=2, point_count=8, include_proprio=True
        )


def test_load_proprio_stats_requires_train_only_25d_manifest(tmp_path):
    root = _write_task(tmp_path, "LiftCube-v0", include_proprio=True)

    mean, std = load_proprio_stats(root)

    np.testing.assert_array_equal(mean, np.arange(25, dtype=np.float32))
    np.testing.assert_array_equal(std, np.arange(1, 26, dtype=np.float32))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["proprio_normalization"]["episode_ids"] = [1]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="training split"):
        load_proprio_stats(root)


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


def test_rgb_sequence_uses_same_indices_as_xyz(tmp_path):
    root = _write_task(
        tmp_path, "LiftCube-v0", frames=9, points=8, include_rgb=True
    )
    dataset = ManiSkillSequenceDataset(
        root,
        T=3,
        point_count=4,
        skip_frame=2,
        include_rgb=True,
    )

    points, actions = dataset[0]

    assert points.shape == (4, 4, 6)
    np.testing.assert_allclose(
        points[..., 3:].numpy(),
        np.asarray(points[..., :3].numpy() % 251, dtype=np.uint8) / 255.0,
        rtol=0,
        atol=1e-7,
    )
    assert actions.shape == (30 // 10 * 2, 7)
    assert float(points[..., 3:].min()) >= 0.0
    assert float(points[..., 3:].max()) <= 1.0


def test_rgb_mode_rejects_missing_or_misaligned_rgb(tmp_path):
    root = _write_task(tmp_path, "LiftCube-v0")
    with pytest.raises(ValueError, match="rgb"):
        ManiSkillSequenceDataset(root, T=3, point_count=4, include_rgb=True)

    root = _write_task(
        tmp_path, "PickCube-v0", include_rgb=True
    )
    path = root / "episode_0000.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    arrays["rgb"] = arrays["rgb"][:, :-1]
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="rgb"):
        ManiSkillSequenceDataset(root, T=3, point_count=4, include_rgb=True)


def test_xyz_only_mode_remains_bitwise_compatible_with_rgb_present(tmp_path):
    root = _write_task(tmp_path, "LiftCube-v0", include_rgb=True)
    with_rgb_file = ManiSkillSequenceDataset(
        root, T=3, point_count=4, include_rgb=False
    )[0][0]
    path = root / "episode_0000.npz"
    with np.load(path, allow_pickle=False) as archive:
        np.savez_compressed(
            tmp_path / "xyz_only.npz",
            xyz=archive["xyz"],
            action=archive["action"],
        )
    with np.load(tmp_path / "xyz_only.npz", allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    np.savez_compressed(path, **arrays)
    xyz_only_file = ManiSkillSequenceDataset(
        root, T=3, point_count=4, include_rgb=False
    )[0][0]

    assert np.array_equal(with_rgb_file.numpy(), xyz_only_file.numpy())


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


def test_contact_sidecar_aligns_all_frames_and_marks_phase_repeats(tmp_path):
    root = _write_task(
        tmp_path,
        "LiftCube-v0",
        train=(0,),
        val=(1,),
        frames=7,
        points=8,
        include_proprio=True,
    )
    sidecar = _write_contact_sidecar(
        tmp_path / "contact.npz",
        (0,),
        {0: [0, 0, 1, 1, 1, 1, 1]},
    )

    dataset = ManiSkillSequenceDataset(
        root,
        split="train",
        T=2,
        skip_frame=2,
        point_count=8,
        include_proprio=True,
        contact_sidecar=sidecar,
        phase_repeat=4,
    )

    np.testing.assert_array_equal(
        dataset[0]["cube_robot_contact"].numpy(), [0.0, 1.0, 1.0]
    )
    assert dataset.sample_repeats == [4, 4, 1]


@pytest.mark.parametrize(
    ("mode", "message"),
    [("missing", "missing contact label"), ("duplicate", "duplicate")],
)
def test_contact_sidecar_rejects_missing_or_duplicate_frame_keys(
    tmp_path, mode, message
):
    root = _write_task(
        tmp_path,
        "LiftCube-v0",
        train=(0,),
        val=(1,),
        frames=7,
        points=8,
        include_proprio=True,
    )
    labels = {0: [0, 0, 1, 1, 1, 1, 1]}
    if mode == "missing":
        labels[0] = labels[0][:-1]
    sidecar = _write_contact_sidecar(
        tmp_path / "contact.npz",
        (0,),
        labels,
        duplicate=mode == "duplicate",
    )

    with pytest.raises(ValueError, match=message):
        ManiSkillSequenceDataset(
            root,
            split="train",
            T=2,
            skip_frame=2,
            point_count=8,
            include_proprio=True,
            contact_sidecar=sidecar,
            phase_repeat=4,
        )


def test_contact_sidecar_rejects_alignment_error(tmp_path):
    root = _write_task(
        tmp_path,
        "LiftCube-v0",
        train=(0,),
        val=(1,),
        frames=7,
        points=8,
        include_proprio=True,
    )
    sidecar = _write_contact_sidecar(
        tmp_path / "contact.npz",
        (0,),
        {0: [0, 0, 1, 1, 1, 1, 1]},
        error=2e-4,
    )

    with pytest.raises(ValueError, match="alignment error"):
        ManiSkillSequenceDataset(
            root,
            split="train",
            T=2,
            skip_frame=2,
            point_count=8,
            include_proprio=True,
            contact_sidecar=sidecar,
            contact_alignment_tolerance_m=1e-4,
        )


def test_cross_episode_sampler_repeats_phase_windows_without_collisions():
    dataset = SimpleNamespace(
        index=[(0, 0), (1, 0), (2, 0)],
        sample_repeats=[4, 1, 1],
    )
    sampler = CrossEpisodeBatchSampler(
        dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        seed=0,
    )

    batches = list(sampler)

    assert Counter(index for batch in batches for index in batch) == Counter(
        {0: 4, 1: 1, 2: 1}
    )
    assert len(sampler) == 3
    for batch in batches:
        episodes = [dataset.index[index][0] for index in batch]
        assert len(episodes) == len(set(episodes))
