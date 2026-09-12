import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.eval_liftcube_frozen_probe import (
    ContactSidecar,
    ExtractedFeatures,
    ProbeWindowDataset,
    append_current_proprio,
    attach_contact_labels,
    evaluate_probe_suite,
    extract_frozen_features,
    load_contact_sidecar,
    pool_scene_tokens,
    save_probe_outputs,
)


def _write_probe_root(root: Path, *, frames: int = 9, points: int = 4) -> Path:
    root.mkdir(parents=True)
    manifest = {
        "task": "LiftCube-v0",
        "episode_ids": [0, 1],
        "train_episode_ids": [0],
        "val_episode_ids": [1],
        "proprio_normalization": {
            "source_split": "train",
            "episode_ids": [0],
            "mean": [0.0] * 25,
            "std": [1.0] * 25,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for episode_id in (0, 1):
        frame_value = np.arange(frames, dtype=np.float32) + 100 * episode_id
        xyz = np.broadcast_to(frame_value[:, None, None], (frames, points, 3)).copy()
        action = np.zeros((frames - 1, 7), dtype=np.float32)
        action[:, 0] = np.arange(frames - 1) + 10 * episode_id
        qpos = np.zeros((frames, 9), dtype=np.float32)
        qpos[:, -2] = frame_value * 0.001
        qpos[:, -1] = -frame_value * 0.002
        qvel = np.broadcast_to(frame_value[:, None], (frames, 9)).copy()
        tcp_pose = np.zeros((frames, 7), dtype=np.float32)
        tcp_pose[:, :3] = np.stack(
            [frame_value, frame_value + 1, frame_value + 2], axis=-1
        )
        tcp_pose[:, 3] = 1.0
        object_pos = np.zeros((frames, 3), dtype=np.float32)
        object_pos[:, 2] = frame_value + 0.5
        np.savez(
            root / f"episode_{episode_id:04d}.npz",
            xyz=xyz,
            action=action,
            qpos=qpos,
            qvel=qvel,
            tcp_pose=tcp_pose,
            object_pos=object_pos,
        )
    return root


def test_probe_dataset_aligns_target_frame_actions_and_physical_labels(tmp_path):
    root = _write_probe_root(tmp_path / "LiftCube-v0")
    dataset = ProbeWindowDataset(
        root,
        split="train",
        T=2,
        point_count=4,
        skip_frame=2,
    )

    item = dataset[0]

    assert dataset.episode_ids == [0]
    assert item["episode_id"].item() == 0
    assert item["frame_id"].item() == 4
    assert item["actions"][:, 0].tolist() == pytest.approx([0, 1, 2, 3])
    assert item["points"][:, 0, 0].tolist() == pytest.approx([0, 2, 4])
    assert item["object_height"].item() == pytest.approx(4.5)
    assert item["tcp_position"].tolist() == pytest.approx([4, 5, 6])
    assert item["gripper_gap"].item() == pytest.approx(0.012)
    assert item["gripper_closed"].item() == 1


def test_probe_dataset_keeps_train_and_validation_episodes_disjoint(tmp_path):
    root = _write_probe_root(tmp_path / "LiftCube-v0")
    train = ProbeWindowDataset(root, split="train", T=2, point_count=4, skip_frame=2)
    val = ProbeWindowDataset(root, split="val", T=2, point_count=4, skip_frame=2)

    assert train.episode_ids == [0]
    assert val.episode_ids == [1]
    assert set(train.episode_ids).isdisjoint(val.episode_ids)
    assert {int(train[index]["episode_id"]) for index in range(len(train))} == {0}
    assert {int(val[index]["episode_id"]) for index in range(len(val))} == {1}


def test_pooling_and_current_proprio_feature_dimensions():
    tokens = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    proprio = torch.ones(2, 25)

    pooled = pool_scene_tokens(tokens)
    full_state = append_current_proprio(pooled, proprio)

    torch.testing.assert_close(pooled, tokens.mean(dim=1))
    assert pooled.shape == (2, 5)
    assert full_state.shape == (2, 30)
    torch.testing.assert_close(full_state[:, :5], pooled)
    torch.testing.assert_close(full_state[:, 5:], proprio)


class _ScaleProprio:
    def normalize(self, value):
        return value * 0.1


class _FakeProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(3.0))
        self.cfg = SimpleNamespace(T=2, proprio=SimpleNamespace(enabled=True, dim=25))
        self.proprio_encoder = _ScaleProprio()

    def forward(self, points, actions, env, proprio, object_pos):
        del actions, env
        target = points[:, -1, :, :2] + self.anchor * 0.0
        current = points[:, self.cfg.T - 1, :, :2]
        return {
            "target": target,
            "pred": target + 1.0,
            "latent2": current + 2.0,
            "next_proprio": proprio[:, 1:],
            "object_pos": object_pos[:, 1:],
        }


def test_extract_frozen_features_records_all_boundaries_without_gradients(tmp_path):
    root = _write_probe_root(tmp_path / "LiftCube-v0")
    dataset = ProbeWindowDataset(root, split="train", T=2, point_count=4, skip_frame=2)
    model = _FakeProbeModel()
    before = model.anchor.detach().clone()

    extracted = extract_frozen_features(
        model,
        dataset,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert extracted.representations["true_spatial_next"].shape == (len(dataset), 2)
    assert extracted.representations["predicted_spatial_next"].shape == (
        len(dataset),
        2,
    )
    assert extracted.representations["temporal_current"].shape == (len(dataset), 2)
    assert extracted.representations["current_proprio_normalized"].shape == (
        len(dataset),
        25,
    )
    np.testing.assert_allclose(
        extracted.representations["predicted_spatial_next"],
        extracted.representations["true_spatial_next"] + 1.0,
    )
    np.testing.assert_allclose(
        extracted.labels["object_height"],
        np.asarray([4.5, 5.5, 6.5, 7.5, 8.5]),
    )
    np.testing.assert_array_equal(extracted.episode_id, np.zeros(len(dataset), dtype=np.int64))
    np.testing.assert_array_equal(extracted.frame_id, np.arange(4, 9, dtype=np.int64))
    assert extracted.existing_heads["object_position"].shape == (len(dataset), 3)
    assert extracted.existing_heads["next_proprio"].shape == (len(dataset), 25)
    assert extracted.labels["object_position"].shape == (len(dataset), 3)
    assert model.training is False
    assert model.anchor.grad is None
    torch.testing.assert_close(model.anchor.detach(), before)


def _write_sidecar(path: Path, *, duplicate: bool = False, bad_error: bool = False):
    episode_id = np.asarray([0, 0, 1], dtype=np.int64)
    frame_id = np.asarray([4, 5, 4], dtype=np.int64)
    if duplicate:
        episode_id[-1], frame_id[-1] = 0, 4
    np.savez(
        path,
        episode_id=episode_id,
        frame_id=frame_id,
        cube_robot_contact=np.asarray([0, 1, 1], dtype=np.uint8),
        object_pos_error_m=np.asarray([0.0, 2e-6, 2e-3 if bad_error else 1e-6]),
        tcp_pos_error_m=np.asarray([0.0, 1e-6, 1e-6]),
    )


def test_contact_sidecar_joins_exact_episode_frame_keys(tmp_path):
    path = tmp_path / "contact.npz"
    _write_sidecar(path)
    sidecar = load_contact_sidecar(path, alignment_tolerance_m=1e-4)
    extracted = ExtractedFeatures(
        representations={},
        labels={},
        existing_heads={},
        episode_id=np.asarray([1, 0, 0]),
        frame_id=np.asarray([4, 5, 4]),
    )

    joined = attach_contact_labels(extracted, sidecar)

    np.testing.assert_array_equal(joined.labels["cube_robot_contact"], [1, 1, 0])


@pytest.mark.parametrize("mode, message", [("duplicate", "duplicate"), ("error", "alignment")])
def test_contact_sidecar_rejects_duplicates_and_alignment_failures(tmp_path, mode, message):
    path = tmp_path / "contact.npz"
    _write_sidecar(path, duplicate=mode == "duplicate", bad_error=mode == "error")

    with pytest.raises(ValueError, match=message):
        load_contact_sidecar(path, alignment_tolerance_m=1e-4)


def test_contact_sidecar_rejects_missing_requested_key():
    sidecar = ContactSidecar(
        episode_id=np.asarray([0]),
        frame_id=np.asarray([4]),
        cube_robot_contact=np.asarray([1]),
        object_pos_error_m=np.asarray([0.0]),
        tcp_pos_error_m=np.asarray([0.0]),
    )
    extracted = ExtractedFeatures(
        representations={},
        labels={},
        existing_heads={},
        episode_id=np.asarray([0]),
        frame_id=np.asarray([5]),
    )

    with pytest.raises(ValueError, match="missing contact label"):
        attach_contact_labels(extracted, sidecar)


def _synthetic_extracted(count: int, *, episode_offset: int) -> ExtractedFeatures:
    x0 = np.linspace(-2.0, 2.0, count)
    x1 = np.sin(np.linspace(0.0, 2.0, count))
    true_scene = np.stack([x0, x1], axis=1)
    predicted_scene = true_scene + np.asarray([0.35, -0.2])
    proprio = np.zeros((count, 25), dtype=np.float64)
    proprio[:, 0] = x0
    object_position = np.stack([x1, -x1, 0.2 + 0.03 * x0], axis=1)
    tcp_position = np.stack([x0, x1, x0 - x1], axis=1)
    gap = 0.02 - 0.006 * x0
    qpos = np.zeros((count, 9), dtype=np.float64)
    qpos[:, -2:] = gap[:, None] / 2.0
    next_proprio = np.zeros((count, 25), dtype=np.float64)
    next_proprio[:, :9] = qpos
    next_proprio[:, 18:21] = tcp_position
    # The final two entries are TCP quaternion components, not gripper joints.
    # Keep them deliberately large so an accidental [:, -2:] read fails loudly.
    next_proprio[:, 23:25] = np.asarray([0.6, 0.8])
    labels = {
        "object_position": object_position,
        "object_height": object_position[:, 2],
        "tcp_position": tcp_position,
        "gripper_gap": gap,
        "gripper_closed": (gap <= 0.02).astype(np.int64),
        "cube_robot_contact": (x0 > 0).astype(np.int64),
    }
    return ExtractedFeatures(
        representations={
            "true_spatial_next": true_scene,
            "predicted_spatial_next": predicted_scene,
            "temporal_current": true_scene * np.asarray([0.8, 1.2]),
            "current_proprio_normalized": proprio,
        },
        labels=labels,
        existing_heads={
            "object_position": object_position.copy(),
            "next_proprio": next_proprio,
        },
        episode_id=np.arange(count, dtype=np.int64) + episode_offset,
        frame_id=np.arange(count, dtype=np.int64) + 10,
    )


def test_probe_suite_reuses_true_spatial_probe_on_predicted_latents():
    train = _synthetic_extracted(40, episode_offset=0)
    val = _synthetic_extracted(20, episode_offset=100)

    summary, predictions = evaluate_probe_suite(
        train,
        val,
        ridge_alpha=1e-6,
        logistic_l2=1e-4,
        logistic_max_iter=500,
    )

    object_representations = summary["targets"]["object_height"]["representations"]
    true_result = object_representations["true_spatial_next"]
    predicted_result = object_representations["predicted_spatial_next_transfer"]
    assert true_result["probe_sha256"] == predicted_result["probe_sha256"]
    assert predicted_result["probe_fitted_on"] == "train:true_spatial_next"
    assert "object_height__true_spatial_next" in predictions
    assert "object_height__predicted_spatial_next_transfer" in predictions
    assert summary["existing_heads"]["object_height"]["rmse"] == pytest.approx(0.0)
    assert summary["existing_heads"]["tcp_position"]["rmse"] == pytest.approx(0.0)
    assert summary["existing_heads"]["gripper_gap"]["rmse"] == pytest.approx(0.0)
    assert summary["existing_heads"]["gripper_closed"][
        "balanced_accuracy"
    ] == pytest.approx(1.0)


def test_probe_outputs_have_versioned_json_and_npz_schema(tmp_path):
    train = _synthetic_extracted(30, episode_offset=0)
    val = _synthetic_extracted(12, episode_offset=100)
    summary, predictions = evaluate_probe_suite(train, val)

    save_probe_outputs(tmp_path, summary=summary, predictions=predictions, val=val)

    loaded_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert loaded_summary["schema_version"] == 1
    assert (tmp_path / "REPORT.md").is_file()
    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as archive:
        assert archive["episode_id"].shape == (12,)
        assert archive["frame_id"].shape == (12,)
        assert "label__object_height" in archive.files
        assert "prediction__object_height__true_spatial_next" in archive.files
