"""CPU tests for the Sonata CLS adapter using a fake frozen backbone."""

from pathlib import Path

import pytest
import torch
from torch import nn

from encoders.sonata_cls import SonataCLSEncoder, file_sha256, lift_encoder_features


class _FakeBackbone(nn.Module):
    def __init__(self, in_channels, enc_channels, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_channels, enc_channels[-1], bias=False)

    def forward(self, data):
        return {"feat": self.linear(data["feat"])}


def _write_checkpoint(path: Path, *, valid_state: bool = True) -> str:
    config = {"in_channels": 9, "enc_channels": (4,)}
    backbone = _FakeBackbone(**config)
    state_dict = backbone.state_dict() if valid_state else {}
    torch.save({"config": config, "state_dict": state_dict}, path)
    return file_sha256(path)


def _build(tmp_path: Path) -> SonataCLSEncoder:
    checkpoint = tmp_path / "sonata-small.pth"
    digest = _write_checkpoint(checkpoint)
    return SonataCLSEncoder(
        checkpoint_path=str(checkpoint),
        expected_sha256=digest,
        grid_size=0.1,
        lift_concat_levels=0,
        backbone_factory=_FakeBackbone,
    )


def _points(batch: int = 2, count: int = 8) -> torch.Tensor:
    points = torch.rand(batch, count, 6)
    points[..., :3] = points[..., :3] * 0.5
    return points


def test_output_is_one_finite_cls_vector_per_sample(tmp_path):
    model = _build(tmp_path).eval()
    points = _points()
    with torch.no_grad():
        output = model(points)
    assert output.shape == (2, 192)
    assert torch.isfinite(output).all()


def test_lift_encoder_features_concatenates_requested_levels():
    stage0 = {"feat": torch.tensor([[1.0], [2.0], [3.0], [4.0]])}
    stage1 = {
        "feat": torch.tensor([[10.0, 11.0], [20.0, 21.0]]),
        "pooling_parent": stage0,
        "pooling_inverse": torch.tensor([0, 0, 1, 1]),
    }
    stage2 = {
        "feat": torch.tensor([[100.0, 101.0, 102.0]]),
        "pooling_parent": stage1,
        "pooling_inverse": torch.tensor([0, 0]),
    }
    lifted = lift_encoder_features(stage2, concat_levels=2)
    assert lifted.shape == (4, 6)
    torch.testing.assert_close(
        lifted[0], torch.tensor([1.0, 10.0, 11.0, 100.0, 101.0, 102.0])
    )


def test_eval_forward_is_deterministic(tmp_path):
    model = _build(tmp_path).eval()
    points = _points()
    with torch.no_grad():
        first, second = model(points), model(points)
    torch.testing.assert_close(first, second)


def test_validation_rejects_bad_shape_rgb_and_hash(tmp_path):
    checkpoint = tmp_path / "sonata-small.pth"
    digest = _write_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="six channels"):
        SonataCLSEncoder(
            checkpoint_path=str(checkpoint),
            expected_sha256=digest,
            in_dim=3,
            lift_concat_levels=0,
            backbone_factory=_FakeBackbone,
        )
    model = SonataCLSEncoder(
        checkpoint_path=str(checkpoint),
        expected_sha256=digest,
        lift_concat_levels=0,
        backbone_factory=_FakeBackbone,
    ).eval()
    bad = _points()
    bad[..., 3] = 1.1
    with pytest.raises(ValueError, match="RGB"):
        model(bad)
    with pytest.raises(ValueError, match="SHA-256"):
        SonataCLSEncoder(
            checkpoint_path=str(checkpoint),
            expected_sha256="0" * 64,
            lift_concat_levels=0,
            backbone_factory=_FakeBackbone,
        )


def test_backbone_stays_frozen_and_eval_after_train(tmp_path):
    model = _build(tmp_path).train()
    output = model(_points())
    output.square().mean().backward()
    assert model.backbone.training is False
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.projection.parameters())


def test_cls_sequence_uses_unique_voxels_not_original_point_count(tmp_path):
    model = _build(tmp_path).eval()

    class _Capture(nn.Module):
        def forward(self, sequence, src_key_padding_mask=None):
            self.sequence = sequence
            self.mask = src_key_padding_mask
            return sequence

    capture = _Capture()
    model.cls_transformer = capture
    points = torch.zeros(2, 8, 6)
    points[..., 3:] = 0.5
    points[1, :, :3] = torch.arange(8, dtype=torch.float32).view(8, 1) * 0.2
    with torch.no_grad():
        model(points)

    # Sample 0 collapses to one voxel; sample 1 has eight. The padded batch
    # sequence is therefore CLS + max(unique voxels), not CLS + 1200 points.
    assert capture.sequence.shape == (2, 1 + 8, 192)
    assert capture.mask.shape == (2, 1 + 8)
    assert capture.mask[0].tolist() == [False, False] + [True] * 7
    assert capture.mask[1].tolist() == [False] * 9


def test_strict_checkpoint_load_is_enforced(tmp_path):
    checkpoint = tmp_path / "sonata-small.pth"
    digest = _write_checkpoint(checkpoint, valid_state=False)
    with pytest.raises(RuntimeError, match="Missing key"):
        SonataCLSEncoder(
            checkpoint_path=str(checkpoint),
            expected_sha256=digest,
            lift_concat_levels=0,
            backbone_factory=_FakeBackbone,
        )
