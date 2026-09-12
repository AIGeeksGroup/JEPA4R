from pathlib import Path

from addict import Dict
import pytest
import torch
from torch import nn

from models.ordering import OrderingConfig
from models.sonata_aggregator import (
    SonataAggregatorConfig,
    SonataSpatialAggregator,
    center_shift_xyz,
    file_sha256,
    lift_encoder_features,
)


class _FakeBackbone(nn.Module):
    def __init__(self, in_channels, enc_channels, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_channels, enc_channels[-1], bias=False)

    def forward(self, data):
        return Dict(feat=self.linear(data["feat"]))


def _write_checkpoint(path: Path, *, valid_state=True) -> str:
    config = {
        "in_channels": 9,
        "order": ("z",),
        "stride": (),
        "enc_depths": (1,),
        "enc_channels": (4,),
        "enc_num_head": (1,),
        "enc_patch_size": (16,),
        "dec_depths": (),
        "dec_channels": (),
        "dec_num_head": (),
        "dec_patch_size": (),
        "mlp_ratio": 2,
        "qkv_bias": True,
        "qk_scale": None,
        "attn_drop": 0.0,
        "proj_drop": 0.0,
        "drop_path": 0.0,
        "layer_scale": None,
        "pre_norm": True,
        "shuffle_orders": False,
        "enable_rpe": False,
        "enable_flash": False,
        "upcast_attention": False,
        "upcast_softmax": False,
        "traceable": True,
        "mask_token": False,
        "enc_mode": True,
        "freeze_encoder": False,
    }
    backbone = _FakeBackbone(**config)
    state_dict = backbone.state_dict() if valid_state else {}
    torch.save({"config": config, "state_dict": state_dict}, path)
    return file_sha256(path)


def _build_fake(tmp_path: Path) -> SonataSpatialAggregator:
    checkpoint = tmp_path / "sonata-small.pth"
    digest = _write_checkpoint(checkpoint)
    cfg = SonataAggregatorConfig(
        checkpoint_path=str(checkpoint),
        expected_sha256=digest,
        grid_size=0.1,
        lift_concat_levels=0,
    )
    return SonataSpatialAggregator(
        cfg,
        in_channels=6,
        out_channels=3,
        ordering_cfg=OrderingConfig(
            order="z", grid_size=0.1, origin=(-1.0, -1.0, -1.0), depth=8
        ),
        backbone_factory=_FakeBackbone,
    )


def test_center_shift_uses_xy_midpoint_and_z_min():
    coord = torch.tensor(
        [
            [[-2.0, 1.0, 0.4], [4.0, 5.0, 1.4]],
            [[10.0, -3.0, -0.2], [14.0, 1.0, 0.8]],
        ]
    )

    shifted = center_shift_xyz(coord)

    torch.testing.assert_close(shifted[0], torch.tensor([[-3.0, -2.0, 0.0], [3.0, 2.0, 1.0]]))
    torch.testing.assert_close(shifted[1], torch.tensor([[-2.0, -2.0, 0.0], [2.0, 2.0, 1.0]]))


def test_lift_encoder_features_concatenates_two_deep_levels():
    stage0 = Dict(feat=torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
    stage1 = Dict(
        feat=torch.tensor([[10.0, 11.0], [20.0, 21.0]]),
        pooling_parent=stage0,
        pooling_inverse=torch.tensor([0, 0, 1, 1]),
    )
    stage2 = Dict(
        feat=torch.tensor([[100.0, 101.0, 102.0]]),
        pooling_parent=stage1,
        pooling_inverse=torch.tensor([0, 0]),
    )

    lifted = lift_encoder_features(stage2, concat_levels=2)

    assert lifted.shape == (4, 6)
    torch.testing.assert_close(
        lifted[0], torch.tensor([1.0, 10.0, 11.0, 100.0, 101.0, 102.0])
    )
    torch.testing.assert_close(
        lifted[3], torch.tensor([4.0, 20.0, 21.0, 100.0, 101.0, 102.0])
    )


def test_lift_encoder_features_broadcasts_remaining_levels():
    stage0 = Dict(feat=torch.zeros(3, 1))
    stage1 = Dict(
        feat=torch.zeros(2, 2),
        pooling_parent=stage0,
        pooling_inverse=torch.tensor([0, 0, 1]),
    )
    stage2 = Dict(
        feat=torch.tensor([[7.0, 8.0, 9.0]]),
        pooling_parent=stage1,
        pooling_inverse=torch.tensor([0, 0]),
    )

    lifted = lift_encoder_features(stage2, concat_levels=0)

    torch.testing.assert_close(
        lifted, torch.tensor([[7.0, 8.0, 9.0]]).expand(3, -1)
    )


def test_checkpoint_hash_mismatch_fails_before_backbone_construction(tmp_path):
    checkpoint = tmp_path / "sonata.pth"
    _write_checkpoint(checkpoint)
    called = False

    def factory(**kwargs):
        nonlocal called
        called = True
        return _FakeBackbone(**kwargs)

    cfg = SonataAggregatorConfig(
        checkpoint_path=str(checkpoint), expected_sha256="0" * 64
    )
    with pytest.raises(ValueError, match="SHA-256"):
        SonataSpatialAggregator(
            cfg, in_channels=6, out_channels=3, backbone_factory=factory
        )
    assert called is False


def test_checkpoint_load_is_strict(tmp_path):
    checkpoint = tmp_path / "sonata.pth"
    digest = _write_checkpoint(checkpoint, valid_state=False)
    cfg = SonataAggregatorConfig(
        checkpoint_path=str(checkpoint),
        expected_sha256=digest,
        lift_concat_levels=0,
    )

    with pytest.raises(RuntimeError, match="Missing key"):
        SonataSpatialAggregator(
            cfg, in_channels=6, out_channels=3, backbone_factory=_FakeBackbone
        )


def test_frozen_backbone_stays_eval_and_receives_no_grad(tmp_path):
    model = _build_fake(tmp_path).train()
    points = torch.rand(2, 8, 6)
    points[..., :3] = points[..., :3] * 0.5

    tokens, _ = model(points)
    tokens.square().mean().backward()

    assert model.backbone.training is False
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.projection.parameters())


def test_aggregator_restores_points_and_returns_ordered_world_positions(tmp_path):
    model = _build_fake(tmp_path).eval()
    points = torch.rand(2, 8, 6)
    points[..., :3] = points[..., :3] * 0.5
    original_xyz = points[..., :3].clone()

    tokens, positions = model(points)
    order = model.ordering(original_xyz)
    expected_positions = original_xyz.gather(
        1, order.unsqueeze(-1).expand(-1, -1, 3)
    )

    assert tokens.shape == (2, 8, 3)
    assert positions.shape == (2, 8, 3)
    torch.testing.assert_close(positions, expected_positions)
    assert torch.isfinite(tokens).all()


def test_diagnostic_features_expose_raw_and_projected_tokens(tmp_path):
    model = _build_fake(tmp_path).eval()
    points = torch.rand(2, 8, 6)
    points[..., :3] *= 0.5

    result = model.diagnostic_features(points)

    assert result["raw_spatial"].shape == (2, 8, 4)
    assert result["projected_spatial"].shape == (2, 8, 3)
    assert result["positions"].shape == (2, 8, 3)


def test_sonata_requires_rgb_input_width_and_frozen_backbone(tmp_path):
    checkpoint = tmp_path / "sonata.pth"
    digest = _write_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="six-channel"):
        SonataSpatialAggregator(
            SonataAggregatorConfig(
                checkpoint_path=str(checkpoint), expected_sha256=digest
            ),
            in_channels=3,
            out_channels=3,
            backbone_factory=_FakeBackbone,
        )
    with pytest.raises(ValueError, match="frozen"):
        SonataSpatialAggregator(
            SonataAggregatorConfig(
                checkpoint_path=str(checkpoint),
                expected_sha256=digest,
                freeze_backbone=False,
            ),
            in_channels=6,
            out_channels=3,
            backbone_factory=_FakeBackbone,
        )
