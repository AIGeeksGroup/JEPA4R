"""Frozen Sonata PTv3 adapted to LeWM's single-vector encoder contract.

The input is a batched point cloud ``(B, N, 6)`` containing world-frame xyz
and normalized rgb.  Sonata operates on unique 2 cm voxels, and a small CLS
Transformer turns the variable number of voxel features into one ``(B, D)``
vector for the existing LeWM projector and predictor.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from .ppt import SinusoidalPosEmbedder


SONATA_REVISION = "18c09ff8d713494f78a8213792262b910977a65d"


def file_sha256(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def center_shift_xyz(coord: torch.Tensor) -> torch.Tensor:
    """Apply Sonata/Pointcept CenterShift: xy midpoint and z minimum."""
    if coord.dim() != 3 or coord.shape[-1] != 3:
        raise ValueError(f"coord must be [B,K,3], got {tuple(coord.shape)}")
    minimum = coord.amin(dim=1)
    maximum = coord.amax(dim=1)
    center = (minimum + maximum) * 0.5
    center[:, 2] = minimum[:, 2]
    return coord - center[:, None]


def _point_feat(point) -> torch.Tensor:
    return point.feat if hasattr(point, "feat") else point["feat"]


def _set_point_feat(point, feat: torch.Tensor) -> None:
    if hasattr(point, "feat"):
        point.feat = feat
    else:
        point["feat"] = feat


def lift_encoder_features(point, concat_levels: int = 2) -> torch.Tensor:
    """Lift encoder-only Sonata features back to the input voxel resolution."""
    if concat_levels < 0:
        raise ValueError("concat_levels must be non-negative")
    for _ in range(concat_levels):
        if "pooling_parent" not in point or "pooling_inverse" not in point:
            raise ValueError("Sonata output has too few traceable pooling levels")
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        lifted = _point_feat(point)[inverse]
        _set_point_feat(parent, torch.cat([_point_feat(parent), lifted], dim=-1))
        point = parent
    while "pooling_parent" in point:
        if "pooling_inverse" not in point:
            raise ValueError("traceable Sonata parent is missing pooling_inverse")
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        _set_point_feat(parent, _point_feat(point)[inverse])
        point = parent
    return _point_feat(point)


def _scatter_mean(source: torch.Tensor, index: torch.Tensor, group_count: int) -> torch.Tensor:
    output = source.new_zeros((group_count, source.shape[-1]))
    output.index_add_(0, index, source)
    counts = torch.bincount(index, minlength=group_count).to(source.dtype)
    return output / counts.clamp_min_(1).unsqueeze(-1)


class SonataCLSEncoder(nn.Module):
    """Frozen Sonata backbone plus a trainable voxel-level CLS readout.

    The Sonata backbone is kept in ``eval`` mode and runs under ``no_grad``.
    Only the projection and the two-layer CLS Transformer are optimized by
    LeWM.  The output is always one vector per input frame: ``(B, out_dim)``.
    """

    def __init__(
        self,
        checkpoint_path: str,
        expected_sha256: str,
        in_dim: int = 6,
        out_dim: int = 192,
        grid_size: float = 0.02,
        lift_concat_levels: int = 2,
        cls_depth: int = 2,
        cls_heads: int = 8,
        cls_mlp_ratio: float = 4.0,
        enable_flash: bool = False,
        *,
        freeze_backbone: bool = True,
        backbone_factory: Callable[..., nn.Module] | None = None,
    ):
        super().__init__()
        if in_dim != 6:
            raise ValueError("SonataCLSEncoder requires six channels: xyz plus rgb")
        if not freeze_backbone:
            raise ValueError("SonataCLSEncoder requires a frozen backbone")
        if grid_size <= 0:
            raise ValueError("Sonata grid_size must be positive")
        if cls_depth < 1 or cls_heads < 1 or cls_mlp_ratio <= 0:
            raise ValueError("CLS Transformer depth, heads and mlp_ratio must be positive")
        if out_dim % cls_heads != 0:
            raise ValueError("out_dim must be divisible by cls_heads")

        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing Sonata checkpoint: {checkpoint}")
        actual_sha256 = file_sha256(checkpoint)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Sonata checkpoint SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if set(state) != {"config", "state_dict"}:
            raise ValueError("Sonata checkpoint must contain exactly config and state_dict")
        backbone_config = dict(state["config"])
        if int(backbone_config.get("in_channels", -1)) != 9:
            raise ValueError("Sonata checkpoint must have a nine-channel input stem")
        enc_channels = tuple(int(value) for value in backbone_config.get("enc_channels", ()))
        if not enc_channels:
            raise ValueError("Sonata checkpoint config is missing enc_channels")
        if not 0 <= lift_concat_levels < len(enc_channels):
            raise ValueError("lift_concat_levels must be smaller than Sonata encoder stages")

        backbone_config.update(
            enable_flash=bool(enable_flash),
            shuffle_orders=False,
            drop_path=0.0,
            traceable=True,
            enc_mode=True,
        )
        if backbone_factory is None:
            from sonata.model import PointTransformerV3

            backbone_factory = PointTransformerV3
        self.backbone = backbone_factory(**backbone_config)
        self.backbone.load_state_dict(state["state_dict"], strict=True)
        self.freeze_backbone = bool(freeze_backbone)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        lifted_channels = sum(enc_channels[-(lift_concat_levels + 1) :])
        self.projection = nn.Sequential(
            nn.LayerNorm(lifted_channels),
            nn.Linear(lifted_channels, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.pos_embedder = SinusoidalPosEmbedder(out_dim, n_dim=3)
        layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=cls_heads,
            dim_feedforward=int(out_dim * cls_mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cls_transformer = nn.TransformerEncoder(
            layer, num_layers=cls_depth, enable_nested_tensor=False
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, out_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.projection.apply(self._init_weights)
        self.cls_transformer.apply(self._init_weights)

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.grid_size = float(grid_size)
        self.lift_concat_levels = int(lift_concat_levels)
        self.checkpoint_path = str(checkpoint)
        self.checkpoint_sha256 = actual_sha256
        self.checkpoint_config = backbone_config

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _unique_voxels(
        self, centered_xyz: torch.Tensor, rgb: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor | float], torch.Tensor]:
        batch_size, point_count, _ = centered_xyz.shape
        minimum = centered_xyz.amin(dim=1, keepdim=True)
        grid = torch.div(
            centered_xyz - minimum, self.grid_size, rounding_mode="floor"
        ).long()
        batch = torch.arange(
            batch_size, device=centered_xyz.device, dtype=torch.long
        ).repeat_interleave(point_count)
        key = torch.cat([batch[:, None], grid.reshape(-1, 3)], dim=1)
        key, inverse = torch.unique(key, sorted=True, return_inverse=True, dim=0)
        group_count = key.shape[0]
        flat_xyz = centered_xyz.reshape(-1, 3)
        flat_rgb = rgb.reshape(-1, 3)
        unique_xyz = _scatter_mean(flat_xyz, inverse, group_count)
        unique_rgb = _scatter_mean(flat_rgb, inverse, group_count)
        normal = torch.zeros_like(unique_rgb)
        feat = torch.cat([unique_xyz, unique_rgb, normal], dim=-1)
        if feat.shape[-1] != int(self.checkpoint_config["in_channels"]):
            raise RuntimeError("constructed Sonata feature width differs from checkpoint")
        return (
            {
                "coord": unique_xyz,
                "feat": feat,
                "batch": key[:, 0],
                "grid_coord": key[:, 1:].int(),
                "grid_size": float(self.grid_size),
            },
            key[:, 0],
        )

    def _pad_voxel_tokens(
        self, tokens: torch.Tensor, positions: torch.Tensor, batch_ids: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = torch.bincount(batch_ids, minlength=batch_size)
        if bool((counts == 0).any()):
            raise ValueError("each sample must contain at least one point")
        max_voxels = int(counts.max().item())
        padded = tokens.new_zeros((batch_size, max_voxels, self.out_dim))
        padded_positions = positions.new_zeros((batch_size, max_voxels, 3))
        padding_mask = torch.ones(
            (batch_size, max_voxels), dtype=torch.bool, device=tokens.device
        )
        start = 0
        for sample, count in enumerate(counts.tolist()):
            end = start + count
            padded[sample, :count] = tokens[start:end]
            padded_positions[sample, :count] = positions[start:end]
            padding_mask[sample, :count] = False
            start = end
        position_encoding = self.pos_embedder(padded_positions).to(dtype=padded.dtype)
        padded = padded + position_encoding
        return padded, padding_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, N, 6)`` point clouds into ``(B, out_dim)`` CLS vectors."""
        if x.dim() != 3 or x.shape[-1] != self.in_dim:
            raise ValueError(f"points must be [B,N,{self.in_dim}], got {tuple(x.shape)}")
        if x.shape[1] == 0:
            raise ValueError("SonataCLSEncoder requires at least one point")
        x = x.float()
        if not torch.isfinite(x).all():
            raise ValueError("Sonata points must be finite")
        rgb = x[..., 3:6]
        if bool((rgb < 0).any()) or bool((rgb > 1).any()):
            raise ValueError("Sonata RGB must be normalized to [0,1]")

        centered_xyz = center_shift_xyz(x[..., :3])
        data, batch_ids = self._unique_voxels(centered_xyz, rgb)
        with torch.no_grad():
            encoded = self.backbone(data)
            raw_unique = lift_encoder_features(
                encoded, concat_levels=self.lift_concat_levels
            )
        if raw_unique.dim() != 2 or raw_unique.shape[0] != data["coord"].shape[0]:
            raise RuntimeError(
                "Sonata encoder output must align one feature with every unique voxel"
            )
        projected = self.projection(raw_unique)
        voxel_tokens, padding_mask = self._pad_voxel_tokens(
            projected, data["coord"], batch_ids, x.shape[0]
        )
        cls = self.cls_token.to(dtype=voxel_tokens.dtype).expand(x.shape[0], -1, -1)
        sequence = torch.cat([cls, voxel_tokens], dim=1)
        sequence_mask = torch.cat(
            [torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device), padding_mask],
            dim=1,
        )
        encoded_sequence = self.cls_transformer(
            sequence, src_key_padding_mask=sequence_mask
        )
        return encoded_sequence[:, 0]

    def provenance(self) -> dict[str, object]:
        return {
            "backend": "sonata_ptv3_cls",
            "source_revision": SONATA_REVISION,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_config": self.checkpoint_config,
            "feature_layout": ["centered_xyz", "rgb", "zero_normal"],
            "grid_size": self.grid_size,
            "lift_concat_levels": self.lift_concat_levels,
            "frozen_backbone": True,
        }
