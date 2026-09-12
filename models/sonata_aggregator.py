"""Frozen official Sonata PTv3 encoder adapted to the world-model token contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from models.ordering import Ordering, OrderingConfig


SONATA_REVISION = "18c09ff8d713494f78a8213792262b910977a65d"


@dataclass
class SonataAggregatorConfig:
    checkpoint_path: str = ""
    expected_sha256: str = ""
    freeze_backbone: bool = True
    enable_flash: bool = False
    grid_size: float = 0.02
    lift_concat_levels: int = 2


def file_sha256(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def center_shift_xyz(coord: torch.Tensor) -> torch.Tensor:
    """Apply Sonata/Pointcept CenterShift with xy midpoint and z minimum."""
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
    """Lift encoder-only Sonata features using its published traceable chain."""
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


def _scatter_mean(
    source: torch.Tensor, index: torch.Tensor, group_count: int
) -> torch.Tensor:
    output = source.new_zeros((group_count, source.shape[-1]))
    output.index_add_(0, index, source)
    counts = torch.bincount(index, minlength=group_count).to(source.dtype)
    return output / counts.clamp_min_(1).unsqueeze(-1)


class SonataSpatialAggregator(nn.Module):
    """Map `[B,K,xyz+rgb]` to ordered `[B,K,D]` tokens and world XYZ."""

    def __init__(
        self,
        cfg: SonataAggregatorConfig,
        in_channels: int,
        out_channels: int,
        *,
        ordering_cfg: OrderingConfig | None = None,
        backbone_factory: Callable[..., nn.Module] | None = None,
    ):
        super().__init__()
        if in_channels != 6:
            raise ValueError(
                "Sonata requires six-channel model points: xyz plus aligned rgb"
            )
        if not cfg.freeze_backbone:
            raise ValueError("the first Sonata experiment requires a frozen backbone")
        if cfg.grid_size <= 0:
            raise ValueError("Sonata grid_size must be positive")
        checkpoint_path = Path(cfg.checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing Sonata checkpoint: {checkpoint_path}")
        actual_sha256 = file_sha256(checkpoint_path)
        if actual_sha256 != cfg.expected_sha256:
            raise ValueError(
                "Sonata checkpoint SHA-256 mismatch: "
                f"expected {cfg.expected_sha256}, got {actual_sha256}"
            )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if set(state) != {"config", "state_dict"}:
            raise ValueError(
                "Sonata checkpoint must contain exactly config and state_dict"
            )
        backbone_config = dict(state["config"])
        if int(backbone_config.get("in_channels", -1)) != 9:
            raise ValueError("Sonata checkpoint must have a nine-channel input stem")
        enc_channels = tuple(int(value) for value in backbone_config["enc_channels"])
        if not 0 <= cfg.lift_concat_levels < len(enc_channels):
            raise ValueError(
                "lift_concat_levels must be smaller than Sonata encoder stages"
            )
        backbone_config.update(
            enable_flash=bool(cfg.enable_flash),
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
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        lifted_channels = sum(enc_channels[-(cfg.lift_concat_levels + 1) :])
        self.projection = nn.Sequential(
            nn.LayerNorm(lifted_channels),
            nn.Linear(lifted_channels, out_channels),
            nn.LayerNorm(out_channels),
        )
        self.cfg = cfg
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.checkpoint_sha256 = actual_sha256
        self.checkpoint_config = backbone_config
        self.ordering = Ordering(ordering_cfg or OrderingConfig())

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
            centered_xyz - minimum,
            self.cfg.grid_size,
            rounding_mode="floor",
        ).long()
        batch = torch.arange(
            batch_size, device=centered_xyz.device, dtype=torch.long
        ).repeat_interleave(point_count)
        key = torch.cat([batch[:, None], grid.reshape(-1, 3)], dim=1)
        key, inverse = torch.unique(
            key, sorted=True, return_inverse=True, dim=0
        )
        group_count = key.shape[0]
        unique_xyz = _scatter_mean(
            centered_xyz.reshape(-1, 3), inverse, group_count
        )
        unique_rgb = _scatter_mean(rgb.reshape(-1, 3), inverse, group_count)
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
                "grid_size": float(self.cfg.grid_size),
            },
            inverse,
        )

    def _encode(
        self, points: torch.Tensor, *, return_raw: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        if points.dim() != 3 or points.shape[-1] != self.in_channels:
            raise ValueError(
                f"points must be [B,K,{self.in_channels}], got {tuple(points.shape)}"
            )
        world_xyz = points[..., :3]
        rgb = points[..., 3:6]
        if not torch.isfinite(points).all():
            raise ValueError("Sonata points must be finite")
        if bool((rgb < 0).any()) or bool((rgb > 1).any()):
            raise ValueError("Sonata RGB must be normalized to [0,1]")
        centered_xyz = center_shift_xyz(world_xyz)
        data, voxel_inverse = self._unique_voxels(centered_xyz, rgb)
        with torch.no_grad():
            encoded = self.backbone(data)
            raw_unique = lift_encoder_features(
                encoded, concat_levels=self.cfg.lift_concat_levels
            )
            raw_points = raw_unique[voxel_inverse].view(
                points.shape[0], points.shape[1], -1
            )
        tokens = self.projection(raw_points)
        order = self.ordering(world_xyz)
        index = order.unsqueeze(-1)
        tokens = tokens.gather(1, index.expand(-1, -1, self.out_channels))
        positions = world_xyz.gather(1, index.expand(-1, -1, 3))
        if return_raw:
            raw_points = raw_points.gather(
                1, index.expand(-1, -1, raw_points.shape[-1])
            )
            return raw_points, tokens, positions
        return tokens, positions

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encode(points)

    @torch.no_grad()
    def diagnostic_features(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        raw, projected, positions = self._encode(points, return_raw=True)
        return {
            "raw_spatial": raw,
            "projected_spatial": projected,
            "positions": positions,
        }

    def provenance(self) -> dict[str, object]:
        return {
            "backend": "sonata_ptv3",
            "source_revision": SONATA_REVISION,
            "checkpoint_path": self.cfg.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_config": self.checkpoint_config,
            "adapter_config": asdict(self.cfg),
            "feature_layout": ["centered_xyz", "rgb_div_255", "zero_normal"],
            "strict_load": True,
            "frozen_backbone": True,
        }
