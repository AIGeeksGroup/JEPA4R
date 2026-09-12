"""Normalization and encoding for optional robot-state conditioning."""

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class ProprioConfig:
    enabled: bool = False
    dim: int = 25
    hidden_dim: int = 128
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)


class ProprioEncoder(nn.Module):
    def __init__(self, cfg: ProprioConfig, D: int):
        super().__init__()
        if cfg.dim < 1 or cfg.hidden_dim < 1:
            raise ValueError("proprio dim and hidden_dim must be positive")
        if len(cfg.mean) != cfg.dim or len(cfg.std) != cfg.dim:
            raise ValueError(
                f"enabled proprio mean/std must each contain {cfg.dim} values"
            )
        mean = torch.tensor(cfg.mean, dtype=torch.float32)
        std = torch.tensor(cfg.std, dtype=torch.float32)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("proprio mean/std must be finite")
        if not (std > 0).all():
            raise ValueError("proprio std must be positive")
        self.dim = int(cfg.dim)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(1e-6))
        self.net = nn.Sequential(
            nn.Linear(self.dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, D),
        )
        self.robot_type = nn.Parameter(torch.zeros(1, 1, D))

    def _validate(self, value: torch.Tensor) -> None:
        if value.shape[-1] != self.dim:
            raise ValueError(
                f"proprio last dimension must be {self.dim}, got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError("proprio must be finite")

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        self._validate(value)
        return (value - self.mean) / self.std

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        self._validate(normalized)
        return normalized * self.std + self.mean

    def forward_normalized(self, normalized: torch.Tensor) -> torch.Tensor:
        self._validate(normalized)
        return self.net(normalized) + self.robot_type

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.forward_normalized(self.normalize(value))
