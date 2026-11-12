"""Frozen-encoder interface shared by every policy learner and every encoder row.

    enc = jepa4r.methods.load_frozen_encoder(cfg)   # -> cfg.method.impl.load_frozen_encoder(cfg)
    enc.obs           "pointcloud" | "image"
    enc.latent_dim    int
    enc.encode(batch) -> float tensor [B, D]          no grad, eval mode, on cfg.device
"""
from typing import Any, Protocol


class FrozenEncoder(Protocol):
    obs: str
    latent_dim: int

    def encode(self, batch: Any) -> Any: ...
