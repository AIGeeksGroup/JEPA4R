"""PointNet copied from DexArt (Kami-code/dexart-release,
``stable_baselines3/networks/pretrain_nets.py``, used by ``PointNetImaginationExtractorGP``):

    smallpn : Linear(C->64) -> GELU -> Linear(64->256)                                  (default in DexArt)
    mediumpn: Linear(C->64) -> GELU -> Linear(64->64) -> GELU -> Linear(64->128) -> GELU -> Linear(128->256)
    largepn : ... -> 64 -> 64 -> 128 -> 128 -> 256 -> 256 (GELU between layers)

then max over the points. No normalisation layers, no T-Net, no activation
after the last per-point layer. Weights: trunc_normal(std 0.02), zero bias.
DexArt feeds xyz only (C = 3); here C follows the data (6 = xyz + rgb).

Adaptations to LeWM: an affine head ``Linear(mlp_spec[-1] -> out_dim)`` after
the max-pool (192-d latent), and widths scaled to match the ViT-tiny encoder's
5.50 M parameters while keeping each variant's depth and width ratios:
    small  [64, 256]                  -> [1080, 4320]
    medium [64, 64, 128, 256]         -> [672, 672, 1344, 2688]
    large  [64, 64, 128, 128, 256, 256] -> [408, 408, 816, 816, 1632, 1632]
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

DEXART_SPECS = {
    'smallpn': (64, 256),
    'mediumpn': (64, 64, 128, 256),
    'largepn': (64, 64, 128, 128, 256, 256),
}


class DexArtPointNet(nn.Module):
    def __init__(
        self,
        in_dim: int = 6,
        mlp_spec: Sequence[int] = DEXART_SPECS['smallpn'],
        out_dim: int = 192,
        chunk_points: int | None = 300,
    ):
        """chunk_points: see encoders.pointnet_ms2 -- point chunks + activation checkpointing
        while training; per-point MLP + max-pool make it identical to the unchunked forward."""
        super().__init__()
        self.chunk_points = chunk_points
        layers, d = [], in_dim
        for i, h in enumerate(mlp_spec):
            layers.append(nn.Linear(d, h))
            if i < len(mlp_spec) - 1:
                layers.append(nn.GELU())
            d = h
        self.local_mlp = nn.Sequential(*layers)
        self.head = nn.Linear(d, out_dim)
        self.reset_parameters_()

    def reset_parameters_(self):  # DexArt init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _body_max(self, x):  # (B, n, C) -> (B, d)
        return self.local_mlp(x).max(dim=1).values

    def forward(self, x):
        """x: (B, N, C) -> (B, out_dim)"""
        x = x.float()
        if self.chunk_points and x.shape[1] > self.chunk_points:
            use_ckpt = self.training and torch.is_grad_enabled()
            parts = [checkpoint(self._body_max, xc, use_reentrant=False) if use_ckpt else self._body_max(xc)
                     for xc in x.split(self.chunk_points, dim=1)]
            feat = torch.stack(parts, dim=-1).max(dim=-1).values
        else:
            feat = self._body_max(x)
        return self.head(feat)
