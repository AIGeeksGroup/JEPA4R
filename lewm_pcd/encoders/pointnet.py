"""Reference point-cloud encoder: a minimal PointNet.

Per-point shared MLP -> max-pool over points -> linear head. Permutation
invariant, no dependence on point order or count. Meant to exercise the
training pipeline end to end; swap in any architecture that follows the same
``(B, N, C) -> (B, out_dim)`` contract.
"""

from torch import nn


class PointNet(nn.Module):
    def __init__(
        self,
        in_dim: int = 6,
        out_dim: int = 192,
        hidden: tuple[int, ...] = (64, 128, 256, 512),
    ):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
            d = h
        self.point_mlp = nn.Sequential(*layers)
        self.head = nn.Linear(d, out_dim)

    def forward(self, x):
        """x: (B, N, C) -> (B, out_dim)"""
        feats = self.point_mlp(x.float())      # (B, N, d)
        pooled = feats.max(dim=1).values       # (B, d)
        return self.head(pooled)
