"""Shared trunk of the decoders: a fixed-size set generated from one latent vector."""

import torch
import torch.nn.functional as F
from torch import nn


class FiLMBlock(nn.Module):
    """Residual per-element MLP modulated by the latent (FiLM)."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, h, gamma, beta):
        # h: (B, M, dim); gamma / beta: (B, 1, dim)
        return h + self.fc2(F.gelu(self.fc1(self.norm(h) * (1.0 + gamma) + beta)))


class SetDecoder(nn.Module):
    """latent (B, D) -> (B, M, out_dim).

    M learned queries say "which element of the set am I"; the latent modulates every block. Elements are
    decoded independently of each other. The last projection runs in float32: its output is coordinates."""

    def __init__(self, latent_dim=192, num_queries=1200, out_dim=6, dim=512, hidden=1024, depth=3, query_dim=128):
        super().__init__()
        self.dim, self.depth = dim, depth
        self.queries = nn.Parameter(torch.randn(num_queries, query_dim) * 0.02)
        self.q_proj = nn.Linear(query_dim, dim)
        self.in_norm = nn.LayerNorm(latent_dim)
        self.cond = nn.Sequential(nn.Linear(latent_dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU())
        self.film = nn.Linear(dim, 2 * dim * depth)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.blocks = nn.ModuleList([FiLMBlock(dim, hidden) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, out_dim)

    def forward(self, latent):
        B = latent.size(0)
        mods = self.film(self.cond(self.in_norm(latent))).view(B, self.depth, 2, 1, self.dim)
        h = self.q_proj(self.queries).unsqueeze(0).expand(B, -1, -1)
        for i, blk in enumerate(self.blocks):
            h = blk(h, mods[:, i, 0], mods[:, i, 1])
        h = self.out_norm(h)
        with torch.autocast(device_type=h.device.type, enabled=False):
            return self.out(h.float())
