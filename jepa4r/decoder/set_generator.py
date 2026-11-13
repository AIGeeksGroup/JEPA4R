"""Set generator (new in JEPA4R, Sec. 4.4): replaces the backbone's image input.

256 learnable queries -> 3 residual MLP blocks modulated by z_t through FiLM [perez2018film]
-> 16 x 16 grid of 1,024-d tokens (cfg.decoder.set_generator).

``SetDecoder`` is the recon branch's set_decoder.py moved unchanged; every backbone decoder (jepa4r/decoder/backbones)
builds it with num_queries = grid^2 and out_dim = the backbone's token width.
"""
import torch
import torch.nn.functional as F
from torch import nn


# Moved from lewm-pcd-recon-page4d@d3c2d07:lewm_pcd/recon/decoders/set_decoder.py:FiLMBlock, SetDecoder (lines 8-50;
# verbatim)
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


def build_set_generator(decoder_cfg):
    """SetDecoder with cfg.decoder.set_generator, as the backbone decoders build it (PAGE4DDecoder.__init__)."""
    sg = decoder_cfg.set_generator
    return SetDecoder(
        latent_dim=int(decoder_cfg.latent_dim),
        num_queries=int(sg.num_queries),
        out_dim=int(sg.token_dim),
        dim=int(sg.dim),
        hidden=int(sg.hidden),
        depth=int(sg.num_blocks),
        query_dim=int(sg.query_dim),
    )
