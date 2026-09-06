"""时间聚合 Transformer（新加，PPRL 原仓库没有）。

输入 H 帧 latent1（T-H+1, ..., T，从旧到新），因果自注意力，输出每个位置的 latent2；
位置 k 的 latent2 只依赖 <= k 的帧。latent2 与 latent1 同维。H 由 history_len 指定。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class TemporalBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout)
        )

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        h = self.norm_1(x)
        x = x + self.attention(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.mlp(self.norm_2(x))
        return x


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        history_len: int,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.history_len = history_len
        self.time_embed = nn.Parameter(torch.zeros(1, history_len, embed_dim))
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        self.blocks = nn.ModuleList([TemporalBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, latents: Tensor) -> Tensor:
        """latents: [B, h, D]（h <= history_len，从旧到新）-> latent2 序列 [B, h, D]；取 [:, -1] 即当前帧的 latent2。"""
        B, h, D = latents.shape
        assert 1 <= h <= self.history_len, f"输入帧数 {h} 超出 history_len {self.history_len}"
        assert D == self.embed_dim, f"latent 维度 {D} 与 embed_dim {self.embed_dim} 不符"
        attn_mask = torch.triu(torch.ones(h, h, dtype=torch.bool, device=latents.device), diagonal=1)
        x = latents + self.time_embed[:, :h]
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.norm(x)
