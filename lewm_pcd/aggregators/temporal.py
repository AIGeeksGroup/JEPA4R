"""Default aggregator: LeWM's ARPredictor temporal machinery without the action conditioning.

Learned positional embedding over up to ``max_frames`` frames (positions 0..K-1, exactly like
``module.ARPredictor`` for a K-frame context), causal transformer made of upstream ``module.Block``
(plain pre-norm blocks, no AdaLN), and the last frame's token is the output. Accepts 1..max_frames
frames, so a short context at episode start is handled the way upstream's predictor handles it.
"""

import torch
from torch import nn

from module import Block, Transformer


class TemporalTransformerAggregator(nn.Module):
    def __init__(self, dim=192, max_frames=3, depth=2, heads=16, dim_head=64, mlp_dim=2048, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        self.max_frames = max_frames
        self.pos_embedding = nn.Parameter(torch.randn(1, max_frames, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, dim, dim, depth, heads, dim_head, mlp_dim, dropout, block_class=Block)

    def forward(self, x):
        """x: (B, K, D), K <= max_frames -> (B, D)"""
        K = x.size(1)
        assert 1 <= K <= self.max_frames, f'got {K} frames, max {self.max_frames}'
        x = x + self.pos_embedding[:, :K]
        x = self.transformer(self.dropout(x))  # causal attention inside module.Attention
        return x[:, -1]
