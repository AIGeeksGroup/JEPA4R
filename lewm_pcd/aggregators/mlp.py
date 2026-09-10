"""concat(K frames) -> MLP -> latent2. Contexts shorter than max_frames are left-padded by
repeating the oldest frame (as if the world had been stationary before)."""

import torch
from torch import nn


class MLPAggregator(nn.Module):
    def __init__(self, dim=192, max_frames=3, hidden_dim=2048):
        super().__init__()
        self.max_frames = max_frames
        self.net = nn.Sequential(nn.Linear(max_frames * dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x):
        K = x.size(1)
        if K < self.max_frames:
            x = torch.cat([x[:, :1].expand(-1, self.max_frames - K, -1), x], dim=1)
        return self.net(x.flatten(1))
