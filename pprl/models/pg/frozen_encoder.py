"""冻结的 encoder 包装：用于把训练好的世界模型 encoder 原样接进 PPO，权重不更新、BatchNorm 统计量不更新。

- 参数全部 requires_grad=False，forward 在 no_grad 下执行（不建图，省显存）；
- 任何 .train() 调用都会被压回 eval 模式，保证采样与更新时输出一致（世界模型验证也是 eval 模式）；
- embed_dim 透传，PointCloudPgModel 无需改动。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FrozenEncoder(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    @property
    def embed_dim(self) -> int:
        return self.encoder.embed_dim

    def train(self, mode: bool = True) -> "FrozenEncoder":
        super().train(mode)
        self.encoder.eval()  # 无论外层 train/eval，encoder 始终 eval
        return self

    def forward(self, observation):
        with torch.no_grad():
            return self.encoder(observation)
