"""预测器：结构与 PPRL 价值头（pprl/models/sac/q_and_pi_heads.py 的 QMlpHead）相同：
输入 latent 拼动作 -> MlpModel；这里 latent 为 latent2，输出维度从 1 改为 latent1 的维度。原 Q 头文件不改。"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from parllel.torch.models import MlpModel
from parllel.torch.utils import infer_leading_dims, restore_leading_dims
from torch import Tensor


class LatentPredictor(nn.Module):
    def __init__(
        self,
        input_size: int,
        action_size: int,
        output_size: int,
        hidden_sizes: int | Sequence[int] | None,
        hidden_nonlinearity: str,
    ):
        super().__init__()
        hidden_nonlinearity: type[nn.Module] = getattr(nn, hidden_nonlinearity)
        self.mlp = MlpModel(
            input_size=input_size + action_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            hidden_nonlinearity=hidden_nonlinearity,
        )

    def forward(self, latent2: Tensor, action: Tensor) -> Tensor:
        lead_dim, T, B, _ = infer_leading_dims(latent2, 1)
        x = torch.cat([latent2.reshape(T * B, -1), action.reshape(T * B, -1)], dim=1)
        pred = self.mlp(x)
        return restore_leading_dims(pred, lead_dim, T, B)
