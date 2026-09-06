"""PPO 用的策略-价值模型：encoder（PPRL 原有，与 SAC / 世界模型完全相同）-> latent；
策略头复用 PPRL 的 PiMlpHead（输出 mean / log_std）；价值头 V(s) 是同结构的 MLP，输入不拼动作、输出 1 维。
返回 parllel GaussianPgAgent 要求的 ModelOutputs(dist_params={mean, log_std}, value)。
BatchNorm 不做任何特殊处理，与原仓库 SAC / 世界模型中的 encoder 行为一致（采样 eval 模式用运行统计量，更新 train 模式用批统计量）。"""

from __future__ import annotations

from typing import Sequence

import torch.nn as nn
from parllel.torch.agents.gaussian import ModelOutputs
from parllel.torch.distributions.gaussian import DistParams
from parllel.torch.models import MlpModel
from parllel.torch.utils import infer_leading_dims, restore_leading_dims
from torch import Tensor

from pprl.models.sac.q_and_pi_heads import PiMlpHead


class PointCloudPgModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        action_size: int,
        action_space,
        pi_hidden_sizes: Sequence[int],
        value_hidden_sizes: Sequence[int],
        hidden_nonlinearity: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        latent_dim = encoder.embed_dim
        self.pi = PiMlpHead(
            input_size=latent_dim,
            action_size=action_size,
            action_space=action_space,
            hidden_sizes=list(pi_hidden_sizes),
            hidden_nonlinearity=hidden_nonlinearity,
        )
        self.value = MlpModel(
            input_size=latent_dim,
            hidden_sizes=list(value_hidden_sizes),
            output_size=1,
            hidden_nonlinearity=getattr(nn, hidden_nonlinearity),
        )

    def forward(self, observation) -> ModelOutputs:
        encoding: Tensor = self.encoder(observation)
        pi_out = self.pi(encoding)
        lead_dim, T, B, _ = infer_leading_dims(encoding, 1)
        value = self.value(encoding.reshape(T * B, -1)).squeeze(-1)
        value = restore_leading_dims(value, lead_dim, T, B)
        return ModelOutputs(
            dist_params=DistParams(mean=pi_out["mean"], log_std=pi_out["log_std"]),
            value=value,
        )
