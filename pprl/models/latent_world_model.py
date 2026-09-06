"""潜空间世界模型。

encoder（PPRL 原有）-> latent1
时间聚合 Transformer（新加）-> latent2（同维）
预测器（PPRL Q 头结构，输出 latent1 维）：(latent2, 动作块) -> 下一帧 latent1

训练损失照 LeWM 官方 train.py 的 lejepa_forward：
    emb = (B, T, D)，T = history_size + num_preds
    ctx = emb[:, :history_size]，pred = predict(ctx)（每个历史位置各预测一次），tgt = emb[:, num_preds:]
    pred_loss = (pred - tgt).pow(2).mean()          # 目标不 detach
    sigreg_loss = SIGReg(emb.transpose(0, 1))        # (T, B, D)
    loss = pred_loss + lambda * sigreg_loss
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from pprl.models.modules.latent_predictor import LatentPredictor
from pprl.models.modules.lewm_sigreg import SIGReg
from pprl.models.modules.temporal_transformer import TemporalTransformer


class LatentWorldModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        history_size: int = 3,
        num_preds: int = 1,
        temporal_depth: int = 2,
        temporal_num_heads: int = 8,
        temporal_mlp_ratio: float = 4.0,
        temporal_dropout: float = 0.0,
        predictor_hidden_sizes: Sequence[int] = (1024, 1024, 1024),
        predictor_nonlinearity: str = "ELU",
        sigreg_weight: float = 0.09,
        sigreg_knots: int = 17,
        sigreg_num_proj: int = 1024,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.history_size = history_size
        self.num_preds = num_preds
        self.sigreg_weight = sigreg_weight
        latent_dim = encoder.embed_dim
        self.temporal = TemporalTransformer(
            embed_dim=latent_dim, history_len=history_size, depth=temporal_depth,
            num_heads=temporal_num_heads, mlp_ratio=temporal_mlp_ratio, dropout=temporal_dropout,
        )
        self.predictor = LatentPredictor(
            input_size=latent_dim, action_size=action_dim, output_size=latent_dim,
            hidden_sizes=list(predictor_hidden_sizes), hidden_nonlinearity=predictor_nonlinearity,
        )
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

    @property
    def latent_dim(self) -> int:
        return self.encoder.embed_dim

    @property
    def num_steps(self) -> int:
        """每个样本的帧数 = history_size + num_preds（同官方 dataset.num_steps）"""
        return self.history_size + self.num_preds

    def encode(self, observation) -> Tensor:
        return self.encoder(observation)

    def predict(self, emb: Tensor, action: Tensor) -> Tensor:
        """同官方 JEPA.predict：emb (B, h, D) 历史 latent1，action (B, h, A) 对应动作块 -> 每个位置预测的下一帧 latent1 (B, h, D)"""
        latent2 = self.temporal(emb)
        return self.predictor(latent2, action)

    def forward(self, observation, action: Tensor) -> dict[str, Tensor]:
        """
        :param observation: 组批观测，样本数 B*T，样本 b 的 T 帧连续且从旧到新
        :param action: (B, T, A) 每帧对应的动作块（同官方，最后一帧的动作块不参与预测）
        """
        B, T, _ = action.shape
        assert T == self.num_steps, f"每样本帧数 {T} 应为 history_size + num_preds = {self.num_steps}"
        emb = self.encode(observation).view(B, T, -1)
        ctx_emb, ctx_act = emb[:, : self.history_size], action[:, : self.history_size]
        tgt_emb = emb[:, self.num_preds :]
        pred_emb = self.predict(ctx_emb, ctx_act)

        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        sigreg_loss = self.sigreg(emb.transpose(0, 1))
        loss = pred_loss + self.sigreg_weight * sigreg_loss
        return {"loss": loss, "pred_loss": pred_loss, "sigreg_loss": sigreg_loss, "emb": emb, "pred_emb": pred_emb}

    @torch.no_grad()
    def rollout(self, emb: Tensor, action: Tensor) -> Tensor:
        """同官方 JEPA.rollout：emb (N, h0, D) 初始 latent1 历史，action (N, T, A) 动作块序列（前 h0 块对应历史帧）
        -> 自回归预测后的 latent1 序列 (N, T+1, D)，最后一帧为执行完全部动作块后的预测。"""
        HS = self.history_size
        h0 = emb.size(1)
        n_steps = action.size(1) - h0
        for t in range(n_steps + 1):
            act = action[:, : h0 + t]
            pred = self.predict(emb[:, -HS:], act[:, -HS:])[:, -1:]
            emb = torch.cat([emb, pred], dim=1)
        return emb
