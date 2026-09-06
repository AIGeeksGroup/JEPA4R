"""TemporalEncoder：latent1 [B, T, K, D] -> latent2 [B, K, D]。

三个参考仓库都不涉及这一部分，按文档取最简单的实现：
- 位置编码按排序索引 k、时间编码按帧索引 t，各自可选可学习（nn.Parameter）或固定正余弦；
- 主干直接用 torch.nn.TransformerEncoder（pre-norm，GELU），全注意力（跨帧、跨点）；
- 因果掩码按帧划分（第 t 帧的 K 个 token 只能看到帧 <= t 的 token），默认关；
- 读出 last_frame 取最后一帧的 K 个向量；query 用 K 个可学习查询做一层 nn.MultiheadAttention 交叉注意力。
"""

import math
from dataclasses import dataclass

import torch
from torch import nn


def sinusoidal_table(n: int, dim: int) -> torch.Tensor:
    """标准正余弦编码表 [n, dim]。"""
    position = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    table = torch.zeros(n, dim)
    table[:, 0::2] = torch.sin(position * div)
    table[:, 1::2] = torch.cos(position * div[: dim // 2])
    return table


class _Encoding(nn.Module):
    """长度为 n 的索引编码表：learned -> nn.Parameter；sinusoidal -> 固定 buffer。"""

    def __init__(self, kind: str, n: int, dim: int):
        super().__init__()
        assert kind in ("learned", "sinusoidal"), f"编码方式应为 learned | sinusoidal，得到 {kind}"
        if kind == "learned":
            self.table = nn.Parameter(torch.zeros(n, dim))
            nn.init.trunc_normal_(self.table, std=0.02)
        else:
            self.register_buffer("table", sinusoidal_table(n, dim))

    def forward(self) -> torch.Tensor:
        return self.table


@dataclass
class TemporalConfig:
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    causal: bool = False  # 按帧的因果掩码
    pos_encoding: str = "learned"  # learned | sinusoidal（按排序索引 k）
    time_encoding: str = "learned"  # learned | sinusoidal（按帧索引 t）
    readout: str = "last_frame"  # last_frame | query


class TemporalEncoder(nn.Module):
    def __init__(self, cfg: TemporalConfig, K: int, T: int, D: int):
        super().__init__()
        assert D % cfg.num_heads == 0 and cfg.readout in ("last_frame", "query")
        self.cfg, self.K, self.T, self.D = cfg, K, T, D
        self.pos = _Encoding(cfg.pos_encoding, K, D)
        self.time = _Encoding(cfg.time_encoding, T, D)
        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=cfg.num_heads, dim_feedforward=int(D * cfg.mlp_ratio), dropout=cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers, enable_nested_tensor=False)
        if cfg.readout == "query":
            self.query = nn.Parameter(torch.zeros(K, D))
            nn.init.trunc_normal_(self.query, std=0.02)
            self.readout_attn = nn.MultiheadAttention(D, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
        frame = torch.arange(T).repeat_interleave(K)  # 每个展平 token 所属的帧
        self.register_buffer("causal_mask", frame[None, :] > frame[:, None], persistent=False)  # True = 不可见

    def forward(self, latent1: torch.Tensor, *, return_sequence: bool = False, causal: bool | None = None) -> torch.Tensor:
        assert latent1.dim() == 4 and latent1.shape[1:] == (self.T, self.K, self.D), \
            f"latent1 应为 [B, {self.T}, {self.K}, {self.D}]，得到 {tuple(latent1.shape)}"
        B = latent1.shape[0]
        x = latent1 + self.pos()[None, None] + self.time()[None, :, None]
        x = x.reshape(B, self.T * self.K, self.D)
        use_causal = self.cfg.causal if causal is None else causal
        x = self.encoder(x, mask=self.causal_mask if use_causal else None)
        encoded = x.view(B, self.T, self.K, self.D)
        if return_sequence:
            return encoded
        if self.cfg.readout == "last_frame":
            latent2 = encoded[:, -1]
        else:
            latent2, _ = self.readout_attn(self.query.expand(B, -1, -1), x, x, need_weights=False)
        assert latent2.shape == (B, self.K, self.D)
        return latent2
