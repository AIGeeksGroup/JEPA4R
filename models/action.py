"""ActionEncoder：原始动作 [B, M, A_env] -> 通用动作向量 [B, M, D]。

移植自 HMA（仓库 B）``hma/model/st_mask_git.py``：
- ActionStat：按环境登记的均值/方差做归一化（去掉了原版把 stride 折进最后一维的 (s d) 重排，这里最后一维就是该环境的动作维度）；
- BasicMLP：两层 MLP 词干（Linear -> LayerNorm -> ReLU -> Linear）；首层保留 gain=0.01，末层恢复 gain=1.0；
- skip-frame 模式可保留每个原子动作 token，并可在 chunk 内做顺序敏感的 Transformer 编码。
- 按环境名索引的模块字典（每个环境一个词干，主干共享），环境名由前向的 ``env`` 参数指定，新环境只加配置。
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch import nn


@dataclass
class EnvSpec:
    dim: int  # 该环境的动作维度
    mean: List[float] = field(default_factory=list)  # 长度 = dim
    std: List[float] = field(default_factory=list)


@dataclass
class ActionConfig:
    envs: Dict[str, EnvSpec] = field(default_factory=dict)
    output_scale: float = 1.0
    token_encoder_layers: int = 0
    token_encoder_heads: int = 4
    token_mlp_ratio: float = 2.0
    token_position_encoding: str = "learned"
    token_position_scale: float = 1.0


def sinusoidal_position_table(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    table = torch.zeros(length, dim)
    table[:, 0::2] = torch.sin(position * frequency)
    table[:, 1::2] = torch.cos(position * frequency[: table[:, 1::2].shape[1]])
    return table


class ActionStat(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-10)


class BasicMLP(nn.Module):
    def __init__(self, d_action, d_model):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_action, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True),
        )
        self.apply(self._init_weights)
        nn.init.xavier_uniform_(self.model[3].weight, gain=1.0)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class ActionEncoder(nn.Module):
    def __init__(self, cfg: ActionConfig, D: int, chunk_size: int = 1, preserve_tokens: bool = False):
        super().__init__()
        assert len(cfg.envs) > 0, "至少登记一个环境（action.envs）"
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if cfg.output_scale <= 0:
            raise ValueError("action output_scale must be > 0")
        if cfg.token_encoder_layers < 0 or cfg.token_mlp_ratio <= 0:
            raise ValueError("token_encoder_layers must be non-negative and token_mlp_ratio must be positive")
        if cfg.token_position_encoding not in {"learned", "sinusoidal"}:
            raise ValueError("token_position_encoding must be learned or sinusoidal")
        if cfg.token_position_scale <= 0:
            raise ValueError("token_position_scale must be positive")
        self.chunk_size = int(chunk_size)
        self.preserve_tokens = bool(preserve_tokens)
        self.output_scale = float(cfg.output_scale)
        self.token_position_scale = float(cfg.token_position_scale)
        self.dims = {}
        self.stat, self.mlp = nn.ModuleDict(), nn.ModuleDict()
        for env, spec in cfg.envs.items():
            assert len(spec.mean) == len(spec.std) == spec.dim, f"环境 {env} 的 mean/std 长度应等于 dim={spec.dim}"
            self.dims[env] = spec.dim
            self.stat[env] = ActionStat(spec.mean, spec.std)
            self.mlp[env] = BasicMLP(spec.dim, D)
        self.chunk_gru = nn.GRU(D, D, batch_first=True) if self.chunk_size > 1 and not self.preserve_tokens else None
        if self.preserve_tokens:
            if cfg.token_position_encoding == "learned":
                self.chunk_position = nn.Parameter(torch.empty(1, self.chunk_size, D))
                nn.init.normal_(self.chunk_position, std=0.02)
            else:
                self.register_buffer(
                    "chunk_position", sinusoidal_position_table(self.chunk_size, D).unsqueeze(0)
                )
        else:
            self.register_parameter("chunk_position", None)
        if self.preserve_tokens and cfg.token_encoder_layers > 0:
            if D % cfg.token_encoder_heads:
                raise ValueError(f"D={D} must be divisible by token_encoder_heads={cfg.token_encoder_heads}")
            layer = nn.TransformerEncoderLayer(
                d_model=D,
                nhead=cfg.token_encoder_heads,
                dim_feedforward=int(D * cfg.token_mlp_ratio),
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.token_encoder = nn.TransformerEncoder(
                layer,
                num_layers=cfg.token_encoder_layers,
                norm=nn.LayerNorm(D),
                enable_nested_tensor=False,
            )
        else:
            self.token_encoder = None

    def forward(self, actions: torch.Tensor, env: str) -> torch.Tensor:
        assert env in self.mlp, f"未登记的环境 {env!r}，已登记：{list(self.mlp)}"
        assert actions.dim() == 3 and actions.shape[-1] == self.dims[env], \
            f"actions 应为 [B, M, {self.dims[env]}]，得到 {tuple(actions.shape)}"
        out = self.mlp[env](self.stat[env](actions.float()))
        assert out.shape[:2] == actions.shape[:2]
        return out

    def encode_chunks(self, actions: torch.Tensor, env: str) -> torch.Tensor:
        """Encode ordered atomic actions into one embedding per action chunk."""
        assert actions.dim() == 3 and actions.shape[1] % self.chunk_size == 0, (
            f"actions 应为 [B, M, A] 且 M 可被 chunk_size={self.chunk_size} 整除，得到 {tuple(actions.shape)}"
        )
        atom = self(actions, env)
        if self.chunk_size == 1:
            return atom * self.output_scale
        B, M, D = atom.shape
        chunks = atom.view(B, M // self.chunk_size, self.chunk_size, D)
        sequence = chunks.reshape(B * (M // self.chunk_size), self.chunk_size, D)
        _, hidden = self.chunk_gru(sequence)
        return hidden[-1].view(B, M // self.chunk_size, D) * self.output_scale

    def encode_tokens(self, actions: torch.Tensor, env: str) -> torch.Tensor:
        """Encode every ordered atomic action as [B, num_chunks, chunk_size, D]."""
        if not self.preserve_tokens:
            raise RuntimeError("encode_tokens requires preserve_tokens=True")
        assert actions.dim() == 3 and actions.shape[1] % self.chunk_size == 0, (
            f"actions 应为 [B, M, A] 且 M 可被 chunk_size={self.chunk_size} 整除，得到 {tuple(actions.shape)}"
        )
        atom = self(actions, env)
        B, M, D = atom.shape
        chunks = atom.view(B, M // self.chunk_size, self.chunk_size, D)
        tokens = chunks + self.token_position_scale * self.chunk_position[:, None]
        if self.token_encoder is not None:
            sequence = tokens.reshape(B * (M // self.chunk_size), self.chunk_size, D)
            tokens = self.token_encoder(sequence).view(B, M // self.chunk_size, self.chunk_size, D)
        return tokens.mul(self.output_scale)
