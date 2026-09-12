"""Predictor：latent2 [B, K, D] + 动作向量 [B, 1, D] -> 下一帧 latent1 的点估计 [B, K, D]。

移植自 HMA（仓库 B）：
- ``hma/model/attention.py`` BasicSelfAttention：去掉顶层对 xformers 的硬依赖，内部换成 torch 原生缩放点积注意力；
- ``hma/model/st_transformer.py`` Mlp / STBlock / STTransformerDecoder：解码器整体照搬（零 dropout 默认、xavier gain=0.1 初始化），
  latent2 视作时间长度 1、空间长度 K 的输入；时间长度为 1 时时间注意力退化为残差线性层，保留原样；
- ``hma/model/st_mask_git.py`` ModulateLayer：动作调制融合（HMA 的 "modulate"）。

改动：muP 缩放开关显式关掉（use_mup=False）；提供按 batch/time 对齐的逐层动作 cross-attention；
输出头换成一个线性回归层（原版是离散 token 分类或扩散头，没有均值路径可取）。
"""

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

FUSIONS = ("add", "modulate", "cross_attention")


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class BasicSelfAttention(nn.Module):
    def __init__(self, num_heads, d_model, qkv_bias=False, proj_bias=True, qk_norm=True, use_mup=False, attn_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # Scaling by 8 to be equal when head_dim=64（muP）；关掉 muP 时就是标准的 1/sqrt(head_dim)
        self.scale = 8 / self.head_dim if use_mup else self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.qk_norm = qk_norm
        if self.qk_norm:
            self.norm = nn.LayerNorm(self.head_dim, eps=1e-05)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = self.norm(q).to(dtype=v.dtype)
            k = self.norm(k).to(dtype=v.dtype)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0, is_causal=causal, scale=self.scale
        )
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, d_model, mlp_ratio=4.0, mlp_bias=True, mlp_drop=0.0):
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=mlp_bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model, bias=mlp_bias)
        self.drop = nn.Dropout(mlp_drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class ModulateLayer(nn.Module):
    """DiT 风格的逐 token 调制（HMA ModulateLayer）。"""

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(out_channels, elementwise_affine=False, eps=1e-6)
        self.linear_out = nn.Linear(out_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(model_channels, model_channels), nn.SiLU(), nn.Linear(model_channels, 2 * out_channels, bias=True)
        )
        self.apply(self._init_weights)

    def forward(self, x, c):
        """x: (b s) t d；c: b t d。"""
        x_shape = x.shape
        x = rearrange(x, "(b s) t d -> b s t d", b=len(c))
        c = c[:, None, : x_shape[1]]
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.linear_out(modulate(self.norm_final(x), shift, scale))
        return x.view(x_shape)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)


class STBlock(nn.Module):
    # See Figure 4 of https://arxiv.org/pdf/2402.15391.pdf
    def __init__(self, num_heads, d_model, qkv_bias, proj_bias, qk_norm, use_mup, attn_drop, mlp_ratio, mlp_bias, mlp_drop, action_processing):
        super().__init__()
        assert action_processing in FUSIONS
        self.norm1 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        attn_kw = dict(num_heads=num_heads, d_model=d_model, qkv_bias=qkv_bias, proj_bias=proj_bias, qk_norm=qk_norm, use_mup=use_mup, attn_drop=attn_drop)
        self.spatial_attn = BasicSelfAttention(**attn_kw)  # sequence dim is over each frame's spatial tokens
        self.temporal_attn = BasicSelfAttention(**attn_kw)  # sequence dim is over time sequence
        self.action_processing = action_processing
        if action_processing == "cross_attention":
            self.action_norm = nn.LayerNorm(d_model, eps=1e-05)
            self.action_cross_attn = nn.MultiheadAttention(
                d_model, num_heads, dropout=attn_drop, batch_first=True,
            )
        else:
            self.action_norm = None
            self.action_cross_attn = None
        self.norm2 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        self.mlp = Mlp(d_model=d_model, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        self.action_projectors = None  # 按环境名索引的 ModuleDict，由 Predictor 挂上

    def forward(
        self,
        x_TSC: torch.Tensor,
        action_ids: torch.Tensor = None,
        domain: str = None,
        robot_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        T, S = x_TSC.size(1), x_TSC.size(2)
        x_SC = rearrange(x_TSC, "B T S C -> (B T) S C")
        x_SC = x_SC + self.spatial_attn(self.norm1(x_SC))

        if self.action_processing == "cross_attention" and action_ids is not None:
            assert action_ids.dim() == 4 and action_ids.shape[:2] == x_TSC.shape[:2]
            action_SC = rearrange(action_ids, "B T A C -> (B T) A C")
            condition = action_SC
            if robot_tokens is not None:
                assert robot_tokens.shape == (
                    x_TSC.shape[0],
                    T,
                    x_TSC.shape[-1],
                ), (
                    "robot_tokens must be [B,T,D], got "
                    f"{tuple(robot_tokens.shape)}"
                )
                robot_SC = rearrange(robot_tokens, "B T C -> (B T) 1 C")
                condition = torch.cat([robot_SC, action_SC], dim=1)
            query = self.action_norm(x_SC)
            key_value = self.action_norm(condition)
            action_update, _ = self.action_cross_attn(query, key_value, key_value, need_weights=False)
            x_SC = x_SC + action_update

        x_TC = rearrange(x_SC, "(B T) S C -> (B S) T C", T=T)
        if self.action_processing != "cross_attention" and action_ids is not None and domain is not None and self.action_projectors is not None:
            if self.action_processing == "add":  # action_ids: [B, T, D]，加到该帧全部空间 token 上
                action_ids = self.action_projectors[domain](action_ids)
                x_TC = rearrange(x_TC, "(B S) T C -> B S T C", S=S)
                x_TC = x_TC + action_ids[:, None, : x_TC.shape[2]]
                x_TC = rearrange(x_TC, "B S T C -> (B S) T C", S=S)
            else:  # modulate
                x_TC = x_TC + self.action_projectors[domain](x_TC, action_ids)

        x_TC = x_TC + self.temporal_attn(x_TC, causal=True)
        x_TC = x_TC + self.mlp(self.norm2(x_TC))
        return rearrange(x_TC, "(B S) T C -> B T S C", S=S)


class STTransformerDecoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, qkv_bias=False, proj_bias=True, qk_norm=True, use_mup=False,
                 attn_drop=0.0, mlp_ratio=4.0, mlp_bias=True, mlp_drop=0.0, action_processing="add"):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                STBlock(num_heads, d_model, qkv_bias, proj_bias, qk_norm, use_mup, attn_drop, mlp_ratio, mlp_bias, mlp_drop, action_processing)
                for _ in range(num_layers)
            ]
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, tgt, action_ids=None, domain="", robot_tokens=None):
        x = tgt
        for layer in self.layers:
            x = layer(
                x,
                action_ids=action_ids,
                domain=domain,
                robot_tokens=robot_tokens,
            )
        return x


@dataclass
class PredictorConfig:
    num_layers: int = 2
    num_heads: int = 4
    fusion: str = "add"  # add（HMA 默认）| modulate | cross_attention
    qkv_bias: bool = False
    proj_bias: bool = True
    qk_norm: bool = True
    attn_drop: float = 0.0
    mlp_ratio: float = 4.0
    mlp_bias: bool = True
    mlp_drop: float = 0.0


class Predictor(nn.Module):
    def __init__(self, cfg: PredictorConfig, D: int, envs: List[str]):
        super().__init__()
        assert cfg.fusion in FUSIONS and D % cfg.num_heads == 0
        self.D = D
        self.decoder = STTransformerDecoder(
            num_layers=cfg.num_layers, num_heads=cfg.num_heads, d_model=D, qkv_bias=cfg.qkv_bias, proj_bias=cfg.proj_bias,
            qk_norm=cfg.qk_norm, use_mup=False, attn_drop=cfg.attn_drop, mlp_ratio=cfg.mlp_ratio, mlp_bias=cfg.mlp_bias,
            mlp_drop=cfg.mlp_drop, action_processing=cfg.fusion,
        )
        if cfg.fusion != "cross_attention":
            for layer in self.decoder.layers:  # HMA：每层、每个环境各一份融合模块
                layer.action_projectors = nn.ModuleDict(
                    {env: (nn.Identity() if cfg.fusion == "add" else ModulateLayer(D, D)) for env in envs}
                )
        self.head = nn.Linear(D, D)  # 线性回归头（替换 HMA 的分类 / 扩散头）

    def forward(
        self,
        latent2: torch.Tensor,
        action: torch.Tensor,
        env: str,
        *,
        robot_tokens: torch.Tensor | None = None,
        return_features: bool = False,
    ):
        assert latent2.shape[-1] == self.D, f"latent 输入最后一维应为 {self.D}，得到 {tuple(latent2.shape)}"
        if robot_tokens is not None and self.decoder.layers[0].action_processing != "cross_attention":
            raise ValueError("robot tokens require predictor fusion='cross_attention'")
        if latent2.dim() == 3:
            if self.decoder.layers[0].action_processing == "cross_attention":
                assert action.dim() == 4 and action.shape[:2] == (latent2.shape[0], 1) and action.shape[-1] == self.D, (
                    f"单帧 token action 应为 [B, 1, S, {self.D}]，得到 {tuple(action.shape)}"
                )
            else:
                assert action.shape == (latent2.shape[0], 1, self.D), (
                    f"单帧 action 应为 [B, 1, {self.D}]，得到 {tuple(action.shape)}"
                )
            if robot_tokens is not None:
                assert robot_tokens.shape == (latent2.shape[0], 1, self.D)
            x = self.decoder(
                latent2.unsqueeze(1),
                action_ids=action,
                domain=env,
                robot_tokens=robot_tokens,
            )  # [B, 1, K, D]
            features = x[:, 0]
            pred = self.head(features)
            assert pred.shape == latent2.shape
            return (pred, features) if return_features else pred

        assert latent2.dim() == 4, f"latent 应为 [B, K, D] 或 [B, T, K, D]，得到 {tuple(latent2.shape)}"
        B, T, K, _ = latent2.shape
        if self.decoder.layers[0].action_processing == "cross_attention":
            assert action.dim() == 4 and action.shape[:2] == (B, T) and action.shape[-1] == self.D, (
                f"序列 token action 应为 [B, T, S, {self.D}]，得到 {tuple(action.shape)}"
            )
        else:
            assert action.shape == (B, T, self.D), (
                f"序列 action 应为 [B, T, {self.D}]，得到 {tuple(action.shape)}"
            )
        if robot_tokens is not None:
            assert robot_tokens.shape == (B, T, self.D), (
                f"robot_tokens 应为 [B, T, {self.D}]，得到 {tuple(robot_tokens.shape)}"
            )
        x = self.decoder(
            latent2,
            action_ids=action,
            domain=env,
            robot_tokens=robot_tokens,
        )  # [B, T, K, D]
        pred = self.head(x)
        assert pred.shape == latent2.shape
        return (pred, x) if return_features else pred
