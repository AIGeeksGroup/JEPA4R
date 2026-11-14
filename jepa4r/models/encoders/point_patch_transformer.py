"""PointPatchTransformer of PPRL [gyenes2024pprl] for the Tab. 4b ablation, sized to ~5.51M parameters.

Original module docstring:
PointPatchTransformer copied from PPRL / PointPatchRL (balazsgyenes/pprl, ``conf/model/ppt.yaml``),
re-implemented in plain PyTorch (batched, fixed 1200 points) so it needs neither
parllel nor torch_geometric / torch_cluster.

Defaults are PPRL's:
  tokenizer  : furthest-point sampling (ratio 0.0625 -> 75 centres for 1200 points,
               random start while training) + kNN groups of 32; per group
               mlp_1 = MLP[C, 128, 256] on (pos_j - centre, colour_j), max over the
               group concatenated back to every point, mlp_2 = MLP[512, 512, E],
               max over the group -> token. MLPs follow torch_geometric.nn.MLP:
               Linear -> BatchNorm1d -> ReLU between layers, plain last layer.
  pos embed  : sinusoidal on the centre xyz (temperature 1e4, scale 1)
  encoder    : depth 3 blocks of  x = LN(x); x = x + MHA(x); x = LN(x); x = x + MLP(x)
               with MLP[E, 3E, E] (GELU, no norm), 6 heads, pos added before every block
  pooling    : SequencePooling = LN -> softmax(Linear(E, 1)) attention -> weighted sum
  init       : trunc_normal(0.02) on Linear weights, LN weight 1 / bias 0

Adaptations for JEPA4R: an affine head Linear(E -> out_dim) (192-d latent), and
E = 402 instead of 384 so the parameter count matches the ViT-tiny encoder (5.50 M).
The state vector PPRL concatenates after pooling is not part of this module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _pyg_mlp(channels: Sequence[int], act: str = "relu", norm: bool = True, dropout: float = 0.0) -> nn.Sequential:
    """torch_geometric.nn.MLP(channel_list, act, norm, plain_last=True) equivalent."""
    acts = {"relu": nn.ReLU, "gelu": nn.GELU}
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Linear(channels[i], channels[i + 1]))
        if i < len(channels) - 2:  # plain last layer
            if norm:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(acts[act]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def farthest_point_sampling(xyz: torch.Tensor, num: int, random_start: bool) -> torch.Tensor:
    """xyz (B, N, 3) -> indices (B, num), same greedy rule as torch_cluster.fps."""
    B, N, _ = xyz.shape
    idx = torch.zeros(B, num, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), float("inf"), device=xyz.device)
    far = (
        torch.randint(0, N, (B,), device=xyz.device)
        if random_start
        else torch.zeros(B, dtype=torch.long, device=xyz.device)
    )
    ar = torch.arange(B, device=xyz.device)
    for i in range(num):
        idx[:, i] = far
        d = ((xyz - xyz[ar, far][:, None]) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(-1)
    return idx


class SinusoidalPosEmbedder(nn.Module):
    def __init__(self, token_dim: int, n_dim: int = 3, temperature: float = 10000.0, scale: float = 1.0):
        super().__init__()
        self.n_dim = n_dim
        self.num_pos_features = token_dim // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = token_dim - self.num_pos_features * n_dim
        self.scale = scale * 2 * math.pi
        self.token_dim = token_dim

    def forward(self, xyz):  # (B, G, 3) -> (B, G, token_dim)
        B, G, _ = xyz.shape
        dim_t = torch.pow(
            self.temperature, 2 * torch.arange(0, self.num_pos_features // 2, device=xyz.device) / self.num_pos_features
        )
        pos_divided = (xyz * self.scale).unsqueeze(-1) / dim_t
        pos_emb = torch.zeros(B, G, self.token_dim - self.padding, dtype=torch.float32, device=xyz.device)
        pos_emb[..., 0::2] = pos_divided.sin().reshape(B, G, -1)
        pos_emb[..., 1::2] = pos_divided.cos().reshape(B, G, -1)
        return F.pad(pos_emb, (0, self.padding))


class Tokenizer(nn.Module):
    def __init__(
        self,
        point_dim,
        embed_dim,
        mlp_1=(128, 256),
        mlp_2=(512, 512),
        group_size=32,
        sampling_ratio=0.0625,
        random_start=True,
    ):
        super().__init__()
        self.mlp_1 = _pyg_mlp([point_dim, *mlp_1], act="relu")
        self.mlp_2 = _pyg_mlp([*mlp_2, embed_dim], act="relu")
        assert mlp_1[-1] * 2 == mlp_2[0], "last layer of mlp_1 must be half the first layer of mlp_2"
        self.group_size, self.sampling_ratio, self.random_start = group_size, sampling_ratio, random_start

    def forward(self, pos, color=None):  # pos (B, N, 3), color (B, N, C-3) -> tokens (B, G, E), centres (B, G, 3)
        B, N, _ = pos.shape
        G = math.ceil(N * self.sampling_ratio)
        centre_idx = farthest_point_sampling(pos, G, self.random_start and self.training)
        ar = torch.arange(B, device=pos.device)[:, None]
        centres = pos[ar, centre_idx]  # (B, G, 3)
        nn_idx = torch.cdist(centres, pos).topk(self.group_size, dim=-1, largest=False).indices  # (B, G, M)
        arg = torch.arange(B, device=pos.device)[:, None, None]
        neighborhood = pos[arg, nn_idx] - centres[:, :, None]  # (B, G, M, 3), local coords
        if color is not None:
            neighborhood = torch.cat([neighborhood, color[arg, nn_idx]], dim=-1)
        M = self.group_size
        msg = self.mlp_1(neighborhood.reshape(B * G * M, -1)).reshape(B * G, M, -1)
        msg = torch.cat([msg.max(dim=1, keepdim=True).values.expand(-1, M, -1), msg], dim=2)
        msg = self.mlp_2(msg.reshape(B * G * M, -1)).reshape(B, G, M, -1)
        return msg.max(dim=2).values, centres


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=3, attn_dropout=0.0, mlp_dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, bias=True, batch_first=True)
        self.mlp = _pyg_mlp(
            [embed_dim, int(mlp_ratio * embed_dim), embed_dim], act="gelu", norm=False, dropout=mlp_dropout
        )
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm_1(x)
        x = x + self.attention(x, x, x, need_weights=False)[0]
        x = self.norm_2(x)
        x = x + self.mlp(x)
        return x


class SequencePooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attention_pool = nn.Linear(embed_dim, 1)

    def forward(self, x):  # (B, G, E) -> (B, E)
        x = self.norm(x)
        weights = F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2)
        return torch.matmul(weights, x).squeeze(-2)


class PointPatchTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int = 6,
        embed_dim: int = 402,
        depth: int = 3,
        num_heads: int = 6,
        mlp_ratio: float = 3,
        group_size: int = 32,
        sampling_ratio: float = 0.0625,
        random_start: bool = True,
        tokenizer_mlp_1: Sequence[int] = (128, 256),
        tokenizer_mlp_2: Sequence[int] = (512, 512),
        pos_temperature: float = 10000.0,
        pos_scale: float = 1.0,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        out_dim: int = 192,
    ):
        super().__init__()
        self.tokenizer = Tokenizer(
            in_dim, embed_dim, tuple(tokenizer_mlp_1), tuple(tokenizer_mlp_2), group_size, sampling_ratio, random_start
        )
        self.pos_embedder = SinusoidalPosEmbedder(embed_dim, 3, pos_temperature, pos_scale)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, attn_dropout, mlp_dropout) for _ in range(depth)]
        )
        self.pooling = SequencePooling(embed_dim)
        self.apply(self._init_weights)
        self.head = nn.Linear(embed_dim, out_dim)  # adaptation to the 192-d latent

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """x: (B, N, C) with xyz first -> (B, out_dim)"""
        x = x.float()
        pos, color = x[..., :3], (x[..., 3:] if x.shape[-1] > 3 else None)
        tokens, centres = self.tokenizer(pos, color)
        pe = self.pos_embedder(centres)
        for block in self.blocks:
            tokens = block(tokens + pe)
        return self.head(self.pooling(tokens))


def build(enc_cfg):
    """configs/encoder/point_patch_transformer.yaml (hparams = JEPA4R's PPT encoder config)."""
    h = enc_cfg.hparams
    return PointPatchTransformer(
        in_dim=int(enc_cfg.in_dim),
        embed_dim=int(h.embed_dim),
        depth=int(h.depth),
        num_heads=int(h.num_heads),
        mlp_ratio=h.mlp_ratio,
        group_size=int(h.group_size),
        sampling_ratio=float(h.sampling_ratio),
        random_start=bool(h.random_start),
        out_dim=int(enc_cfg.out_dim),
    )
