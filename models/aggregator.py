"""SpatialAggregator：单帧点云 [B, K, C] -> 逐点 token [B, K, D] + 重排后的坐标 [B, K, 3]。

骨干移植自 Pointcept（仓库 A）的 Point Transformer V3 v3m2（Sonata 所用版本）：
- ``pointcept/models/point_transformer_v3/point_transformer_v3m2_sonata.py``：
  SerializedAttention / Block / GridPooling / GridUnpooling / Embedding / PointTransformerV3；
- ``pointcept/models/utils/structure.py``：Point（serialization、sparsify）；
- ``pointcept/models/modules.py``：PointModule / PointSequential；
- ``pointcept/models/utils/misc.py``：offset/batch 互转。

相对 v3m2 的改动（README「改动栏」同步记录）：
1. 只保留普通注意力分支，去掉 flash-attn 分支；分块大小 = min(批内最少点数, patch_size)，
   因此同一样本的输出会轻微依赖同批其他样本（无法避免）；
2. 序列化顺序的随机打乱开关 ``shuffle_orders`` 补传到每一级 GridPooling（原版只在顶层，下采样层固定为 True）；
3. 序列化深度固定为 OrderingConfig.depth（原版按样本自适应）；
4. 进网前按（样本索引, 格点坐标）体素去重，同体素的点取坐标和特征的均值（本仓库自己的选择），
   骨干只跑唯一体素，输出按逆映射散回 K 个点；
5. 删掉与本任务无关的分支：RPE、LayerScale、mask_token、enc_mode、freeze_encoder、traceable 上采样、ocnn；
6. torch_scatter 的 segment_csr / scatter_mean 用 torch 原生算子等价实现（见下方两个函数）。
"""

from dataclasses import dataclass, field
from typing import List

import spconv.pytorch as spconv
import torch
from addict import Dict
from timm.layers import DropPath
from torch import nn
from torch.nn.init import trunc_normal_

from models.ordering import Ordering, OrderingConfig, encode
from models.sonata_aggregator import SonataAggregatorConfig


# --------------------------------------------------------------------------- Point 结构
@torch.no_grad()
def offset2bincount(offset):
    return torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long))


@torch.no_grad()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(bincount)


@torch.no_grad()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


# PTv3 原版用 torch_scatter.segment_csr / scatter_mean；这里用 torch 原生算子等价实现，不依赖 torch-scatter
# （2026-09 起 PyG 的预编译轮子托管 data.pyg.org 已被禁用，无法一条命令安装）。
def segment_csr(src, indptr, reduce):
    """按 CSR 段规约：src [N, C]，indptr [V+1]（段非空）-> [V, C]。"""
    return torch.segment_reduce(src, reduce, offsets=indptr, axis=0)


def scatter_mean(src, index, dim_size):
    """按 index 分组求均值：src [N, C]，index [N] -> [dim_size, C]。"""
    out = src.new_zeros(dim_size, src.shape[1]).index_add_(0, index, src)
    return out / torch.bincount(index, minlength=dim_size).clamp_(min=1).to(src.dtype)[:, None]


class Point(Dict):
    """批量点云字典：coord / feat / batch(offset) / grid_coord / grid_size + 序列化与稀疏张量缓存。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order, depth, shuffle_orders=False):
        """按每种曲线编码并排序；格点坐标由 Ordering 事先算好塞进字典，这里不再重算。"""
        assert {"grid_coord", "batch"}.issubset(self.keys())
        assert depth <= 16 and depth * 3 + len(self.offset).bit_length() <= 63
        self["order"] = order
        self["serialized_depth"] = depth
        code = torch.stack([encode(self.grid_coord, self.batch, depth, order=o) for o in order])
        order_ = torch.argsort(code)
        inverse = torch.zeros_like(order_).scatter_(
            dim=1, index=order_, src=torch.arange(0, code.shape[1], device=order_.device).repeat(code.shape[0], 1)
        )
        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code, order_, inverse = code[perm], order_[perm], inverse[perm]
        self["serialized_code"], self["serialized_order"], self["serialized_inverse"] = code, order_, inverse

    def sparsify(self, pad=96):
        """把 feat 包成 spconv.SparseConvTensor（indices = [batch, x, y, z]）。"""
        assert {"feat", "batch", "grid_coord"}.issubset(self.keys())
        sparse_shape = torch.add(torch.max(self.grid_coord, dim=0).values, pad).tolist()
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat([self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )


class PointModule(nn.Module):
    """占位基类：子类的 forward 接收/返回 Point。"""


class PointSequential(PointModule):
    """顺序容器：PointModule 直接吃 Point；spconv 模块作用在稀疏张量上；普通 torch 模块作用在 feat 上。"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
        for name, module in kwargs.items():
            self.add_module(name, module)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        self.add_module(str(len(self._modules)) if name is None else name, module)

    def forward(self, point: Point) -> Point:
        for module in self._modules.values():
            if isinstance(module, PointModule):
                point = module(point)
            elif spconv.modules.is_spconv_module(module):
                if not point.feat.is_cuda:
                    module.algo = spconv.ConvAlgo.Native  # spconv 在 CPU 上只有 Native 算法（且只有前向）
                point.sparse_conv_feat = module(point.sparse_conv_feat)
                point.feat = point.sparse_conv_feat.features
            else:
                point.feat = module(point.feat)
                if "sparse_conv_feat" in point.keys():
                    point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


# --------------------------------------------------------------------------- PTv3 组件
class SerializedAttention(PointModule):
    """序列上的分块注意力（普通注意力分支）。"""

    def __init__(self, channels, num_heads, patch_size, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, order_index=0):
        super().__init__()
        assert channels % num_heads == 0
        self.channels, self.num_heads, self.order_index = channels, num_heads, order_index
        self.scale = (channels // num_heads) ** -0.5
        self.patch_size_max = patch_size  # 实际分块大小在前向里取 min(批内最少点数, patch_size_max)
        self.patch_size = 0
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        """把每个样本补齐到 patch_size 的整数倍（重复自己尾部的点），返回 pad / unpad 索引。"""
        if "pad" not in point.keys() or "unpad" not in point.keys():
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = torch.div(bincount + self.patch_size - 1, self.patch_size, rounding_mode="trunc") * self.patch_size
            mask_pad = bincount > self.patch_size  # 只有点数多于 patch_size 的样本才补齐
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[_offset_pad[i + 1] - self.patch_size + (bincount[i] % self.patch_size) : _offset_pad[i + 1]] = pad[
                        _offset_pad[i + 1] - 2 * self.patch_size + (bincount[i] % self.patch_size) : _offset_pad[i + 1] - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
            point["pad"], point["unpad"] = pad, unpad
        return point["pad"], point["unpad"]

    def forward(self, point: Point) -> Point:
        self.patch_size = min(offset2bincount(point.offset).min().tolist(), self.patch_size_max)
        H, K, C = self.num_heads, self.patch_size, self.channels
        pad, unpad = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        qkv = self.qkv(point.feat)[order]
        # (N', K, 3, H, C') => (3, N', H, K, C')
        q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
        attn = self.attn_drop(self.softmax(attn)).to(qkv.dtype)
        feat = (attn @ v).transpose(1, 2).reshape(-1, C)[inverse]
        point.feat = self.proj_drop(self.proj(feat))
        return point


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(PointModule):
    """xCPE(子流形稀疏卷积) -> 分块注意力 -> MLP，均为前置 LayerNorm 的残差块。"""

    def __init__(self, channels, num_heads, patch_size, mlp_ratio, qkv_bias, attn_drop, proj_drop, drop_path, order_index, cpe_indice_key):
        super().__init__()
        # 稀疏卷积不带 bias（原版 bias=True）：紧随其后的 Linear 的 bias 能完全吸收它，参数化等价；
        # 这样 spconv 的 CPU 前向（不支持 fused bias）也能跑形状测试。
        self.cpe = PointSequential(
            spconv.SubMConv3d(channels, channels, kernel_size=3, bias=False, indice_key=cpe_indice_key),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
        )
        self.norm1 = PointSequential(nn.LayerNorm(channels))
        self.attn = SerializedAttention(channels, num_heads, patch_size, qkv_bias, attn_drop, proj_drop, order_index)
        self.norm2 = PointSequential(nn.LayerNorm(channels))
        self.mlp = PointSequential(MLP(channels, int(channels * mlp_ratio), channels, drop=proj_drop))
        self.drop_path = PointSequential(DropPath(drop_path) if drop_path > 0.0 else nn.Identity())

    def forward(self, point: Point) -> Point:
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        point = self.drop_path(self.attn(self.norm1(point)))
        point.feat = shortcut + point.feat
        shortcut = point.feat
        point = self.drop_path(self.mlp(self.norm2(point)))
        point.feat = shortcut + point.feat
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class GridPooling(PointModule):
    """按体素合并下采样（记录归属 pooling_inverse 供上采样还原）。"""

    def __init__(self, in_channels, out_channels, stride=2, reduce="max", shuffle_orders=False):
        super().__init__()
        assert reduce in ["sum", "mean", "min", "max"]
        self.stride, self.reduce, self.shuffle_orders = stride, reduce, shuffle_orders
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = PointSequential(nn.LayerNorm(out_channels))
        self.act = PointSequential(nn.GELU())

    def forward(self, point: Point) -> Point:
        grid_coord = torch.div(point.grid_coord, self.stride, rounding_mode="trunc")
        grid_coord = grid_coord | point.batch.view(-1, 1) << 48  # 样本索引放高位：唯一体素按样本有序
        grid_coord, cluster, counts = torch.unique(grid_coord, sorted=True, return_inverse=True, return_counts=True, dim=0)
        grid_coord = grid_coord & ((1 << 48) - 1)
        _, indices = torch.sort(cluster)  # 按簇排序的点索引，供 segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]
        order, depth = point.order, point.serialized_depth
        point = Point(
            feat=segment_csr(self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce),
            coord=segment_csr(point.coord[indices], idx_ptr, reduce="mean"),
            grid_coord=grid_coord,
            batch=point.batch[head_indices],
            grid_size=point.grid_size * self.stride,
            pooling_inverse=cluster,
            pooling_parent=point,
        )
        point = self.act(self.norm(point))
        point.serialization(order=order, depth=depth, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        return point


class GridUnpooling(PointModule):
    """按记录的归属关系把粗层特征散回父层并与跳连相加。"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels), nn.LayerNorm(out_channels), nn.GELU())
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels), nn.LayerNorm(out_channels), nn.GELU())

    def forward(self, point: Point) -> Point:
        assert {"pooling_parent", "pooling_inverse"}.issubset(point.keys())
        parent = point.pop("pooling_parent")
        inverse = point.pooling_inverse
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + self.proj(point).feat[inverse]
        parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)
        return parent


class Embedding(PointModule):
    def __init__(self, in_channels, embed_channels):
        super().__init__()
        self.stem = PointSequential(linear=nn.Linear(in_channels, embed_channels), norm=nn.LayerNorm(embed_channels), act=nn.GELU())

    def forward(self, point: Point) -> Point:
        return self.stem(point)


class PointTransformerV3(PointModule):
    """完整编码-解码 U-Net：输入 Point 字典，输出逐点（逐体素）特征 point.feat，通道数 = dec_channels[0]。"""

    def __init__(
        self,
        in_channels,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        patch_size=1024,
        mlp_ratio=4,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=False,
        depth=16,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else list(order)
        self.shuffle_orders, self.depth = shuffle_orders, depth
        assert self.num_stages == len(stride) + 1 == len(enc_channels) == len(enc_num_head)
        assert self.num_stages == len(dec_depths) + 1 == len(dec_channels) + 1 == len(dec_num_head) + 1

        self.embedding = Embedding(in_channels, enc_channels[0])
        block_kw = dict(patch_size=patch_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)

        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            dp = enc_drop_path[sum(enc_depths[:s]) : sum(enc_depths[: s + 1])]
            enc = PointSequential()
            if s > 0:
                enc.add(GridPooling(enc_channels[s - 1], enc_channels[s], stride=stride[s - 1], shuffle_orders=shuffle_orders), name="down")
            for i in range(enc_depths[s]):
                enc.add(
                    Block(enc_channels[s], enc_num_head[s], drop_path=dp[i], order_index=i % len(self.order), cpe_indice_key=f"stage{s}", **block_kw),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(enc, name=f"enc{s}")

        dec_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))]
        self.dec = PointSequential()
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        for s in reversed(range(self.num_stages - 1)):
            dp = dec_drop_path[sum(dec_depths[:s]) : sum(dec_depths[: s + 1])]
            dp.reverse()
            dec = PointSequential()
            dec.add(GridUnpooling(dec_channels[s + 1], enc_channels[s], dec_channels[s]), name="up")
            for i in range(dec_depths[s]):
                dec.add(
                    Block(dec_channels[s], dec_num_head[s], drop_path=dp[i], order_index=i % len(self.order), cpe_indice_key=f"stage{s}", **block_kw),
                    name=f"block{i}",
                )
            self.dec.add(dec, name=f"dec{s}")
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, spconv.SubMConv3d)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, data_dict) -> Point:
        point = Point(data_dict)
        point = self.embedding(point)
        point.serialization(order=self.order, depth=self.depth, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        point = self.enc(point)
        point = self.dec(point)
        return point


# --------------------------------------------------------------------------- 聚合模块
@dataclass
class AggregatorConfig:
    backend: str = "scratch_ptv3"
    sonata: SonataAggregatorConfig = field(default_factory=SonataAggregatorConfig)
    ordering: OrderingConfig = field(default_factory=OrderingConfig)
    order: List[str] = field(default_factory=lambda: ["z", "z-trans", "hilbert", "hilbert-trans"])  # 骨干各 Block 轮换的曲线
    stride: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    enc_depths: List[int] = field(default_factory=lambda: [2, 2, 2, 6, 2])
    enc_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    enc_num_head: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    dec_depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    dec_channels: List[int] = field(default_factory=lambda: [64, 64, 128, 256])  # dec_channels[0] 必须等于 D
    dec_num_head: List[int] = field(default_factory=lambda: [4, 4, 8, 16])
    patch_size: int = 1024  # 分块大小上限（各级共用）
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    drop_path: float = 0.3
    shuffle_orders: bool = False  # 关掉序列化顺序的随机打乱（顶层与各级下采样一起）


class SpatialAggregator(nn.Module):
    """单帧调用：points [B, K, C] -> (tokens [B, K, D], positions [B, K, 3])，二者已按 Ordering 的曲线序重排。"""

    def __init__(self, cfg: AggregatorConfig, in_channels: int, out_channels: int):
        super().__init__()
        assert in_channels >= 3, "C >= 3，前三维是 xyz"
        assert cfg.dec_channels[0] == out_channels, f"解码器末级通道数 dec_channels[0]={cfg.dec_channels[0]} 必须等于 D={out_channels}"
        self.in_channels, self.out_channels = in_channels, out_channels
        self.ordering = Ordering(cfg.ordering)
        self.backbone = PointTransformerV3(
            in_channels=in_channels,
            order=cfg.order,
            stride=cfg.stride,
            enc_depths=cfg.enc_depths,
            enc_channels=cfg.enc_channels,
            enc_num_head=cfg.enc_num_head,
            dec_depths=cfg.dec_depths,
            dec_channels=cfg.dec_channels,
            dec_num_head=cfg.dec_num_head,
            patch_size=cfg.patch_size,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
            drop_path=cfg.drop_path,
            shuffle_orders=cfg.shuffle_orders,
            depth=cfg.ordering.depth,
        )

    def forward(self, points: torch.Tensor):
        assert points.dim() == 3 and points.shape[2] == self.in_channels, f"points 应为 [B, K, {self.in_channels}]，得到 {tuple(points.shape)}"
        B, K, C = points.shape
        coord = points[..., :3]
        order = self.ordering(coord)  # [B, K]

        # 体素去重：键 = (样本索引, 格点)，样本索引在最前面保证唯一体素按样本有序
        batch = torch.arange(B, device=points.device).repeat_interleave(K)
        key = torch.cat([batch[:, None], self.ordering.voxelize(coord).reshape(-1, 3).long()], dim=1)
        key, inverse = torch.unique(key, dim=0, sorted=True, return_inverse=True)
        feat = scatter_mean(points.reshape(-1, C), inverse, dim_size=key.shape[0])  # 同体素取均值

        point = self.backbone(
            dict(coord=feat[:, :3], feat=feat, batch=key[:, 0], grid_coord=key[:, 1:].int(), grid_size=self.ordering.cfg.grid_size)
        )
        tokens = point.feat[inverse].view(B, K, self.out_channels)  # 按逆映射散回 K 个点

        idx = order.unsqueeze(-1)
        tokens = tokens.gather(1, idx.expand(-1, -1, self.out_channels))
        positions = coord.gather(1, idx.expand(-1, -1, 3))
        assert tokens.shape == (B, K, self.out_channels) and positions.shape == (B, K, 3)
        return tokens, positions


def build_spatial_aggregator(
    cfg: AggregatorConfig, in_channels: int, out_channels: int
) -> nn.Module:
    """Build the selected spatial backend while preserving the scratch default."""
    if cfg.backend == "scratch_ptv3":
        return SpatialAggregator(cfg, in_channels, out_channels)
    if cfg.backend == "sonata_ptv3":
        from models.sonata_aggregator import SonataSpatialAggregator

        return SonataSpatialAggregator(
            cfg.sonata,
            in_channels,
            out_channels,
            ordering_cfg=cfg.ordering,
        )
    raise ValueError(f"unknown aggregator backend: {cfg.backend!r}")
