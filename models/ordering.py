"""Ordering：把一帧点云沿空间填充曲线排成确定的一维序列。

移植自 Pointcept（仓库 A）：
- ``pointcept/models/utils/serialization/z_order.py``   ：OCNN 的查表式 Z 序编码（只保留编码）；
- ``pointcept/models/utils/serialization/hilbert.py``   ：Skilling 算法的向量化 Hilbert 编码（只保留编码）；
- ``pointcept/models/utils/serialization/default.py``  ：``encode``（曲线种类 + 转置版 + 样本索引高位）。

与 Pointcept 的差别：
1. 体素原点固定为世界系包围盒最小角、曲线立方体深度固定在配置里，都不随样本自适应；
2. 格点坐标截断到 [0, 2**depth - 1]；
3. 排序用稳定排序：同体素的点编码相同，保持输入次序。
"""

import math
from dataclasses import dataclass, field
from typing import List

import torch
from torch import nn

ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")


# --------------------------------------------------------------------------- Z 序（OCNN 查表法）
class _KeyLUT:
    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        self._encode = {
            torch.device("cpu"): (
                self._xyz2key(r256, zero, zero, 8),
                self._xyz2key(zero, r256, zero, 8),
                self._xyz2key(zero, zero, r256, 8),
            )
        }

    def encode_lut(self, device):
        if device not in self._encode:
            self._encode[device] = tuple(e.to(device) for e in self._encode[torch.device("cpu")])
        return self._encode[device]

    @staticmethod
    def _xyz2key(x, y, z, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = key | ((x & mask) << (2 * i + 2)) | ((y & mask) << (2 * i + 1)) | ((z & mask) << (2 * i + 0))
        return key


_key_lut = _KeyLUT()


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16) -> torch.Tensor:
    """grid_coord [N, 3] 整数格点 -> [N] int64 Z 序编码（depth <= 16）。"""
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    EX, EY, EZ = _key_lut.encode_lut(x.device)
    mask = 255 if depth > 8 else (1 << depth) - 1
    key = EX[x & mask] | EY[y & mask] | EZ[z & mask]
    if depth > 8:
        mask = (1 << (depth - 8)) - 1
        key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask] | EZ[(z >> 8) & mask]
        key = key16 << 24 | key
    return key


# --------------------------------------------------------------------------- Hilbert 曲线（Skilling）
def _right_shift(binary: torch.Tensor, k: int = 1) -> torch.Tensor:
    if binary.shape[-1] <= k:
        return torch.zeros_like(binary)
    return nn.functional.pad(binary[..., :-k], (k, 0), mode="constant", value=0)


def _gray2binary(gray: torch.Tensor) -> torch.Tensor:
    shift = 2 ** (math.ceil(math.log2(gray.shape[-1])) - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, _right_shift(gray, shift))
        shift //= 2
    return gray


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16) -> torch.Tensor:
    """grid_coord [N, 3] 整数格点 -> [N] int64 Hilbert 编码（num_dims=3, num_bits=depth <= 16）。"""
    num_dims, num_bits = 3, depth
    assert grid_coord.shape[-1] == num_dims and num_dims * num_bits <= 63
    bitpack_mask = 1 << torch.arange(0, 8, device=grid_coord.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    # 64 位整数拆成 8 个字节，再拆成比特，截到 num_bits 位
    locs_uint8 = grid_coord.long().contiguous().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)
    gray = locs_uint8.unsqueeze(-1).bitwise_and(bitpack_mask_rev).ne(0).byte().flatten(-2, -1)[..., -num_bits:]

    for bit in range(0, num_bits):
        for dim in range(0, num_dims):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], mask[:, None])
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(gray[:, dim, bit + 1 :], to_flip)
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))
    hh_bin = _gray2binary(gray)
    padded = nn.functional.pad(hh_bin, (64 - num_bits * num_dims, 0), "constant", 0)
    hh_uint8 = (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask).sum(2).type(torch.uint8)
    return hh_uint8.view(torch.int64).reshape(-1)


@torch.no_grad()
def encode(grid_coord: torch.Tensor, batch: torch.Tensor = None, depth: int = 16, order: str = "z") -> torch.Tensor:
    """grid_coord [N, 3] -> [N] int64 曲线编码；给了 batch [N] 就把样本索引放到高位（先按样本、再按曲线）。"""
    assert order in ORDERS, f"order 应为 {ORDERS} 之一，得到 {order}"
    if order.endswith("-trans"):
        grid_coord = grid_coord[:, [1, 0, 2]]
    code = z_order_encode(grid_coord, depth) if order.startswith("z") else hilbert_encode(grid_coord, depth)
    if batch is not None:
        code = batch.long() << (depth * 3) | code
    return code


# --------------------------------------------------------------------------- Ordering 模块
@dataclass
class OrderingConfig:
    order: str = "hilbert"  # z | z-trans | hilbert | hilbert-trans
    grid_size: float = 0.02  # 体素边长（世界单位）
    origin: List[float] = field(default_factory=lambda: [-1.0, -1.0, -1.0])  # 世界系包围盒最小角，固定
    depth: int = 16  # 曲线立方体深度：每轴 2**depth 个体素，固定（<= 16）


class Ordering(nn.Module):
    """输入点坐标 [B, K, 3]，输出排序索引 [B, K]（按该索引 gather 即得曲线序）。"""

    def __init__(self, cfg: OrderingConfig):
        super().__init__()
        assert cfg.order in ORDERS and 1 <= cfg.depth <= 16 and cfg.grid_size > 0 and len(cfg.origin) == 3
        self.cfg = cfg
        self.register_buffer("origin", torch.tensor(cfg.origin, dtype=torch.float32), persistent=False)

    def voxelize(self, coord: torch.Tensor) -> torch.Tensor:
        """坐标 [..., 3] -> 整数格点 [..., 3] (int32)，相对固定原点、截断到 [0, 2**depth - 1]。"""
        grid = torch.div(coord - self.origin, self.cfg.grid_size, rounding_mode="floor")
        return grid.clamp_(0, 2**self.cfg.depth - 1).int()

    def forward(self, coord: torch.Tensor) -> torch.Tensor:
        assert coord.dim() == 3 and coord.shape[-1] == 3, f"coord 应为 [B, K, 3]，得到 {tuple(coord.shape)}"
        B, K, _ = coord.shape
        code = encode(self.voxelize(coord).reshape(-1, 3), depth=self.cfg.depth, order=self.cfg.order).view(B, K)
        order = torch.argsort(code, dim=1, stable=True)  # 稳定排序：同体素的点保持输入次序
        assert order.shape == (B, K)
        return order
