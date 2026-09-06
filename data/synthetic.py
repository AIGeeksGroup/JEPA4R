"""合成数据集：随机生成 T+1 帧、每帧 K 个点、点之间无对应关系的点云序列和动作，只用于冒烟测试。"""

import torch
from torch.utils.data import Dataset


class SyntheticSequences(Dataset):
    """样本 i 由 seed + i 决定：points [T+1, K, C]（xyz 均匀落在 box 内，其余通道均匀 [0, 1]），actions [T, action_dim]。"""

    def __init__(self, num_samples: int, T: int, K: int, C: int, action_dim: int, box=(-1.0, 1.0), seed: int = 0):
        assert C >= 3
        self.num_samples, self.T, self.K, self.C, self.action_dim, self.box, self.seed = num_samples, T, K, C, action_dim, box, seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.seed + i)
        points = torch.rand(self.T + 1, self.K, self.C, generator=g)
        points[..., :3] = points[..., :3] * (self.box[1] - self.box[0]) + self.box[0]
        actions = torch.randn(self.T, self.action_dim, generator=g)
        return points, actions
