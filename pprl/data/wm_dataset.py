"""离线回合数据集与子轨迹采样，照 LeWM 官方：
- 每个样本 = num_steps 帧（history_size + num_preds，默认 4）、帧间隔 frameskip（默认 5）；
- 每帧的动作块 = 该帧起 frameskip 个连续动作拼接（官方 action_encoder.input_dim = frameskip * action_dim）；
  回合末尾不足一个动作块时用 0 填充（官方：序列边界的 NaN 动作置 0）；
- 动作按列 z-score 标准化，统计量取自训练集（官方 get_column_normalizer）。
观测（点云 + 状态）保持 pprl 环境包装后的原样，作为 encoder 输入。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_episode(path: Path) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


class ColumnNormalizer:
    """z-score：(x - mean) / std，统计量按列在训练集上计算（同官方 ZScoreNormalizer）"""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)

    @classmethod
    def fit(cls, rows: np.ndarray) -> "ColumnNormalizer":
        return cls(rows.mean(0), rows.std(0) + 1e-8)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}


class WindowDataset(Dataset):
    def __init__(self, episodes: list[dict], num_steps: int, frameskip: int, action_norm: ColumnNormalizer):
        self.episodes, self.num_steps, self.frameskip, self.action_norm = episodes, num_steps, frameskip, action_norm
        self.index = []  # (episode_idx, 起始帧 t)
        span = (num_steps - 1) * frameskip
        for i, ep in enumerate(episodes):
            n_frames = len(ep["ptr"]) - 1
            for t in range(0, n_frames - span):
                self.index.append((i, t))

    def __len__(self) -> int:
        return len(self.index)

    def action_block(self, ep: dict, k: int) -> np.ndarray:
        a = ep["action"][k : k + self.frameskip]
        if len(a) < self.frameskip:  # 边界：不足一块的动作置 0（官方 NaN -> 0）
            a = np.concatenate([a, np.zeros((self.frameskip - len(a), ep["action"].shape[1]), np.float32)], 0)
        return self.action_norm(a).reshape(-1)

    def __getitem__(self, idx: int) -> dict:
        i, t = self.index[idx]
        ep = self.episodes[i]
        frames = [t + j * self.frameskip for j in range(self.num_steps)]
        return {
            "points": [ep["points"][ep["ptr"][k] : ep["ptr"][k + 1]].astype(np.float32) for k in frames],
            "state": np.stack([ep["state"][k] for k in frames]).astype(np.float32),
            "action": np.stack([self.action_block(ep, k) for k in frames]).astype(np.float32),
        }


def collate_windows(samples: list[dict]) -> tuple[dict, torch.Tensor]:
    """组批：观测展平为 B*T 个点云（pyg 风格 pos/ptr，与 pprl 的 dict_to_batched_data 约定一致），动作为 (B, T, A_block)。"""
    points = [p for s in samples for p in s["points"]]
    lengths = np.array([len(p) for p in points])
    obs = {
        "points": {
            "pos": torch.from_numpy(np.concatenate(points, 0)),
            "ptr": torch.from_numpy(np.concatenate([[0], np.cumsum(lengths)])).long(),
        },
        "state": torch.from_numpy(np.concatenate([s["state"] for s in samples], 0)),
    }
    action = torch.from_numpy(np.stack([s["action"] for s in samples]))
    return obs, action


def to_device(obs: dict, device) -> dict:
    return {"points": {k: v.to(device, non_blocking=True) for k, v in obs["points"].items()}, "state": obs["state"].to(device, non_blocking=True)}
