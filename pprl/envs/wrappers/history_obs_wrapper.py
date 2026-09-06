"""历史帧观测包装：把单帧观测 {"points": 点云, "state": 向量} 变成按世界模型口径取的 T 帧历史
{"points_0", "state_0", ..., "points_{T-1}", "state_{T-1}"}（下标 0 最旧，T-1 为当前帧），
相邻两帧间隔 frameskip 个环境步（与 scripts/train_latent_wm.py 的 WindowDataset 一致：t-2·fs, t-fs, t）。
回合开头历史不足时用首帧重复填充。点云仍是变长数组，parllel 按 PointCloudSpace 建 jagged 缓冲。
frameskip 应与世界模型训练时的 data.frameskip 一致（正确设计是 1，即连续帧）；2026-09-06 的 LiftCube 世界模型按 5 训练，
所以配套 PPO 必须 frameskip=5（train_ppo_frozen_wm.py 从检查点读取）。"""
from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import gymnasium.spaces as spaces


class HistoryObsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, num_frames: int = 3, frameskip: int = 5, points_key: str = "points", state_key: str = "state") -> None:
        super().__init__(env)
        assert num_frames >= 1 and frameskip >= 1
        self.num_frames, self.frameskip = num_frames, frameskip
        self.points_key, self.state_key = points_key, state_key
        inner = env.observation_space
        assert isinstance(inner, spaces.Dict) and points_key in inner.spaces and state_key in inner.spaces, f"需要 {{{points_key}, {state_key}}} 字典观测，得到 {inner}"
        self.span = (num_frames - 1) * frameskip + 1  # 需要缓存的最近帧数
        self.buffer: deque = deque(maxlen=self.span)
        d = {}
        for k in range(num_frames):
            d[f"{points_key}_{k}"] = inner.spaces[points_key]
            d[f"{state_key}_{k}"] = inner.spaces[state_key]
        self.observation_space = spaces.Dict(d)

    def _stack(self) -> dict[str, Any]:
        frames = list(self.buffer)
        out = {}
        for k in range(self.num_frames):
            idx = len(frames) - 1 - (self.num_frames - 1 - k) * self.frameskip
            obs = frames[idx]
            out[f"{self.points_key}_{k}"] = obs[self.points_key]
            out[f"{self.state_key}_{k}"] = obs[self.state_key]
        return out

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.buffer.clear()
        for _ in range(self.span):  # 开局用首帧填满历史
            self.buffer.append(obs)
        return self._stack(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.buffer.append(obs)
        return self._stack(), reward, terminated, truncated, info
