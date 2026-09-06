"""时限截断的 bootstrap（batch transform）。

parllel 的优势估计把 done = terminated | truncated 一律当作终止，下一状态价值记 0。
但时限截断（truncated 且非 terminated）并不是任务结束，正确做法是在该步奖励上加 γ·V(下一状态)，
再交给优势估计（partial-episode bootstrapping）。V 用当前 agent 的价值头在采样时参数下无梯度计算。
放在奖励归一化之后、优势估计之前，使 V 与归一化后的奖励量纲一致。
"""

from __future__ import annotations

import numpy as np
import torch
from parllel import Array, ArrayDict, dict_map
from parllel.transforms.transform import Transform


class BootstrapTruncated(Transform):
    def __init__(self, agent, discount: float) -> None:
        self.agent = agent
        self.discount = discount

    def __call__(self, sample_tree: ArrayDict[Array]) -> ArrayDict[Array]:
        terminated = np.asarray(sample_tree["terminated"])
        truncated = np.asarray(sample_tree["truncated"])
        mask = truncated & ~terminated
        if not mask.any():
            return sample_tree
        t_idx, b_idx = np.nonzero(mask)
        next_obs = sample_tree["next_observation"][t_idx, b_idx]
        next_obs = next_obs.to_ndarray()
        next_obs = dict_map(torch.from_numpy, next_obs)
        next_obs = next_obs.to(device=self.agent.device)
        with torch.no_grad():
            value = self.agent.model(next_obs)["value"].detach().cpu().numpy()
        reward = np.asarray(sample_tree["reward"])
        reward[t_idx, b_idx] += self.discount * value
        return sample_tree
