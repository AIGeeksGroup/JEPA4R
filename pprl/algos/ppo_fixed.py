"""parllel 的 PPO.loss 在触发 kl_divergence_limit 时只返回 loss 张量，而 optimize_agent 按二元组解包会报错。
这里子类化修正返回值，原仓库文件不改。"""

from __future__ import annotations

from parllel import ArrayDict
from parllel.torch.algos.ppo import PPO
from torch import Tensor


class PPOFixed(PPO):
    def loss(self, batch: ArrayDict[Tensor]):
        result = super().loss(batch)
        if self.early_stopping and not isinstance(result, tuple):
            return result, None
        return result
