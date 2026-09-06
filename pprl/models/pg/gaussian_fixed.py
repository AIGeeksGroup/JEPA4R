"""修正 parllel Gaussian 分布的 likelihood_ratio：原实现对各动作维度的比值做了求和（torch.sum），
对角高斯的似然比应为各维比值的乘积；动作维度为 7 时原实现在新旧参数相同时也返回约 7，导致 PPO 的 approx_kl 恒为 4 左右、
clip_fraction 恒为 1。这里用 exp(sum(log ratio)) 计算乘积，数值上更稳。原仓库文件不改。"""

from __future__ import annotations

import torch
from parllel.torch.distributions.gaussian import EPS, DistParams, Gaussian
from torch import Tensor


class GaussianFixed(Gaussian):
    def likelihood_ratio(
        self,
        x: Tensor,
        /,
        old_dist_params: DistParams,
        new_dist_params: DistParams,
    ) -> Tensor:
        if self.fixed_std is None:
            old_log_std = old_dist_params["log_std"]
            new_log_std = new_dist_params["log_std"]
            if self.min_log_std is not None or self.max_log_std is not None:
                old_log_std = torch.clamp(old_log_std, min=self.min_log_std, max=self.max_log_std)
                new_log_std = torch.clamp(new_log_std, min=self.min_log_std, max=self.max_log_std)
            old_std = torch.exp(old_log_std)
            new_std = torch.exp(new_log_std)
            old_z = (x - old_dist_params["mean"]) / (old_std + EPS)
            new_z = (x - new_dist_params["mean"]) / (new_std + EPS)
            # log(L_n/L_o) 按维求和 = log(s_o/s_n) - (z_n^2 - z_o^2)/2
            log_ratios = (old_log_std - new_log_std) - (new_z**2 - old_z**2) / 2
        else:
            old_X = x - old_dist_params["mean"]
            new_X = x - new_dist_params["mean"]
            log_ratios = -(new_X**2 - old_X**2) / 2 / self.fixed_std**2
        # 求和后 clamp 到 [-20, 20] 再取指数：上溢会让 approx_kl 变成 NaN 并静默毁掉参数，clamp 后 approx_kl 仍会很大并正常触发 KL 上限
        return torch.exp(torch.sum(log_ratios, dim=-1).clamp(-20.0, 20.0))
