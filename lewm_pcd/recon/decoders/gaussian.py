"""Gaussian decoder: the latent -> a mixture of K anisotropic 3D Gaussians."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from recon.losses import gaussian_nll, sample_gaussians

from .set_decoder import SetDecoder


class GaussianDecoder(nn.Module):
    """(B, D) -> dict: mu (B, K, 3), log_s (B, K, 3), quat (B, K, 4) wxyz, weight_logit (B, K), rgb (B, K, 3) or None.

    A density model of the point cloud, fitted by maximum likelihood (``recon.losses.gaussian_nll``): the softmax
    of ``weight_logit`` is the share of the points a component explains. No renderer is involved in training,
    so the weight is a mass, not an opacity.

    ``min_scale`` / ``max_scale`` bound each standard deviation and are given in METRES; ``target_scale`` is the
    scale of the target frame (``data.target_preprocessor.scale``). Points sampled from surfaces have no
    thickness, so the thinnest axis of most components settles on ``min_scale``: it sets the sheet thickness."""

    def __init__(self, latent_dim=192, num_gauss=1024, color=True, min_scale=0.003, max_scale=0.03, target_scale=4.0,
                 color_weight=0.1, spread_weight=1.0, **trunk):
        super().__init__()
        self.color = color
        self.dec = SetDecoder(latent_dim=latent_dim, num_queries=num_gauss, out_dim=14 if color else 11, **trunk)
        self.log_min, self.log_max = math.log(min_scale * target_scale), math.log(max_scale * target_scale)
        self.color_weight, self.spread_weight = color_weight, spread_weight

    def forward(self, latent):
        o = self.dec(latent)
        return dict(
            mu=o[..., 0:3],
            log_s=self.log_min + (self.log_max - self.log_min) * torch.sigmoid(o[..., 3:6] - 1.0),
            quat=F.normalize(o[..., 6:10] + o.new_tensor([1.0, 0.0, 0.0, 0.0]), dim=-1),
            weight_logit=o[..., 10],
            rgb=torch.sigmoid(o[..., 11:14]) if self.color else None,
        )

    def loss(self, output, target):
        return gaussian_nll(output, target, self.color_weight, self.spread_weight)

    def to_points(self, output, n=1200):
        return sample_gaussians(output, n)
