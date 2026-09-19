"""Point-cloud decoder: the latent -> N explicit points."""

import torch
from torch import nn

from recon.losses import chamfer, sliced_wasserstein

from .set_decoder import SetDecoder


class PointDecoder(nn.Module):
    """(B, D) -> (B, num_points, out_dim): xyz in the target frame, plus rgb in [0, 1] when ``out_dim`` is 6.

    Loss = Chamfer (where each point is) + ``sw_weight`` x sliced Wasserstein on xyz (how many points each
    region gets -- a nearest-neighbour match alone lets many real points share one decoded point)."""

    def __init__(self, latent_dim=192, num_points=1200, out_dim=6, color_weight=0.1, sw_weight=1.0,
                 sw_projections=100, **trunk):
        super().__init__()
        assert out_dim in (3, 6), 'xyz or xyz + rgb'
        self.dec = SetDecoder(latent_dim=latent_dim, num_queries=num_points, out_dim=out_dim, **trunk)
        self.color_weight, self.sw_weight, self.sw_projections = color_weight, sw_weight, sw_projections

    def forward(self, latent):
        o = self.dec(latent)
        return o if o.size(-1) == 3 else torch.cat([o[..., :3], torch.sigmoid(o[..., 3:])], dim=-1)

    def loss(self, output, target):
        loss, logs = chamfer(output, target, self.color_weight)
        if self.sw_weight > 0:
            sw = sliced_wasserstein(output, target, self.sw_projections)
            loss, logs['sw'] = loss + self.sw_weight * sw, sw.detach()
        return loss, logs

    def to_points(self, output, n=None):
        return output
