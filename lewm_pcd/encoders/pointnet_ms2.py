"""PointNet copied from ManiSkill2-Learn (configs/mfrl/ppo/maniskill2_pn.py):

    dict(type="PointNet", feat_dim="pcd_all_channel", mlp_spec=[64, 128, 512], feature_transform=[])

Per point: Conv1d(kernel 1, no bias) -> LayerNorm over channels (eps 1e-6, affine)
-> ReLU, for every entry of ``mlp_spec`` (the last layer is normalised and
activated too, ``inactivated_output=False``); then max over the points. No T-Net.
With the original spec the body has 75,520 parameters, exactly as upstream.
Implementation note: a 1x1 Conv1d without bias is exactly a per-point Linear, so
the body is stored in (B, N, C) layout with ``nn.Linear`` and ``nn.LayerNorm``
on the channel dim -- same maths, no transposes of the 600k-point activations.

The only adaptation to LeWM is the output size: a linear head maps the pooled
feature to ``out_dim`` (= the 192-d LeWM latent). Width is a hyper-parameter:
``mlp_spec=[512, 1024, 4096]`` gives 5.52 M parameters, matching the ViT-tiny
encoder of the pixel model (5.50 M).
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ManiSkillPointNet(nn.Module):
    def __init__(
        self,
        in_dim: int = 6,
        mlp_spec: Sequence[int] = (64, 128, 512),
        out_dim: int = 192,
        eps: float = 1e-6,
        chunk_points: int | None = 300,
    ):
        """chunk_points: with wide layers the per-point activations of a 128 x 4-frame batch
        (614k points) do not fit in GPU memory; the body is therefore run on point chunks with
        activation checkpointing while training. LayerNorm is per point and max-pool is
        associative, so the result is identical to the unchunked forward."""
        super().__init__()
        self.chunk_points = chunk_points
        layers, d = [], in_dim
        for h in mlp_spec:
            # Conv1d(d, h, 1, bias=False) == Linear(d, h, bias=False) applied per point; LN1d == LayerNorm(h) per point
            layers += [nn.Linear(d, h, bias=False), nn.LayerNorm(h, eps=eps), nn.ReLU(inplace=True)]
            d = h
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(d, out_dim)

    def _body_max(self, x):  # (B, n, C) -> (B, C')
        return self.conv(x).max(dim=1).values

    def forward(self, x):
        """x: (B, N, C) -> (B, out_dim)"""
        x = x.float()
        if self.chunk_points and x.shape[1] > self.chunk_points:
            use_ckpt = self.training and torch.is_grad_enabled()
            parts = [checkpoint(self._body_max, xc, use_reentrant=False) if use_ckpt else self._body_max(xc)
                     for xc in x.split(self.chunk_points, dim=1)]
            feat = torch.stack(parts, dim=-1).max(dim=-1).values
        else:
            feat = self._body_max(x)
        return self.head(feat)
