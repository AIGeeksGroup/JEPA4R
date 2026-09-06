"""Shared PPO/WM PointNet with its final activation replaced by Identity.

The original PointNet forward, earlier ReLUs, normalization, max pooling,
parameter names and tensor shapes are retained. This external registration
leaves all official source files and historical tanh configurations untouched.
"""
from torch import nn

from maniskill2_learn.networks.backbones.pointnet import PointNet
from maniskill2_learn.networks.builder import BACKBONES


@BACKBONES.register_module()
class PointNetLastIdentity(PointNet):
    """ReLU, ReLU, Identity, followed by the official global max pooling."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.feature_transform or not hasattr(self, "conv"):
            raise ValueError("PointNetLastIdentity requires the selected PPO feature_transform=[] configuration.")
        final_block = self.conv.mlp[-1]
        if not isinstance(final_block.act, nn.ReLU):
            raise ValueError("Expected the official final ReLU before its explicit Identity replacement.")
        final_block.act = nn.Identity()
        final_block.act_cfg = {"type": "Identity"}
