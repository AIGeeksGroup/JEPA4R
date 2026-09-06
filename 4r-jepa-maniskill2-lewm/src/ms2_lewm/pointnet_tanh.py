"""The selected official PointNet with only its final ReLU replaced by tanh.

Registered outside vendor so PPO and the world model can instantiate the same
configuration. The inherited forward, pooling, weights and earlier activations
are unchanged. No parameters or state-dict keys are introduced.
"""
from torch import nn

from maniskill2_learn.networks.backbones.pointnet import PointNet
from maniskill2_learn.networks.builder import BACKBONES


@BACKBONES.register_module()
class PointNetLastTanh(PointNet):
    """ReLU, ReLU, tanh followed by the official global maximum pooling."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.feature_transform or not hasattr(self, "conv"):
            raise ValueError("PointNetLastTanh requires the selected PPO feature_transform=[] configuration.")
        final_block = self.conv.mlp[-1]
        if not isinstance(final_block.act, nn.ReLU):
            raise ValueError("Expected the official final ReLU before its explicit tanh replacement.")
        final_block.act = nn.Tanh()
        final_block.act_cfg = {"type": "Tanh"}
