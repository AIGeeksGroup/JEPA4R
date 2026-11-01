"""Pluggable observation encoders for the JEPA world model.

Contract (what ``jepa.JEPA.encode`` expects from ``self.encoder``):

    input  x   : float tensor ``(B, ...)`` -- one observation per row, batch and
                 time already flattened. Pixels: ``(B, 3, H, W)``; point clouds:
                 ``(B, N, C)`` with N points and C = 3 (xyz) or 6 (xyz + rgb).
    output emb : float tensor ``(B, embed_dim)``; ``embed_dim`` is the latent
                 size the projector / predictor / SIGReg were built for (192).

Anything satisfying this can be dropped in through ``model.encoder._target_``
in the Hydra config; nothing else in the model or training loop changes.
"""

from .dgcnn import DGCNN
from .pointnet import PointNet
from .pointnet2 import PointNet2
from .pointnet_dexart import DexArtPointNet
from .pointnet_ms2 import ManiSkillPointNet
from .ppt import PointPatchTransformer
from .sonata_cls import SonataCLSEncoder
from .vit import ViTCLS

__all__ = ['DGCNN', 'DexArtPointNet', 'ManiSkillPointNet', 'PointNet', 'PointNet2', 'PointPatchTransformer',
           'SonataCLSEncoder', 'ViTCLS']
