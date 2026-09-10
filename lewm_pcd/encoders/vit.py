"""Pixel baseline encoder: the LeWM ViT, returning its CLS token.

Wraps ``stable_pretraining.backbone.utils.vit_hf`` so that it satisfies the
generic encoder contract (``(B, 3, H, W) -> (B, embed_dim)``). Behaviour is
exactly what ``jepa.py`` used to do inline (``interpolate_pos_encoding=True``,
CLS token); only the state-dict prefix moves from ``encoder.`` to ``encoder.vit.``.
"""

import stable_pretraining as spt
from torch import nn


class ViTCLS(nn.Module):
    def __init__(
        self,
        size: str = 'tiny',
        patch_size: int = 14,
        image_size: int = 224,
        pretrained: bool = False,
        use_mask_token: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.vit = spt.backbone.utils.vit_hf(
            size,
            patch_size=patch_size,
            image_size=image_size,
            pretrained=pretrained,
            use_mask_token=use_mask_token,
            **kwargs,
        )

    def forward(self, x):
        return self.vit(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
