"""ViT encoders. ViT-tiny/14 on 224x224 images, trained from scratch, is the LeWM encoder (Sec. 3, 5.50M)
and the PLDM / LeJEPA encoder; ViT-B/16 is used for Random (ViT-B). Released VISReg / V-JEPA 2.1 weights are
loaded by their own modules in jepa4r/baselines.

Original module docstring:
Pixel baseline encoder: the LeWM ViT, returning its CLS token.

Wraps ``stable_pretraining.backbone.utils.vit_hf`` so that it satisfies the
generic encoder contract (``(B, 3, H, W) -> (B, embed_dim)``). Behaviour is
exactly what ``jepa.py`` used to do inline (``interpolate_pos_encoding=True``,
CLS token); only the state-dict prefix moves from ``encoder.`` to ``encoder.vit.``.
"""
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
        import stable_pretraining as spt

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


def build(enc_cfg):
    """configs/encoder/vit_*.yaml -> ViTCLS (stable-pretraining's vit_hf; LeWM's ViT encoder
    config)."""
    return ViTCLS(size=str(enc_cfg.variant), patch_size=int(enc_cfg.patch_size),
                  image_size=int(enc_cfg.get('image_size', 224)), pretrained=bool(enc_cfg.get('pretrained', False)),
                  use_mask_token=False)
