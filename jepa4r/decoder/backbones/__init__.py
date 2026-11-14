"""Frozen 3D reconstruction backbones for the decoder; ``build_backbone(bb_cfg, **kwargs)`` imports ``bb_cfg.impl``.

Each module holds the whole decoder with its backbone, as the paper's code builds it (set generator -> the backbone's
frozen layers, the generated tokens replacing the output of its patch embedding (+ LoRA in stage B) -> affine map, rgb
branch, pixel selection): page4d.PAGE4DDecoder (default), vggt.VGGTDecoder, depth_anything3.DA3Decoder (Tab. 4a).
``build(bb_cfg, **kwargs)`` returns it with the released weights ``bb_cfg.weights`` (frozen, reloaded, never written
into decoder checkpoints); kwargs: jepa4r.decoder.model.decoder_kwargs. The decoder maps z [B, 192] to the dense point
map [B, 224 x 224, 6] in the target frame (``forward``), its training loss (``loss``) and 1,200 points (``to_points``).
"""
import importlib


def build_backbone(bb_cfg, **kwargs):
    return importlib.import_module(bb_cfg.impl).build(bb_cfg, **kwargs)
