"""Frozen 3D reconstruction backbones for the decoder; ``build_backbone(bb_cfg)`` imports ``bb_cfg.impl``.

Contract: forward(tokens [B, 256, 1024]) -> point map [B, 224, 224, 3] (+ features for the rgb branch);
tokens replace the output of the patch embedding. attention_blocks() lists the LoRA targets.
"""
import importlib


def build_backbone(bb_cfg):
    return importlib.import_module(bb_cfg.impl).build(bb_cfg)
