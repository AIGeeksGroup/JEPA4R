"""Observation encoders. ``build_encoder(enc_cfg)`` imports ``enc_cfg.impl`` and calls its ``build(enc_cfg)``.

pointnet                 JEPA4R default (Sec. 4.2), ~5.51M
pointnet2 / dgcnn /
point_patch_transformer  Tab. 4b ablations, same parameter count
vit                      ViT-tiny/14 for LeWM, PLDM, LeJEPA (5.50M); ViT-B/16, ViT-L/16 for Random (ViT-B)
impala                   PLDM on OGBench Cube
Contract: forward(obs) -> [B, out_dim]; point encoders take [B, N, 6], image encoders [B, 3, 224, 224].
"""
import importlib


def build_encoder(enc_cfg):
    return importlib.import_module(enc_cfg.impl).build(enc_cfg)
