"""V-JEPA 2.1 [murlabadia2026vjepa21]: the authors' released weights, frozen; ViT-B/16, ViT-L/16 on Cube
(cfg.method.encoder / cfg.method.weights after the per-env override).

ViT-B/16 (MetaWorld, Scene, Puzzle): single frames through the image path at 384, the mean of the last layer's patch
tokens, 768-d: jepa4r/policy/bc/jepa_baselines.py load_vjepa21 (facebookresearch/vjepa2 @ 204698b at
JEPA4R_VJEPA2). ViT-L/16 on Cube: the Cube workflow's loader (jepa4r/baselines/cube/released_encoders.py).
"""


def load_frozen_encoder(cfg):
    from jepa4r.baselines import frozen
    from jepa4r.policy.frozen import checkpoint

    if cfg.method.encoder.name != "vit_b16":
        from jepa4r.policy.gciql import load_frozen_vjepa21

        return load_frozen_vjepa21(cfg)
    from jepa4r.policy.bc.jepa_baselines import load_vjepa21

    return frozen(load_vjepa21(str(checkpoint(cfg))), cfg)
