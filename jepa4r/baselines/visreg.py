"""VISReg [wu2026visreg]: the authors' released ViT-B/16 weights (cfg.method.weights), frozen.

timm's vit_base_patch16_224, the CLS after the final norm, 768-d, input 224 with ImageNet normalisation:
jepa4r/policy/bc/jepa_baselines.py load_visreg (cortexbench@9331494, as CortexBench and DQC read it).
"""


def load_frozen_encoder(cfg):
    from jepa4r.baselines import frozen
    from jepa4r.policy.bc.jepa_baselines import load_visreg
    from jepa4r.policy.frozen import checkpoint

    return frozen(load_visreg(str(checkpoint(cfg))), cfg)
