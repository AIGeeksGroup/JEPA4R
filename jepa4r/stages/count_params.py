"""Stage ``count_params`` -- parameter counts quoted in the paper (paper/expected/params.yaml).

method=M   encoder_params_M  encoder incl. its linear head (PointNet 5.51M vs ViT-tiny 5.50M, Sec. 4.2)
           total_params_M    encoder, projectors, predictor, action encoder (JEPA4R ~18.0M)
decoder=D  new_params_M      stage-A trainables: set generator, affine map, rgb branch (5.75M, Sec. 4.4)
           lora_params_M     stage-B LoRA adapters (0.88M)
           backbone_params_B frozen backbone (PAGE-4D ~0.94B)
Builds modules from their configs only: no data, no checkpoints.

The world model is built with the action block of 5-D actions (OGBench; frameskip 5 -> 25-D) unless an env is
selected (env=E uses that task's action dimension, ``env.action_dim``); the action encoder's input layer is the only
part that depends on it (18,040,364 parameters with 25-D blocks, 18,040,314 with MetaWorld's 20-D blocks).
"""


def _count(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def run(cfg, out_dir):
    if "method" in cfg:
        from jepa4r.models.encoders import build_encoder
        from jepa4r.models.world_model import build_world_model

        if cfg.method.kind != "jepa_wm":
            encoder = build_encoder(cfg.method.encoder)
            return {"encoder_params_M": _count(encoder) / 1e6}
        action_dim = int(cfg.env.action_dim) if "env" in cfg and "action_dim" in cfg.env else 5
        model = build_world_model(cfg.method, action_dim)
        return {"encoder_params_M": _count(model.encoder) / 1e6, "total_params_M": _count(model) / 1e6}
    from jepa4r.decoder.model import count_parameters

    return count_parameters(cfg.decoder)
