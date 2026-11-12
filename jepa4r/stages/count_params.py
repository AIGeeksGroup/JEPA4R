"""Stage ``count_params`` -- parameter counts quoted in the paper (paper/expected/params.yaml).

method=M   encoder_params_M  encoder incl. its linear head (PointNet 5.51M vs ViT-tiny 5.50M, Sec. 4.2)
           total_params_M    encoder, projectors, predictor, action encoder (JEPA4R ~18.0M)
decoder=D  new_params_M      stage-A trainables: set generator, affine map, rgb branch (5.75M, Sec. 4.4)
           lora_params_M     stage-B LoRA adapters (0.88M)
           backbone_params_B frozen backbone (PAGE-4D ~0.94B)
Builds modules from their configs only: no data, no checkpoints.
"""


def run(cfg, out_dir):
    raise NotImplementedError(
        "TODO(integrate): build jepa4r.models.world_model / jepa4r.decoder.model from cfg and count parameters")
