"""Decoder training (stage train_decoder), linear-probe-then-fine-tune order [kumar2022lpft]:

stage A  train the new parameters (set generator, affine map, rgb branch; 5.75M), backbone frozen
stage B  add rank-4 LoRA on q / v of all attention blocks and continue
Targets: observed clouds of data/wm/<env>; inputs: latents of the frozen world model cfg.decoder.wm (epoch 10).
Returns {"val_nn_stage_a": cm, "val_nn_stage_b": cm} (paper: 0.87 -> 0.79).
"""


def train(cfg, out_dir) -> dict:
    raise NotImplementedError("TODO(integrate): two-stage decoder training")
