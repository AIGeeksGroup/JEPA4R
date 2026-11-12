"""PLDM [sobal2025pldm]: trained on our image data for 10 epochs with its own recipe; ViT-tiny/14,
Impala on OGBench Cube.
"""


def train(cfg, out_dir) -> dict:
    raise NotImplementedError("TODO(integrate): PLDM training")


def load_frozen_encoder(cfg):
    raise NotImplementedError("TODO(integrate)")
