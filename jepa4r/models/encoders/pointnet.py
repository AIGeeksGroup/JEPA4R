"""PointNet encoder, JEPA4R's default (Sec. 4.2): the DexArt [bao2023dexart] variant widened to match ViT-tiny.

shared per-point MLP 6 -> 1,080 -> 4,320 (GELU) -> order-invariant max-pool -> Linear(4,320 -> 192)
in  [B, 1200, 6] (normalised xyz + rgb)    out [B, 192] (the LeWM projector follows)
Parameters: 6*1080+1080 + 1080*4320+4320 + 4320*192+192 = 5,507,112 (+ norms, if any) ~ 5.51M.
"""


def build(enc_cfg):
    raise NotImplementedError("TODO(integrate): PointNet encoder")
