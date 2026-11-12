"""ViT encoders. ViT-tiny/14 on 224x224 images, trained from scratch, is the LeWM encoder (Sec. 3, 5.50M)
and the PLDM / LeJEPA encoder; ViT-B/16 is used for Random (ViT-B). Released VISReg / V-JEPA 2.1 weights are
loaded by their own modules in jepa4r/baselines.
"""


def build(enc_cfg):
    raise NotImplementedError("TODO(integrate): ViT (from LeWM / stable-worldmodel)")
