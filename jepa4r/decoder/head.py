"""Output head (Sec. 4.4): a learnable affine map moves the backbone's 224x224 point map into the target frame,
a new branch adds rgb, and 1,200 pixels form P_hat_t (selection rule: cfg.decoder.head.pixel_selection).
"""


def build_head(decoder_cfg):
    raise NotImplementedError("TODO(integrate)")
