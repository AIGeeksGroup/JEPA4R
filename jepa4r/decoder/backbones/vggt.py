"""VGGT [wang2025vggt] backbone for Tab. 4a.

PAGE-4D is VGGT's architecture re-trained on dynamic scenes, so ``VGGTDecoder`` is PAGE4DDecoder unchanged with
VGGT-1B's released weights (its point head: a point map in the camera frame).
"""
import torch

import jepa4r.decoder.backbones.page4d as p4
from jepa4r.decoder.backbones.page4d import PAGE4DDecoder, checkpoint_path


def load_vggt(weights):
    """PAGE-4D's model class (= VGGT plus a dynamics-mask head that the decoder drops) with VGGT-1B's released
    weights."""
    m = p4.PAGE4D(enable_camera=False, enable_depth=False, enable_track=False)
    sd = torch.load(checkpoint_path(weights), map_location='cpu', weights_only=False)
    sd = sd.get('model', sd)
    missing, _ = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if 'spatial_mask_head' not in k]
    assert not missing, f'VGGT weights missing: {missing[:5]}'
    return m


class VGGTDecoder(PAGE4DDecoder):
    def __init__(self, *args, weights='vggt/model.pt', **kwargs):
        loader, p4.load_page4d = p4.load_page4d, load_vggt  # PAGE4DDecoder.__init__ looks the loader up at call time
        try:
            super().__init__(*args, weights=weights, **kwargs)
        finally:
            p4.load_page4d = loader


def build(bb_cfg, **kwargs):
    """The decoder with this backbone (kwargs: jepa4r.decoder.model.decoder_kwargs)."""
    return VGGTDecoder(weights=str(bb_cfg.weights), **kwargs)
