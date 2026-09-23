"""The reconstruction decoder: the per-frame latent of a frozen world model -> explicit 3D geometry.

Contract (what ``recon.model.recon_forward`` expects from the decoder):

    forward(latent)          latent ``(B, embed_dim)`` -> the decoder's own output (a tensor or a dict)
    loss(output, target)     target ``(B, N, C)`` point cloud in the normalised target frame, C = 3 (xyz) or
                             6 (xyz + rgb in [0, 1]) -> ``(scalar loss, dict of detached terms to log)``
    to_points(output, n)     the output as a point cloud ``(B, n, C)``, so every decoder can be looked at and
                             compared the same way

``PAGE4DDecoder`` (``page4d.py``, ``config/recon/decoder/page4d.yaml``): a token grid generated from the latent
(``SetDecoder``) through PAGE-4D's pretrained, frozen backbone and point head, with small LoRA adapters (``lora.py``)
in stage B. Nothing here depends on the scene: no object labels, no colour thresholds, colour itself is optional.
"""

from .page4d import PAGE4DDecoder
from .set_decoder import SetDecoder

__all__ = ['PAGE4DDecoder', 'SetDecoder']
