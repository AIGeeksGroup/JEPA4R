"""Pluggable reconstruction decoders: the per-frame latent of a frozen world model -> explicit 3D geometry.

Contract (what ``recon.model.recon_forward`` expects from the decoder):

    forward(latent)          latent ``(B, embed_dim)`` -> the decoder's own output (a tensor or a dict)
    loss(output, target)     target ``(B, N, C)`` point cloud in the normalised target frame, C = 3 (xyz) or
                             6 (xyz + rgb in [0, 1]) -> ``(scalar loss, dict of detached terms to log)``
    to_points(output, n)     the output as a point cloud ``(B, n, C)``, so every decoder can be looked at and
                             compared the same way

A new decoder is a class in this package plus a yaml in ``config/recon/decoder/``; train with ``decoder=<name>``.
Nothing here depends on the scene: no object labels, no colour thresholds, colour itself is optional.
"""

from .gaussian import GaussianDecoder
from .point import PointDecoder
from .set_decoder import SetDecoder

__all__ = ['GaussianDecoder', 'PointDecoder', 'SetDecoder']
