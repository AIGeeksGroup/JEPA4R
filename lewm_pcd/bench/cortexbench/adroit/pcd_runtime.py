"""Runtime adapter for Adroit Pen point-cloud observations in official BC."""

from __future__ import annotations

import numpy as np

from .pcd import AdroitPCDConfig
from .smoke import capture_adroit_frame


def seed_adroit_pcd(wrapper, seed=None):
    """Seed both the native Adroit environment and the PCD sampling RNG."""
    wrapper._adroit_pcd_seed = seed
    wrapper._adroit_pcd_rng = np.random.default_rng(seed)
    return wrapper.env.seed(seed)


def get_adroit_pcd(wrapper, *, capture_fn=capture_adroit_frame, config=None):
    """Return the current Pen frame as float32 XYZRGB with 1200 points."""
    if getattr(getattr(wrapper, "spec", None), "id", None) != "pen-v0":
        raise RuntimeError("Adroit PCD observations are implemented for pen-v0 only")
    if config is None:
        config = AdroitPCDConfig()
    rng = getattr(wrapper, "_adroit_pcd_rng", None)
    if rng is None:
        rng = np.random.default_rng(0)
        wrapper._adroit_pcd_rng = rng
    _, xyz, rgb, _ = capture_fn(wrapper.env, rng, config)
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    expected = (config.num_points, 3)
    if xyz.shape != expected or rgb.shape != expected:
        raise ValueError(f"invalid Adroit PCD frame shape: xyz={xyz.shape}, rgb={rgb.shape}")
    if not np.isfinite(xyz).all():
        raise ValueError("Adroit PCD frame contains NaN or Inf")
    return np.concatenate([xyz, rgb.astype(np.float32)], axis=1)
