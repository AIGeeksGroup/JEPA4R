import os
from pathlib import Path

import numpy as np
import pytest
import torch

from lewm_cortex import load_lewm_pcd


CHECKPOINT = Path(os.environ.get(
    "ADROIT_PCD_CHECKPOINT",
    "/root/autodl-tmp/workspace/users/lewm/adroit_full/pen_official_10000_150_20260927/"
    "training_full_v1/home/checkpoints/adroit_pen_full_v1",
))


def test_adroit_epoch10_pcd_checkpoint_reloads_with_its_preprocessor():
    if not (CHECKPOINT / "weights_epoch_10.pt").exists():
        pytest.skip(f"Adroit epoch-10 PCD checkpoint is unavailable: {CHECKPOINT}")
    model, dim, transform, metadata = load_lewm_pcd(
        CHECKPOINT,
        epoch=10,
        center=(0.075, -0.02, 0.225),
        scale=4.0,
        use_rgb=True,
    )
    frame = np.zeros((1200, 6), dtype=np.float32)
    encoded = transform(frame).unsqueeze(0)
    with torch.no_grad():
        latent = model(encoded)
    assert list(encoded.shape) == [1, 1200, 6]
    assert list(latent.shape) == [1, dim]
    assert torch.isfinite(latent).all()
    assert metadata["checkpoint"].endswith("weights_epoch_10.pt")
