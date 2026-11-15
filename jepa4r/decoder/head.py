"""Output head (Sec. 4.4): a learnable affine map moves the backbone's 224x224 point map into the target frame,
a new branch adds rgb, and 1,200 pixels form P_hat_t (selection rule: cfg.decoder.head.pixel_selection).

The head is built inside each backbone decoder (jepa4r/decoder/backbones: ``frame`` = the affine map, ``rgb`` = the
colour branch, ``_pick`` = the pixel selection: a fresh random subset of the dense map while training, a fixed one --
torch.randperm with a seed-0 generator -- otherwise). The affine map starts as a virtual camera looking at the
workspace: ``make_cam`` below.
"""

import numpy as np
import torch


BG = (0.09, 0.10, 0.12)


def make_cam(az_deg=45.0, el_deg=40.0, dist=1.2, target=(0.36, 0.0, 0.2), fovy_deg=45.0, size=320, device="cuda"):
    """World -> camera rotation / translation and intrinsics (OpenCV axes: x right, y down, z forward)."""
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    pos = np.asarray(target, float) + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    fwd = np.asarray(target, float) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(fwd, right), fwd])
    f = 0.5 * size / np.tan(np.deg2rad(fovy_deg) / 2)
    return dict(
        R=torch.tensor(R, dtype=torch.float32, device=device),
        t=torch.tensor(-R @ pos, dtype=torch.float32, device=device),
        f=f,
        c=size / 2,
        size=size,
        device=device,
    )
