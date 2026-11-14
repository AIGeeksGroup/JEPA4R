"""Checkpoint loading and encoding of the JEPA world models the Cube workflow trains (JEPA4R, LeWM), with LeWM's
loader."""

from pathlib import Path
import os
import sys

FAMILY, DIM = "jepa_wm", 192


def load_frozen(checkpoint, device="cuda"):
    code = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(code), str(code / "third_party/stable-worldmodel")]
    import hydra.utils
    from stable_worldmodel.wm.utils import load_pretrained
    from jepa4r.models.world_model import _retarget  # config.json's LeWM targets (jepa.JEPA, ...) -> our classes

    cache = os.environ.get("CUBE_WORKFLOW_FACTORY_CACHE")
    if not cache:
        raise ValueError("An explicit fresh workflow factory cache is required")
    instantiate = hydra.utils.instantiate
    hydra.utils.instantiate = lambda config, *args, **kwargs: instantiate(_retarget(config), *args, **kwargs)
    try:
        return load_pretrained(str(Path(checkpoint).resolve()), cache_dir=cache).to(device).eval().requires_grad_(False)
    finally:
        hydra.utils.instantiate = instantiate


def encode(model, inputs):
    import torch

    with torch.inference_mode():
        return model.encode({model.obs_key: inputs[:, None]})["emb"][:, 0]
