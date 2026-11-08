"""Original LeWM local-checkpoint factory/encode API for fresh workflow weights."""
from pathlib import Path
import os
import sys

FAMILY, DIM = 'lewm', 192


def load_frozen(checkpoint, device='cuda'):
    code = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(code), str(code/'third_party/stable-worldmodel')]
    from stable_worldmodel.wm.utils import load_pretrained
    cache = os.environ.get('CUBE_WORKFLOW_FACTORY_CACHE')
    if not cache:
        raise ValueError('An explicit fresh workflow factory cache is required')
    return load_pretrained(str(Path(checkpoint).resolve()), cache_dir=cache).to(device).eval().requires_grad_(False)


def encode(model, inputs):
    import torch
    with torch.inference_mode():
        return model.encode({model.obs_key: inputs[:, None]})['emb'][:, 0]
