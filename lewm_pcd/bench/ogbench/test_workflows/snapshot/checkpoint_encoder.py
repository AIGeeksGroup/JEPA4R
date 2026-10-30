"""Reuse the actual LeWM factory and preprocessing; no dynamics or GCBC head."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def source_paths():
    code = ROOT / 'source/lewm_pcd'
    sys.path[:0] = [str(code), str(code/'third_party/stable-worldmodel')]


def load_frozen(checkpoint, device='cuda'):
    source_paths()
    from stable_worldmodel.wm.utils import load_pretrained
    model = load_pretrained(str(Path(checkpoint).resolve()), cache_dir=str(ROOT/'factory_cache'))
    return model.to(device).eval().requires_grad_(False)


def preprocess(batch, modality):
    source_paths()
    from utils import get_img_preprocessor, get_pcd_preprocessor
    if modality == '2d':
        batch = dict(batch)
        if batch['pixels'].ndim == 4 and batch['pixels'].shape[-1] == 3:
            batch['pixels'] = batch['pixels'].permute(0, 3, 1, 2)
        return get_img_preprocessor('pixels', 'pixels', 224)(batch)
    if modality == '3d':
        return get_pcd_preprocessor(center=(.425, 0., .2), scale=4., use_rgb=True)(dict(batch))
    raise ValueError(modality)


def encode(model, inputs):
    import torch
    with torch.no_grad():
        info = model.encode({model.obs_key: inputs[:, None]})
    output = info['emb'][:, 0]
    if output.shape != (len(inputs), 192) or not torch.isfinite(output).all():
        raise ValueError('Invalid frozen LeWM feature output')
    return output
