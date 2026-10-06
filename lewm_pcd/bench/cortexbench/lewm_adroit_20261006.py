"""2026-10-06 Pen loaders with the f1cc56 construction and RNG contract.

Reuse the existing embedding/encoder/transform classes; keep the historical
loader's checkpoint remapping and random-PCD API intact for older configs.
"""
import json
from pathlib import Path

import lewm_cortex as _base


def build_lewm_embedding(checkpoint_dir, epoch=10, use_projector=True, random_init=False, seed=0):
    ckdir = Path(checkpoint_dir)
    cfg = json.loads((ckdir / 'config.json').read_text())
    enc_cfg = {k: v for k, v in cfg['encoder'].items() if k != '_target_'}
    assert cfg['encoder']['_target_'] in ('encoders.ViTCLS', 'encoders.vit.ViTCLS'), cfg['encoder']['_target_']
    with _base.torch.random.fork_rng(devices=[]):
        _base.torch.manual_seed(seed)
        encoder = _base.ViTCLS(**enc_cfg)
        projector = _base.instantiate(cfg['projector']) if use_projector else None
    if not random_init:
        sd = _base.torch.load(ckdir / f'weights_epoch_{epoch}.pt', map_location='cpu')
        encoder.load_state_dict({k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')})
        if projector is not None:
            projector.load_state_dict({k[len('projector.'):]: v for k, v in sd.items() if k.startswith('projector.')})
    return _base.LeWMEmbedding(encoder, projector).eval()


def load_lewm(checkpoint_dir: str, epoch: int = 10, use_projector: bool = True,
              random_init: bool = False, seed: int = 0):
    ckdir = Path(checkpoint_dir)
    model = build_lewm_embedding(ckdir, epoch, use_projector, random_init, seed)
    with _base.torch.no_grad():
        dim = model(_base.torch.zeros(2, 3, 224, 224)).shape[-1]
    transforms = _base.T.Compose([
        _base.T.Resize((224, 224), interpolation=_base.T.InterpolationMode.BICUBIC, antialias=True),
        _base.T.ToTensor(),
        _base.T.Normalize(*_base.IMAGENET),
    ])
    metadata = {
        'algo': 'random' if random_init else 'lewm',
        'model': 'vit_tiny_patch14 (LeWM encoder + projector)',
        'data': [] if random_init else _base._train_data(ckdir),
        'checkpoint': None if random_init else str(ckdir / f'weights_epoch_{epoch}.pt'),
    }
    return model, int(dim), transforms, metadata


def load_lewm_pcd(checkpoint_dir: str, epoch: int = 10, center=(0.0, 0.65, 0.2),
                  scale: float = 4.0, use_rgb: bool = True):
    # The old pretrained branch is the f1cc56 implementation: construct once,
    # load strict weights, and retain the caller RNG consumption (no reseeding).
    return _base.load_lewm_pcd(checkpoint_dir, epoch, center, scale, use_rgb, random_init=False)
