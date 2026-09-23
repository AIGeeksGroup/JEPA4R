"""The frozen world model as ``observation -> latent1``, and the training forward of the decoders."""

from pathlib import Path

import hydra
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from torch import nn

from recon.losses import nn_distance


class FrozenLatent(nn.Module):
    """A trained world model reduced to its per-frame latent. Never trains, never leaves eval mode (the
    projector's BatchNorm statistics and every dropout stay as trained) -- also when Lightning calls ``train()``."""

    def __init__(self, world_model):
        super().__init__()
        assert hasattr(world_model, 'obs_key') and hasattr(world_model, 'encode'), \
            f'{type(world_model).__name__} is not a jepa.JEPA world model'
        self.world_model = world_model.requires_grad_(False).eval()
        self.obs_key = world_model.obs_key

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, obs):
        """obs (B, T, ...) -> latent1 (B, T, D); ``JEPA.encode`` for 2D / 3D, inherited unchanged by the 4D model."""
        return self.world_model.encode({self.obs_key: obs})['emb']


def load_world_model(ckpt):
    return FrozenLatent(swm.wm.utils.load_pretrained(ckpt))


def input_pipeline(cfg):
    """Columns and preprocessor that feed the encoder. By default those the checkpoint was trained with, read
    from its run's ``config.yaml`` (pixels, a point-cloud affine, Sonata's identity affine, ...);
    ``input_keys`` / ``input_preprocessor`` in the config override them."""
    keys, preprocessor = cfg.get('input_keys'), cfg.get('input_preprocessor')
    if keys is None or preprocessor is None:
        ckpt = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.ckpt)
        run_cfg = (ckpt.parent if ckpt.suffix == '.pt' else ckpt) / 'config.yaml'
        assert run_cfg.exists(), f'{run_cfg} not found: set input_keys and input_preprocessor explicitly'
        data = OmegaConf.load(run_cfg).data
        if keys is None:  # observation columns, by the rule train.py uses
            keys = [k for k in data.dataset.keys_to_load if k.startswith('pixels') or k.startswith('pcd')]
        if preprocessor is None:
            preprocessor = data.preprocessor
    return list(keys), hydra.utils.instantiate(preprocessor)


def load_recon(name):
    """A trained reconstruction run -> (frozen world model, decoder, the run's config).
    ``name``: ``<run>`` (its last epoch) or ``<run>/weights_epoch_N.pt``, under ``$STABLEWM_HOME/checkpoints/recon/``."""
    path = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), 'recon', name)
    if path.suffix != '.pt':
        weights = sorted(path.glob('weights_epoch_*.pt'), key=lambda w: int(w.stem.rsplit('_', 1)[1]))
        assert weights, f'no weights_epoch_N.pt in {path}'
        path = weights[-1]
    cfg = OmegaConf.load(path.parent / 'config.yaml')
    decoder = swm.wm.utils.load_pretrained(str(Path('recon', path.parent.name, path.name))).eval().requires_grad_(False)
    return load_world_model(cfg.ckpt), decoder, cfg


def open_run(name, input=None, target=None, device='cuda'):
    """Everything the evaluation scripts need from a trained run: (world model, decoder, config, ``Episodes``).
    ``input`` / ``target`` replace the run's training datasets, e.g. by other trajectories -- for a pixel checkpoint
    the pixel dataset and the point clouds of the same rows, checked exactly as in training."""
    from recon.data import Episodes, build_dataset

    world_model, decoder, cfg = load_recon(name)
    if input is not None:
        cfg.data.input = input
    if target is not None:
        cfg.data.target = target
    input_keys, input_preprocessor = input_pipeline(cfg)
    frames = build_dataset(cfg.data, world_model.obs_key, input_keys, input_preprocessor)
    return world_model.to(device), decoder.to(device), cfg, Episodes(frames)


def recon_forward(self, batch, stage, cfg):
    """encode the frame with the frozen world model, decode its latent, compare with the target point cloud."""
    latent = self.world_model(batch[self.world_model.obs_key])[:, 0]   # (B, D)
    target = batch['target'][:, 0]                                      # (B, N, C)

    decoded = self.model(latent)
    loss, terms = self.model.loss(decoded, target)
    output = {'loss': loss, **terms}
    if stage != 'fit':  # the same readout for every decoder: two-sided nearest-neighbour distance, in cm
        points = self.model.to_points(decoded, target.size(1))
        output['nn_cm'] = nn_distance(points, target).mean() / cfg.data.target_preprocessor.scale * 100

    self.log_dict({f'{stage}/{k}': v.detach() for k, v in output.items()}, on_step=True, sync_dist=True)
    return output
