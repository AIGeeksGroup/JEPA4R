"""Thin raw-row / frozen-LeWM boundary around pinned official GCIQL."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

OBS_FIELDS = ('observations', 'next_observations', 'value_goals', 'actor_goals')
AGENT_FIELDS = OBS_FIELDS + ('actions', 'rewards', 'masks')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def setup(config):
    """Pin namespace resolution; never replace the official utils package."""
    w = Path(config['workspace'])
    official = w / config['official']
    pcd = w / config['repo'] / 'lewm_pcd'
    for path in reversed([official / 'impls', official, pcd, pcd / 'third_party/stable-worldmodel']):
        sys.path.insert(0, str(path))
    result = {}
    for rel, expected in config['locked_sha256'].items():
        actual = sha256(w / rel)
        if actual != expected:
            raise ValueError(f'Source/data lock mismatch: {rel}: {actual}')
        result[rel] = actual
    for rel, expected in config['heads'].items():
        actual = subprocess.check_output(['git', '-C', str(w / rel), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != expected:
            raise ValueError(f'HEAD mismatch: {rel}')
    from utils import datasets
    if Path(datasets.__file__).resolve() != (official / 'impls/utils/datasets.py').resolve():
        raise ValueError(f'Wrong official utils: {datasets.__file__}')
    return result


def load_alias(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preprocess_rows(preprocessor, rows, input_key):
    import torch
    inputs = {k: torch.from_numpy(v) for k, v in rows.items()}
    if 'pixels' in inputs:
        # Same channel conversion as the actual stable HDF5Dataset loader.
        inputs['pixels'] = inputs['pixels'].permute(0, 3, 1, 2)
    return preprocessor(inputs)[input_key]


def agent_config(batch_size):
    from agents.gciql import get_config
    cfg = get_config()
    cfg.update(dict(alpha=1.0, encoder=None, frame_stack=None, p_aug=0.0, batch_size=batch_size))
    return cfg.to_dict()


def load_compact(config, split='train'):
    from ogbench.utils import load_dataset
    suffix = '-val' if split == 'val' else ''
    path = Path(config['workspace']) / config['raw'] / f'cube-single-play-v0{suffix}.npz'
    return load_dataset(str(path), compact_dataset=True)


def make_sampler(compact, cfg, ids=False):
    from utils.datasets import Dataset, GCDataset
    fields = dict(compact)
    if ids:
        fields['observations'] = np.arange(len(compact['observations']), dtype=np.int64)[:, None]
    return GCDataset(Dataset.create(**fields), cfg)


def sample_ids(sampler, batch_size, idxs=None):
    if idxs is not None and np.any(sampler.dataset['valids'][idxs] <= 0):
        raise ValueError('Raw terminal cannot be a current/action row')
    return sampler.sample(batch_size, idxs=idxs)


class RawRows:
    """Reads only policy modality and provenance; IDs remain reader addresses."""
    def __init__(self, config, modality, compact, split='train'):
        self.modality, self.compact = modality, compact
        index = Path(config['workspace']) / config['routes'][modality]['index']
        self.entries = {int(x['episode']): x for x in json.loads(index.read_text())[split]}
        self.starts = np.r_[0, np.flatnonzero(compact['valids'] == 0)[:-1] + 1]
        self.split = split

    def read(self, ids, role):
        import h5py
        import hdf5plugin  # Registers the real shard compression filters.
        ids = np.asarray(ids)
        if ids.dtype != np.int64 or ids.ndim != 2 or ids.shape[1] != 1:
            raise ValueError('Reader IDs must be int64 (B,1)')
        ids = ids[:, 0]
        if np.any(ids < 0) or np.any(ids >= len(self.compact['valids'])):
            raise ValueError('Raw ID outside source')
        if role != 'next_observations' and np.any(self.compact['valids'][ids] <= 0):
            raise ValueError('Raw terminal allowed only as next observation')
        columns = ('pixels',) if self.modality == 'rgb' else ('pcd_xyz', 'pcd_rgb')
        rows = {k: [] for k in columns}
        # Individual reads preserve order and duplicates; no feature cache.
        for row in ids:
            episode = int(np.searchsorted(self.starts, row, side='right') - 1)
            step = int(row - self.starts[episode])
            with h5py.File(self.entries[episode]['path'], 'r') as f:
                for key, expected in [('source_episode', episode), ('source_step', step), ('source_row', row),
                                      ('source_terminal', self.compact['valids'][row] == 0)]:
                    if f[key][step] != expected:
                        raise ValueError(f'{key} mismatch at raw row {row}')
                if role == 'observations':
                    np.testing.assert_array_equal(f['action'][step], self.compact['actions'][row])
                    if not np.isfinite(f['action'][step]).all():
                        raise ValueError('Nonfinite executed current action')
                for key in columns:
                    rows[key].append(f[key][step])
        rows = {k: np.stack(v) for k, v in rows.items()}
        for k, value in rows.items():
            shape = (224, 224, 3) if k == 'pixels' else (1200, 3)
            dtype = np.float32 if k == 'pcd_xyz' else np.uint8
            if value.shape[1:] != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError(f'Invalid real modality: {k} {value.shape} {value.dtype}')
        return rows


class FrozenLatents:
    def __init__(self, config, modality, reader, device='cuda'):
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        w = Path(config['workspace'])
        checkpoint = w / config['routes'][modality]['checkpoint']
        self.world_model = instantiate(json.loads((checkpoint.parent / 'config.json').read_text()))
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if not all(torch.isfinite(x).all() for x in weights.values()):
            raise ValueError('Nonfinite full WM checkpoint')
        self.world_model.load_state_dict(weights, strict=True)
        self.world_model.requires_grad_(False).eval().to(device)
        helper = load_alias('frozen_lewm_gciql_helper', w / config['helper'])
        self.encoder = helper.LeWMFrozenEncoder(world_model=self.world_model).to(device)
        preproc = load_alias('frozen_lewm_preprocessing', w / config['repo'] / 'lewm_pcd/utils.py')
        yaml = OmegaConf.load(checkpoint.parent / 'config.yaml')
        kwargs = dict(yaml.data.preprocessor)
        target = kwargs.pop('_target_').split('.')[-1]
        if modality == 'rgb':
            kwargs['img_size'] = int(yaml.img_size)
        self.preprocessor = getattr(preproc, target)(**kwargs)
        self.reader, self.device = reader, device
        self.input_key = self.world_model.obs_key
        self.sources = {name: sys.modules[name].__file__ for name in ['utils', 'jepa', 'module', 'encoders']}
        self.sources.update(preprocessing=preproc.__file__, helper=helper.__file__)

    def inputs(self, ids, role):
        rows = self.reader.read(ids, role)
        return preprocess_rows(self.preprocessor, rows, self.input_key).to(self.device)

    def encode(self, ids, role):
        return self.encoder(self.inputs(ids, role)).logits.cpu().numpy().astype(np.float32, copy=False)

    def verify_direct(self, ids, role):
        import torch
        inputs = self.inputs(ids, role)
        with torch.no_grad():
            direct = self.world_model.projector(self.world_model.encoder(inputs.float()))
            adapted = self.encoder(inputs).logits
        if direct.shape != (len(ids), 192) or not torch.equal(direct, adapted):
            raise ValueError('Actual WM encoder+projector / existing adapter parity failed')
        return dict(batch_size=len(ids), shape=list(direct.shape), max_abs_error=0.0)

    def batch(self, raw_batch):
        result = {k: raw_batch[k] for k in AGENT_FIELDS}
        for key in OBS_FIELDS:
            result[key] = self.encode(raw_batch[key], key)
        return result

    def snapshot(self):
        return {k: v.detach().cpu().clone() for k, v in self.world_model.state_dict().items()}

    def check_frozen(self, before):
        import torch
        now = self.world_model.state_dict()
        def identical(a, b):
            b = b.detach().cpu()
            return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
                a.reshape(-1).contiguous().view(torch.uint8), b.reshape(-1).contiguous().view(torch.uint8))
        if set(before) != set(now) or not all(identical(before[k], now[k]) for k in before):
            raise ValueError('Frozen full WM parameters/buffers changed')
        if any(p.requires_grad or p.grad is not None for p in self.world_model.parameters()):
            raise ValueError('Frozen WM gradient violation')


def policy_batch(batch):
    return {k: batch[k] for k in AGENT_FIELDS}
