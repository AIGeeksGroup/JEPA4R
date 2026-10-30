"""Versioned PCD reader. Reuse the frozen compact-pair/sampler contract."""
from collections import OrderedDict
import hashlib
import importlib.util
from pathlib import Path
import sys

import torch

BASE_DATASET_SHA = '5c922fad193fbd6cffad58992504ce3107fbd4230d99741d15c4a5c77af07181'


def load_pinned(name, path, expected_sha):
    path = Path(path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('Frozen dependency source SHA differs: ' + str(path))
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise ValueError('Frozen dependency module path differs')
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_pinned('official_rgb_dataset_frozen_5c922', Path(__file__).with_name('dataset.py'), BASE_DATASET_SHA)
LogicalSampler = base.LogicalSampler
OfficialGoalSampler = base.OfficialGoalSampler
IdentityAction = base.IdentityAction


class _PCDSliceView:
    """The inherited pair checker uses 'pixels' as a generic observation alias.

    This alias contains the real processed point tensor, never RGB or features.
    """
    def __init__(self, reader):
        self.reader = reader

    def _load_slice(self, *args):
        result = self.reader._load_slice(*args)
        if not torch.equal(result['source_step'].reshape(-1).long(), torch.arange(args[1], args[2])):
            raise ValueError('PCD source local step differs from requested row')
        result['pixels'] = result['pcd']
        return result

    @property
    def h5_file(self):
        return self.reader.h5_file


class CubeOfficialPCDDataset(base.CubeOfficialDataset):
    def _reader(self, episode):
        from stable_worldmodel.data.formats.hdf5 import HDF5Dataset
        from gcbc_smoke import DataTransform
        if episode in self.readers:
            self.readers.move_to_end(episode)
            return self.readers[episode]
        reader = HDF5Dataset(path=self.entries[episode]['path'], num_steps=1, frameskip=1,
            keys_to_load=['pcd_xyz', 'pcd_rgb', 'action', 'source_row', 'source_episode', 'source_step', 'source_terminal'],
            transform=DataTransform('pcd', self.preprocessor, IdentityAction()))
        if reader.lengths.tolist() != [self.rows] or reader.offsets.tolist() != [0]:
            raise ValueError('Original PCD local episode bounds differ')
        view = _PCDSliceView(reader)
        self.readers[episode] = view
        if len(self.readers) > self.open_files:
            _, retired = self.readers.popitem(last=False)
            if retired.h5_file is not None:
                retired.h5_file.close()
        return view

    def __getitem__(self, pair):
        result = super().__getitem__(pair)
        for key in ('current', 'goal'):
            value = result[key]
            if value.shape != (1200, 6) or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError('Real PCD single-frame1200x6 contract invalid')
        if (result['action'].abs() > 1).any():
            raise ValueError('Published normalized Box action outside[-1,1]')
        return result
