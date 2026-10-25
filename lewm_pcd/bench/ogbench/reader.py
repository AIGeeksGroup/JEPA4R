"""One bounded reader over unchanged stable-worldmodel scalar HDF5 slicing."""
import os
import operator
from pathlib import Path
import sys

import numpy as np

from .contracts import _validate_open_file, _view_identity


class ReadOnlyClipIndex:
    """Original compact storage, scalar tuples and slice-list semantics."""
    def __init__(self, pairs):
        self._pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        self._pairs.setflags(write=False)

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [tuple(map(int, row)) for row in self._pairs[key]]
        episode, start = self._pairs[operator.index(key)]
        return int(episode), int(start)

    def __getstate__(self):
        return self._pairs

    def __setstate__(self, pairs):
        self.__init__(pairs)


def reader_class():
    """Load the sole implementation only when a tensor reader is required."""
    if 'TrainingHDF5Dataset' in globals():
        return globals()['TrainingHDF5Dataset']
    swm = Path(__file__).resolve().parents[2] / 'third_party/stable-worldmodel'
    # Resolve the bundled canonical source, rejecting a preloaded other reader.
    sys.path.insert(0, str(swm))
    import h5py
    from stable_worldmodel.data.formats import hdf5
    if Path(hdf5.__file__).resolve() != swm / 'stable_worldmodel/data/formats/hdf5.py':
        raise ValueError('another stable-worldmodel HDF5 implementation was imported')

    class TrainingHDF5Dataset(hdf5.HDF5Dataset):
        def __init__(self, path, profile, split, num_steps, expected_sha256=None, *,
                     lewm_mode=None, data_root=None, transform=None, cache_dir=None,
                     raw_chunk_cache_nbytes=1 << 20, compact_clip_indices=True):
            self.lewm_mode = lewm_mode
            self.data_root = data_root
            if lewm_mode is None:
                path, selected, expected = _view_identity(path, profile, split, expected_sha256)
            else:
                from . import lewm_data
                path, selected, expected, _ = lewm_data._view_identity(
                    path, profile, lewm_mode, split, data_root, expected_sha256)
            self.profile = profile
            self.split = split
            self._profile = selected
            self.expected_sha256 = expected
            self.raw_chunk_cache_nbytes = int(raw_chunk_cache_nbytes)
            if self.raw_chunk_cache_nbytes <= 0:
                raise ValueError('raw_chunk_cache_nbytes must be positive')
            self.compact_clip_indices = compact_clip_indices
            keys = (['pixels'] if num_steps == 1 else ['pixels', 'action']) if lewm_mode is None else selected['keys_to_load']
            super().__init__(path=path, num_steps=num_steps, frameskip=1 if lewm_mode is None else 5,
                             keys_to_load=keys,
                             keys_to_cache=[] if num_steps == 1 else ['action'],
                             transform=transform, cache_dir=cache_dir)
            self.clip_indices = ReadOnlyClipIndex(self.clip_indices)

        def _open_h5(self):
            if self.is_remote:
                raise ValueError('training views require sealed local HDF5 sources')
            # The original producer validators run again on every worker reopen.
            if self.lewm_mode is None:
                _view_identity(self.h5_path, self.profile, self.split, self.expected_sha256)
            else:
                from . import lewm_data
                _, _, _, manifest = lewm_data._view_identity(
                    self.h5_path, self.profile, self.lewm_mode, self.split,
                    self.data_root, self.expected_sha256)
                lewm_data._check_sources(manifest)
            handle = h5py.File(self.h5_path, 'r', swmr=True,
                               rdcc_nbytes=self.raw_chunk_cache_nbytes)
            try:
                if self.lewm_mode is None:
                    _validate_open_file(handle, self._profile, self.expected_sha256)
                else:
                    lewm_data._validate_open_file(handle, self._profile, self.split,
                                                  self.expected_sha256, manifest)
            except Exception:
                handle.close()
                raise
            return handle

        def _open(self):
            pid = os.getpid()
            if self.h5_file is not None and getattr(self, '_owner_pid', None) != pid:
                self.close()
            super()._open()
            self._owner_pid = pid

        def close(self):
            handle = getattr(self, 'h5_file', None)
            if handle is not None:
                handle.close()
            self.h5_file = None
            self._owner_pid = None

        def __del__(self):
            self.close()

    # Keep multiprocessing spawn/pickle identity stable despite lazy imports.
    TrainingHDF5Dataset.__qualname__ = 'TrainingHDF5Dataset'
    globals()['TrainingHDF5Dataset'] = TrainingHDF5Dataset
    return TrainingHDF5Dataset


def make_reader(path, profile, split, num_steps, expected_sha256=None):
    """Open an accepted production split using the shared scalar SWM reader.

    ``num_steps=16`` selects PLDM RGB/action clips and the original action
    cache. ``num_steps=1`` selects RGB-only Tiny inputs without an action cache.
    Both use frameskip one, terminal-free views, and compact clip indices.
    """
    if type(num_steps) is not int or num_steps not in (1, 16):
        raise ValueError('supported num_steps are 1 (Tiny) and 16 (PLDM)')
    # Fail closed on provenance before importing tensor or simulation libraries.
    _view_identity(path, profile, split, expected_sha256)
    return reader_class()(path, profile, split, num_steps, expected_sha256)


def __getattr__(name):
    # Python's spawn unpickler imports this module in a fresh worker before
    # resolving the dataset class; recreate the same lazy class on demand.
    if name == 'TrainingHDF5Dataset':
        return reader_class()
    raise AttributeError(name)
