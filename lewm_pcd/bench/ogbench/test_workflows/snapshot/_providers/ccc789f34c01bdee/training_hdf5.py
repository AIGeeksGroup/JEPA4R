"""Opt-in local VDS reader; original slicing and model math remain upstream."""
import os
import operator

import h5py
import numpy as np
from double_cube_contract import validate_view_attributes
from stable_worldmodel.data.format import Format, register_format
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset


class ReadOnlyClipIndex:
    """Compact storage with the original scalar tuple and slice-list semantics."""

    def __init__(self, pairs):
        self._pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        self._pairs.setflags(write=False)

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [tuple(map(int, row)) for row in self._pairs[key]]
        ep, start = self._pairs[operator.index(key)]
        return int(ep), int(start)

    def __getstate__(self):
        return self._pairs

    def __setstate__(self, pairs):
        self.__init__(pairs)


class TrainingHDF5Dataset(HDF5Dataset):
    def __init__(self, *args, raw_chunk_cache_nbytes=1 << 20,
                 compact_clip_indices=False, expected_sha256=None, **kwargs):
        import hashlib
        from pathlib import Path
        path = Path(kwargs.get("path", args[0] if args else ""))
        if expected_sha256 is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("sealed DoubleCube VDS checksum differs")
        self.expected_sha256 = expected_sha256
        self.allow_fixture = False
        self.raw_chunk_cache_nbytes = int(raw_chunk_cache_nbytes)
        if self.raw_chunk_cache_nbytes <= 0:
            raise ValueError('raw_chunk_cache_nbytes must be positive')
        super().__init__(*args, **kwargs)
        self.compact_clip_indices = bool(compact_clip_indices)
        if self.compact_clip_indices:
            self.clip_indices = ReadOnlyClipIndex(self.clip_indices)

    def _open_h5(self):
        if self.is_remote:
            raise ValueError('Training views must reference sealed local HDF5 sources')
        handle = h5py.File(self.h5_path, 'r', swmr=True, rdcc_nbytes=self.raw_chunk_cache_nbytes)
        try:
            validate_view_attributes(handle, expected_sha256=self.expected_sha256)
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


@register_format
class TrainingHDF5(Format):
    name = 'hdf5_train_bounded'

    @classmethod
    def detect(cls, path):
        return False  # selected explicitly; never shadows the upstream reader

    @classmethod
    def open_reader(cls, path, **kwargs):
        return TrainingHDF5Dataset(path=path, **kwargs)
