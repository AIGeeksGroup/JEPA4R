"""Source/atomic RGB episode primitives; imports no simulator or encoder."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

import h5py
import hdf5plugin
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(data, handle, indent=2, allow_nan=False); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def extract_numeric_source(receipt, destination):
    """Lossless original numeric NPY cache; no rendered or encoder features."""
    destination = Path(destination)
    if sha(receipt['path']) != receipt['archive_sha256']: raise ValueError('Official archive SHA differs')
    destination.mkdir(parents=True, exist_ok=True)
    mapping = {'observations': 'observation', 'actions': 'action', 'qpos': 'qpos', 'qvel': 'qvel', 'terminals': 'terminals'}
    manifest = {}
    with zipfile.ZipFile(receipt['path']) as archive:
        for original, target in mapping.items():
            path = destination / (target + '.npy')
            expected = receipt['fields'][original + '.npy']['uncompressed_npy_sha256']
            if not path.exists():
                temporary = path.with_suffix('.partial.npy')
                with archive.open(original + '.npy') as source, temporary.open('wb') as output:
                    for block in iter(lambda: source.read(1024 * 1024), b''): output.write(block)
                    output.flush(); os.fsync(output.fileno())
                if sha(temporary) != expected: raise ValueError('Original NPY member SHA differs')
                os.link(temporary, path); temporary.unlink()
            if sha(path) != expected: raise ValueError('Existing numeric cache SHA differs')
            manifest[target] = {'path': str(path.resolve()), 'sha256': expected}
    atomic_json(destination / 'source_cache.json', {'source_npz_sha256': receipt['archive_sha256'], 'fields': manifest,
                                                   'kind': 'original numeric arrays only; no features'})
    return {key: np.load(value['path'], mmap_mode='r', allow_pickle=False) for key, value in manifest.items()}


def episode_fields(raw, episode, expected_rows=1001):
    terminals = raw['terminals']
    ends = np.flatnonzero(terminals)
    starts = np.r_[0, ends[:-1] + 1]
    if episode < 0 or episode >= len(ends): raise ValueError('Source episode outside split')
    first, last = int(starts[episode]), int(ends[episode] + 1)
    if last - first != expected_rows: raise ValueError('Source episode length differs')
    fields = {key: np.array(raw[key][first:last], copy=True) for key in ('observation', 'action', 'qpos', 'qvel', 'terminals')}
    for key, dim in (('observation', 28), ('action', 5), ('qpos', 21), ('qvel', 20)):
        if fields[key].shape != (expected_rows, dim) or fields[key].dtype != np.float32 or not np.isfinite(fields[key]).all():
            raise ValueError('Original numeric contract differs: ' + key)
    if fields['terminals'].dtype != np.bool_ or np.flatnonzero(fields['terminals']).tolist() != [expected_rows - 1]:
        raise ValueError('Original episode terminal boundary differs')
    fields.update(source_row=np.arange(first, last, dtype=np.int64), source_episode=np.full(expected_rows, episode, np.int64),
                  source_step=np.arange(expected_rows, dtype=np.int64), source_terminal=fields['terminals'].copy(),
                  valid_transition=~fields['terminals'], id=np.arange(first, last, dtype=np.int64),
                  ep_idx=np.full(expected_rows, episode, np.int64), step_idx=np.arange(expected_rows, dtype=np.int64))
    return fields


class RGBEpisodeWriter:
    """Exclusive final names, chunk-aligned block writes, committed-prefix resume."""
    def __init__(self, final, fields, protocol_sha256, rgb_shape=(224, 224, 3), chunk_rows=10, attrs=None):
        self.final = Path(final); self.partial = self.final.with_suffix('.partial.h5')
        self.marker = self.final.with_suffix('.prefix.json')
        self.fields, self.n = fields, len(fields['action'])
        self.rgb_shape, self.chunk_rows = tuple(rgb_shape), chunk_rows
        self.protocol = protocol_sha256
        if self.final.exists(): raise FileExistsError(self.final)
        self.final.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.partial, 'r+' if self.partial.exists() else 'x')
        self.digest = hashlib.sha256()
        try:
            if 'pixels' not in self.f:
                for key, values in fields.items():
                    self.f.create_dataset(key, data=values, chunks=(min(chunk_rows, self.n), *values.shape[1:]),
                                          shuffle=True, **dict(hdf5plugin.Zstd(clevel=9)))
                self.f.create_dataset('pixels', shape=(self.n, *self.rgb_shape), dtype='u1',
                                      chunks=(min(chunk_rows, self.n), *self.rgb_shape), shuffle=True, **dict(hdf5plugin.Zstd(clevel=9)))
                self.f['ep_len'] = np.array([self.n], np.int32); self.f['ep_offset'] = np.array([0], np.int64)
                self.f.attrs.update(status='building', completed_rows=0, protocol_sha256=protocol_sha256,
                                    rgb_offsamples=4, modality='pixels_only', pcd_produced=False, **(attrs or {}))
                self._durable()
                atomic_json(self.marker, {'completed_rows': 0, 'rgb_prefix_sha256': self.digest.hexdigest(), 'protocol_sha256': self.protocol})
            if self.f.attrs.get('protocol_sha256') != self.protocol: raise ValueError('Partial protocol differs')
            if not np.array_equal(self.f['ep_len'][:], [self.n]) or not np.array_equal(self.f['ep_offset'][:], [0]):
                raise ValueError('Partial local episode offsets differ')
            if any(self.f.attrs.get(key) != value for key, value in (attrs or {}).items()): raise ValueError('Partial source/recipe attrs differ')
            for key, expected in fields.items():
                if self.f[key].dtype != expected.dtype or not np.array_equal(self.f[key][:], expected): raise ValueError('Partial source differs: ' + key)
            if self.f['pixels'].shape != (self.n, *self.rgb_shape) or self.f['pixels'].chunks[0] != min(self.chunk_rows, self.n):
                raise ValueError('Partial image/chunk shape differs')
            if not self.marker.exists() and int(self.f.attrs.get('completed_rows', -1)) == 0:
                atomic_json(self.marker, {'completed_rows': 0, 'rgb_prefix_sha256': self.digest.hexdigest(), 'protocol_sha256': self.protocol})
            committed = json.loads(self.marker.read_text())
            self.completed = int(committed['completed_rows'])
            if committed['protocol_sha256'] != self.protocol or not 0 <= self.completed <= self.n: raise ValueError('Invalid committed prefix')
            for row in range(0, self.completed, chunk_rows): self.digest.update(self.f['pixels'][row:min(row + chunk_rows, self.completed)].tobytes())
            if self.digest.hexdigest() != committed['rgb_prefix_sha256']: raise ValueError('Committed RGB prefix differs')
        except Exception:
            self.f.close(); raise

    def _durable(self):
        self.f.flush()
        with self.partial.open('rb') as handle: os.fsync(handle.fileno())

    def append(self, pixels):
        if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[1:] != self.rgb_shape or not 0 < len(pixels) <= self.chunk_rows:
            raise ValueError('RGB block shape/dtype/length differs')
        end = self.completed + len(pixels)
        if end > self.n: raise ValueError('RGB block exceeds original episode')
        if len(pixels) != min(self.chunk_rows, self.n - self.completed): raise ValueError('Nonfinal RGB writes must fill a chunk block')
        self.f['pixels'][self.completed:end] = pixels
        self.digest.update(pixels.tobytes()); self._durable()
        atomic_json(self.marker, {'completed_rows': end, 'rgb_prefix_sha256': self.digest.hexdigest(), 'protocol_sha256': self.protocol})
        self.completed = end
        self.f.attrs.modify('completed_rows', end); self._durable()

    def commit(self):
        if self.completed != self.n: raise ValueError('Incomplete episode cannot commit')
        self.f.attrs.modify('completed_rows', self.n)
        self.f.attrs.modify('status', 'complete'); self._durable(); self.close()
        receipt = {'path': str(self.final.resolve()), 'sha256': sha(self.partial), 'rows': self.n, 'clips': self.n - 3,
                   'rgb_sha256': self.digest.hexdigest(), 'protocol_sha256': self.protocol}
        # Publish receipt first: a failed receipt write leaves only a resumable
        # private partial. The complete name never precedes its durable receipt.
        atomic_json(self.final.with_suffix('.receipt.json'), receipt)
        os.link(self.partial, self.final); self.partial.unlink()
        return receipt

    def close(self):
        if self.f: self.f.close()


def recover_final_receipt(final, fields, protocol_sha256):
    """Recover only a verified published final whose receipt write was interrupted."""
    final = Path(final)
    receipt_path = final.with_suffix('.receipt.json')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['protocol_sha256'] != protocol_sha256 or sha(final) != receipt['sha256']:
            raise ValueError('Existing completed shard/receipt differs')
        return receipt
    marker = json.loads(final.with_suffix('.prefix.json').read_text())
    n = len(fields['action'])
    digest = hashlib.sha256()
    with h5py.File(final, 'r') as f:
        if f.attrs.get('protocol_sha256') != protocol_sha256 or f.attrs.get('status') != 'complete' or f.attrs.get('completed_rows') != n:
            raise ValueError('Unreceipted final is not complete or source-bound')
        for key, value in fields.items():
            if f[key].dtype != value.dtype or not np.array_equal(f[key][:], value): raise ValueError('Unreceipted final source differs')
        for row in range(0, n, 10): digest.update(f['pixels'][row:row + 10].tobytes())
    if marker['completed_rows'] != n or marker['protocol_sha256'] != protocol_sha256 or marker['rgb_prefix_sha256'] != digest.hexdigest():
        raise ValueError('Unreceipted final committed RGB differs')
    receipt = dict(path=str(final.resolve()), sha256=sha(final), rows=n, clips=n - 3,
                   rgb_sha256=digest.hexdigest(), protocol_sha256=protocol_sha256, recovered_receipt_only=True)
    atomic_json(receipt_path, receipt)
    return receipt
