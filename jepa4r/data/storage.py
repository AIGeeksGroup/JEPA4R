"""On-disk layout of the paired datasets in data/wm/<env>/ and data/policy/<env>/.

Format (stable-worldmodel's HDF5 layout, as written by ``World.collect``; kept from the integrated collection code):
one HDF5 file per dataset, every column a dataset of rows (one row = the state before an action, plus that action),
episodes delimited by ``ep_len`` / ``ep_offset``; ``ep_idx`` (row -> episode) and flat column names are added after
merging (``add_eval_columns``). A paired collection writes the same rollouts twice, row for row:
  <name>.h5      2D: ``pixels`` [224, 224, 3] uint8 (gzip) + every state / action column
  <name>_pcd.h5  3D: ``pcd_xyz`` [1200, 3] float32 (metres, world frame), ``pcd_rgb`` [1200, 3] uint8 + the same columns
Common columns: ``action`` [A] float32 (NaN in the row after the last action when it is kept), ``qpos``, ``qvel``,
``observation`` (proprioception / state), ``step_idx``, ``id``, ``terminated``, ``truncated``, ``reward``; per suite the
recipe columns (``sample_type``, ``rand_action``, ``noise_std``, ``branch_t``, ``branch_len``, ``episode_seed``, ...).
The train / validation split of the world models is drawn at training time (jepa4r.data.clips; whole held-out episode
files for Cube1-4).
Readers: data/clips.py (world-model clips), policy/features.py (latent caching),
decoder/train.py and eval/recon.py (point-cloud targets).
"""
import os
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np

try:  # stable-worldmodel / h5py are needed to write datasets, not to import this module
    import h5py
    from stable_worldmodel.data.formats.hdf5 import HDF5Writer, _is_string_col
except ImportError:  # pragma: no cover
    h5py, HDF5Writer, _is_string_col = None, object, None

# Moved from cortexbench@9331494:lewm_pcd/scripts/data/collect_cube_pix.py:GzipHDF5Writer (lines 35-56; verbatim)
class GzipHDF5Writer(HDF5Writer):
    """HDF5Writer whose datasets use multi-row chunks and gzip for the listed columns (schema creation only;
    episode writing, append validation and bookkeeping are inherited unchanged)."""

    def __init__(self, path, *, mode='append', chunk_rows=20, gzip_cols=('pixels',), gzip_chunk_rows=5, level=4):
        super().__init__(path, mode=mode)
        self.chunk_rows, self.gzip_cols, self.gzip_chunk_rows, self.level = chunk_rows, set(gzip_cols), gzip_chunk_rows, level

    def _init_schema(self, sample_ep: dict) -> None:
        for col, vals in sample_ep.items():
            if _is_string_col(vals[0]):
                self._f.create_dataset(col, shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(), chunks=(self.chunk_rows,))
                continue
            sample = np.asarray(vals[0])
            if col in self.gzip_cols:
                kw = dict(chunks=(self.gzip_chunk_rows, *sample.shape), compression='gzip', compression_opts=self.level, shuffle=True)
            else:
                kw = dict(chunks=(self.chunk_rows, *sample.shape))
            self._f.create_dataset(col, shape=(0, *sample.shape), maxshape=(None, *sample.shape), dtype=sample.dtype, **kw)
        self._f.create_dataset('ep_len', shape=(0,), maxshape=(None,), dtype=np.int32)
        self._f.create_dataset('ep_offset', shape=(0,), maxshape=(None,), dtype=np.int64)



# Moved from cortexbench@9331494:lewm_pcd/scripts/data/collect_cube_pcd.py:PIX_COLS, PCD_COLS, PairedWriter (lines 40-65; verbatim)
PIX_COLS = ('pixels', 'render_time')    # swm's AddPixelsWrapper: 2D dataset only
PCD_COLS = ('pcd_', 'pixels/', 'cam/')  # prefixes of PointCloudWrapper's keys (pixels/, cam/ only with keep_rgb): 3D dataset only


class PairedWriter:
    """``World.collect`` writer that stores every episode in two datasets: all columns but the point cloud in the
    2D one, all columns but the pixels in the 3D one -- the same episodes in the same order, row for row."""

    def __init__(self, pix_writer, pcd_writer):
        self.pix, self.pcd = pix_writer, pcd_writer

    def __enter__(self):
        with ExitStack() as stack:  # if the second file cannot be opened, the first one is closed again
            stack.enter_context(self.pix)
            stack.enter_context(self.pcd)
            self._files = stack.pop_all()
        return self

    def __exit__(self, *exc):
        return self._files.__exit__(*exc)

    def write_episodes(self, episodes):
        for ep in episodes:
            self.pix.write_episode({k: v for k, v in ep.items() if not k.startswith(PCD_COLS)})
            self.pcd.write_episode({k: v for k, v in ep.items() if k not in PIX_COLS})



# Moved from cortexbench@9331494:lewm_pcd/scripts/data/collect_cube_pcd_sharded.py:check_paired (lines 32-40; verbatim)
def check_paired(pcd, pix):
    """The 3D and 2D datasets of one paired collection hold the same episodes, row for row."""
    with h5py.File(pcd, 'r') as a, h5py.File(pix, 'r') as b:
        every = slice(None, None, 997)  # a sample of rows: the 3D file still has one-row chunks here
        for col, rows in [('ep_len', slice(None)), ('ep_offset', slice(None)), ('id', every), ('qpos', every), ('action', every)]:
            if not np.array_equal(a[col][rows], b[col][rows], equal_nan=a[col].dtype.kind == 'f'):
                sys.exit(f'{pcd.name} and {pix.name} differ in {col!r}: not the same rollouts')
    print(f'{pcd.name} (3D) and {pix.name} (2D) hold the same episodes row for row', flush=True)


# Moved from OGBench-cube@4fbb0c6:lewm_pcd/scripts/data/merge_h5_shards.py:CHUNK, datasets, append_shard (lines 23-74; verbatim)
CHUNK = 20_000  # frames per copy step


def datasets(f):
    """All dataset paths, recursing by name so hard-linked aliases are listed too
    (h5py.visititems reports each object only once)."""
    cols = []
    def walk(g, prefix=''):
        for k in g.keys():
            o = g[k]
            if isinstance(o, h5py.Dataset):
                cols.append(prefix + k)
            else:
                walk(o, prefix + k + '/')
    walk(f)
    return sorted(c for c in cols if c not in ('ep_len', 'ep_offset'))


def append_shard(out_path, shard_path, preserve_episode_indices=False):
    with h5py.File(out_path, 'a', libver='latest') as fo, h5py.File(shard_path, 'r') as fs:
        co, cs = datasets(fo), datasets(fs)
        if 'ep_idx' in cs and not preserve_episode_indices:
            raise ValueError(f'{shard_path} already has ep_idx (shard-local episode numbers); run add_eval_columns.py only after merging')
        if co != cs:
            raise ValueError(f'schema mismatch: only in out {sorted(set(co) - set(cs))}, '
                             f'only in shard {sorted(set(cs) - set(co))}')
        if preserve_episode_indices:
            if 'ep_idx' not in co:
                raise ValueError('preserve_episode_indices requires original ep_idx')
            old_ids = fo['ep_idx'][fo['ep_offset'][:]]
            new_ids = fs['ep_idx'][fs['ep_offset'][:]]
            if np.intersect1d(old_ids,new_ids).size:
                raise ValueError('overlapping original episode IDs')
        n_old = int(fo['ep_len'][:].sum())
        T = int(fs['ep_len'][:].sum())
        for c in cs:
            do, ds = fo[c], fs[c]
            if do.shape[1:] != ds.shape[1:] or do.dtype != ds.dtype:
                raise ValueError(f"column '{c}': {do.shape[1:]}/{do.dtype} vs {ds.shape[1:]}/{ds.dtype}")
            do.resize(n_old + T, axis=0)
            for s in range(0, T, CHUNK):
                e = min(T, s + CHUNK)
                do[n_old + s:n_old + e] = ds[s:e]
        lens, offs = fs['ep_len'][:], fs['ep_offset'][:]
        k = fo['ep_len'].shape[0]
        fo['ep_len'].resize(k + len(lens), axis=0)
        fo['ep_len'][k:] = lens
        fo['ep_offset'].resize(k + len(offs), axis=0)
        fo['ep_offset'][k:] = offs + n_old
        return len(lens), T



# Moved from OGBench-cube@4fbb0c6:lewm_pcd/scripts/data/merge_h5_shards.py:main (lines 75-103; verbatim except: argparse replaced by the arguments of merge_shards)
def merge_shards(out, shards, delete=False, preserve_episode_indices=False):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shards = [str(s) for s in shards]
    if not out.exists():
        shutil.move(shards.pop(0), out)
        print(f'moved first shard -> {out}', flush=True)
    for shard in shards:
        n, T = append_shard(out, shard, preserve_episode_indices)
        print(f'appended {n:5d} episodes / {T:8d} frames from {shard}', flush=True)
        if delete:
            os.remove(shard)
    with h5py.File(out, 'r') as f:
        lens, offs = f['ep_len'][:], f['ep_offset'][:]
        ok = np.all(offs == np.concatenate([[0], np.cumsum(lens)[:-1]])) and all(
            f[c].shape[0] == lens.sum() for c in datasets(f))
        print(f'-> {out}: {len(lens)} episodes, {int(lens.sum())} frames, '
              f'{out.stat().st_size/1e9:.2f} GB, offsets/lengths consistent: {bool(ok)}')
        if not ok:
            raise SystemExit('merged file inconsistent')




# Moved from cortexbench@9331494:lewm_pcd/scripts/data/add_eval_columns.py:main (lines 18-42; verbatim except: renamed add_eval_columns)
def add_eval_columns(path):
    with h5py.File(path, 'a') as f:
      lens, offs = f['ep_len'][:], f['ep_offset'][:]  # noqa: F841  (moved verbatim)
      n = int(lens.sum())
      if 'ep_idx' not in f:
          ep_idx = np.repeat(np.arange(len(lens), dtype=np.int64), lens)
          assert len(ep_idx) == n
          f.create_dataset('ep_idx', data=ep_idx, chunks=(min(n, 1 << 16),))
          print(f'added ep_idx ({n} rows, {len(lens)} episodes)')
      nested = []
      f.visititems(lambda k, o: nested.append(k) if isinstance(o, h5py.Dataset) and '/' in k else None)
      added = 0
      for k in nested:
          flat = k.replace('/', '_')
          if flat not in f:
              f[flat] = f[k]  # hard link, data is not copied
              added += 1
          elif f[flat] != f[k]:
              raise SystemExit(f'name clash: top-level {flat!r} exists and is not the same object as {k!r}; not touching the file')
      groups = sorted({k.split('/')[0] for k in nested})
      for g in groups:
          del f[g]  # drop the group link; the datasets live on under their flat names
      print(f'flattened {len(nested)} nested datasets ({added} new links); removed groups {groups}')
      print('top-level columns:', sorted(k for k in f.keys()))



def open_dataset(path, split="train"):
    """A world-model / policy HDF5 dataset for reading (h5py). ``path`` is the file, or a data directory plus the
    file name convention of the collection (``<dir>/<split>.h5``)."""
    path = Path(path)
    if path.is_dir():
        path = path / f'{split}.h5'
    return h5py.File(path, 'r')


class Writer:
    def __init__(self, path, env_cfg):
        """The writer of the paired collection: 2D file ``path`` (gzip pixels) and 3D file ``path`` with ``_pcd``,
        both never appended to (``mode='error'``), as collect_cube_pcd.py / collect_mw.py / collect_ogb.py open them."""
        path = Path(path)
        pcd = path.with_name(path.stem + '_pcd' + path.suffix)
        self.paired = PairedWriter(GzipHDF5Writer(path, mode='error'), GzipHDF5Writer(pcd, mode='error', gzip_cols=()))

    def __enter__(self):
        self.paired.__enter__()
        return self.paired

    def __exit__(self, *exc):
        return self.paired.__exit__(*exc)
