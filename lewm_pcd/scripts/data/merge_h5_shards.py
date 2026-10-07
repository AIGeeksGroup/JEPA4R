"""Append shard h5 files (written by World.collect / HDF5Writer) into one dataset.

    python scripts/data/merge_h5_shards.py --out <final.h5> <shard0.h5> <shard1.h5> ... [--delete]

If ``final.h5`` does not exist yet the first shard is *moved* there (no copy),
then every further shard is appended column by column with ep_len/ep_offset
re-based, which is exactly the layout one long collection produces. With
``--delete`` each shard is removed right after it has been appended, so peak
disk usage is the final file plus one shard.

(Upstream HDF5Writer(mode='append') cannot be used here: its schema check
compares top-level h5 groups with flat column names and rejects any dataset
that has nested columns such as ``privileged/block_0_pos``.)
"""
import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np

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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('shards', nargs='+')
    ap.add_argument('--delete', action='store_true', help='remove each shard after appending it')
    ap.add_argument('--preserve-episode-indices', action='store_true', help='keep verified global replay IDs; reject overlapping shards')
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shards = list(args.shards)
    if not out.exists():
        shutil.move(shards.pop(0), out)
        print(f'moved first shard -> {out}', flush=True)
    for shard in shards:
        n, T = append_shard(out, shard, args.preserve_episode_indices)
        print(f'appended {n:5d} episodes / {T:8d} frames from {shard}', flush=True)
        if args.delete:
            os.remove(shard)
    with h5py.File(out, 'r') as f:
        lens, offs = f['ep_len'][:], f['ep_offset'][:]
        ok = np.all(offs == np.concatenate([[0], np.cumsum(lens)[:-1]])) and all(
            f[c].shape[0] == lens.sum() for c in datasets(f))
        print(f'-> {out}: {len(lens)} episodes, {int(lens.sum())} frames, '
              f'{out.stat().st_size/1e9:.2f} GB, offsets/lengths consistent: {bool(ok)}')
        if not ok:
            raise SystemExit('merged file inconsistent')


if __name__ == '__main__':
    main()
