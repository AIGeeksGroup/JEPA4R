"""Rewrite a World.collect() HDF5 dataset with larger chunks (same data, same columns).

Upstream's HDF5Writer stores every column with chunks=(1, ...) i.e. one chunk per
step. With 2M steps that is 2M chunks per column: HDF5 keeps a multi-GB chunk
index per process (13 GB just to read the small cached columns), and every clip
read costs ~40 tiny random IOs. This copies each dataset with ``rows`` steps per
chunk (default 20 = one training clip span) and recreates hard-linked aliases.

    python scripts/data/rechunk_h5.py <in.h5> <out.h5> [--rows 20]
"""
import argparse
import time
import h5py
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('--rows', type=int, default=20)
ap.add_argument('--block', type=int, default=20_000, help='rows copied per step')
a = ap.parse_args()

with h5py.File(a.src, 'r') as fi, h5py.File(a.dst, 'w', libver='latest') as fo:
    names = []
    def walk(g, prefix=''):  # by name, so hard-linked aliases are visited too
        for k in g.keys():
            o = g[k]
            names.append(prefix + k) if isinstance(o, h5py.Dataset) else walk(o, prefix + k + '/')
    walk(fi)
    done = {}  # object address -> first name written (hard-linked aliases share it)
    t0 = time.time()
    for k in names:
        d = fi[k]
        key = h5py.h5o.get_info(d.id).addr
        if key in done:
            fo[k] = fo[done[key]]  # recreate alias as a hard link
            continue
        n = d.shape[0]
        if k in ('ep_len', 'ep_offset') or d.ndim == 1 and n < 100_000:
            fo.create_dataset(k, data=d[:], chunks=True, maxshape=(None, *d.shape[1:]))
        else:
            rows = a.rows if d.nbytes / max(n, 1) > 1000 else 4096  # small per-row columns get big chunks
            chunks = (min(rows, n), *d.shape[1:])
            o = fo.create_dataset(k, shape=d.shape, dtype=d.dtype, chunks=chunks, maxshape=(None, *d.shape[1:]))
            for s in range(0, n, a.block):
                o[s:s + a.block] = d[s:s + a.block]
        done[key] = k
        print(f'{k:32s} {str(d.shape):22s} chunks {fi[k].chunks} -> {fo[k].chunks}   ({time.time()-t0:.0f}s)', flush=True)
    for name, val in fi.attrs.items():
        fo.attrs[name] = val
print('done', a.dst)
