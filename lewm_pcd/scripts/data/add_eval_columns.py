"""Give a World.collect() HDF5 dataset the two things the official LeWM eval.py
expects from the paper-era layout, without touching the data:

  * ``ep_idx``            per-row episode index (derived from ep_len / ep_offset)
  * flat column layout     nested columns such as ``privileged/block_0_pos`` become
                           top-level ``privileged_block_0_pos`` (HDF5 hard link, zero
                           extra bytes) and the group entries are removed, because
                           HEAD's HDF5Dataset reader assumes a flat file (it treats
                           every top-level key as a dataset). ``goal_privileged_block_0_pos``
                           etc. then resolve exactly as in config/eval/cube.yaml.

    python scripts/data/add_eval_columns.py <dataset.h5>
"""
import sys
import h5py
import numpy as np

def main(path):
    with h5py.File(path, 'a') as f:
      lens, offs = f['ep_len'][:], f['ep_offset'][:]
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


if __name__ == '__main__':
    main(sys.argv[1])
