"""Sanity checks on a pilot point-cloud h5."""
import sys
from pathlib import Path
import h5py, numpy as np
H5 = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'datasets/ogbench/pilot_cube_single_expert_pcd.h5'
with h5py.File(H5, 'r') as f:
    print('file:', H5, f'({H5.stat().st_size/1e6:.1f} MB)')
    print('ep_len:', f['ep_len'][:], ' ep_offset:', f['ep_offset'][:])
    rows = []
    f.visititems(lambda n, o: rows.append((n, o.shape, str(o.dtype))) if isinstance(o, h5py.Dataset) else None)
    for n, s, d in rows:
        print(f'  {n:34s} {str(s):24s} {d}')
    xyz = f['pcd_xyz'][:]; blk = f['privileged/block_0_pos'][:]
    print('pcd_xyz finite:', bool(np.isfinite(xyz).all()),
          '| x %.2f..%.2f  y %.2f..%.2f  z %.2f..%.2f' % (xyz[...,0].min(), xyz[...,0].max(), xyz[...,1].min(), xyz[...,1].max(), xyz[...,2].min(), xyz[...,2].max()))
    d = np.linalg.norm(xyz - blk[:, None, :], axis=-1)
    near = d < 0.04
    cube_n = near.sum(1)
    print('points within 4 cm of cube centre per frame: min %d  median %d  max %d' % (cube_n.min(), np.median(cube_n), cube_n.max()))
    err = np.array([np.linalg.norm(xyz[t][near[t]].mean(0) - blk[t]) for t in range(len(xyz)) if cube_n[t] > 5])
    print('cube centroid error: mean %.2f cm  max %.2f cm  (single view was 1.3-1.6 cm)' % (100 * err.mean(), 100 * err.max()))
    below = (xyz[..., 2] < 0.0).mean()
    print('fraction of points below table (z<0): %.4f' % below)
