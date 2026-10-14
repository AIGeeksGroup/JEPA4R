"""OGBench's puzzle-3x3-play data -> every state re-rendered with the LeWM data's camera -> the frozen LeWM 2D / 3D
encoders -> one 192-d latent per step (DQC's training observations, run_dqc.py).
* The official data stores states only (OGBench's state observation + qpos / qvel / button_states), no images. Each
  step's qpos / qvel / button_states are set back into the simulator (env.set_state) and the frame is rendered as the
  LeWM data is (envs/ogb_puzzle_env.py: the opaque arm of ob_type='states', front_pixels, 224; 2D anti-aliased, 3D the
  same camera's depth with the same rules, 1200 points, the point sampling seeded per frame).
* Encoders (lewm_obs.Encoders, the given checkpoints' epoch): the projector's 192-d output, LeWM's latent; the world
  model is not trained further.
* Only the latents are stored (float32, N x 192, row for row with the npz), written in chunks, restartable.
    python extract_feats.py --ckpt-2d $STABLEWM_HOME/checkpoints/puzzle_3x3_pix --ckpt-3d .../puzzle_3x3_pcd_dexart \
        --out $STABLEWM_HOME/datasets/ogbench/dqc_feats/puzzle_3x3 --split train --workers 6
    python extract_feats.py ... --check   # re-render our test set from its stored states, compare with its frames / latents
"""
import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from lewm_obs import Encoders, load_puzzle  # this script's directory is on sys.path

PCD_SEED = 20261011
_W = {}


def _init_worker():
    pdat = load_puzzle()
    env = pdat.make_env()
    env.reset(seed=0)
    _W.update(pdat=pdat, u=env.unwrapped, cam=pdat.Camera(env.unwrapped._model), env=env)


def _render_chunk(args):
    """Set a run of states back into the simulator one by one and render each: (first row, 2D frames, point-cloud xyz,
    point-cloud rgb, visible pixels of the button caps)."""
    start, split_id, qpos, qvel, bstates = args
    pdat, u, cam = _W['pdat'], _W['u'], _W['cam']
    n = len(qpos)
    pix = np.empty((n, pdat.IMG, pdat.IMG, 3), np.uint8)
    xyz = np.empty((n, pdat.NUM_POINTS, 3), np.float32)
    col = np.empty((n, pdat.NUM_POINTS, 3), np.uint8)
    cap = np.empty((n, 9), np.int32)
    for k in range(n):
        u.set_state(qpos[k].astype(np.float64), qvel[k].astype(np.float64), np.asarray(bstates[k]).astype(int))
        pixels, rgb, depth, seg, cap_px, R, t = cam.render(u._data)
        rng = np.random.default_rng((PCD_SEED, split_id, start + k))  # each frame's point sampling fixed on its own
        pix[k] = pixels
        xyz[k], col[k] = pdat.point_cloud(cam, rgb, depth, seg, R, t, rng)
        cap[k] = cap_px
    return start, pix, xyz, col, cap


def check(a):
    """Our own test set stores the frames / point clouds and qpos / qvel / button states: re-render from the stored
    states and compare with what was stored."""
    import h5py
    D = Path(os.environ['STABLEWM_HOME']) / 'datasets' / 'ogbench'
    with h5py.File(D / f'{a.test_set}.h5', 'r') as f:
        n = int(f['ep_len'][:].sum())
        rows = np.arange(0, min(n, 2002), 4)  # the first two episodes, every 4th frame
        pix0 = f['pixels'][rows]
        qpos, qvel, bst = f['qpos'][rows], f['qvel'][rows], f['button_states'][rows]
        cap0 = f['cap_visible_px'][rows]
    with h5py.File(D / f'{a.test_set}_pcd.h5', 'r') as f:
        xyz0, col0 = f['pcd_xyz'][rows], f['pcd_rgb'][rows]
    _init_worker()
    t = time.time()
    _, pix1, xyz1, col1, cap1 = _render_chunk((0, 9, qpos, qvel, bst))
    dt = (time.time() - t) / len(rows)
    print(f'{len(rows)} frames re-rendered, {dt * 1000:.1f} ms per frame (one process)')
    print(f'2D frames: share of identical pixels {(pix1 == pix0).all(-1).mean():.5f}, '
          f'mean abs difference {np.abs(pix1.astype(int) - pix0).mean():.4f} (0-255)')
    print(f'button caps visible pixels identical: {(cap1 == cap0).all(1).mean():.4f}')
    enc = Encoders(a.ckpt_2d, a.ckpt_3d, a.epoch)
    cos = lambda x, y: np.sum(x * y, 1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
    for name, z0, z1 in (('2D', enc.z2d(pix0), enc.z2d(pix1)), ('3D', enc.z3d(xyz0, col0), enc.z3d(xyz1, col1))):
        rnd = np.random.default_rng(0).permutation(len(z0))
        print(f'{name} latents, stored vs re-rendered: cosine mean {cos(z0, z1).mean():.4f} min {cos(z0, z1).min():.4f}; '
              f'max abs difference {np.abs(z0 - z1).max():.4f}; reference: cosine between different frames '
              f'{cos(z0, z0[rnd]).mean():.4f}')


def main():
    home = Path(os.environ['STABLEWM_HOME'])
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-2d', required=True, help='checkpoint directory of the 2D LeWM')
    ap.add_argument('--ckpt-3d', required=True, help='checkpoint directory of the 3D LeWM')
    ap.add_argument('--epoch', type=int, default=10)
    ap.add_argument('--data', default=str(home / 'datasets' / 'ogbench' / 'official'), help='the official npz files')
    ap.add_argument('--out', default=None, help='output directory (required unless --check)')
    ap.add_argument('--split', choices=['train', 'val'], default='train')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--chunk', type=int, default=250)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--test-set', default='puzzle_3x3_noisy_test', help='--check: our test set under datasets/ogbench')
    a = ap.parse_args()
    if a.check:
        return check(a)
    name = 'puzzle-3x3-play-v0' + ('-val' if a.split == 'val' else '')
    f = np.load(Path(a.data) / f'{name}.npz')
    qpos, qvel, bst = f['qpos'], f['qvel'], f['button_states']
    n = len(qpos) if a.limit is None else a.limit
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = a.split if a.limit is None else f'{a.split}_first{n}'
    z_paths = {k: out / f'{tag}_z{k}.npy' for k in ('2d', '3d')}
    done_path = out / f'{tag}_done.npy'
    starts = list(range(0, n, a.chunk))
    if done_path.exists():
        Z = {k: np.load(p, mmap_mode='r+') for k, p in z_paths.items()}
        done = np.load(done_path)
    else:
        Z = {k: np.lib.format.open_memmap(p, mode='w+', dtype=np.float32, shape=(n, 192)) for k, p in z_paths.items()}
        done = np.zeros(len(starts), bool)
    todo = [i for i, s in enumerate(starts) if not done[i]]
    print(f'{name}: {n} rows, {len(starts)} chunks, {len(todo)} to do', flush=True)
    enc = Encoders(a.ckpt_2d, a.ckpt_3d, a.epoch)
    split_id = 0 if a.split == 'train' else 1
    tasks = ((starts[i], split_id, qpos[starts[i]: starts[i] + a.chunk][: n - starts[i]],
              qvel[starts[i]: starts[i] + a.chunk][: n - starts[i]], bst[starts[i]: starts[i] + a.chunk][: n - starts[i]])
             for i in todo)
    t0, n_done = time.time(), 0
    with mp.get_context('spawn').Pool(a.workers, initializer=_init_worker) as pool:
        for j, (s, pix, xyz, col, _) in enumerate(pool.imap(_render_chunk, tasks)):
            Z['2d'][s: s + len(pix)] = enc.z2d(pix)
            Z['3d'][s: s + len(pix)] = enc.z3d(xyz, col)
            done[s // a.chunk] = True
            n_done += len(pix)
            if j % 20 == 0 or done.all():
                for z in Z.values():
                    z.flush()
                np.save(done_path, done)
                rate = n_done / (time.time() - t0)
                print(f'{done.sum()}/{len(starts)} chunks, {rate:.0f} frames/s', flush=True)
    for z in Z.values():
        z.flush()
    np.save(done_path, done)
    assert done.all()
    for k, p in z_paths.items():
        z = np.load(p, mmap_mode='r')
        print(k, z.shape, 'nan' if np.isnan(z).any() else 'ok', float(np.abs(z).mean()))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
