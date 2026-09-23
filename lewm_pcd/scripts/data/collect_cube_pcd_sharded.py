"""Run collect_cube_pcd.py (or --script collect_cube_pix.py) as K parallel shards, then merge them into one h5
per dataset.

    python scripts/data/collect_cube_pcd_sharded.py --shards 40 --out-pix ogbench/cube_single_expert.h5
           [--num-traj 10000] [--seed 3072] [--out ogbench/cube_single_expert_pcd.h5] [--no-merge]
           [extra hydra overrides...]

Every shard is an unmodified ``collect_cube_pcd.py`` run (10 envs x 200 steps,
same env / expert / point-cloud config); only ``num_traj``, the World seed and
the output names differ. ``collect_cube_pcd.py`` writes the 3D (``--out``) and
the 2D (``--out-pix``) dataset of the same rollouts; both shard sets are merged
in the same order and checked to hold the same episodes row for row.
Shard seeds are drawn from ``default_rng(seed)`` one after another, the
way upstream stable-worldmodel sharded its own cube collection (commit 93b1a86).
Per-shard logs go to ``$STABLEWM_HOME/logs/collect/<run>/shard_XX.log``.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def check_paired(pcd, pix):
    """The 3D and 2D datasets of one paired collection hold the same episodes, row for row."""
    with h5py.File(pcd, 'r') as a, h5py.File(pix, 'r') as b:
        every = slice(None, None, 997)  # a sample of rows: the 3D file still has one-row chunks here
        for col, rows in [('ep_len', slice(None)), ('ep_offset', slice(None)), ('id', every), ('qpos', every), ('action', every)]:
            if not np.array_equal(a[col][rows], b[col][rows], equal_nan=a[col].dtype.kind == 'f'):
                sys.exit(f'{pcd.name} and {pix.name} differ in {col!r}: not the same rollouts')
    print(f'{pcd.name} (3D) and {pix.name} (2D) hold the same episodes row for row', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shards', type=int, required=True)
    ap.add_argument('--num-traj', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=3072)
    ap.add_argument('--out', default='ogbench/cube_single_expert_pcd.h5',
                    help='final dataset name, relative to $STABLEWM_HOME/datasets')
    ap.add_argument('--out-pix', default=None,
                    help='final 2D dataset name (relative to $STABLEWM_HOME/datasets) written by collect_cube_pcd.py '
                         'from the same rollouts as --out; required with that script, not accepted with collect_cube_pix.py')
    ap.add_argument('--cache-dir', default=None, help='override $STABLEWM_HOME')
    ap.add_argument('--no-merge', action='store_true')
    ap.add_argument('--threads', type=int, default=2, help='OMP/MKL threads per shard')
    ap.add_argument('--max-parallel', type=int, default=None, help='shards running at once (default: all)')
    ap.add_argument('--script', default='collect_cube_pcd.py', help='per-shard collection script in scripts/data (collect_cube_pix.py for the pixel dataset)')
    ap.add_argument('overrides', nargs='*', help='extra hydra overrides for every shard')
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir or os.environ.get('STABLEWM_HOME', '~/.stable-wm')).expanduser()
    run = datetime.now().strftime('%Y-%m-%d/%H-%M-%S') + f'_shards{args.shards}'
    log_dir = cache_dir / 'logs' / 'collect' / run
    log_dir.mkdir(parents=True, exist_ok=True)
    final = cache_dir / 'datasets' / args.out
    shard_dir = final.parent / 'shards'
    paired = args.script == 'collect_cube_pcd.py'  # also writes the 2D dataset of the same rollouts (pix_dataset_name)
    if paired != (args.out_pix is not None):
        sys.exit('--out-pix (the 2D dataset of the same rollouts) is required with collect_cube_pcd.py and only allowed with it')
    final_pix = cache_dir / 'datasets' / args.out_pix if paired else None

    fresh = [final, shard_dir] + ([final_pix] if paired else [])
    if any(p.exists() for p in fresh):
        sys.exit(f'refusing to run: {", ".join(str(p) for p in fresh if p.exists())} already exists (HDF5 output must be fresh)')
    base, rem = divmod(args.num_traj, args.shards)
    counts = [base + (1 if i < rem else 0) for i in range(args.shards)]
    rng = np.random.default_rng(args.seed)
    for attempt in range(100):  # each World consumes seeds [s, s + n + num_envs); require disjoint ranges
        seeds = [rng.integers(0, 1_000_000).item() for _ in range(args.shards)]
        spans = sorted((s, s + n + 64) for s, n in zip(seeds, counts))
        if all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1)):
            break
    else:
        sys.exit('could not draw non-overlapping shard seed ranges')

    env = dict(os.environ, MUJOCO_GL='egl', OMP_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads))
    procs, shards, pix_shards, logs = [], [], [], []
    max_par = args.max_parallel or args.shards
    t0 = time.time()
    for i, (sd, n) in enumerate(zip(seeds, counts)):
        while sum(p.poll() is None for p in procs) >= max_par:
            time.sleep(5)
        shard = shard_dir / f'{final.stem}.shard{i:03d}.h5'
        shards.append(shard)
        outs = [f'dataset_name={shard.relative_to(cache_dir / "datasets")}']
        if paired:  # this shard's 2D dataset
            pix_shards.append(shard_dir / f'{final_pix.stem}.shard{i:03d}.h5')
            outs.append(f'pix_dataset_name={pix_shards[-1].relative_to(cache_dir / "datasets")}')
        cmd = [sys.executable, str(HERE / args.script),
               f'num_traj={n}', f'world_seed={sd}', f'seed={args.seed}',
               f'cache_dir={cache_dir}', *outs,
               f'hydra.run.dir={log_dir / f"shard_{i:03d}"}', *args.overrides]
        log = open(log_dir / f'shard_{i:03d}.log', 'w'); logs.append(log)
        procs.append(subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
    print(f'launched {len(procs)} shards ({counts[0]}..{counts[-1]} episodes each), logs: {log_dir}', flush=True)

    failed = []
    for i, p in enumerate(procs):
        if p.wait() != 0:
            failed.append(i)
    for log in logs:
        log.close()
    print(f'all shards finished in {(time.time()-t0)/60:.1f} min; failed: {failed or "none"}', flush=True)
    if failed:
        sys.exit(1)
    if args.no_merge:
        return
    merges = [(final, shards)] + ([(final_pix, pix_shards)] if paired else [])  # same shard order for both datasets
    for out, parts in merges:
        merge = [sys.executable, str(HERE / 'merge_h5_shards.py'), '--out', str(out), '--delete', *map(str, parts)]
        subprocess.run(merge, cwd=ROOT, env=env, check=True)
    shard_dir.rmdir()
    if paired:
        check_paired(final, final_pix)
    print(f'total wall {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
