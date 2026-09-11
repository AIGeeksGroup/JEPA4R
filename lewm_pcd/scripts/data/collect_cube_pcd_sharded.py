"""Run collect_cube_pcd.py (or --script collect_cube_pix.py) as K parallel shards, then merge them into one h5.

    python scripts/data/collect_cube_pcd_sharded.py --shards 40 [--num-traj 10000] [--seed 3072]
           [--out ogbench/cube_single_expert_pcd.h5] [--no-merge] [extra hydra overrides...]

Every shard is an unmodified ``collect_cube_pcd.py`` run (10 envs x 200 steps,
same env / expert / point-cloud config); only ``num_traj`` and the World seed
differ. Shard seeds are drawn from ``default_rng(seed)`` one after another, the
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

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shards', type=int, required=True)
    ap.add_argument('--num-traj', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=3072)
    ap.add_argument('--out', default='ogbench/cube_single_expert_pcd.h5',
                    help='final dataset name, relative to $STABLEWM_HOME/datasets')
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

    if final.exists() or shard_dir.exists():
        sys.exit(f'refusing to run: {final} or {shard_dir} already exists (HDF5 output must be fresh)')
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
    procs, shards, logs = [], [], []
    max_par = args.max_parallel or args.shards
    t0 = time.time()
    for i, (sd, n) in enumerate(zip(seeds, counts)):
        while sum(p.poll() is None for p in procs) >= max_par:
            time.sleep(5)
        shard = shard_dir / f'{final.stem}.shard{i:03d}.h5'
        shards.append(shard)
        cmd = [sys.executable, str(HERE / args.script),
               f'num_traj={n}', f'world_seed={sd}', f'seed={args.seed}',
               f'cache_dir={cache_dir}', f'dataset_name={shard.relative_to(cache_dir / "datasets")}',
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
    merge = [sys.executable, str(HERE / 'merge_h5_shards.py'), '--out', str(final), '--delete', *map(str, shards)]
    subprocess.run(merge, cwd=ROOT, env=env, check=True)
    shard_dir.rmdir()
    print(f'total wall {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
