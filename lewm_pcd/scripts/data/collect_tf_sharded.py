"""collect_tf.py in shards, as collect_cube_pcd_sharded.py runs collect_mw.py for MetaWorld: the episodes split over
--shards runs (each shard one task of --tasks), shard world seeds in consecutive disjoint ranges from --seed, at most
--max-parallel at a time; then both shard sets merged in the same order (merge_h5_shards.py --delete) and the 2D / 3D
pairing checked. In the lewm_tf python, as scripts/run_tf_pipeline.sh runs it:
    STABLEWM_HOME=... PYTHONPATH=.:third_party/stable-worldmodel $PY scripts/data/collect_tf_sharded.py \
        --num-traj 120000 --tasks move reach --shards 56 --max-parallel 14 --seed 3072 --out trifinger/tf_cube_pcd.h5
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from collect_cube_pcd_sharded import check_paired  # this script's directory is on sys.path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # lewm_pcd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-traj', type=int, required=True, help='episodes per task')
    ap.add_argument('--tasks', nargs='+', default=['move'], choices=('move', 'reach'),
                    help='each shard collects one task; the shards are split equally over the tasks, in this order')
    ap.add_argument('--shards', type=int, required=True)
    ap.add_argument('--max-parallel', type=int, required=True)
    ap.add_argument('--seed', type=int, default=3072)
    ap.add_argument('--out', required=True, help='final 3D dataset name under $STABLEWM_HOME/datasets (with _pcd)')
    ap.add_argument('rest', nargs='*', help='after --: extra collect_tf.py arguments for every shard')
    args = ap.parse_args()

    home = Path(os.environ['STABLEWM_HOME'])
    final = home / 'datasets' / args.out
    final_pix = final.with_name(final.name.replace('_pcd', '', 1))
    shard_dir = final.parent / 'shards'
    if any(p.exists() for p in (final, final_pix, shard_dir)):
        sys.exit(f'refusing to run: {final}, {final_pix} or {shard_dir} exists (HDF5 output must be fresh)')
    per_task, odd = divmod(args.shards, len(args.tasks))
    if odd:
        sys.exit(f'--shards {args.shards} does not split equally over the tasks {args.tasks}')
    base, rem = divmod(args.num_traj, per_task)
    counts = [base + (1 if i < rem else 0) for _ in args.tasks for i in range(per_task)]
    tasks = [t for t in args.tasks for _ in range(per_task)]
    # shard i resets its episodes with seeds [s_i, s_i + n_i) (+ num_envs, World's pool): consecutive disjoint ranges
    # from --seed (2 x 120000 episodes span ~0.25 M seeds: clear of the test set's 2000000). Random draws as in
    # collect_cube_pcd_sharded.py overlap at thousands of episodes per shard.
    seeds = list(args.seed + np.concatenate([[0], np.cumsum([n + 64 for n in counts])[:-1]]).astype(int))

    log_dir = home / 'logs' / 'collect' / time.strftime('%Y-%m-%d_%H-%M-%S_shards')
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    procs, shards, pix_shards, logs = [], [], [], []
    t0 = time.time()
    for i, (sd, n) in enumerate(zip(seeds, counts)):
        while sum(p.poll() is None for p in procs) >= args.max_parallel:
            time.sleep(5)
        shard = shard_dir / f'{final.stem}.shard{i:03d}.h5'
        shards.append(shard)
        pix_shards.append(shard_dir / f'{final_pix.stem}.shard{i:03d}.h5')
        cmd = [sys.executable, str(HERE / 'collect_tf.py'), '--task', tasks[i], '--num-traj', str(n), '--world-seed', str(sd),
               '--out', str(shard.relative_to(home / 'datasets')), *args.rest]
        log = open(log_dir / f'shard_{i:03d}.log', 'w')
        logs.append(log)
        procs.append(subprocess.Popen(['nice', '-n', '10', *cmd], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
        print(f'{time.strftime("%F %T")} shard {i}: {tasks[i]}, {n} episodes, world seed {sd}', flush=True)
    failed = [i for i, p in enumerate(procs) if p.wait() != 0]
    for log in logs:
        log.close()
    print(f'all shards finished in {(time.time() - t0) / 60:.1f} min; failed: {failed or "none"}', flush=True)
    if failed:
        sys.exit(1)
    for out, parts in ((final, shards), (final_pix, pix_shards)):  # same shard order for both datasets
        subprocess.run([sys.executable, str(ROOT / 'scripts' / 'data' / 'merge_h5_shards.py'), '--out', str(out),
                        '--delete', *map(str, parts)], cwd=ROOT, env=env, check=True)
    shard_dir.rmdir()
    check_paired(final, final_pix)
    print(f'total wall {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
