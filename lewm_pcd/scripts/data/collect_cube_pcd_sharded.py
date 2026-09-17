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
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def shard_path(final, index):
    return final.parent / 'shards' / f'{final.stem}.shard{index:03d}.h5'


def partial_shard_path(final, index):
    return final.parent / 'shards' / f'{final.stem}.shard{index:03d}.partial.h5'


def merge_partial_path(final):
    return final.with_name(final.name + '.merge.partial.h5')


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return parsed


def acquire_shard_lock(shard_dir):
    lock = open(shard_dir / '.collection.lock', 'a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f'another launcher already holds the shard lock: {shard_dir}')
    return lock


def release_shard_lock(lock):
    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    lock.close()


def _episode_counts(num_traj, shards):
    base, rem = divmod(num_traj, shards)
    return [base + (1 if i < rem else 0) for i in range(shards)]


def _world_seeds(seed, counts):
    rng = np.random.default_rng(seed)
    for _ in range(100):
        seeds = [rng.integers(0, 1_000_000).item() for _ in counts]
        spans = sorted((s, s + n + 64) for s, n in zip(seeds, counts))
        if all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1)):
            return seeds
    raise ValueError('could not draw non-overlapping shard seed ranges')


def make_manifest(args, final):
    counts = _episode_counts(args.num_traj, args.shards)
    return {
        'version': 1,
        'out': str(final.resolve()),
        'script': args.script,
        'seed': args.seed,
        'shards': args.shards,
        'num_traj': args.num_traj,
        'episode_counts': counts,
        'world_seeds': _world_seeds(args.seed, counts),
        'overrides': list(args.overrides),
    }


def write_manifest_atomic(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    with open(partial, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(partial, path)


def load_manifest(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def assert_resume_matches(manifest, args, final):
    expected = {
        'version': 1,
        'out': str(final.resolve()),
        'script': args.script,
        'seed': args.seed,
        'shards': args.shards,
        'num_traj': args.num_traj,
        'episode_counts': _episode_counts(args.num_traj, args.shards),
        'overrides': list(args.overrides),
    }
    try:
        counts, seeds = manifest['episode_counts'], manifest['world_seeds']
        valid_counts = (isinstance(counts, list) and len(counts) == args.shards
                        and all(type(count) is int and count > 0 for count in counts))
        valid_seeds = (isinstance(seeds, list) and len(seeds) == args.shards
                       and all(type(seed) is int for seed in seeds))
        spans = sorted((seed, seed + count + 64) for seed, count in zip(seeds, counts))
        non_overlapping = all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
    except (KeyError, TypeError):
        valid_counts = valid_seeds = non_overlapping = False
    if not (valid_counts and valid_seeds and non_overlapping):
        raise ValueError('invalid collection manifest')
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected:
        raise ValueError('resume arguments do not match collection manifest')


def _datasets(group, prefix=''):
    paths = []
    for name in group:
        item = group[name]
        if isinstance(item, h5py.Dataset):
            paths.append(prefix + name)
        else:
            paths.extend(_datasets(item, prefix + name + '/'))
    return paths


def validate_shard(path, expected_episodes):
    with h5py.File(path, 'r') as f:
        if 'ep_len' not in f or 'ep_offset' not in f:
            raise ValueError(f'{path} is missing ep_len or ep_offset')
        lens, offsets = f['ep_len'][:], f['ep_offset'][:]
        if lens.ndim != 1 or offsets.ndim != 1:
            raise ValueError(f'{path}: invalid episode arrays')
        if not np.issubdtype(lens.dtype, np.integer):
            raise ValueError(f'{path}: ep_len must be an integer array')
        if not np.issubdtype(offsets.dtype, np.integer):
            raise ValueError(f'{path}: ep_offset must be an integer array')
        if len(lens) != expected_episodes:
            raise ValueError(f'{path}: expected {expected_episodes} episodes, found {len(lens)}')
        if len(offsets) != len(lens):
            raise ValueError(f'{path}: invalid episode arrays')
        if np.any(lens <= 0):
            raise ValueError(f'{path}: ep_len must be positive')
        expected_offsets = np.concatenate(([0], np.cumsum(lens)[:-1]))
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError(f'{path}: non-contiguous ep_offset')
        rows = int(lens.sum())
        for column in _datasets(f):
            if column in ('ep_len', 'ep_offset'):
                continue
            if f[column].ndim == 0:
                raise ValueError(f'{path}: scalar column {column} is not supported')
            if f[column].shape[0] != rows:
                raise ValueError(f'{path}: column {column} has {f[column].shape[0]} rows; expected {rows}')
    return len(lens), rows


def remove_incomplete_partial(path, shard_dir):
    path = path.resolve()
    shard_dir = shard_dir.resolve()
    if path.parent != shard_dir:
        raise ValueError(f'refusing to remove partial outside shard directory: {path}')
    if path.suffixes[-2:] != ['.partial', '.h5']:
        raise ValueError(f'refusing to remove non-partial shard: {path}')
    path.unlink()


def recover_shards(manifest, final, shard_dir):
    complete = set()
    for index, expected_episodes in enumerate(manifest['episode_counts']):
        shard = shard_path(final, index)
        partial = partial_shard_path(final, index)
        if shard.exists():
            validate_shard(shard, expected_episodes)
            complete.add(index)
        elif partial.exists():
            try:
                validate_shard(partial, expected_episodes)
            except (OSError, ValueError):
                remove_incomplete_partial(partial, shard_dir)
            else:
                os.replace(partial, shard)
                complete.add(index)
    return complete


def merge_shards_atomic(final, shards, expected_episodes, env):
    partial = merge_partial_path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(shards[0], partial)
    if len(shards) > 1:
        merge = [sys.executable, str(HERE / 'merge_h5_shards.py'), '--out', str(partial),
                 *map(str, shards[1:])]
        subprocess.run(merge, cwd=ROOT, env=env, check=True)
    validate_shard(partial, expected_episodes)
    os.replace(partial, final)


def terminate_active_children(procs):
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def _raise_interrupted(_signum, _frame):
    raise KeyboardInterrupt


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shards', type=positive_int, required=True)
    ap.add_argument('--num-traj', type=positive_int, default=10000)
    ap.add_argument('--seed', type=int, default=3072)
    ap.add_argument('--out', default='ogbench/cube_single_expert_pcd.h5',
                    help='final dataset name, relative to $STABLEWM_HOME/datasets')
    ap.add_argument('--cache-dir', default=None, help='override $STABLEWM_HOME')
    ap.add_argument('--no-merge', action='store_true')
    ap.add_argument('--resume', action='store_true', help='resume from the existing collection manifest')
    ap.add_argument('--threads', type=positive_int, default=2, help='OMP/MKL threads per shard')
    ap.add_argument('--max-parallel', type=positive_int, default=None, help='shards running at once (default: all)')
    ap.add_argument('--script', default='collect_cube_pcd.py', help='per-shard collection script in scripts/data (collect_cube_pix.py for the pixel dataset)')
    ap.add_argument('overrides', nargs='*', help='extra hydra overrides for every shard')
    return ap


def main():
    args = build_parser().parse_args()

    cache_dir = Path(args.cache_dir or os.environ.get('STABLEWM_HOME', '~/.stable-wm')).expanduser()
    run = datetime.now().strftime('%Y-%m-%d/%H-%M-%S') + f'_shards{args.shards}'
    final = cache_dir / 'datasets' / args.out
    shard_dir = final.parent / 'shards'
    manifest_path = shard_dir / 'collection_manifest.json'
    if final.exists():
        sys.exit(f'refusing to run: final dataset already exists: {final}')
    if args.resume:
        if not shard_dir.exists():
            sys.exit(f'cannot resume without collection manifest: {manifest_path}')
    else:
        try:
            shard_dir.mkdir(parents=True)
        except FileExistsError:
            sys.exit(f'refusing to run: {shard_dir} already exists (HDF5 output must be fresh)')
    try:
        lock = acquire_shard_lock(shard_dir)
    except RuntimeError as exc:
        sys.exit(str(exc))
    old_handlers = {signum: signal.signal(signum, _raise_interrupted)
                    for signum in (signal.SIGINT, signal.SIGTERM)}
    procs, logs = [], []
    try:
        if args.resume:
            if not manifest_path.exists():
                sys.exit(f'cannot resume without collection manifest: {manifest_path}')
            manifest = load_manifest(manifest_path)
            try:
                assert_resume_matches(manifest, args, final)
            except ValueError as exc:
                sys.exit(str(exc))
            stale_merge = merge_partial_path(final)
            if stale_merge.exists():
                stale_merge.unlink()
        else:
            manifest = make_manifest(args, final)
            write_manifest_atomic(manifest_path, manifest)
        log_dir = cache_dir / 'logs' / 'collect' / run
        log_dir.mkdir(parents=True, exist_ok=True)
        counts, seeds = manifest['episode_counts'], manifest['world_seeds']
        completed = recover_shards(manifest, final, shard_dir) if args.resume else set()

        env = dict(os.environ, MUJOCO_GL='egl', OMP_NUM_THREADS=str(args.threads),
                   MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads))
        max_par = args.max_parallel or args.shards
        t0 = time.time()
        launched = []
        for i, (sd, n) in enumerate(zip(seeds, counts)):
            if i in completed:
                continue
            while sum(p.poll() is None for p in procs) >= max_par:
                time.sleep(5)
            partial = partial_shard_path(final, i)
            cmd = [sys.executable, str(HERE / args.script),
                   f'num_traj={n}', f'world_seed={sd}', f'seed={args.seed}',
                   f'cache_dir={cache_dir}', f'dataset_name={partial.relative_to(cache_dir / "datasets")}',
                   f'hydra.run.dir={log_dir / f"shard_{i:03d}"}', *args.overrides]
            log = open(log_dir / f'shard_{i:03d}.log', 'w'); logs.append(log)
            procs.append(subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),)))
            launched.append(i)
        print(f'launched {len(procs)} shards ({counts[0]}..{counts[-1]} episodes each), logs: {log_dir}', flush=True)

        failed = []
        for i, proc in zip(launched, procs):
            if proc.wait() != 0:
                failed.append(i)
                continue
            partial = partial_shard_path(final, i)
            try:
                validate_shard(partial, counts[i])
                os.replace(partial, shard_path(final, i))
            except (OSError, ValueError) as exc:
                print(f'shard {i:03d} validation failed: {exc}', file=sys.stderr, flush=True)
                failed.append(i)
        print(f'all shards finished in {(time.time()-t0)/60:.1f} min; failed: {failed or "none"}', flush=True)
        if failed:
            sys.exit(1)
        if args.no_merge:
            return
        shards = [shard_path(final, i) for i in range(args.shards)]
        merge_shards_atomic(final, shards, args.num_traj, env)
        print(f'total wall {(time.time()-t0)/60:.1f} min', flush=True)
    except BaseException:
        terminate_active_children(procs)
        raise
    finally:
        for log in logs:
            log.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        release_shard_lock(lock)


if __name__ == '__main__':
    main()
