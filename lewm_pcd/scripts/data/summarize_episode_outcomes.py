"""Read-only source-episode outcome statistics; not a training label or CEM score."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np


def classify_episode(success, cube, target):
    success = np.asarray(success)
    if success.dtype.kind not in 'bifu' or not np.isfinite(success).all() or not np.isin(success, [0, 1]).all():
        raise ValueError('success must contain finite boolean or 0/1 values')
    success = success.astype(bool).reshape(-1)
    cube, target = np.asarray(cube), np.asarray(target)
    if len(success) < 2 or cube.shape != (len(success), 3) or target.shape != cube.shape:
        raise ValueError('inconsistent episode shapes')
    if not np.isfinite(cube).all() or not np.isfinite(target).all():
        raise ValueError('nonfinite cube/goal coordinates')
    reached_goal = bool(success[1:].any())
    lifted = bool(cube[1:, 2].max() > .035)
    return {
        'reached_goal': reached_goal,
        'lifted': lifted,
        'initial_goal_error': float(np.linalg.norm(cube[0] - target[0])),
        'final_goal_error': float(np.linalg.norm(cube[-1] - target[-1])),
        'failure_reason': 'none' if reached_goal else ('lifted_not_placed' if lifted else 'never_lifted'),
    }


def _column(handle, name):
    return handle[name if name in handle else name.replace('/', '_')]


def summarize_dataset(path):
    records, seen, duplicates = [], set(), 0
    with h5py.File(path, 'r') as f:
        lengths, offsets = f['ep_len'][:], f['ep_offset'][:]
        if lengths.ndim != 1 or offsets.ndim != 1 or lengths.dtype.kind not in 'iu' or offsets.dtype.kind not in 'iu':
            raise ValueError('episode lengths/offsets must be one-dimensional integer arrays')
        if len(lengths) == 0 or not np.all(lengths > 1):
            raise ValueError('empty or invalid episode lengths')
        if not np.array_equal(offsets, np.r_[0, np.cumsum(lengths[:-1])]):
            raise ValueError('noncontiguous episode offsets')
        cube = _column(f, 'privileged/block_0_pos')
        target = _column(f, 'privileged/target_block_pos')
        for col in (f['success'], cube, target, f['qpos'], f['qvel']):
            if col.shape[0] != int(lengths.sum()):
                raise ValueError('column length mismatch')
        for episode, (offset, length) in enumerate(zip(offsets, lengths)):
            offset, length = int(offset), int(length)
            sl = slice(offset, offset + length)
            result = classify_episode(f['success'][sl], cube[sl], target[sl])
            digest = hashlib.sha256()
            for values in (f['qpos'][offset], f['qvel'][offset], target[offset]):
                if not np.isfinite(values).all():
                    raise ValueError('nonfinite initial state')
                digest.update(np.asarray(values, dtype='<f8').tobytes())
            state_hash = digest.hexdigest()
            duplicates += int(state_hash in seen)
            seen.add(state_hash)
            records.append({'episode': episode, 'initial_state_sha256': state_hash, **result})
    successes = sum(r['reached_goal'] for r in records)
    lifted = sum(r['lifted'] for r in records)
    return {
        'path': str(Path(path).resolve()),
        'definition': 'any post-reset success frame in the 200-step source episode; not oracle subtask success',
        'episodes': len(records), 'successes': successes, 'failures': len(records) - successes,
        'success_rate': successes / len(records), 'lifted': lifted, 'lift_rate': lifted / len(records),
        'duplicate_initial_states': duplicates,
        'initially_within_goal': sum(r['initial_goal_error'] <= .04 for r in records),
        'failure_reasons': dict(Counter(r['failure_reason'] for r in records if not r['reached_goal'])),
        'records': records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve() == args.dataset.resolve() or (args.out.exists() and args.out.samefile(args.dataset)):
        parser.error('report output must not alias the source dataset')
    if args.out.exists():
        parser.error('report already exists; choose a fresh output path')
    report = summarize_dataset(args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=args.out.parent, prefix=args.out.name + '.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(args.out)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, sort_keys=True), flush=True)
    if report['duplicate_initial_states']:
        raise SystemExit('duplicate initial-state/goal combinations found; dataset not accepted')


if __name__ == '__main__':
    main()
