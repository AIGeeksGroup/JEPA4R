# Moved from OGBench-cube@4fbb0c6:scripts/prepare_official_assets.py (verbatim except: imports / paths adapted to the
# package; lines wrapped at 120 columns, unused imports removed)
#!/usr/bin/env python3
"""Prepare final Relocate assets from the original official 100-demo pickle.

Only ``images`` is removed. The original 1536-D cached ``observations`` and
39-D physical ``state_obs`` remain distinct and unchanged, as do raw actions,
initial states and every other nonvisual field. No simulator is imported.
Formal preparation requires the canonical official raw file and the recorded
WM environment with NumPy 1.26.4 for reproducible protocol-4 initialization hashes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from numbers import Real
from pathlib import Path
import pickle
import re
import sys
import traceback

import numpy as np

# This entry also works from scripts/ without a caller-provided PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from jepa4r.data.relocate import official_contract as official_content

demo_content_record = official_content.demo_content_record
content_digest = official_content.content_digest
validate_content_records = official_content.validate_content_records


OFFICIAL_COUNT = 100
OFFICIAL_RAW_SHA256 = 'c71c7af5dda7b9a3d133a7e2baaa821bfe1ab7c4445dfde5624b947d13865694'
OFFICIAL_INITIALS_SEMANTIC_SHA256 = '064a33e46fdc7407c6b0599f16413c85aab579c8fcccc957606e39da067ad2a8'
OFFICIAL_NUMPY_VERSION = '1.26.4'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def initials_semantic_sha256(records):
    selected = [{key: row[key] for key in
        ('demo_index', 'original_init_sha256', 'obj_pos', 'target_pos')} for row in records]
    payload = json.dumps(selected, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def require_canonical_initials(records):
    digest = initials_semantic_sha256(records)
    if len(records) != OFFICIAL_COUNT or digest != OFFICIAL_INITIALS_SEMANTIC_SHA256:
        raise ValueError('canonical official initial-state digest mismatch: '
            f'expected {OFFICIAL_INITIALS_SEMANTIC_SHA256}, got {digest}; '
            f'formal preparation requires the recorded WM NumPy {OFFICIAL_NUMPY_VERSION}')
    return digest


def write_json(path, value, *, atomic=False):
    path = Path(path)
    target = path.with_suffix(path.suffix + '.tmp') if atomic else path
    with target.open('w' if atomic else 'x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    if atomic:
        target.replace(path)


def finite_real(value, shape, name, *, allow_object=False, allow_bool=False):
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f'{name} must have shape {shape}')
    checked = array
    if array.dtype.kind == 'O' and allow_object:
        if not all((isinstance(v, Real) and (allow_bool or not isinstance(v, (bool, np.bool_))))
                   or (allow_bool and isinstance(v, (bool, np.bool_))) for v in array.flat):
            raise ValueError(f'{name} must contain real numeric values')
        checked = np.asarray(array, dtype=np.float64)
    elif array.dtype.kind not in ('f', 'i', 'u') and not (allow_bool and array.dtype.kind == 'b'):
        raise ValueError(f'{name} must contain real numeric values')
    if not np.isfinite(checked).all():
        raise ValueError(f'{name} must contain finite values')
    return array


def validate_demo(demo):
    if not isinstance(demo, dict):
        raise ValueError('each official demonstration must be a dictionary')
    for key, shape in (('actions', (200, 30)), ('state_obs', (200, 39)),
                       ('observations', (200, 1536)), ('rewards', (200,))):
        finite_real(demo.get(key), shape, key)
    infos = demo.get('env_infos')
    if not isinstance(infos, dict):
        raise ValueError('env_infos must be a dictionary')
    finite_real(infos.get('proprio'), (200, 30), 'env_infos.proprio')
    achieved = finite_real(infos.get('goal_achieved'), (200,), 'env_infos.goal_achieved',
                           allow_object=True, allow_bool=True)
    if not np.isin(achieved, [0, 1]).all():
        raise ValueError('env_infos.goal_achieved must contain boolean success values')
    initial = demo.get('init_state_dict')
    if not isinstance(initial, dict):
        raise ValueError('init_state_dict must be a dictionary')
    for key, shape in (('hand_qpos', (30,)), ('obj_pos', (3,)), ('target_pos', (3,)),
                       ('palm_pos', (3,)), ('qpos', (36,)), ('qvel', (36,))):
        finite_real(initial.get(key), shape, 'init_state_dict.' + key, allow_object=True)
    return {key: np.asarray(initial[key], dtype=np.float64).tolist()
            for key in ('obj_pos', 'target_pos')}


def prepare_assets(source, source_sha256, output, *, expected_count=OFFICIAL_COUNT):
    """Export immutable per-demo copies; a smaller count is a CPU test hook.

    The command-line interface accepts only the paper's 100 demonstrations.
    Smaller test fixtures are explicitly marked ``formal_ready=false``.
    """
    source = Path(source).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    expected_sha = str(source_sha256).lower()
    if not re.fullmatch('[0-9a-f]{64}', expected_sha):
        raise ValueError('source SHA256 must contain 64 hexadecimal characters')
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError('expected_count must be a positive integer')
    output.mkdir(parents=True, exist_ok=False)
    producer = Path(__file__).resolve()
    state = dict(stage='hash_source', started_utc=datetime.now(timezone.utc).isoformat(),
        updated_utc=None, completed_count=0, failed_count=0, actual_exit_code=None,
        configuration=dict(source=str(source), expected_source_sha256=expected_sha,
                           expected_count=expected_count, output=str(output)))

    def update(stage, **values):
        state.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
        write_json(output / 'STATUS.json', state, atomic=True)
        with (output / 'events.jsonl').open('a', encoding='utf-8') as log:
            event = dict(stage=stage, updated_utc=state['updated_utc'],
                         completed_count=state['completed_count'])
            event.update(values)
            log.write(json.dumps(event, allow_nan=False) + '\n')

    try:
        update('validate_official_source')
        if expected_count == OFFICIAL_COUNT:
            if expected_sha != OFFICIAL_RAW_SHA256:
                raise ValueError('formal 100-demo preparation requires the canonical official raw '
                                 f'SHA256 {OFFICIAL_RAW_SHA256}')
            if np.__version__ != OFFICIAL_NUMPY_VERSION:
                raise ValueError(f'formal preparation requires recorded WM NumPy {OFFICIAL_NUMPY_VERSION}; '
                                 f'found NumPy {np.__version__}')
        update('hash_source')
        before = source.stat()
        digest = hashlib.sha256()
        read_bytes = 0
        next_report = 64 << 20
        with source.open('rb') as stream:
            for block in iter(lambda: stream.read(4 << 20), b''):
                digest.update(block)
                read_bytes += len(block)
                if read_bytes >= next_report:
                    update('hash_source', bytes_read=read_bytes)
                    next_report = read_bytes + (64 << 20)
            actual_sha = digest.hexdigest()
            if actual_sha != expected_sha:
                raise ValueError(f'source SHA256 mismatch: expected {expected_sha}, got {actual_sha}')
            update('load_source', source_sha256=actual_sha, bytes_read=read_bytes)
            stream.seek(0)
            demos = pickle.load(stream)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('official source changed during preparation')
        if not isinstance(demos, list) or len(demos) != expected_count:
            raise ValueError(f'require {expected_count} official demonstrations in their original order')
        demo_dir = output / 'demos'
        demo_dir.mkdir()
        records, initial_records, policy_records = [], [], []
        for index, demo in enumerate(demos):
            coordinates = validate_demo(demo)
            policy_records.append(demo_content_record(index, demo))
            initial_sha = hashlib.sha256(pickle.dumps(demo['init_state_dict'], protocol=4)).hexdigest()
            nonvisual = {key: value for key, value in demo.items() if key != 'images'}
            payload = pickle.dumps(nonvisual, protocol=4)
            path = demo_dir / f'demo_{index:04d}.pickle'
            with path.open('xb') as stream:
                stream.write(payload)
            produced_sha = file_sha256(path)
            if produced_sha != hashlib.sha256(payload).hexdigest():
                raise OSError(f'written demonstration differs: {path.name}')
            records.append(dict(demo_index=index, relative_path=path.relative_to(output).as_posix(),
                bytes=path.stat().st_size, sha256=produced_sha, original_init_sha256=initial_sha))
            initial_records.append(dict(demo_index=index, source_id=f'official-relocate-{index:03d}',
                                        original_init_sha256=initial_sha, **coordinates))
            update('export_demos', completed_count=index + 1)
        semantic_sha = (require_canonical_initials(initial_records) if expected_count == OFFICIAL_COUNT
                        else initials_semantic_sha256(initial_records))
        update('validate_initials', initials_semantic_sha256=semantic_sha)
        policy_sha = content_digest(policy_records)
        policy_validation = (validate_content_records(policy_records)
                             if expected_count == OFFICIAL_COUNT else None)
        update('validate_policy_content', policy_critical_content_sha256=policy_sha)
        manifest = dict(schema='official-adroit-relocate-light-source-v1', source=str(source),
            source_sha256=actual_sha, source_bytes=before.st_size, producer=str(producer),
            source_raw_sha256=actual_sha, canonical_raw_sha256=OFFICIAL_RAW_SHA256,
            initials_semantic_sha256=semantic_sha,
            policy_critical_content_sha256=policy_sha,
            policy_critical_content_validation=policy_validation,
            content_checker_sha256=file_sha256(official_content.__file__),
            producer_sha256=file_sha256(producer), python=sys.version, numpy=np.__version__,
            demo_count=expected_count, formal_ready=expected_count == OFFICIAL_COUNT,
            removed_fields=['images'], nonvisual_fields_unchanged=True,
            observations_semantics='Original 1536-D visual embedding; physical state_obs is 39-D.',
            original_init_sha256_algorithm='sha256(pickle.dumps(original_init_state_dict,protocol=4))',
            demo_files=records)
        manifest_path = output / 'manifest.json'
        write_json(manifest_path, manifest)
        write_json(output / 'rh56_official_task_init.json', dict(schema='official-task-init-v1',
            source_manifest=str(manifest_path), source_sha=file_sha256(manifest_path),
            source_raw_sha256=actual_sha, canonical_raw_sha256=OFFICIAL_RAW_SHA256,
            initials_semantic_sha256=semantic_sha,
            policy_critical_content_sha256=policy_sha,
            records=initial_records))
        update('complete', actual_exit_code=0, formal_ready=manifest['formal_ready'])
        (output / 'actual_exit_code').write_text('0\n', encoding='ascii')
        return manifest
    except Exception as error:
        update('failed', failed_count=1, error=f'{type(error).__name__}: {error}', actual_exit_code=1)
        (output / 'actual_exit_code').write_text('1\n', encoding='ascii')
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path, help='A new output directory.')
    parser.add_argument('--expected-count', type=int, choices=[OFFICIAL_COUNT], default=OFFICIAL_COUNT,
                        help='Fixed to the final paper protocol: 100 official demonstrations.')
    args = parser.parse_args(argv)
    try:
        manifest = prepare_assets(args.source, args.source_sha256, args.output,
                                  expected_count=args.expected_count)
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps(dict(output=str(args.output.resolve()), demo_count=manifest['demo_count'],
                          source_sha256=manifest['source_sha256'], actual_exit_code=0)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
