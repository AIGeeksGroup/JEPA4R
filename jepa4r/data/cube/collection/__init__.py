# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/collection/__init__.py (verbatim except: imports / paths
# adapted to the package; lines wrapped at 120 columns)
"""Current-paper Cube paired collection; explicit execution only."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
HERE = Path(__file__).resolve().parent
LEWM = HERE.parents[3]   # this repository (jepa4r/, third_party/)
REPO = LEWM
SOURCES = LEWM / 'configs' / 'env' / 'cube' / 'collection' / 'sources.json'   # the pinned collection sources
OGBENCH = Path(os.environ.get('JEPA4R_OGBENCH') or LEWM / 'third_party/ogbench')   # OGBench a977253 checkout
ENV_TYPES = ('single', 'double', 'triple', 'quadruple')
HORIZONS = (200, 500, 1000, 600)
STRATEGIES = (
    ('expert_low_noise', 400, 40),
    ('expert_high_noise', 800, 80),
    ('expert_random_branch100', 200, 20),
    ('expert_random_branch45', 200, 20),
    ('random', 400, 40),
)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda : stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def _save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')
    temporary.replace(path)

def source_manifest():
    return json.loads((SOURCES).read_text(encoding='utf8'))

def _safe(root, name):
    name = str(name)
    pure = PurePosixPath(name)
    if pure.is_absolute() or '..' in pure.parts or '\\' in name or (':' in name):
        raise ValueError('unsafe source path: ' + name)
    candidate = (Path(root) / name).resolve()
    if not candidate.is_relative_to(Path(root).resolve()):
        raise ValueError('source path escapes root: ' + name)
    return candidate

def materialize(output, *, manifest=None):
    """Validate every source before creating a fresh fixed-child package tree."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = source_manifest() if manifest is None else manifest
    (copies, reused) = ([], {})
    seen = set()
    for (destination, row) in manifest['files'].items():
        target = _safe(output, destination)
        if str(target).casefold() in seen:
            raise ValueError('duplicate source path: ' + destination)
        seen.add(str(target).casefold())
        source = (_safe(OGBENCH, row['repository_path'][len('third_party/ogbench/'):])
                  if row['repository_path'].startswith('third_party/ogbench/') else _safe(REPO, row['repository_path']))
        if sha(source) != row['sha256']:
            raise ValueError('frozen collection source differs: ' + str(source))
        copies.append((source, target, row['sha256']))
    for (name, expected) in manifest['reused_swm_files'].items():
        source = _safe(REPO, name)
        if sha(source) != expected:
            raise ValueError('bundled SWM source differs: ' + name)
        reused[str(source)] = expected
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for (source, target, expected) in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as incoming, target.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        if sha(target) != expected:
            raise ValueError('materialized source differs: ' + str(target))
        files[str(target)] = expected
    return dict(files=files, reused_files=reused, source_manifest_sha256=sha(SOURCES))

def _columns(handle):
    import h5py
    result = []

    def walk(group, prefix=''):
        for name in sorted(group):
            value = group[name]
            if isinstance(value, h5py.Group):
                walk(value, prefix + name + '/')
            else:
                result.append(prefix + name)
    walk(handle)
    return [name for name in result if name not in ('ep_len', 'ep_offset')]

def validate_pair(rgb, pcd, *, episodes, horizon, cube_count, world_seed):
    """Read actual rows, executed actions and both modalities; no simulator imports."""
    import h5py
    import numpy as np
    rows = episodes * (horizon + 1)
    with h5py.File(rgb, 'r') as a, h5py.File(pcd, 'r') as b:
        for handle in (a, b):
            if not np.array_equal(handle['ep_len'][:], np.full(episodes, horizon + 1)) or not np.array_equal(
                handle['ep_offset'][:], np.arange(episodes) * (horizon + 1)
            ):
                raise ValueError('episode length/offset differs')
            if any((handle[name].shape[0] != rows for name in _columns(handle))):
                raise ValueError('column row alignment differs')
        schema = (
            (a, 'pixels', (rows, 224, 224, 3), 'uint8'),
            (b, 'pcd_xyz', (rows, 1200, 3), 'float32'),
            (b, 'pcd_rgb', (rows, 1200, 3), 'uint8'),
        )
        for (handle, name, shape, dtype) in schema:
            if handle[name].shape != shape or handle[name].dtype != np.dtype(dtype):
                raise ValueError('visual schema differs: ' + name)
        for handle in (a, b):
            for name, width in (
                ('action', 5),
                ('observation', 19 + 9 * cube_count),
                ('qpos', 14 + 7 * cube_count),
                ('qvel', 14 + 6 * cube_count),
            ):
                if handle[name].shape != (rows, width):
                    raise ValueError('state/action schema differs: ' + name)
            if handle['action'].dtype != np.dtype('float32'):
                raise ValueError('executed action must be float32')
        shared = sorted(set(_columns(a)) & set(_columns(b)))
        if not {'action', 'observation', 'qpos', 'qvel', 'collection_reset_seed', 'step_idx'} <= set(shared):
            raise ValueError('paired state/action provenance missing')
        for name in shared:
            for start in range(0, rows, 256):
                (left, right) = (a[name][start:start + 256], b[name][start:start + 256])
                if not np.array_equal(left, right, equal_nan=left.dtype.kind in 'fc'):
                    raise ValueError('paired column differs: ' + name)
        (seeds, initial) = ([], [])
        for episode in range(episodes):
            start = episode * (horizon + 1)
            action = a['action'][start:start + horizon + 1]
            if not np.isfinite(action[:-1]).all() or not np.isnan(action[-1]).all():
                raise ValueError('executed action or terminal sentinel differs')
            seed = world_seed + episode
            if not np.all(a['collection_reset_seed'][start : start + horizon + 1] == seed) or not np.array_equal(
                a['step_idx'][start : start + horizon + 1], np.arange(horizon + 1)
            ):
                raise ValueError('actual reset seed/step sequence differs')
            state = np.concatenate([np.asarray(a[key][start], dtype='<f8').reshape(-1) for key in ('qpos', 'qvel')])
            seeds.append(seed)
            initial.append(hashlib.sha256(state.tobytes()).hexdigest())
        if len(set(initial)) != episodes:
            raise ValueError('duplicate physical initial state within collection shard')
        for (handle, names) in ((a, ('observation', 'qpos', 'qvel')), (b, ('pcd_xyz',))):
            for name in names:
                for start in range(0, rows, 256):
                    if not np.isfinite(handle[name][start:start + 256]).all():
                        raise ValueError('nonfinite actual data: ' + name)
    return dict(episodes=episodes, rows=rows, reset_seeds=seeds, initial_state_sha256=initial, actual_exit_code=0)

def merge_raw(shards, output):
    """Concatenate all original raw columns as VDS; retain every physical shard."""
    import h5py
    import numpy as np
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    (sources, lengths, keys, schemas) = ([], [], None, None)
    for path in map(Path, shards):
        with h5py.File(path, 'r') as file:
            columns = _columns(file)
            schema = {name: (file[name].shape[1:], file[name].dtype) for name in columns}
            if keys is not None and (columns != keys or schema != schemas):
                raise ValueError('raw merge source schema differs')
            (keys, schemas) = (columns, schema)
            lens = file['ep_len'][:]
            if not np.array_equal(file['ep_offset'][:], np.r_[0, np.cumsum(lens)[:-1]]):
                raise ValueError('raw merge source offsets differ')
            sources.append((path.resolve(), int(lens.sum())))
            lengths.extend(map(int, lens))
    if not sources:
        raise ValueError('raw merge requires shards')
    output.parent.mkdir(parents=True, exist_ok=True)
    total = sum((rows for (_, rows) in sources))
    with h5py.File(output, 'x', libver='latest') as file:
        for name in keys:
            (shape, dtype) = schemas[name]
            layout = h5py.VirtualLayout((total, *shape), dtype=dtype)
            offset = 0
            for (path, rows) in sources:
                source = h5py.VirtualSource(str(path), name, shape=(rows, *shape))
                layout[offset:offset + rows] = source
                offset += rows
            file.create_virtual_dataset(name, layout)
        file['ep_len'] = np.asarray(lengths, dtype=np.int32)
        file['ep_offset'] = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
    return str(output.resolve())

def _file_record(path):
    path = Path(path)
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('file changed during hash: ' + str(path))
    return dict(sha256=digest, bytes=after.st_size, mtime_ns=after.st_mtime_ns)

def _part_acceptance(plan, part, runtime):
    """The same producer/QC checks apply to a fresh part and a resumed part."""
    code = runtime / 'code/lewm_pcd'
    directory = Path(part['directory'])
    configs = {str(directory / 'config.json'): sha(directory / 'config.json')}
    if plan['cube_count'] == 4:
        configs.update(_validate_adapter_binding(runtime, directory))
    manifests = []
    for name in ('rgb', 'pcd'):
        path = Path(part[name] + '.manifest.json')
        value = json.loads(path.read_text(encoding='utf8'))
        if (value.get('stage') != 'complete' or value.get('exit_code') != 0 or
                value.get('completed') != part['episodes'] or
                value.get('pix_episodes_written') != part['episodes'] or
                value.get('pcd_episodes_written') != part['episodes']):
            raise ValueError('native producer manifest did not close successfully')
        cfg = value['config']
        for key, wanted in (('task_id', part['task']), ('strategy', part['strategy']),
                            ('split', part['split']), ('world_seed', part['world_seed']),
                            ('policy_seed', part['policy_seed']),
                            ('horizon_actions', plan['raw_horizon']),
                            ('requested_episodes', part['episodes'])):
            if cfg.get(key) != wanted:
                raise ValueError('native producer config differs: ' + key)
        if (cfg.get('collector_sha256') != sha(code / 'scripts/data/collect_cube_pcd.py') or
                cfg.get('protocol_sha256') != sha(code / 'scripts/data/collection_protocol.py')):
            raise ValueError('actual native producer source binding differs')
        if plan['cube_count'] == 4:
            override, camera = cfg.get('horizon_override', {}), cfg.get('camera_override', {})
            if (override.get('actions') != 600 or override.get('observations') != 601 or
                    override.get('effective_TimeLimit') != 600 or override.get('terminal_source_enum') != 2 or
                    override.get('adapter_SHA256') != sha(runtime / 'collection_adapters/quad_h600.py') or
                    camera.get('local_Z_pullback_m') != 0.1 or
                    camera.get('adapter_sha256') != sha(runtime / 'collection_adapters/camera_pullback.py')):
                raise ValueError('actual Quad H600/camera overlay binding differs')
        manifests.append(value)
        configs[str(path)] = sha(path)
    if manifests[0] != manifests[1]:
        raise ValueError('paired producer manifests differ')
    qc = validate_pair(part['rgb'], part['pcd'], episodes=part['episodes'],
                       horizon=plan['raw_horizon'], cube_count=plan['cube_count'],
                       world_seed=part['world_seed'])
    return dict(part=part, qc=qc, producer_manifest=manifests[0], configs=configs,
                data_files={part[name]: _file_record(part[name]) for name in ('rgb', 'pcd')})


def _accepted_parts(plan):
    records = {}
    for canonical in plan['parts']:
        marker = Path(canonical['directory']) / 'ACCEPTED.json'
        if not marker.exists():
            continue
        value = json.loads(marker.read_text(encoding='utf8'))
        if value.get('canonical_part') != canonical or value.get('actual_exit_code') != 0:
            raise ValueError('completed part identity changed: ' + str(marker))
        actual = value['part']
        if any(actual.get(k) != v for k, v in canonical.items()
               if k not in ('directory', 'rgb', 'pcd')):
            raise ValueError('completed part seed/task/split identity changed')
        root = Path(canonical['directory']).resolve()
        if not Path(actual['directory']).resolve().is_relative_to(root):
            raise ValueError('completed part directory escapes its canonical shard')
        wanted_paths = {
            'rgb': str(Path(actual['directory']) / 'datasets/pair_pix.h5'),
            'pcd': str(Path(actual['directory']) / 'datasets/pair_pcd.h5'),
        }
        if any(actual.get(key) != path for key, path in wanted_paths.items()):
            raise ValueError('completed part raw paths differ from its actual directory')
        if set(value['data_files']) != set(wanted_paths.values()):
            raise ValueError('completed part hashed data files differ from merge paths')
        for name, expected in value['configs'].items():
            if sha(name) != expected:
                raise ValueError('completed producer/config checksum changed: ' + name)
        for name, expected in value['data_files'].items():
            if sha(name) != expected['sha256']:
                raise ValueError('completed raw data checksum changed: ' + name)
        if (Path(actual['directory']) / 'actual_exit_code').read_text().strip() != '0':
            raise ValueError('completed child exit changed')
        records[canonical['index']] = value
    return records


def _attempt_part(canonical):
    """Keep failed output bytes; retry the same seeded part in a new directory."""
    directory = Path(canonical['directory'])
    if directory.exists():
        attempts = [directory]
        attempts.extend(sorted((directory / 'attempts').glob('attempt_*')))
        for previous in attempts:
            exit_path = previous / 'actual_exit_code'
            if not exit_path.is_file():
                raise FileExistsError(
                    'prior collection attempt has no exit evidence; inspect it before recovery: ' + str(previous)
                )
            try:
                int(exit_path.read_text().strip())
            except ValueError as error:
                raise ValueError('prior collection attempt exit evidence is malformed') from error
        parent = directory / 'attempts'
        parent.mkdir(exist_ok=True)
        index = 1
        while (parent / f'attempt_{index:03d}').exists():
            index += 1
        directory = parent / f'attempt_{index:03d}'
    directory.mkdir(parents=True, exist_ok=False)
    return dict(canonical, directory=str(directory),
                rgb=str(directory / 'datasets/pair_pix.h5'),
                pcd=str(directory / 'datasets/pair_pcd.h5'))


def run(plan, *, resume=False, part_ids=None):
    """Explicit execution. Completion requires actual child exits and data QC."""
    expected = build_plan(
        plan['cube_count'], plan['output_root'], plan['mode'], plan['python'], collection_seed=plan['collection_seed']
    )
    if plan != expected:
        raise ValueError('collection plan differs from fixed protocol')
    root = Path(plan['output_root'])
    if part_ids is not None:
        if len(part_ids) != len(set(part_ids)):
            raise ValueError('duplicate requested part IDs')
        if any(type(i) is not int or i not in range(len(plan['parts'])) for i in part_ids):
            raise ValueError('requested part ID is outside the canonical plan')
    if root.exists():
        if not resume:
            raise FileExistsError(root)
        if json.loads((root / 'plan.json').read_text(encoding='utf8')) != plan:
            raise ValueError('resume plan differs from the original canonical plan')
        state = json.loads((root / 'status.json').read_text(encoding='utf8'))
        if state.get('driver_sha256') != sha(__file__):
            raise ValueError('collection driver source changed; preserve the frozen producer')
        with (root / 'status_history.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(state) + '\n')
    else:
        root.mkdir(parents=True, exist_ok=False)
        started = datetime.now(timezone.utc).isoformat()
        state = dict(stage='materializing', started_at=started, updated_at=started,
                     actual_exit_code=None, completed_count=0, failed_count=0, plan=plan,
                     driver_sha256=sha(__file__))
        _save(root / 'plan.json', plan)
        _save(root / 'status.json', state)
    started = state['started_at']
    try:
        runtime = root / 'runtime'
        binding_path = root / 'runtime_binding.json'
        if binding_path.exists():
            binding = json.loads(binding_path.read_text(encoding='utf8'))
            for name, expected_sha in {**binding['files'], **binding['reused_files']}.items():
                if sha(name) != expected_sha:
                    raise ValueError('collection runtime source changed: ' + name)
        else:
            binding = materialize(runtime)
            _save(binding_path, binding)
        code = runtime / 'code/lewm_pcd'
        env = dict(
            os.environ,
            CUBE_COLLECTION_RUNTIME_ROOT=str(runtime),
            MUJOCO_GL='egl',
            PYOPENGL_PLATFORM='egl',
            EGL_PLATFORM='surfaceless',
            STABLEWM_HOME=str(root / 'native_outputs'),
            OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',
        )
        env['PYTHONPATH'] = os.pathsep.join(
            (
                str(code),
                str(code / 'scripts/data'),
                str(LEWM / 'third_party/stable-worldmodel'),
                str(runtime / 'third_party/ogbench'),
            )
        )
        accepted = _accepted_parts(plan)
        if part_ids is not None and set(part_ids).intersection(accepted):
            raise ValueError('explicit retry of a completed part is refused')
        selected = list(range(len(plan['parts']))) if part_ids is None else part_ids
        for part_index in selected:
            if part_index in accepted:
                continue
            canonical = plan['parts'][part_index]
            part = _attempt_part(canonical)
            directory = Path(part['directory'])
            command = child_command(plan, part, runtime)
            config = dict(
                part=part,
                command=command,
                raw_horizon=plan['raw_horizon'],
                env_type=plan['env_type'],
                mode=plan['mode'],
            )
            _save(directory / 'config.json', config)
            state.update(
                stage='collecting', active_part=part['index'], updated_at=datetime.now(timezone.utc).isoformat()
            )
            _save(root / 'status.json', state)
            with (directory / 'stdout_stderr.log').open('xb') as log:
                result = subprocess.run(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            (directory / 'actual_exit_code').write_text(str(result.returncode) + '\n')
            if result.returncode != 0:
                raise RuntimeError(f"collection child {part['index']} exited {result.returncode}")
            record = _part_acceptance(plan, part, runtime)
            for previous in accepted.values():
                if (set(record['qc']['reset_seeds']).intersection(previous['qc']['reset_seeds']) or
                        set(record['qc']['initial_state_sha256']).intersection(previous['qc']['initial_state_sha256'])):
                    raise ValueError('duplicate reset seed or full initial state across shards')
            record.update(canonical_part=canonical, actual_exit_code=0)
            _save(directory / 'QC.json', record['qc'])
            marker = Path(canonical['directory']) / 'ACCEPTED.json'
            with marker.open('x', encoding='utf8') as stream:
                json.dump(record, stream, indent=2)
            accepted[part_index] = record
            state['completed_count'] = len(accepted)
            _save(root / 'status.json', state)
        if len(accepted) != len(plan['parts']):
            state.update(stage='parts_completed', actual_exit_code=0,
                         completed_count=len(accepted), updated_at=datetime.now(timezone.utc).isoformat())
            _save(root / 'status.json', state)
            (root / 'actual_exit_code').write_text('0\n')
            return dict(schema='cube_collection_partial_v1', actual_exit_code=0, complete=False,
                        completed_count=len(accepted), required_count=len(plan['parts']))
        (records, data_files, configs, seen_seeds, seen_initial) = ([], {}, {}, set(), set())
        actual_parts = []
        for canonical in plan['parts']:
            record = accepted[canonical['index']]
            part, qc = record['part'], record['qc']
            if seen_seeds.intersection(qc['reset_seeds']) or seen_initial.intersection(qc['initial_state_sha256']):
                raise ValueError('duplicate reset seed or full initial state across shards')
            seen_seeds.update(qc['reset_seeds'])
            seen_initial.update(qc['initial_state_sha256'])
            data_files.update(record['data_files'])
            configs.update(record['configs'])
            records.append(dict(part=part, qc=qc, producer_manifest=record['producer_manifest']))
            actual_parts.append(part)
        if (root / 'COLLECTION.json').exists():
            completed = json.loads((root / 'COLLECTION.json').read_text(encoding='utf8'))
            if state.get('collection_manifest_sha256') != sha(root / 'COLLECTION.json'):
                raise ValueError('completed collection manifest checksum changed or is unaccepted')
            expected_header = dict(schema='cube_fixed_collection_v1', actual_exit_code=0,
                cube_count=plan['cube_count'], mode=plan['mode'], raw_horizon=plan['raw_horizon'],
                protocol=plan['protocol'], collection_seed=plan['collection_seed'],
                split_counts=plan['split_counts'], train_episodes=plan['split_counts']['train'],
                validation_episodes=plan['split_counts']['validation'])
            if any(completed.get(k) != v for k, v in expected_header.items()):
                raise ValueError('completed collection receipt protocol/count binding differs')
            paths = completed.get('paths', {})
            if (set(paths) != {'train_rgb', 'train_pcd', 'validation_rgb', 'validation_pcd'} or
                    len(set(paths.values())) != 4 or completed.get('records') != records or
                    set(completed['data_files']) != set(data_files).union(paths.values())):
                raise ValueError('completed collection receipt paths/producer binding differs')
            for name, expected_file in completed['data_files'].items():
                if sha(name) != expected_file['sha256']:
                    raise ValueError('completed collection data checksum changed: ' + name)
            return completed
        paths = {}
        state.update(stage='merging_raw_vds', updated_at=datetime.now(timezone.utc).isoformat())
        _save(root / 'status.json', state)
        merge_root = root / 'datasets'
        if merge_root.exists():
            index = 1
            while (root / 'merge_attempts' / f'attempt_{index:03d}').exists():
                index += 1
            merge_root = root / 'merge_attempts' / f'attempt_{index:03d}' / 'datasets'
        for split in ('train', 'validation'):
            for modality in ('rgb', 'pcd'):
                key = split + '_' + modality
                paths[key] = merge_raw(
                    [p[modality] for p in actual_parts if p['split'] == split], merge_root / (key + '.h5')
                )
                data_files[paths[key]] = _file_record(paths[key])
        for split in ('train', 'validation'):
            selected = [p for p in actual_parts if p['split'] == split]
            import h5py
            import numpy as np
            with h5py.File(paths[split + '_rgb'], 'r') as file:
                actual = file['collection_reset_seed'][::plan['raw_horizon'] + 1]
                wanted = [s for p in selected for s in range(p['world_seed'], p['world_seed'] + p['episodes'])]
                if not np.array_equal(actual, wanted) or len(actual) != plan['split_counts'][split]:
                    raise ValueError('merged raw source/episode identity differs')
        for (path, expected_file) in data_files.items():
            current = Path(path).stat()
            if (current.st_size, current.st_mtime_ns) != (expected_file['bytes'], expected_file['mtime_ns']):
                raise ValueError('raw data changed after validation: ' + path)
        files = {**binding['files'], **binding['reused_files'], **configs}
        files[str(HERE / '__init__.py')] = sha(HERE / '__init__.py')
        files[str(SOURCES)] = sha(SOURCES)
        files[str(root / 'plan.json')] = sha(root / 'plan.json')
        if any((sha(path) != expected_sha for (path, expected_sha) in files.items())):
            raise ValueError('collection source/config changed during execution')
        manifest = dict(
            schema='cube_fixed_collection_v1',
            cube_count=plan['cube_count'],
            mode=plan['mode'],
            raw_horizon=plan['raw_horizon'],
            paths=paths,
            collection_seed=plan['collection_seed'],
            train_episodes=plan['split_counts']['train'],
            validation_episodes=plan['split_counts']['validation'],
            split_counts=plan['split_counts'],
            collection_protocol=plan['collection_protocol'],
            producer_binding=dict(
                files=files,
                config_sha256=files[str(root / 'plan.json')],
                source_manifest_sha256=binding['source_manifest_sha256'],
            ),
            records=records,
            data_files=data_files,
            actual_exit_code=0,
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            validation_limitations=[
                'bounded smoke is not formal numerical reproduction'
                if plan['mode'] == 'smoke'
                else 'collection acceptance does not establish trained or evaluated policy quality',
                'no training or downstream result established by collection',
            ],
            shards_retained=True,
            protocol=plan['protocol'],
            seed_schedule=plan['seed_schedule'],
        )
        _save(root / 'COLLECTION.json', manifest)
        state.update(stage='complete', actual_exit_code=0, updated_at=manifest['finished_at'],
                     collection_manifest_sha256=sha(root / 'COLLECTION.json'))
        _save(root / 'status.json', state)
        (root / 'actual_exit_code').write_text('0\n')
        return manifest
    except BaseException as error:
        state.update(
            stage='failed',
            actual_exit_code=1,
            failed_count=1,
            error=repr(error),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        _save(root / 'status.json', state)
        (root / 'actual_exit_code').write_text('1\n')
        raise
PROTOCOL = 'paper_cube_fixed_mix_v1'

def build_plan(cube_count, output_root, mode='formal', python=None, collection_seed=0, *, source_manifest_sha256=None):
    if type(cube_count) is not int or cube_count not in (1, 2, 3, 4):
        raise ValueError('cube_count must be an integer in 1..4')
    if type(collection_seed) is not int or collection_seed < 0:
        raise ValueError('collection_seed must be an explicit nonnegative integer')
    if mode not in ('formal', 'smoke'):
        raise ValueError('mode must be formal or smoke')
    root = Path(output_root).resolve()
    world_base = 1000000 * cube_count + collection_seed
    policy_base = 100000000 + 1000000 * cube_count + collection_seed
    if mode == 'formal':
        jobs = [
            (task, strategy, split, count)
            for task in range(1, 6)
            for (strategy, train, val) in STRATEGIES
            for (split, count) in (('train', train), ('validation', val))
        ]
        counts = {'train': 10000, 'validation': 1000}
    else:
        jobs = [
            (1, 'expert_low_noise', 'train', 1),
            (2, 'expert_low_noise', 'train', 1),
            (3, 'expert_low_noise', 'validation', 1),
        ]
        counts = {'train': 2, 'validation': 1}
    (parts, preceding) = ([], 0)
    for (logical, (task, strategy, split, count)) in enumerate(jobs):
        for offset in range(0, count, 100):
            index = len(parts)
            directory = root / 'shards' / f'part_{index:03d}'
            parts.append(
                dict(
                    index=index,
                    logical_job_index=logical,
                    task=task,
                    strategy=strategy,
                    split=split,
                    episodes=min(100, count - offset),
                    world_seed=world_base + preceding + 100 * logical + offset,
                    policy_seed=policy_base + 17 * index,
                    directory=str(directory),
                    rgb=str(directory / 'datasets/pair_pix.h5'),
                    pcd=str(directory / 'datasets/pair_pcd.h5'),
                )
            )
        preceding += count
    if source_manifest_sha256 is None:
        source_manifest_sha256 = hashlib.sha256((SOURCES).read_bytes()).hexdigest()
    return dict(
        schema='cube_fixed_collection_plan_v1',
        protocol=PROTOCOL,
        cube_count=cube_count,
        env_type=ENV_TYPES[cube_count - 1],
        collection_seed=collection_seed,
        mode=mode,
        raw_horizon=HORIZONS[cube_count - 1],
        split_counts=counts,
        output_root=str(root),
        python=str(python or sys.executable),
        parts=parts,
        source_manifest_sha256=source_manifest_sha256,
        seed_schedule=dict(
            world_seed_base=world_base,
            policy_seed_base=policy_base,
            world_seed_margin=100,
            policy_rng_scope='physical_part',
            job_order='task; strategy; train then validation; part',
            world_seed_rule='base + preceding logical episodes + 100*logical index + part offset',
            policy_seed_rule='base + 17*part index',
        ),
        collection_protocol=dict(
            name=PROTOCOL,
            training_view_rows=HORIZONS[cube_count - 1],
            independent_episode_train_validation=True,
            camera=dict(
                source='front_pixels' if cube_count == 1 else 'front',
                render_name='front_pixels',
                pitch_up_degrees=0.0 if cube_count == 1 else 5.0,
                pullback_m=0.1 if cube_count == 4 else 0.0,
                width=224,
                height=224,
                RGB_AA=4,
                geometry_AA=0,
            ),
            mixture=dict(expert_low_noise=0.2, expert_high_noise=0.4, mixed_expert_random=0.2, random=0.2),
            mixed_variants=dict(branch100=0.1, branch45=0.1),
            failures_filtered=False,
            numerical_reproduction_verified=False,
            smoke_scope='two train starts and one validation start, low-noise diagnostic only'
            if mode == 'smoke'
            else None,
            action_alignment='action[t] drives observation[t+1]; terminal action is NaN',
        ),
    )

def child_command(plan, part, runtime):
    """One current-paper YAML, four explicit horizons, no old recipe names."""
    runtime = Path(runtime)
    cube = plan['cube_count']
    command = [plan['python'], str(runtime / 'code/lewm_pcd/scripts/data/collect_cube_pcd.py')]
    if cube == 4:
        command = [
            plan['python'],
            str(runtime / 'collection_adapters/quad_h600.py'),
            '--base-root',
            str(runtime),
            '--horizon-actions',
            '600',
            '--camera-pullback-m',
            '0.10',
            '--adapter-binding',
            str(Path(part['directory']) / 'adapter_binding.json'),
            '--',
        ]
    command += [
        '--config-name',
        'paper_cube',
        'env_type=' + plan['env_type'],
        f"fixed_task_id={part['task']}",
        'strategy=' + part['strategy'],
        'split=' + part['split'],
        f"num_traj={part['episodes']}",
        f"seed={part['policy_seed']}",
        f"world_seed={part['world_seed']}",
        'cache_dir=' + part['directory'],
        'dataset_name=pair_pcd.h5',
        'pix_dataset_name=pair_pix.h5',
        'pcd.keep_rgb=false',
        'world.num_envs=1',
        f"world.max_episode_steps={plan['raw_horizon']}",
        'hydra.run.dir=' + str(Path(part['directory']) / 'hydra'),
    ]
    if cube > 1:
        command += [
            'sealed_capture.frame_source_path=' + str(runtime / 'external/native_pcd_frame.py'),
            'sealed_capture.provider_path=' + str(runtime / 'external/pcd_geometry.py'),
            'sealed_capture.core_path=' + str(runtime / 'code/lewm_pcd/envs/pcd_wrapper.py'),
        ]
    if cube > 2:
        command += ['pcd.crop_box=[[-0.25,-0.5,-0.02],[0.75,0.5,0.9]]']
    return command

def _validate_adapter_binding(runtime, directory):
    receipt = directory / 'adapter_binding.json'
    paths = (
        runtime / 'collection_adapters/quad_h600.py',
        runtime / 'collection_adapters/camera_pullback.py',
        runtime / 'code/lewm_pcd/scripts/data/collect_cube_pcd.py',
    )
    expected = {str(path.resolve()): sha(path) for path in paths}
    value = json.loads(receipt.read_text(encoding='utf8'))
    if (
        value.get('schema') != 'paper_cube_adapter_binding_v1'
        or value.get('stage') != 'source_binding_prepared'
        or value.get('raw_horizon') != 600
        or (value.get('pullback_m') != 0.1)
        or (value.get('files') != expected)
    ):
        raise ValueError('actual Quad adapter source binding differs')
    return {str(receipt): sha(receipt), **expected}
