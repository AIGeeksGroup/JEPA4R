# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/relocate/rh56_workflow.py (verbatim except: imports / paths adapted to
# the package; lines wrapped at 120 columns)
"""Thin dispatcher for the final simulated RM75-B/RH56 Relocate protocol.

Planning imports no simulator or learning framework. Existing RH56 bridges own
collection, numerical checks, optimization, strict reload and online rollouts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


LEWM_ROOT = Path(__file__).resolve().parents[3]
ENTRY = LEWM_ROOT / 'policy/bc/relocate/run_rh56.py'
BATCH_SIZE = 250
OFFICIAL_RAW_SHA256 = 'c71c7af5dda7b9a3d133a7e2baaa821bfe1ab7c4445dfde5624b947d13865694'
# Canonical ordered task positions/init identities verified from that raw source.
# Paths and source_id labels are intentionally outside this content identity.
OFFICIAL_TASK_CONTENT_SHA256 = '064a33e46fdc7407c6b0599f16413c85aab579c8fcccc957606e39da067ad2a8'


def _sha(path):
    with Path(path).open('rb') as stream:
        return (
            hashlib.file_digest(stream, 'sha256').hexdigest()
            if hasattr(hashlib, 'file_digest')
            else _stream_sha(stream)
        )


def _stream_sha(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(4 << 20), b''):
        digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _episode_batches(mode):
    counts = (('train', 2, 61000000), ('val', 1, 62000000)) if mode == 'smoke' else (
        ('train', 9000, 61000000), ('val', 1000, 62000000))
    episodes = [dict(split=split, seed=base + index, coverage='protocol')
                for split, count, base in counts for index in range(count)]
    return [episodes[start:start + BATCH_SIZE] for start in range(0, len(episodes), BATCH_SIZE)]


def _official_content_digest(payload):
    _require(payload.get('schema') == 'official-task-init-v1', 'official initialization schema differs')
    records = payload['records']
    _require(len(records) == 100 and all(type(r['demo_index']) is int for r in records)
             and [r['demo_index'] for r in records] == list(range(100)),
             'official initialization must retain all100 sources in order')
    _require(all(isinstance(r['source_id'], str) and r['source_id'] for r in records)
             and len({r['source_id'] for r in records}) == 100, 'duplicate/invalid official source IDs')
    hashes = [r['original_init_sha256'] for r in records]
    _require(len(set(hashes)) == 100, 'duplicate official initialization hashes')
    for record in records:
        value = record['original_init_sha256']
        _require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                 'invalid original initialization hash')
        for key in ('obj_pos', 'target_pos'):
            _require(len(record[key]) == 3 and all(math.isfinite(v) for v in record[key]),
                     'official positions must be finite XYZ')
    semantic = [{key: record[key] for key in ('demo_index', 'original_init_sha256', 'obj_pos', 'target_pos')}
                for record in records]
    digest = hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()
    _require(digest == OFFICIAL_TASK_CONTENT_SHA256, 'canonical official task content digest differs')
    return digest


def plan(args, assets: dict) -> list[dict]:
    """Return executable stage arguments without reading assets or creating files."""
    _require(args.env == 'rh56-relocate', 'RH56 workflow requires env=rh56-relocate')
    _require(args.model in ('jepa4r', 'lewm'), 'RH56 supports jepa4r and lewm only')
    _require(args.mode in ('smoke', 'full'), 'mode must be smoke or full')
    for key in ('rh56_expert_root', 'eai_vc_root', 'rh56_official_task_init'):
        _require(bool(assets.get(key)), 'missing RH56 asset: ' + key)
    route = 'pcd' if args.model == 'jepa4r' else 'rgb'
    base = Path(args.output).resolve() / 'rh56'
    sim_python = str(args.sim_python)
    bc_python = str(args.sim_python if getattr(args, 'sim_python_override', False)
                    else assets.get('rh56_bc_python', args.sim_python))
    environment = {str(k): str(v) for k, v in assets.get('environment', {}).items()}
    paths = [str(LEWM_ROOT.parent)] + [str(p) for p in assets.get('pythonpath', [])]
    if environment.get('PYTHONPATH'):
        paths.append(environment['PYTHONPATH'])
    environment['PYTHONPATH'] = os.pathsep.join(paths)
    environment.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    environment.setdefault('MUJOCO_GL', 'egl')
    environment.setdefault('PYOPENGL_PLATFORM', 'egl')
    common = ['--baseline-root', str(LEWM_ROOT)]
    expert = ['--expert-root', str(assets['rh56_expert_root'])]
    stages = []

    def stage(name, python, arguments, *, entry=ENTRY):
        stages.append(dict(name=name, command=[str(python), str(entry), *map(str, arguments)],
                           cwd=str(LEWM_ROOT.parent), environment=dict(environment)))

    configuration = dict(mode=args.mode, output=str(base), assets=assets)
    stage('rh56_prepare', sim_python, ['prepare', '--config-json', json.dumps(configuration)],
          entry=Path(__file__).resolve())
    batches = _episode_batches(args.mode)
    batch_dirs = []
    for index in range(len(batches)):
        directory = base / 'batches' / f'batch_{index:03d}'
        batch_dirs.append(str(directory))
        frozen_plan = base / 'inputs' / f'plan_{index:03d}.json'
        stage(f'rh56_collect_{index:03d}', sim_python,
              ['collect', *common, *expert, '--camera', 'opposite',
               '--plan', frozen_plan, '--output', directory])
        stage(f'rh56_validate_{index:03d}', sim_python,
              ['validate', *common, '--batch', directory, '--plan', frozen_plan])
    counts = {'train': 2, 'val': 1} if args.mode == 'smoke' else {'train': 9000, 'val': 1000}
    stage('rh56_publish', sim_python, ['publish', *common, '--batches', *batch_dirs,
          '--output', base / 'data', '--expected-counts', json.dumps(counts)])
    wm_mode = 'smoke' if args.mode == 'smoke' else 'formal'
    wm_arguments = ['wm', *common, '--route', route, '--mode', wm_mode, '--device', 'cuda',
                    '--seed', '3072',
                    '--manifest', base / 'data/manifest.json',
                    '--dataset', base / f'data/paired_{route}.h5', '--output', base / 'wm']
    if args.mode == 'smoke':
        wm_arguments += ['--batch-size', '2', '--workers', '0']
    else:
        wm_arguments += ['--workers', str(assets.get('rh56_wm_workers', 24)), '--prefetch-factor', '2']
    stage('rh56_wm', args.wm_python, wm_arguments)
    # Resolve the run-local copied initialization hash immediately before calling
    # the original entry. No stale source path or unexpanded template is executed.
    stage('rh56_official_demo', sim_python, ['official-demo', '--inputs', base / 'inputs',
          'official-init', *common, *expert,
          '--init-source', base / 'inputs/official_task_init.json',
          '--count', '1' if args.mode == 'smoke' else '100', '--start-demo', '0',
          '--seed-base', '64500000', '--output', base / 'demos'], entry=Path(__file__).resolve())
    epoch = 1 if args.mode == 'smoke' else 10
    checkpoint = base / f'wm/worldmodel_cache/checkpoints/rh56_relocate_{route}_{wm_mode}_v1'
    for seed in ((100,) if args.mode == 'smoke' else (100, 200, 300)):
        # Existing smoke uses the old native camera. Profile has the same small
        # budget and the final opposite camera, without changing the BC core.
        stage(f'rh56_bc_{seed}', bc_python, ['bc', *common, *expert,
              '--eai-vc-root', assets['eai_vc_root'], '--route', route,
              '--mode', 'profile' if args.mode == 'smoke' else 'formal', '--device', 'cuda',
              '--checkpoint', checkpoint, '--epoch', str(epoch),
              '--demos-manifest', base / 'demos/bc_demos_manifest.json',
              '--train-seed', str(seed), '--out', base / f'bc_seed{seed}'])
        if assets.get('policy_evidence', False):
            stages[-1]['command'].append('--policy-evidence')
    return stages


def prepare(configuration):
    """Create only this run's plans; preserve source data and original metadata."""
    assets, mode = configuration['assets'], configuration['mode']
    _require(mode in ('smoke', 'full'), 'unsupported RH56 mode')
    source = Path(assets['rh56_official_task_init']).expanduser().resolve()
    digest = _sha(source)
    if assets.get('rh56_official_task_init_sha256'):
        _require(digest == assets['rh56_official_task_init_sha256'], 'official initialization SHA differs')
    payload = _read(source)
    manifest = Path(assets.get('rh56_official_source_manifest', payload['source_manifest'])).expanduser().resolve()
    _require(_sha(manifest) == payload['source_sha'], 'original official source manifest SHA differs')
    official_digest = _official_content_digest(payload)
    payload['source_manifest'] = str(manifest)
    directory = Path(configuration['output']) / 'inputs'
    directory.mkdir(parents=True, exist_ok=False)
    files = []

    def write(name, value):
        path = directory / name
        path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
        files.append(dict(path=str(path.resolve()), sha256=_sha(path)))

    for index, episodes in enumerate(_episode_batches(mode)):
        write(f'plan_{index:03d}.json', dict(episodes=episodes))
    write('official_task_init.json', payload)
    write('input_receipt.json', dict(status='PASS_INPUTS', mode=mode, files=list(files),
          original_init_path=str(source), original_init_sha256=digest,
          source_manifest=str(manifest), source_sha256=payload['source_sha'],
          official_task_content_sha256=official_digest, canonical_raw_sha256=OFFICIAL_RAW_SHA256))


def _completed(directory, state_name='state.json'):
    state = _read(directory / state_name)
    _require(state['phase'] == 'complete' and state['exit_code'] == 0,
             'backend did not complete with actual exit0')
    return state


def _bound_file(path, digest):
    _require(_sha(path) == digest, 'artifact SHA changed: ' + str(path))


def verify_stage(stage, args, assets) -> dict:
    """Check structured numerical receipts and their actual artifact bindings."""
    try:
        name, base = stage['name'], Path(args.output).resolve() / 'rh56'
        route = 'pcd' if args.model == 'jepa4r' else 'rgb'
        smoke = args.mode == 'smoke'
        if name == 'rh56_prepare':
            report = _read(base / 'inputs/input_receipt.json')
            _require(report['status'] == 'PASS_INPUTS' and report['mode'] == args.mode, 'input gate failed')
            for item in report['files']:
                _bound_file(item['path'], item['sha256'])
            _bound_file(report['original_init_path'], report['original_init_sha256'])
            _bound_file(report['source_manifest'], report['source_sha256'])
            for payload in (_read(report['original_init_path']), _read(base / 'inputs/official_task_init.json')):
                _official_content_digest(payload)
                _require(payload['source_sha'] == report['source_sha256'], 'official source manifest binding differs')
            if 'official_task_content_sha256' in report:
                _require(report['official_task_content_sha256'] == OFFICIAL_TASK_CONTENT_SHA256,
                         'official content receipt identity differs')
        elif name.startswith(('rh56_collect_', 'rh56_validate_')):
            index = int(name.rsplit('_', 1)[1])
            directory = base / 'batches' / f'batch_{index:03d}'
            state = _completed(directory)
            report = _read(directory / 'manifest.json')
            records = report['episodes']
            expected = _episode_batches(args.mode)[index]
            _require(
                state['completed'] == len(expected) and len(records) == len(expected), 'missing collection episodes'
            )
            _require(
                [(r['split'], r['seed']) for r in records] == [(r['split'], r['seed']) for r in expected],
                'collection seed/partition differs',
            )
            _require(
                len({r['initial_state_sha256'] for r in records}) == len(records), 'duplicate physical initial state'
            )
            _require(
                all(
                    r['camera']['mode'] == 'opposite' and r['success_consistent'] and r['rows'] == r['steps'] + 1
                    for r in records
                ),
                'collection contract failed',
            )
            if name.startswith('rh56_validate_'):
                report = _read(directory / 'batch_acceptance.json')
                _require(
                    report['status'] == 'PASS_BATCH' and report['deep_data_check'],
                    'requires deep paired numerical acceptance',
                )
                _bound_file(base / 'inputs' / f'plan_{index:03d}.json', report['plan_sha256'])
                for item in report['files']:
                    _require(Path(item['path']).stat().st_size == item['bytes'], 'accepted payload size changed')
                    _bound_file(item['path'], item['sha256'])
        elif name == 'rh56_publish':
            report = _read(base / 'data/manifest.json')
            _require(report['source_acceptance_checked'], 'VDS sources were not accepted')
            _require(
                report['registry']['counts'] == ({'train': 2, 'val': 1} if smoke else {'train': 9000, 'val': 1000}),
                'published split differs',
            )
            for item in report['datasets']:
                _bound_file(item['path'], item['sha256'])
            for item in report['source_receipts']:
                _bound_file(Path(item['batch']) / 'batch_acceptance.json', item['acceptance_sha256'])
        elif name == 'rh56_wm':
            _completed(base / 'wm')
            report = _read(base / 'wm/validation.json')
            identity = _read(base / 'wm/run_identity.json')
            _require(
                report['status'] == 'passed' and report['route'] == route and identity['device'] == 'cuda',
                'WM route/device gate differs',
            )
            _require(
                report['evidence']['optimizer_steps']
                == (
                    2
                    if smoke
                    else report['budget']['completed_epochs']
                    * (report['split']['clips']['train'] // report['budget']['batch_size'])
                ),
                'WM actual optimizer budget differs',
            )
            _require(
                report['evidence']['changed_parameter_tensors'] > 0 and report['evidence']['gradient_checks'] > 0,
                'WM did not update with checked gradients',
            )
            _require(
                math.isfinite(report['evidence']['train_loss'])
                and all(math.isfinite(v) for v in report['evidence']['epoch_validation_losses']),
                'WM nonfinite recorded loss',
            )
            _require(
                report['evidence']['epochs'] == list(range(1, (1 if smoke else 10) + 1)),
                'WM completed epoch schedule differs',
            )
            if smoke:
                _require(report['evidence']['train_batches'] == 2 and report['evidence']['val_batches'] == 1,
                         'WM bounded batch budget differs')
            _require(report['checkpoint_epoch'] == (1 if smoke else 10), 'WM checkpoint epoch differs')
            _require(
                report['strict_reload_maxabs'] and all(v == 0 for v in report['strict_reload_maxabs'].values()),
                'strict model/loss reload failed',
            )
            _require(
                report['actual_adam_step_values'] == [report['evidence']['optimizer_steps']], 'WM Adam state differs'
            )
            _bound_file(report['checkpoint'], report['checkpoint_sha256'])
            _bound_file(base / 'data/manifest.json', report['manifest_sha256'])
            _bound_file(base / f'data/paired_{route}.h5', report['dataset_sha256'])
        elif name == 'rh56_official_demo':
            _completed(base / 'demos', 'wrapper_state.json')
            report = _read(base / 'demos/bc_demos_manifest.json')
            _require(
                report['all_requested_sources_succeeded'] and report['requested_count'] == (1 if smoke else 100),
                'official demonstration attempt failed/incomplete',
            )
            _require(
                report['profile_bc_ready'] if smoke else report['formal_bc_ready'],
                'official demos are not ready for this budget',
            )
            for item in report['selected']:
                _require(
                    item['success']
                    and item['coverage'] == 'expert'
                    and item['noise_std'] == 0
                    and item['camera']['mode'] == 'opposite',
                    'official demo protocol differs',
                )
                _bound_file(item['npz'], item['npz_sha256'])
        elif name.startswith('rh56_bc_'):
            seed = int(name.rsplit('_', 1)[1])
            directory = base / f'bc_seed{seed}'
            _completed(directory)
            report = _read(directory / 'validation.json')
            job = _read(directory / 'job_config.json')
            expected_updates = 1 if smoke else 7900
            _require(job['device'] == 'cuda' and job['epochs'] == (1 if smoke else 100) and
                     job['num_demos'] == (1 if smoke else 100), 'BC actual CUDA job budget differs')
            _require(
                report['status'] == ('PASS_MINIMAL_OFFICIAL_BC_CHAIN' if smoke else 'PASS_FORMAL_OFFICIAL_BC_CHAIN'),
                'BC numerical chain did not pass',
            )
            _require(
                report['route'] == route and report['run_spec']['camera'] == 'opposite', 'BC modality/camera differs'
            )
            _require(
                report['train_seed'] == seed and report['input_dim'] == 589 and report['action_dim'] == 12,
                'BC seed/dimensions differ',
            )
            _require(
                report['encoder_frozen_and_unchanged'] and report['head_updated'] and report['finite_head_gradients'],
                'BC model/gradient gate failed',
            )
            _require(
                report['optimizer_evidence']['steps'] == expected_updates
                and report['optimizer_evidence']['finite_gradient_checks'] == expected_updates,
                'BC actual Adam budget differs',
            )
            _require(
                report['actual_online_episodes'] == (1 if smoke else 500)
                and report['actual_train_initial_episodes'] == (1 if smoke else 500)
                and report['online_steps'] > 0,
                'BC actual closed-loop rollout count differs',
            )
            _require(
                report['policy_reload_max_abs'] == 0 and math.isfinite(report['loss']), 'BC policy reload/loss failed'
            )
            _bound_file(directory / 'policy/iterations/best_policy.pickle', report['policy_sha256'])
            _bound_file(directory / 'policy/iterations/final_policy.pickle', report['final_policy_sha256'])
            _bound_file(base / 'demos/bc_demos_manifest.json', report['demos_manifest_sha256'])
            _bound_file(
                Path(report['checkpoint']) / f'weights_epoch_{report["checkpoint_epoch"]}.pt',
                report['checkpoint_sha256'],
            )
        else:
            raise ValueError('unknown RH56 stage: ' + name)
        return dict(passed=True, details=report)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return dict(passed=False, details=str(error))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inputs = commands.add_parser('prepare')
    inputs.add_argument('--config-json', required=True)
    demo = commands.add_parser('official-demo')
    demo.add_argument('--inputs', type=Path, required=True)
    args, forwarded = parser.parse_known_args(argv)
    if args.command == 'prepare':
        _require(not forwarded, 'unexpected input arguments')
        prepare(json.loads(args.config_json))
    else:
        receipt = _read(args.inputs / 'input_receipt.json')
        path = args.inputs / 'official_task_init.json'
        expected = next(item['sha256'] for item in receipt['files'] if Path(item['path']).name == path.name)
        _bound_file(path, expected)
        forwarded += ['--init-source-sha256', expected]
        import sys
        import importlib.util
        sys.path.insert(0, str(LEWM_ROOT.parent))
        specification = importlib.util.spec_from_file_location('_rh56_original_entry', ENTRY)
        original = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(original)
        return original.main(forwarded)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
