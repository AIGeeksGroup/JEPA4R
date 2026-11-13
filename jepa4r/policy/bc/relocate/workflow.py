# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/relocate/workflow.py (verbatim except: imports / paths adapted to the package)
"""Small env/model dispatcher with persistent stage logs and artifact verification."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys  # noqa: F401  (moved verbatim)


ROOT = Path(__file__).resolve().parents[4]
MODELS = {
    'adroit-relocate': ['jepa4r', 'lewm', 'pldm', 'lejepa', 'visreg', 'vjepa'],
    'rh56-relocate': ['jepa4r', 'lewm'],
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def execute_plan(stages, output, verify, identity, resume=False, configuration=None):
    """Execute each stage once; an exit of zero also requires verified artifacts.

    Resume rechecks accepted stages and continues unstarted stages. Partial failed
    stages require a fresh output directory so their evidence is never overwritten.
    """
    names = [stage['name'] for stage in stages]
    if len(names) != len(set(names)) or any(
            not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in names):
        raise ValueError('Stage names must be unique safe filenames')
    output = Path(output).resolve()
    status_path = output / 'STATUS.json'
    if output.exists():
        if not resume:
            raise FileExistsError(f'Output already exists; preserve it or use --resume: {output}')
        state = json.loads(status_path.read_text(encoding='utf-8'))
        if state['identity'] != identity:
            raise ValueError('Frozen source/configuration identity changed; use a fresh output')
        saved_plan = json.loads((output / 'PLAN.json').read_text(encoding='utf-8'))
        if saved_plan['stages'] != stages:
            raise ValueError('Stage plan changed; a subset cannot resume a different run')
    else:
        if resume:
            raise FileNotFoundError(f'Cannot resume an absent output: {output}')
        output.mkdir(parents=True)
        state = {'status': 'running', 'stage': None, 'started_at': _now(),
                 'updated_at': _now(), 'identity': identity, 'completed_count': 0,
                 'failed_count': 0, 'exit_code': None, 'error': None, 'stages': {}}
        _write_json(output / 'PLAN.json', {'identity': identity,
                    'configuration': configuration or {}, 'stages': stages})
    (output / 'logs').mkdir(exist_ok=True)

    def save():
        state['updated_at'] = _now()
        state['completed_count'] = sum(s['status'] == 'complete'
                                       for s in state['stages'].values())
        state['failed_count'] = sum(s['status'] in ('failed', 'verification_failed', 'interrupted')
                                    for s in state['stages'].values())
        _write_json(status_path, state)

    save()
    for stage in stages:
        name = stage['name']
        state['stage'] = name
        previous = state['stages'].get(name)
        if previous and previous['status'] != 'complete':
            raise FileExistsError(f'Preserved partial stage {name}; use a fresh output directory')
        if previous:
            try:
                actual_exit = int((output / 'logs' / f'{name}.exit').read_text(encoding='ascii').strip())
                if actual_exit != 0 or actual_exit != previous.get('exit_code'):
                    raise ValueError('Preserved process exit evidence changed')
                result = verify(stage)
                if result.get('passed') is True and result != previous.get('verification'):
                    previous['recheck_verification'] = result
                    raise ValueError('Accepted artifact/receipt evidence changed')
            except BaseException as error:
                interrupted = isinstance(error, KeyboardInterrupt)
                previous.update(status='interrupted' if interrupted else 'verification_failed',
                                error=f'{type(error).__name__}: {error}')
                state.update(status='interrupted' if interrupted else 'failed',
                             exit_code=130 if interrupted else 1, error=previous['error'])
                save()
                return state['exit_code']
            if result.get('passed') is not True:
                state.update(status='failed', exit_code=1,
                             error=f'Previously accepted stage no longer verifies: {name}')
                previous.update(status='verification_failed', recheck_verification=result)
                save()
                return 1
            continue
        record = {'status': 'running', 'started_at': _now(), 'exit_code': None,
                  'command': stage['command'], 'cwd': stage['cwd'],
                  'environment': stage.get('environment', {})}
        state['stages'][name] = record
        state.update(status='running', exit_code=None, error=None)
        save()
        environment = os.environ.copy()
        environment.update(stage.get('environment', {}))
        try:
            with (output / 'logs' / f'{name}.log').open('xb') as log:
                print(f'Running {name}', flush=True)
                process = subprocess.run(stage['command'], cwd=stage['cwd'],
                                         env=environment, stdout=log, stderr=subprocess.STDOUT)
            record['exit_code'] = process.returncode
            (output / 'logs' / f'{name}.exit').write_text(
                str(process.returncode) + '\n', encoding='ascii')
            if process.returncode != 0:
                record['status'] = 'failed'
                state.update(status='failed', exit_code=process.returncode,
                             error=f'Stage process failed: {name}')
                record['finished_at'] = _now()
                save()
                return process.returncode
            result = verify(stage)
            record['verification'] = result
            record['status'] = 'complete' if result.get('passed') is True else 'verification_failed'
            record['finished_at'] = _now()
            if record['status'] != 'complete':
                state.update(status='failed', exit_code=1,
                             error=f'Stage artifact verification failed: {name}')
                save()
                return 1
            save()
        except BaseException as error:
            interrupted = isinstance(error, KeyboardInterrupt)
            record.update(status='interrupted' if interrupted else 'failed',
                          error=f'{type(error).__name__}: {error}',
                          finished_at=_now())
            state.update(status='interrupted' if interrupted else 'failed',
                         exit_code=130 if interrupted else 1, error=record['error'])
            save()
            return state['exit_code']
    state.update(status='complete', stage=None, exit_code=0, error=None)
    save()
    return 0


def asset_fingerprints(assets):
    """Bind external source and input contents, including files reached via roots.

    Environment package versions are recorded separately by the runtime doctor;
    Python executable bytes and any supplied runtime manifest are bound here.
    Missing paths remain visible in a preparation-only plan.
    """
    fingerprints = {}

    def file_sha(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
        return digest.hexdigest()

    def bind(label, value):
        if not isinstance(value, str) or not value:
            return
        path = Path(os.path.expandvars(value)).expanduser()
        if path.is_file():
            fingerprints[label] = {'path': str(path.resolve()), 'sha256': file_sha(path)}
        elif path.is_dir():
            digest = hashlib.sha256()
            count = 0
            for directory, dirs, names in os.walk(path, followlinks=False):
                dirs[:] = sorted(d for d in dirs if d not in
                                 ('.git', '__pycache__', '.pytest_cache'))
                for name in sorted(names):
                    child = Path(directory) / name
                    if child.suffix in ('.pyc', '.pyo') or not child.is_file():
                        continue
                    digest.update(child.relative_to(path).as_posix().encode())
                    digest.update(file_sha(child).encode())
                    count += 1
            fingerprints[label] = {'path': str(path.resolve()), 'files': count,
                                   'sha256': digest.hexdigest()}
        else:
            fingerprints[label] = {'path': str(path), 'missing': True}

    for key, value in sorted(assets.items()):
        if (key.endswith(('_root', '_checkpoint', '_python', '_manifest'))
                or key in ('adroit_expert', 'adroit_official_demos', 'rh56_official_task_init',
                           'mujoco_path', 'mujoco_key')):
            bind(key, value)
        elif key == 'pythonpath':
            for index, path in enumerate(value):
                bind(f'pythonpath[{index}]', path)
    return fingerprints


def _source_identity(configuration):
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode())
    for path in sorted((ROOT / 'jepa4r').rglob('*')):
        if (path.is_file() and path.suffix in ('.py', '.yaml', '.json')
                and '__pycache__' not in path.parts and '.git' not in path.parts):
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    for name in ('jepa4r/policy/bc/relocate/DEPENDENCIES.json',):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--list', action='store_true', help='List published environments and models')
    result.add_argument('--env', choices=tuple(MODELS), default='adroit-relocate')
    result.add_argument('--model', default='jepa4r')
    result.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    result.add_argument('--assets', type=Path, help='External asset/environment JSON; see assets.example.json')
    result.add_argument('--output', type=Path, help='Fresh absolute run directory outside the checkout')
    result.add_argument('--wm-python', help='World-model environment Python')
    result.add_argument('--sim-python', help='Selected simulator/BC environment Python')
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--stage', default='all', help='One planned stage name, or all')
    result.add_argument('--execute', action='store_true', help='Execute; otherwise print the plan only')
    result.add_argument('--resume', action='store_true', help='Reverify completed stages and continue unstarted ones')
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.list:
        print(json.dumps(MODELS, indent=2))
        return 0
    if args.model not in MODELS[args.env]:
        argument_parser.error(f'{args.env} does not support model {args.model!r}')
    if args.assets is None or args.output is None:
        argument_parser.error('--assets and --output are required')
    if not args.output.is_absolute():
        argument_parser.error('--output must be an absolute path')
    args.output = args.output.resolve()
    if args.output == ROOT or ROOT in args.output.parents:
        argument_parser.error('--output must be outside the source checkout')
    assets = json.loads(args.assets.read_text(encoding='utf-8-sig'))
    args.sim_python_override = args.sim_python is not None
    args.wm_python = args.wm_python if args.wm_python is not None else assets.get('wm_python')
    if args.sim_python is None:
        args.sim_python = assets.get(args.env.split('-')[0] + '_sim_python') or assets.get('sim_python')
    if not args.wm_python or not args.sim_python:
        argument_parser.error('Provide wm_python and sim_python through flags or the asset JSON')
    module_name = 'adroit_workflow' if args.env == 'adroit-relocate' else 'rh56_workflow'
    backend = importlib.import_module(f'jepa4r.policy.bc.relocate.{module_name}')
    stages = backend.plan(args, assets)
    if args.stage != 'all':
        selected = [stage for stage in stages if stage['name'] == args.stage]
        if not selected:
            argument_parser.error(f'Unknown stage {args.stage!r}; choose {[s["name"] for s in stages]}')
        stages = selected
    configuration = {'env': args.env, 'model': args.model, 'mode': args.mode,
                     'seed': args.seed, 'assets': assets, 'wm_python': args.wm_python,
                     'sim_python': args.sim_python, 'output': str(args.output),
                     'sim_python_override': args.sim_python_override,
                     'stage': args.stage,
                     'external_fingerprints': asset_fingerprints(
                         dict(assets, wm_python=args.wm_python, sim_python=args.sim_python))}
    identity = _source_identity(configuration)
    if not args.execute:
        print(json.dumps({'identity': identity, 'configuration': configuration,
                          'stages': stages}, indent=2))
        return 0
    return execute_plan(stages, args.output,
                        lambda stage: backend.verify_stage(stage, args, assets), identity,
                        resume=args.resume, configuration=configuration)


if __name__ == '__main__':
    raise SystemExit(main())
