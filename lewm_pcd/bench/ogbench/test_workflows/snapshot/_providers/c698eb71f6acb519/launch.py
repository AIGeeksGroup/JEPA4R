"""Thin persistent launcher for the byte-exact reference trainer; stdlib only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


CORE_SHA = '64caecbd3f2282d38c159018cb3d6f204079bc60ccecfa41b231d00332937b51'
IO_SHA = '320bff92d774c8499e9a6f69a622a1d6b64a5e43bd038d7124246f4f50cec8b9'


def checked_plan(case, local_only=False):
    plan = json.loads((case / 'plan.json').read_text())
    for name, expected in [('source/train_lejepa_mw.py', CORE_SHA),
                           ('worker_prefetch_v1/native_io.py', IO_SHA)]:
        if hashlib.sha256((case / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Reference source mismatch: {name}')
    freeze_path = case / 'freeze.json'
    if freeze_path.exists() or not local_only:
        frozen = json.loads(freeze_path.read_text())
        for name, expected in frozen['files'].items():
            if hashlib.sha256((case / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Frozen file mismatch: {name}')
    if not local_only:
        for name, expected in frozen['external_dependency_files'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Reference dependency mismatch: {name}')
        if Path(plan['argv'][0]).resolve() != Path(sys.executable).resolve():
            raise ValueError('Run this launcher with the frozen target Python')
    return plan


def reserve_output(path):
    path.mkdir()  # Atomic no-overwrite guard; the persistent parent must already exist.


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-check-only', action='store_true')
    args = parser.parse_args()
    case = Path(__file__).resolve().parent
    plan = checked_plan(case, local_only=args.source_check_only)
    if args.source_check_only:
        print(json.dumps({'stage': 'source_only_verified', 'launched': False}))
        return 0
    from data_preflight import inspect_sources
    metadata = inspect_sources(Path(plan['data']['sealed_view_index']),
                               plan['data']['sealed_view_index_sha256'])
    if not metadata['training_metadata_ready']:
        print(json.dumps(metadata, indent=2), file=sys.stderr)
        return 3
    out = Path(plan['output_root'])
    reserve_output(out)
    for name in ['plan.json', 'freeze.json']:
        shutil.copyfile(case / name, out / name)
    write_json(out / 'argv.json', plan['argv'])
    write_json(out / 'data_preflight.json', metadata)
    state = {'stage': 'launcher_prepared', 'started_at': utc_now(), 'updated_at': utc_now(),
             'config_summary': plan['recipe'], 'completed_count': None, 'failed_count': None,
             'counter_semantics': 'Native epoch/update counts require log and checkpoint readback',
             'error': None, 'actual_exit_code': None, 'launcher_pid': os.getpid(), 'trainer_pid': None}
    write_json(out / 'status.json', state)
    (out / 'launcher_pid').write_text(str(os.getpid()) + '\n')
    (out / 'started_at').write_text(state['started_at'] + '\n')
    env = os.environ.copy()
    env.update(plan['environment'])
    env['IO_PROFILE_RECEIPT'] = str(out / 'io_runtime_receipt.json')
    rc = 125
    try:
        with (out / 'train.log').open('x') as log:
            trainer = subprocess.Popen(plan['argv'], cwd=plan['working_directory'], env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            state.update(stage='running', trainer_pid=trainer.pid, updated_at=utc_now())
            (out / 'trainer_pid').write_text(str(trainer.pid) + '\n')
            (out / 'status').write_text('running\n')
            write_json(out / 'status.json', state)
            rc = trainer.wait()
    except Exception as error:
        state['error'] = repr(error)
    state.update(stage='process_exited_pending_acceptance', actual_exit_code=rc,
                 updated_at=utc_now(), finished_at=utc_now())
    (out / 'actual_exit_code').write_text(str(rc) + '\n')
    (out / 'finished_at').write_text(state['finished_at'] + '\n')
    (out / 'status').write_text(state['stage'] + '\n')
    write_json(out / 'status.json', state)
    return rc if rc >= 0 else 128 - rc


if __name__ == '__main__':
    sys.exit(main())
