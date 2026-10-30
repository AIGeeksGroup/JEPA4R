"""No-overwrite persistent launcher. This file is prepared here; Root launches."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def checked_plan(case, local_only=False):
    plan = json.loads((case / 'plan.json').read_text())
    reference = (case.parent / Path(plan['reference_case']).name) if local_only else Path(plan['reference_case'])
    if hashlib.sha256((reference / 'freeze.json').read_bytes()).hexdigest() != plan['reference_freeze_sha256']:
        raise ValueError('Reference freeze changed')
    spec = importlib.util.spec_from_file_location('recovery_reference_launch_guard', reference / 'launch.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.checked_plan(reference, local_only=local_only)
    freeze_path = case / 'freeze.json'
    if freeze_path.exists() or not local_only:
        frozen = json.loads(freeze_path.read_text())
        for name, expected in frozen['files'].items():
            if hashlib.sha256((case / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Frozen recovery file mismatch: {name}')
    if not local_only and Path(sys.executable).resolve() != Path(plan['argv'][0]).resolve():
        raise ValueError('Use the frozen target interpreter')
    return plan


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-check-only', action='store_true')
    parser.add_argument('--smoke-root', type=Path, default=None,
                        help='Root-only two-batch smoke at a fresh direct child of this case')
    args = parser.parse_args()
    case = Path(__file__).resolve().parent
    plan = checked_plan(case, local_only=args.source_check_only)
    if args.source_check_only:
        print(json.dumps({'stage': 'recovery_source_only_verified', 'launched': False}))
        return 0
    if args.smoke_root is not None:
        smoke = args.smoke_root.resolve()
        if smoke.parent != case:
            raise ValueError('Smoke root must be an explicit direct child of the new recovery case')
        plan['output_root'] = str(smoke)
        plan['verified_export_root'] = str(smoke.with_name(smoke.name + '_verified_exports'))
        plan['environment']['STABLEWM_HOME'] = str(smoke)
        plan['argv'][plan['argv'].index('--verified-export-root') + 1] = plan['verified_export_root']
        plan['argv'] += ['--max-steps', '2']
        plan['effective_mode'] = 'two_batch_resumed_smoke_only'
    from resume_helpers import require_free_space, audit_checkpoint
    binding = plan['recovery']
    audit_checkpoint(binding['parent_model'], binding['parent_sha256'], binding['parent_bytes'])
    reference = Path(plan['reference_case'])
    spec = importlib.util.spec_from_file_location('recovery_data_guard', reference / 'data_preflight.py')
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    metadata = guard.inspect_sources(Path(plan['data']['sealed_view_index']), plan['data']['sealed_view_index_sha256'])
    if not metadata['training_metadata_ready']:
        print(json.dumps(metadata), file=sys.stderr)
        return 3
    out, stage = Path(plan['output_root']), Path(plan['verified_export_root'])
    for path in [out, stage]:
        if path.exists():
            raise FileExistsError(f'Recovery output/stage already exists: {path}')
        require_free_space(path.parent, plan['storage']['minimum_free_bytes_each_filesystem'])
    out.mkdir()
    stage.mkdir()
    for name in ['plan.json', 'freeze.json']:
        shutil.copyfile(case / name, out / name)
    write_json(out / 'argv.json', plan['argv'])
    write_json(out / 'data_preflight.json', metadata)
    state = {'stage': 'launcher_prepared', 'started_at': utc_now(), 'updated_at': utc_now(),
             'config_summary': plan['recipe'], 'recovery': binding,
             'completed_count': 0, 'failed_count': 0, 'error': None,
             'counter_semantics': 'completed_count counts verified new epoch exports; Adam successful updates require separate evidence',
             'actual_exit_code': None, 'launcher_pid': os.getpid(), 'trainer_pid': None,
             'effective_mode': plan.get('effective_mode', 'formal_remaining_three_epochs'),
             'external_backup': 'pending separate receipt', 'shared_mirror_enabled': False}
    write_json(out / 'status.json', state)
    (out / 'launcher_pid').write_text(str(os.getpid()) + '\n')
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
            write_json(out / 'status.json', state)
            rc = trainer.wait()
    except Exception as error:
        state['error'] = repr(error)
    exported = list((out / 'checkpoints' / plan['run_name']).glob('weights_epoch_*.pt.receipt.json'))
    state.update(stage='process_exited_pending_acceptance' if rc == 0 else 'failed', actual_exit_code=rc,
                 updated_at=utc_now(), finished_at=utc_now(), completed_count=len(exported),
                 failed_count=int(rc != 0))
    (out / 'actual_exit_code').write_text(str(rc) + '\n')
    (out / 'finished_at').write_text(state['finished_at'] + '\n')
    write_json(out / 'status.json', state)
    return rc if rc >= 0 else 128 - rc


if __name__ == '__main__':
    sys.exit(main())
