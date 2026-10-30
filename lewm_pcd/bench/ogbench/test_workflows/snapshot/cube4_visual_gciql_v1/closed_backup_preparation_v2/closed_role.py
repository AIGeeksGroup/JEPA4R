"""Final-only Quad role backup; import accepted IO utilities, never its hardcoded main."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
Q = ROOT / 'cube4_visual_gciql_v1'
REFERENCE = ROOT / 'cube23_distance_gciql_v1/state_priority_v1/closed_training_backup_v1/build_closed_remaining.py'
REFERENCE_SHA = '61c8c160e03d9b36fc893d7fc74132e1f8e1496858bca915605cce581cf8285c'
COUNTERS = [1] + list(range(100000, 1000001, 100000))
MODES = {'G11': 'rgb', 'G12': 'pcd'}


def utilities(path):
    path = Path(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA
    spec = importlib.util.spec_from_file_location('_accepted_backup_IO_only', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # __main__ block is not executed.
    assert all(callable(getattr(module, name)) for name in ('sha', 'read', 'stat', 'new'))
    return module


def verify_metadata(cfg, accepted, status, controller, exits, outer):
    """Small metadata only; used by both the selector and the two fixture tests."""
    assert cfg['group_id'] in MODES and cfg['executable'] is True
    assert cfg['cube_count'] == 4 and cfg['modality'] == MODES[cfg['group_id']]
    assert cfg['seed'] == 0 and cfg['batch_size'] == 1024 and cfg['total_updates'] == 1000000
    assert cfg['checkpoint_steps'] == COUNTERS and cfg['save_step1'] is True
    assert cfg['input_contract']['current_dim'] == 211 and cfg['input_contract']['goal_dim'] == 204
    assert cfg['input_contract']['zero_padding'] is False
    assert accepted['role'] == status['role'] == cfg['role']
    assert outer == ['0', '0']
    assert len(exits) == 4 and all(x['actual_exit_code'] == 0 and x['residual_owned_members'] == [] for x in exits)
    assert controller['actual_exit_code'] == 0 and controller['completed_phases'] == 4
    assert controller['trained_groups'] == controller['accepted_groups'] == 1
    assert status['exit_code'] == 0 and status['completed_updates'] == status['adam_count'] == 1000000
    assert status['network_step'] == 1000001 and status['validation_count'] == 200
    assert accepted['actual_exit_code'] == 0 and accepted['full153_state_exact'] and accepted['state']['leaves'] == 153
    assert accepted['Adam_count'] == 1000000 and accepted['network_step'] == 1000001
    assert accepted['CSV_validation_records'] == 200 and accepted['synthetic_zero_padding'] is False
    assert (accepted['actual_current_dim'], accepted['actual_goal_dim']) == (211, 204)
    assert [x['counter'] for x in accepted['checkpoint_transactions']] == COUNTERS


def select(q, group, u, out):
    mode = MODES[group]
    config = q / 'functional_v1/configs' / (f'{group}_cube4_rgb.json' if group == 'G11' else 'G12_cube4_pcd.v2.json')
    cfg = u.read(config)
    run, role = Path(cfg['output']), q / 'control_v1' / group
    assert run == q / f'runs/{group}_cube4_{mode}'
    report = q / f'acceptance/{group}_full_acceptance.json'
    accepted, status = u.read(report), u.read(run / 'status.json')
    phases = [role / 'runtime' / f'{i:02d}_{action}_{group}' for i, action in enumerate(('gate', 'reload', 'train', 'accept'), 1)]
    exits = [u.read(p / 'exit.json') for p in phases]
    verify_metadata(cfg, accepted, status, u.read(role / 'runtime/status.json'), exits,
                    [(p).read_text().strip() for p in (role / 'runtime/outer_exit_code', role / 'wrapper_exit_code')])
    for p in phases:
        assert (p / 'actual_exit_code').read_text().strip() == '0'
    assert (run / 'trainer_exit_code').read_text().strip() == '0'
    resolved = u.read(run / 'resolved.json')
    assert resolved['binding']['config'] == cfg and resolved['resumed_from'] is None and resolved['initial_counter'] == 0
    assert resolved['initial_network_step'] == 1 and resolved['initial_adam_count'] == 0
    assert u.read(run / 'final_state_fingerprint.json') == accepted['state']
    paths = {config, report, role / 'launch_plan.json', role / 'runtime/status.json',
             role / 'runtime/outer_exit_code', role / 'wrapper_exit_code', Path(__file__), Path(u.__file__)}
    paths.update(run / n for n in ('train.csv', 'resolved.json', 'status.json', 'trainer_exit_code', 'final_state_fingerprint.json', 'final_val_probe.npz'))
    for p in phases:
        paths.update(p / n for n in ('actual_argv.json', 'run.log', 'resources.jsonl', 'actual_exit_code', 'exit.json'))
    for rel in ('control_v1/run_role.py', f'control_v1/run_{group}.sh', 'control_v1/release_freeze.json',
                'functional_v1/SOURCE_FREEZE_v1.json', 'functional_v1/HANDOFF_v1.json'):
        paths.add(q / rel)
    optional = [f'control_v1/{group}/dispatch_receipt.json', 'migration_rebind_v1/RGB_CACHE_ADMISSION_v1.json']
    if group == 'G12':
        optional.append('functional_v1/G12_CONFIG_PREPARED_v2.json')
    for rel in optional:
        if (q / rel).is_file():
            paths.add(q / rel)
    for p, expected in resolved['binding']['code'].items():
        assert u.sha(p) == expected
        paths.add(Path(p))
    ledger = []
    for entry in accepted['checkpoint_transactions']:
        counter = entry['counter']
        cp = run / 'checkpoints' / f'update_{counter:09d}'
        manifest = cp / 'manifest.json'
        assert u.sha(manifest) == entry['manifest_sha256']
        item = u.read(manifest)
        assert item['counter'] == counter
        assert (cp / f'params_{counter}.pkl').is_file() and (cp / 'run_state.pkl').is_file()
        paths.add(manifest)
        ledger.append(dict(counter=counter, manifest=str(manifest), manifest_sha256=entry['manifest_sha256'],
            policy=str(cp / f'params_{counter}.pkl'), policy_sha256=item['agent_sha256'],
            sidecar=str(cp / 'run_state.pkl'), sidecar_sha256=item['sidecar_sha256'],
            policy_stat=u.stat(cp / f'params_{counter}.pkl'), sidecar_stat=u.stat(cp / 'run_state.pkl'), included=counter == 1000000))
        if counter == 1000000:
            assert str(cp) == accepted['final_checkpoint'] and u.sha(manifest) == accepted['final_manifest_sha256']
            assert u.sha(cp / f'params_{counter}.pkl') == item['agent_sha256']
            assert u.sha(cp / 'run_state.pkl') == item['sidecar_sha256']
            paths.update((cp / f'params_{counter}.pkl', cp / 'run_state.pkl'))
    u.new('checkpoint_SHA_ledger.json', dict(actual_exit_code=0, group_id=group, checkpoints=ledger,
        only_final_policy_and_sidecar_included=True, all11_policies_and_sidecars_retained_remote=True))
    u.new('closure_readback.json', dict(actual_exit_code=0, group_id=group, CPU_acceptance_sha256=u.sha(report),
        cfg_sha256=u.sha(config), four_phase_exits=exits, reference_IO_source_sha256=REFERENCE_SHA,
        final_model_sha256=ledger[-1]['policy_sha256'], full153_state_exact=True, Adam_count=1000000,
        network_step=1000001, CSV_validation_records=200, checkpoint_transactions=11,
        actual_current_dim=211, actual_goal_dim=204, no_NN_or_training_or_SR=True))
    paths.update((out / 'checkpoint_SHA_ledger.json', out / 'closure_readback.json'))
    return run, paths


def build(args):
    u = utilities(args.reference)
    os.nice(19)
    assert subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], capture_output=True).returncode == 0
    out = args.output
    u.OUT = out  # Reuse the accepted exclusive JSON writer in this private module.
    run, paths = select(args.q, args.group, u, out)
    final = run / 'checkpoints/update_001000000'
    files = []
    for path in sorted(paths):
        assert path.is_file() and path.resolve().is_relative_to(ROOT)
        assert not any('/runs/'+g+'_' in str(path) for g in MODES if g != args.group)
        assert path.suffix not in ('.pt', '.pth', '.ckpt', '.h5', '.hdf5', '.npy', '.pyc')
        if path.suffix == '.pkl':
            assert path.parent == final and path.name in ('params_1000000.pkl', 'run_state.pkl')
        if path.suffix == '.npz':
            assert path == run / 'final_val_probe.npz'
        before = u.stat(path)
        digest = u.sha(path)
        assert u.stat(path) == before
        files.append(dict(path=str(path), member=str(path.relative_to(ROOT)), bytes=before['size'], sha256=digest, before=before))
    u.new('member_manifest.json', dict(files=files, final_only=True, old_periodics_raw_WM_cache_excluded=True))
    archive = out / f'{args.group}_final1M_closed.tar.gz'
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw, mode='w:gz', compresslevel=1, dereference=True) as tar:
            for item in files:
                tar.add(item['path'], arcname=item['member'], recursive=False)
            tar.add(out / 'member_manifest.json', arcname='member_manifest.json')
    for item in files:
        assert u.stat(item['path']) == item['before'] and u.sha(item['path']) == item['sha256']
    with gzip.open(archive, 'rb') as stream:
        while stream.read(1048576):
            pass
    expected = {x['member']: x for x in files}
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == len(files)+1
        for member in members:
            assert member.isfile()
            data = tar.extractfile(member).read()
            digest = hashlib.sha256(data).hexdigest()
            if member.name == 'member_manifest.json':
                assert digest == u.sha(out / 'member_manifest.json')
            else:
                assert len(data) == expected[member.name]['bytes'] and digest == expected[member.name]['sha256']
    u.new('receipt.json', dict(actual_exit_code=0, group_id=args.group, archive_path=str(archive),
        bytes=archive.stat().st_size, sha256=u.sha(archive), members=len(files)+1,
        member_manifest_sha256=u.sha(out / 'member_manifest.json'), source_before_after_stat_SHA_exact=True,
        all_member_SHA_size_exact=True, full_gzip_CRC_passed=True, reference_IO_sha256=REFERENCE_SHA,
        only_final_policy_and_sidecar=True, no_periodic_WM_raw_or_cache_copy=True, completed=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=list(MODES), required=True)
    parser.add_argument('--q', type=Path, default=Q)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--build-child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.q = args.q.resolve()
    args.output = (args.output or args.q / f'closed_training_backup_v2/{args.group}/completed_v1').resolve()
    assert args.q == Q.resolve() and args.output.is_relative_to(args.q / 'closed_training_backup_v2')
    if not args.execute and not args.build_child:
        print(json.dumps(dict(prepared_only=True, group=args.group, output=str(args.output), reference_sha256=REFERENCE_SHA)))
        return 0
    if args.build_child:
        code = 1
        try:
            build(args)
            code = 0
        finally:
            with (args.output / 'builder_actual_exit_code').open('x') as stream:
                stream.write(str(code)+'\n')
        return code
    args.output.mkdir(parents=True, exist_ok=False)  # Refuse any existing attempt or package.
    argv = [sys.executable, '-B', str(Path(__file__).resolve()), '--group', args.group, '--q', str(args.q),
            '--reference', str(args.reference), '--output', str(args.output), '--build-child']
    with (args.output / 'builder_actual_argv.json').open('x') as stream:
        json.dump(dict(argv=argv, started=time.time(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), stream, indent=2)
    with (args.output / 'builder_stdout.log').open('xb') as stdout, (args.output / 'builder_stderr.log').open('xb') as stderr:
        result = subprocess.run(argv, stdout=stdout, stderr=stderr)
    (args.output / 'builder_outer_exit_code').write_text(str(result.returncode)+'\n')
    if result.returncode == 0:
        print((args.output / 'receipt.json').read_text())
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
