"""Closed G01 final-policy backup; periodic policies remain remote."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import subprocess
import tarfile
import time

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
P = ROOT / 'cube23_distance_gciql_v1/state_priority_v1'
OUT = P / 'closed_training_backup_v1/G01'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def stat(path):
    v = Path(path).stat()
    return dict(size=v.st_size, mtime_ns=v.st_mtime_ns, ctime_ns=v.st_ctime_ns,
                device=v.st_dev, inode=v.st_ino)


def new(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def main():
    os.nice(19)
    io = subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], capture_output=True, text=True)
    assert io.returncode == 0, io.stderr
    assert not (OUT / 'G01_final1M_closed.tar.gz').exists() and not (OUT / 'receipt.json').exists()
    new('builder_started.json', dict(started=time.time(), nice=os.getpriority(os.PRIO_PROCESS, 0),
        ionice_actual_exit_code=io.returncode, source_sha256=sha(__file__), training_or_NN_started=False))
    config = read(P / 'configs/G01_cube1_state.json')
    run = Path(config['output'])
    assert run == P / 'runs/G01_cube1_state'
    for phase in ('07_train_G01', '08_accept_G01'):
        x = read(P / 'runtime' / phase / 'exit.json')
        assert x['actual_exit_code'] == 0 and not x['residual_owned_members']
        assert (P / 'runtime' / phase / 'actual_exit_code').read_text().strip() == '0'
    accepted = read(P / 'acceptance/G01/report.json')
    assert accepted['actual_exit_code'] == 0 and accepted['full153_state_exact']
    assert accepted['Adam_count'] == 1000000 and accepted['network_step'] == 1000001
    assert accepted['checkpoint_count'] == 11 and accepted['CSV_validation_records'] == 200
    assert (accepted['actual_current_dim'], accepted['actual_goal_dim']) == (28, 3)
    assert (run / 'trainer_exit_code').read_text().strip() == '0'
    status = read(run / 'status.json')
    assert status['exit_code'] == 0 and status['completed_updates'] == 1000000
    paths = {P / 'acceptance/G01/report.json', P / 'configs/G01_cube1_state.json'}
    for name in ('status.json', 'resolved.json', 'train.csv', 'trainer_exit_code',
                 'final_state_fingerprint.json', 'final_val_probe.npz'):
        paths.add(run / name)
    counters = [1] + list(range(100000, 1000001, 100000))
    assert [v['counter'] for v in accepted['checkpoint_transactions']] == counters
    checkpoint_ledger = []
    for counter, audited in zip(counters, accepted['checkpoint_transactions']):
        folder = run / 'checkpoints' / f'update_{counter:09d}'
        manifest = folder / 'manifest.json'
        assert sha(manifest) == audited['manifest_sha256']
        identity = read(manifest)
        assert identity['counter'] == counter
        sidecar = folder / 'run_state.pkl'
        assert sha(sidecar) == identity['sidecar_sha256']
        policy = folder / f'params_{counter}.pkl'
        checkpoint_ledger.append(dict(counter=counter, manifest_path=str(manifest),
            manifest_sha256=sha(manifest), sidecar_path=str(sidecar), sidecar_sha256=sha(sidecar),
            policy_path=str(policy), policy_stat=stat(policy), policy_sha256=identity['agent_sha256'],
            policy_included=counter == 1000000, periodic_policy_SHA_adopted_from_completed_CPU_accept=True))
        paths.update((manifest, sidecar))
        if counter == 1000000:
            assert policy.stat().st_size == 39379738 and sha(policy) == identity['agent_sha256']
            paths.add(policy)
    for phase in ('07_train_G01', '08_accept_G01'):
        for name in ('actual_argv.json', 'run.log', 'status.json', 'actual_exit_code', 'exit.json', 'resources.jsonl'):
            paths.add(P / 'runtime' / phase / name)
    for rel in ('release_freeze.json', 'FINAL_STATIC_REVIEW_v1.json', 'DATA_ROLE_CONFIG_FREEZE_v1.json',
                'INPUT_FREEZE_v1.json', 'INPUT_HANDOFF_v1.json', 'dispatch_receipt.json'):
        paths.add(P / rel)
    for rel in ('reference/report.json', 'reference/selection.json', 'numerical_comparison.json',
                'throughput_selection.json', 'reload_report.json'):
        paths.add(P / 'gates/G01' / rel)
    release = read(P / 'release_freeze.json')
    assert sha(P / 'release_freeze.json') == '1bc9fff9ffe485cd91f5ea69fdbea4e55ae83a367e98b89ba4129fd3c7e4321d'
    other_config_references = {}
    for path, expected in release['files'].items():
        if '/configs/G04_' in path or '/configs/G07_' in path:
            other_config_references[path] = expected
            continue
        assert sha(path) == expected
        paths.add(Path(path))
    cpu = Path(release['target_cpu_receipt']['path'])
    assert sha(cpu) == release['target_cpu_receipt']['sha256'] and read(cpu)['actual_exit_code'] == 0
    paths.add(cpu)
    for path, expected in read(run / 'resolved.json')['binding']['code'].items():
        assert sha(path) == expected
        paths.add(Path(path))
    new('checkpoint_SHA_ledger.json', dict(actual_exit_code=0, checkpoints=checkpoint_ledger,
        final_policy_only_included=True, all11_policies_retained_remote=True,
        periodic10_policy_contents_not_rehashed_or_copied=True,
        other_group_config_SHA_references_only=other_config_references))
    paths.update((OUT / 'checkpoint_SHA_ledger.json', Path(__file__)))
    files = []
    for path in sorted(paths):
        assert path.is_file() and path.resolve().is_relative_to(ROOT)
        assert 'runtime/status.json' not in str(path) and 'runtime/resources.jsonl' not in str(path)
        assert '09_train_' not in str(path) and '/runs/G04_' not in str(path) and '/runs/G07_' not in str(path)
        if path.suffix in ('.pt', '.pth', '.ckpt', '.h5', '.hdf5', '.npy'):
            raise AssertionError(path)
        if path.suffix == '.pkl':
            assert path.name == 'run_state.pkl' or path == run / 'checkpoints/update_001000000/params_1000000.pkl'
        if path.suffix == '.npz':
            assert path == run / 'final_val_probe.npz'
        before = stat(path)
        h = sha(path)
        assert stat(path) == before
        files.append(dict(path=str(path), member=str(path.relative_to(ROOT)), bytes=before['size'],
                          sha256=h, before=before))
    new('member_manifest.json', dict(group_id='G01', files=files,
        original11checkpoints_retained=True, SR_performed=False, active_files_included=False))
    archive = OUT / 'G01_final1M_closed.tar.gz'
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw, mode='w:gz', compresslevel=1, dereference=True) as tar:
            for item in files:
                tar.add(item['path'], arcname=item['member'], recursive=False)
            tar.add(OUT / 'member_manifest.json', arcname='member_manifest.json')
    for item in files:
        assert stat(item['path']) == item['before'] and sha(item['path']) == item['sha256']
    with gzip.open(archive, 'rb') as stream:
        while stream.read(1024 * 1024):
            pass
    expected = {v['member']: v for v in files}
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == len(files) + 1
        for member in members:
            assert member.isfile()
            b = tar.extractfile(member).read()
            if member.name == 'member_manifest.json':
                assert hashlib.sha256(b).hexdigest() == sha(OUT / 'member_manifest.json')
            else:
                item = expected[member.name]
                assert len(b) == item['bytes'] and hashlib.sha256(b).hexdigest() == item['sha256']
    new('receipt.json', dict(actual_exit_code=0, archive_path=str(archive), bytes=archive.stat().st_size,
        sha256=sha(archive), members=len(files) + 1, source_files=len(files),
        member_manifest_sha256=sha(OUT / 'member_manifest.json'), source_before_after_stat_SHA_exact=True,
        all_member_SHA_size_exact=True, gzip_CRC_full_read=True, low_priority_nice=19, ionice_class=3,
        periodic10_policy_transfer=False, original11policy_preserved=True, NN_or_training_or_SR=False,
        active_group_or_parent_runtime_files_read=False, completed=time.time()))
    print((OUT / 'receipt.json').read_text())


if __name__ == '__main__':
    code = 1
    try:
        main()
        code = 0
    finally:
        (OUT / 'builder_actual_exit_code').write_text(str(code) + '\n')
