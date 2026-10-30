"""Package only closed G01/G04/G07 SR evidence; never launch NN, audits or SR."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
P = ROOT / 'cube23_distance_gciql_v1/state_priority_v1'
SR = P / 'native_sr_v1'
OUT = SR / 'closed_raw_evidence_backup_v1/completed_v1'
RELEASE_SHA = '4011e4fbda6ee680ae23819b615e67d7fd7b5c3584014676028e5df18ee662de'
PLAN_SHA = 'aad96b495d32f2a6444dc2cfce38595f2e35e9873fcc557d6f32378bf8f563be'
GROUPS = ('G01', 'G04', 'G07')
FORBIDDEN = ('.pkl', '.pt', '.pth', '.ckpt', '.h5', '.hdf5', '.npy')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def stat(path):
    v = Path(path).stat()
    return dict(size=v.st_size, mtime_ns=v.st_mtime_ns, ctime_ns=v.st_ctime_ns,
                device=v.st_dev, inode=v.st_ino)


def new(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def closed_paths():
    assert sha(SR / 'release_freeze.json') == RELEASE_SHA
    assert sha(SR / 'launch_plan.json') == PLAN_SHA
    release, plan = read(SR / 'release_freeze.json'), read(SR / 'launch_plan.json')
    status = read(SR / 'runtime/status.json')
    assert status['stage'] == 'three_final1M_SR_and_audits_complete'
    assert status['actual_exit_code'] == 0 and status['completed_phases'] == 12
    for path in (SR / 'runtime/outer_exit_code', SR / 'wrapper_exit_code'):
        assert path.read_text().strip() == '0', path
    assert plan['groups'] == list(GROUPS) and plan['training_updates'] == 0
    paths = {SR / n for n in ('release_freeze.json', 'launch_plan.json', 'dispatch_receipt.json',
                             'dispatch.log', 'wrapper_exit_code', 'runtime/status.json', 'runtime/outer_exit_code')}
    external = {}
    for path, expected in release['reference_pins'].items():
        path = Path(path)
        if path.suffix in FORBIDDEN + ('.npz',):
            external[str(path)] = dict(sha256=expected, not_read_or_copied=True)
            continue
        assert sha(path) == expected, path
        paths.add(path)
    for relative, expected in release['local_files'].items():
        path = SR / relative
        assert path.resolve().is_relative_to(SR) and sha(path) == expected, path
        paths.add(path)
    assert len(plan['phases']) == 12
    for phase in plan['phases']:
        folder = SR / 'runtime' / phase['name']
        result = read(folder / 'exit.json')
        assert result['actual_exit_code'] == 0 and result['residual_owned_members'] == []
        assert (folder / 'actual_exit_code').read_text().strip() == '0'
        assert read(folder / 'actual_argv.json')['argv'] == phase['argv']
        for check in phase.get('checks', []):
            report = read(check['path'])
            assert all(report[k] == v for k, v in check['equals'].items()), phase['name']
        paths.update(folder / n for n in ('actual_argv.json', 'run.log', 'resources.jsonl',
                                         'exit.json', 'actual_exit_code'))
    summaries = {}
    for group in GROUPS:
        for mode, count in (('diagnostic', 1), ('full', 250)):
            output, audit = SR / mode / group, SR / 'audits' / mode / group
            report, accepted = read(output / 'report.json'), read(audit / 'report.json')
            assert report['actual_exit_code'] == accepted['actual_exit_code'] == 0
            assert report['episodes'] == accepted['episodes'] == count
            assert report['unique_seeds'] == accepted['unique_seeds'] == count
            assert report['successes'] == accepted['successes']
            for folder in (output, audit):
                assert (folder / 'actual_exit_code').read_text().strip() == '0'
            paths.update(output / n for n in ('binding.json', 'protocol.json', 'status.json',
                                              'report.json', 'actual_exit_code'))
            paths.update(audit / n for n in ('report.json', 'actual_exit_code'))
            identities = set()
            for task in ([1] if mode == 'diagnostic' else range(1, 6)):
                paths.add(output / f'task{task}/summary.json')
                for episode in range(1 if mode == 'diagnostic' else 50):
                    folder = output / f'task{task}/episode_{episode:03d}'
                    result = read(folder / 'result.json')
                    assert result['actual_exit_code'] == 0 and (folder / 'attempt_exit_code').read_text().strip() == '0'
                    assert result['group_id'] == group and result['task_id'] == task and result['episode'] == episode
                    assert result['diagnostic'] == (mode == 'diagnostic')
                    identities.add(result['seed'])
                    paths.update(folder / n for n in ('reset.json', 'policy_inputs.json', 'steps.npz',
                                                      'result.json', 'attempt_exit_code'))
            assert len(identities) == count
            if mode == 'full':
                summaries[group] = dict(episodes=count, successes=report['successes'],
                    policy_steps=report['policy_steps'], raw_report_sha256=sha(output / 'report.json'),
                    audit_report_sha256=sha(audit / 'report.json'))
    return paths, external, summaries


def main():
    os.nice(19)
    io = subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], capture_output=True, text=True)
    assert io.returncode == 0, io.stderr
    paths, external, summaries = closed_paths()
    refs = Path(__file__).parent / 'TRAINING_EXTERNAL_REFERENCES_v1.json'
    assert read(refs)['all3_final_only_local_backups_closed'] is True
    paths.update((refs, Path(__file__)))
    OUT.mkdir(exist_ok=False)
    new('closure_readback.json', dict(actual_exit_code=0, release_sha256=RELEASE_SHA,
        launch_plan_sha256=PLAN_SHA, formal_episodes=750, groups=summaries,
        old_training_external_pins=external, training_models_or_cache_read=False,
        no_NN_audit_SR_started=True, closed_at=time.time()))
    paths.add(OUT / 'closure_readback.json')
    files = []
    for path in sorted(paths):
        assert path.is_file() and path.resolve().is_relative_to(ROOT)
        assert path.suffix not in FORBIDDEN, path
        if path.suffix == '.npz':
            assert path.name == 'steps.npz' and any(path.is_relative_to(SR / m / g)
                for m in ('full', 'diagnostic') for g in GROUPS), path
        before = stat(path)
        digest = sha(path)
        assert stat(path) == before, path
        files.append(dict(path=str(path), member=str(path.relative_to(ROOT)), bytes=before['size'],
                          sha256=digest, before=before))
    new('member_manifest.json', dict(files=files, formal_episodes=750, diagnostics_excluded=True,
                                    model_cache_dataset_copies=False, active_files_included=False))
    archive = OUT / 'G01_G04_G07_final1M_native_SR_closed.tar.gz'
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
    expected = {item['member']: item for item in files}
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == len(files) + 1
        for member in members:
            assert member.isfile()
            value = tar.extractfile(member).read()
            digest = hashlib.sha256(value).hexdigest()
            if member.name == 'member_manifest.json':
                assert digest == sha(OUT / 'member_manifest.json')
            else:
                item = expected[member.name]
                assert len(value) == item['bytes'] and digest == item['sha256']
    new('receipt.json', dict(actual_exit_code=0, archive_path=str(archive), bytes=archive.stat().st_size,
        sha256=sha(archive), members=len(files)+1, source_files=len(files),
        member_manifest_sha256=sha(OUT / 'member_manifest.json'), source_before_after_stat_SHA_exact=True,
        all_member_SHA_size_exact=True, gzip_CRC_full_read=True, nice=19, ionice_class=3,
        no_model_cache_dataset_copy=True, no_NN_audit_SR_started=True, completed=time.time()))
    print((OUT / 'receipt.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(prepare_only=True, expected_formal_episodes=750, no_remote_reads=True)))
    else:
        code = 1
        try:
            main()
            code = 0
        finally:
            if OUT.is_dir():
                with (OUT / 'builder_actual_exit_code').open('x') as stream:
                    stream.write(str(code)+'\n')
