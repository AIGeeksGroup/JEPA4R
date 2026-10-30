"""One closed archive transport -> scratch -> verified project copy; never build or run NN."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time

ROOT = '/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=('G11', 'G12'), required=True)
    parser.add_argument('--host', choices=('103032Test',), default='103032Test')
    parser.add_argument('--local-output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    remote = ROOT + f'/cube4_visual_gciql_v1/closed_training_backup_v2/{args.group}/completed_v1'
    if not args.execute:
        print(json.dumps(dict(prepared_only=True, remote=remote, local_output=str(args.local_output))))
        return
    args.local_output.mkdir(parents=True, exist_ok=False)  # Duplicate refusal before network copy.
    script = '''import pathlib,json
p=pathlib.Path(remote)
assert (p/'builder_actual_exit_code').read_text().strip()=='0'
assert (p/'builder_outer_exit_code').read_text().strip()=='0'
r=json.loads((p/'receipt.json').read_text());assert r['actual_exit_code']==0 and r['group_id']==group
print(json.dumps(r))
'''
    metadata = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=60', args.host,
        ROOT+'/env_runtime/bin/python -B -'], input=(f'remote={remote!r}\ngroup={args.group!r}\n'+script).encode(), capture_output=True)
    (args.local_output/'metadata.stdout.log').write_bytes(metadata.stdout)
    (args.local_output/'metadata.stderr.log').write_bytes(metadata.stderr)
    (args.local_output/'metadata_actual_exit.json').write_text(json.dumps(dict(
        actual_exit_code=metadata.returncode, completed=time.time()))+'\n')
    assert metadata.returncode == 0, metadata.stderr.decode(errors='replace')
    receipt = json.loads(metadata.stdout)
    assert receipt['source_before_after_stat_SHA_exact'] and receipt['all_member_SHA_size_exact'] and receipt['full_gzip_CRC_passed']
    scratch = Path(tempfile.mkdtemp(prefix=f'quad_{args.group}_closed_'))
    assert 'onedrive' not in str(scratch).lower()
    names = [f'{args.group}_final1M_closed.tar.gz', 'receipt.json', 'member_manifest.json',
             'checkpoint_SHA_ledger.json', 'closure_readback.json', 'builder_actual_exit_code',
             'builder_outer_exit_code', 'builder_actual_argv.json', 'builder_stdout.log', 'builder_stderr.log']
    assert receipt['archive_path'] == remote+'/'+names[0]
    argv = ['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=60',
            *[f'{args.host}:{remote}/{name}' for name in names], str(scratch)]
    with (args.local_output/'transport.stdout.log').open('xb') as stdout, (args.local_output/'transport.stderr.log').open('xb') as stderr:
        copied = subprocess.run(argv, stdout=stdout, stderr=stderr)
    (args.local_output/'TRANSPORT_ACTUAL.json').write_text(json.dumps(dict(actual_exit_code=copied.returncode,
        argv=argv, scratch=str(scratch), completed=time.time()), indent=2)+'\n')
    assert copied.returncode == 0
    archive = scratch/names[0]
    assert archive.stat().st_size == receipt['bytes'] and sha(archive) == receipt['sha256']
    assert json.loads((scratch/'receipt.json').read_text()) == receipt
    assert (scratch/'builder_actual_exit_code').read_text().strip() == (scratch/'builder_outer_exit_code').read_text().strip() == '0'
    manifest = scratch/'member_manifest.json'
    assert sha(manifest) == receipt['member_manifest_sha256']
    expected = {x['member']: x for x in json.loads(manifest.read_text())['files']}
    with gzip.open(archive, 'rb') as stream:
        while stream.read(1048576):
            pass
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == receipt['members'] == len(expected)+1
        for member in members:
            assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            data = tar.extractfile(member).read()
            digest = hashlib.sha256(data).hexdigest()
            if member.name == 'member_manifest.json':
                assert digest == receipt['member_manifest_sha256']
            else:
                assert digest == expected[member.name]['sha256'] and len(data) == expected[member.name]['bytes']
    for name in names:
        source, target = scratch/name, args.local_output/name
        shutil.copyfile(source, target)
        assert source.stat().st_size == target.stat().st_size and sha(source) == sha(target)
    result = dict(actual_exit_code=0, group_id=args.group, archive_path=str((args.local_output/names[0]).resolve()),
        archive_bytes=receipt['bytes'], archive_sha256=receipt['sha256'], members=receipt['members'],
        source_before_after_stat_SHA_exact=True, all_member_SHA_size_exact=True, full_gzip_CRC_passed=True,
        builder_inner_actual_exit_code=0, builder_outer_actual_exit_code=0, metadata_SSH_actual_exit_code=0,
        single_SCP_transport_actual_exit_code=0, single_archive_network_copy=True,
        scratch_outside_OneDrive=str(scratch), projectcopy_wholeSHA_size_exact=True,
        only_final_policy_and_sidecar=True, no_NN_training_SR_or_cache_payload_copy=True, completed=time.time())
    with (args.local_output/'LOCAL_BACKUP_VERIFIED.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
