"""Closed restored-M5 SR evidence only; no NN, simulator, policy/cache copies."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import subprocess
import tarfile
import time

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
BASE = ROOT / 'cube23_distance_gciql_v1/native_sr_visual_v1'
R = BASE / 'm5_restore_v1'
OUT = R / 'closed_sr_backup_v1/completed_v2'
RELEASE = '375e900e00380c8f26ad3242278209f9a121878523ebdc094bf8c87f4f0fd4da'
PLAN = '1152c9bfffa95092d25baeabbb1c891fee3c44c1872e21db4ffc95b1cc98ef36'
GROUPS = {'G05': (2, 'rgb', 'cube2/2d_epoch10', '995a99f4904d87e71424b2de7910e7e0cf79556de8c2cebd08ac582d70880190'),
          'G06': (2, 'pcd', 'cube2/3d_epoch10', 'f391f047aa73fa08b2b4534e6018fde6a219cc3ae6ba4464b907e2b9f9138a46'),
          'G08': (3, 'rgb', '2d_epoch10', '4e16a6c5c58e8c44728d153e4ca7149d05676973379d99774ff7db2e08bbfd4e'),
          'G09': (3, 'pcd', '3d_epoch10', '07bb8d2160b1a0fecda35bba3aede4790071540a6d7a9c34b92d89ae437a04b6')}
FORBIDDEN = ('.pkl', '.pt', '.pth', '.ckpt', '.h5', '.hdf5', '.npy', '.pyc')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stat(path):
    v = Path(path).stat()
    return dict(size=v.st_size, mtime_ns=v.st_mtime_ns, ctime_ns=v.st_ctime_ns,
                device=v.st_dev, inode=v.st_ino)


def read(path):
    return json.loads(Path(path).read_text())


def save(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def collect():
    assert sha(R / 'release_freeze.json') == RELEASE and sha(R / 'launch_plan.json') == PLAN
    plan = read(R / 'launch_plan.json')
    expected = [f'{group}_{mode}' for group in GROUPS for mode in ('smoke', 'full')]
    status = read(R / 'runtime/status.json')
    assert status['stage'] == 'complete' and status['actual_exit_code'] == 0
    assert status['completed_phases'] == expected == [x['name'] for x in plan['stages']]
    assert (R / 'runtime/outer_exit_code').read_text().strip() == (R / 'wrapper_exit_code').read_text().strip() == '0'
    paths = {p for p in R.iterdir() if p.is_file() and p.suffix in ('.py', '.sh', '.json', '.log')}
    paths.update((R / 'wrapper_exit_code', R / 'benchmark.actual_exit_code',
                  R / 'runtime/status.json', R / 'runtime/outer_exit_code'))
    assert (R / 'benchmark.actual_exit_code').read_text().strip() == '0'
    phases, summaries = {}, {}
    pgids = set()
    for stage in plan['stages']:
        group, name = stage['group'], stage['name']
        phase, output = R / 'runtime' / name, Path(stage['output'])
        assert output == R / ('smoke' if stage['expected_episodes'] == 1 else 'full') / group
        assert (phase / 'actual_exit_code').read_text().strip() == (output / 'actual_exit_code').read_text().strip() == '0'
        audit = read(phase / 'audit.json')
        assert audit['actual_exit_code'] == 0 and audit['summary']['episodes'] == stage['expected_episodes']
        assert read(phase / 'actual_argv.json') == stage
        process = read(phase / 'process.json')
        assert process['argv'] == stage['argv']
        pgids.add(process['pid'])
        report, threads = read(output / 'report.json'), read(output / 'runtime_threads.json')
        assert report['actual_exit_code'] == 0 and report['episodes'] == report['unique_seeds'] == stage['expected_episodes']
        assert report['encoding_protocol'] == 'fresh_current_and_goal_each_policy_call_2N'
        assert threads['torch_num_threads'] == 1 and threads['numpy_physics']['BLAS_threads'] == 4
        assert audit['summary']['successes'] == report['successes']
        for root in (phase, output):
            for path in root.rglob('*'):
                if path.is_file():
                    assert not any(x in path.parts for x in ('__pycache__', 'tmp')) and path.suffix != '.tmp'
                    paths.add(path)
        phases[name] = dict(actual_exit_code=0, audit_actual_exit_code=0, owned_PGID=process['pid'])
        if stage['expected_episodes'] == 250:
            results = sorted((output / 'episodes').glob('task*/episode_*/result.json'))
            assert len(results) == 250
            summaries[group] = dict(episodes=250, successes=report['successes'],
                success_rate=report['success_rate'], report_sha256=sha(output / 'report.json'),
                audit_sha256=sha(phase / 'audit.json'))
    residual = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] != 'Z' and int(fields[2]) in pgids:
                residual.append(int(proc.name))
        except (OSError, ProcessLookupError):
            pass
    assert not residual, residual
    release = read(R / 'release_freeze.json')
    for path, expected_sha in release['target_pins'].items():
        assert sha(path) == expected_sha
        paths.add(Path(path))
    external = {}
    ext = ROOT / 'cube23_distance_gciql_v1/five_group_extension_v1'
    for group, (cubes, mode, wm_folder, wm_sha) in GROUPS.items():
        config = ext / f'configs/{group}_cube{cubes}_{mode}.json'
        accepted = ext / f'acceptance/{group}_full_acceptance.json'
        cfg, acceptance = read(config), read(accepted)
        assert acceptance['actual_exit_code'] == 0 and acceptance['full153_state_exact']
        final = Path(acceptance['final_checkpoint'])
        manifest = final / 'manifest.json'
        assert sha(manifest) == acceptance['final_manifest_sha256']
        identity = read(manifest)
        assert identity['counter'] == 1000000
        policy = final / 'params_1000000.pkl'
        wm = ROOT / 'checkpoints' / wm_folder / 'weights_epoch_10.pt'
        external[group] = dict(config_sha256=sha(config), acceptance_sha256=sha(accepted),
            policy=dict(path=str(policy), sha256=identity['agent_sha256'], stat=stat(policy), content_not_read=True),
            world_model=dict(path=str(wm), sha256=wm_sha, stat=stat(wm), content_not_read=True))
        paths.update((config, accepted, manifest))
        for name in ('config.json', 'config.yaml'):
            paths.add(wm.parent / name)
        for path, expected_sha in read(Path(cfg['output']) / 'resolved.json')['binding']['code'].items():
            assert Path(path).suffix not in FORBIDDEN + ('.npz',)
            assert sha(path) == expected_sha
            paths.add(Path(path))
    paths.add(ROOT / 'checkpoint_encoder.py')
    paths.update(ROOT / rel for rel in ('external/native_pcd_frame.py', 'external/pcd_geometry.py',
        'source/lewm_pcd/envs/cube_pcd_env.py', 'source/lewm_pcd/envs/pcd_wrapper.py'))
    paths.add(ROOT / 'assets/imports/single_cube_rl3_101632_20261018_v1/source_layout/coordination/cached_lewm_state_gciql_xyz_109_20261012_v1/native_v1/locked_dependencies/native_runtime_helpers.py')
    for relative in ('jepa.py', 'module.py', 'utils.py', 'encoders/vit.py', 'encoders/__init__.py',
                     'third_party/stable-worldmodel/stable_worldmodel/wm/utils.py'):
        paths.add(ROOT / 'source/lewm_pcd' / relative)
    reference = ROOT / 'cube23_distance_gciql_v1/state_priority_v1/native_sr_v1/configs/G04.json'
    paths.add(reference)
    for path, expected_sha in read(reference)['source_pins'].items():
        if Path(path).suffix in FORBIDDEN + ('.npz',):
            continue
        assert sha(path) == expected_sha
        paths.add(Path(path))
    return paths, dict(actual_exit_code=0, stages=phases, residual_owned_members=[],
        groups=summaries, checkpoint_and_WM_external_references=external,
        source_release_sha256=RELEASE, launch_plan_sha256=PLAN,
        models_cache_raw_or_old_SR_contents_read=False, no_NN_or_audit_or_SR_started=True)


def main():
    os.nice(19)
    io = subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], capture_output=True, text=True)
    assert io.returncode == 0, io.stderr
    assert not OUT.exists(), 'Existing backup; never duplicate it'
    paths, closure = collect()
    OUT.mkdir()
    save('closure_readback.json', closure)
    paths.update((OUT / 'closure_readback.json', Path(__file__)))
    files = []
    for path in sorted(paths):
        assert path.is_file() and path.resolve().is_relative_to(ROOT) and path.suffix not in FORBIDDEN
        if path.suffix == '.npz':
            assert path.name == 'steps.npz' and any(path.is_relative_to(R / mode) for mode in ('full', 'smoke'))
        before = stat(path)
        digest = sha(path)
        assert stat(path) == before
        files.append(dict(path=str(path), member=str(path.relative_to(ROOT)), bytes=before['size'], sha256=digest, before=before))
    save('member_manifest.json', dict(files=files, raw_formal_episodes=1000, smoke_episodes=4,
                                    models_and_dataset_excluded=True, active_files_included=False))
    print(json.dumps(dict(stage='packaging_closed_files', source_files=len(files), source_bytes=sum(x['bytes'] for x in files))), flush=True)
    archive = OUT / 'G05_G06_G08_G09_M5_restored_closed_SR_v1.tar.gz'
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw, mode='w:gz', compresslevel=1, dereference=True) as tar:
            for item in files:
                tar.add(item['path'], arcname=item['member'], recursive=False)
            tar.add(OUT / 'member_manifest.json', arcname='member_manifest.json')
    for item in files:
        assert stat(item['path']) == item['before'] and sha(item['path']) == item['sha256']
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
                assert digest == sha(OUT / 'member_manifest.json')
            else:
                item = expected[member.name]
                assert len(data) == item['bytes'] and digest == item['sha256']
    save('receipt.json', dict(actual_exit_code=0, archive_path=str(archive), bytes=archive.stat().st_size,
        sha256=sha(archive), members=len(files)+1, source_files=len(files), source_bytes=sum(x['bytes'] for x in files),
        member_manifest_sha256=sha(OUT / 'member_manifest.json'), source_before_after_stat_SHA_exact=True,
        all_member_SHA_size_exact=True, full_gzip_CRC_passed=True, nice=19, ionice_class=3,
        no_training_WM_cache_raw_or_old_SR_copy=True, no_NN_or_SR_or_audit_started=True, completed=time.time()))
    print((OUT / 'receipt.json').read_text(), flush=True)


if __name__ == '__main__':
    code = 1
    try:
        main()
        code = 0
    finally:
        if OUT.is_dir():
            with (OUT / 'builder_actual_exit_code').open('x') as stream:
                stream.write(str(code)+'\n')
