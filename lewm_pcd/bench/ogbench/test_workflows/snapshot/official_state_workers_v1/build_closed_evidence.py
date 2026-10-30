"""Archive closed workers evidence once; raw data and diagnostic weights stay remote."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import time

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
PHASES = ('correctness', 'fresh', 'baseline_A', 'prefetch1', 'workers2', 'workers4', 'baseline_B')
DATA_NAMES = {'cube-triple-play-v0.npz', 'cube-triple-play-v0-val.npz'}
MAX_SMALL = 32 * 1024 ** 2


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def stat(path):
    s = path.stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                device=s.st_dev, inode=s.st_ino)


def require_zero(path, child=False):
    if child:
        record = read(path)
        assert record['actual_exit_code'] == 0 and record['residual_members'] == [], str(path)
        return record
    assert path.read_text().strip() == '0', str(path)
    return 0


def verify_named_freeze(path):
    record = read(path)
    for name, item in record['files'].items():
        actual = HERE / name
        assert actual.is_file() and actual.stat().st_size == item['bytes'], str(actual)
        assert digest(actual) == item['sha256'], str(actual)
    return record


def verify_data():
    acceptance_path, freeze_path = HERE / 'data_prepare/ACCEPTANCE.json', HERE / 'data_prepare/stat-freeze.json'
    acceptance, frozen = read(acceptance_path), read(freeze_path)
    assert acceptance['stage'] == 'complete' and acceptance['actual_exit_code'] == frozen['actual_exit_code'] == 0
    assert digest(acceptance_path) == frozen['acceptance_sha256']
    for item in frozen['datasets'].values():
        # Intentionally metadata only: the accepted original NPZ is never rehashed here.
        assert stat(Path(item['path'])) == item['accepted_stat']
    return dict(acceptance_sha256=digest(acceptance_path), stat_freeze_sha256=digest(freeze_path),
                accepted_stat_matches=True, original_NPZ_rehashed=False), frozen


def verify_protected():
    protected = read(HERE / 'root_preflight.json')['protected_source']
    assert len(protected) == 5
    for name, item in protected.items():
        path = CASE / 'source/ogbench' / name
        assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], str(path)
    return protected


def ensure_no_active_owned_jobs():
    scripts = {'measure.py', 'literal_baseline.py', 'run_experiments.py', 'prepare_official_state.py',
               'launch_prepare.py', 'finalize_transported.py', 'launch_finalize_transported.py'}
    active = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = [x.decode(errors='replace') for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if argv and Path(argv[0]).name.startswith('python') and any(
                str(HERE) in arg and Path(arg).name in scripts for arg in argv[1:]):
            active.append(dict(pid=int(proc.name), argv=argv))
    assert not active, active


def eligible(path, out):
    rel = path.relative_to(HERE)
    if out == path or out in path.parents or '__pycache__' in rel.parts:
        return False
    if path.is_symlink():
        raise AssertionError('Refuse symlink in evidence scope: ' + str(path))
    if not path.is_file():
        return False
    if path.name in DATA_NAMES or path.suffix in ('.pyc', '.pkl', '.part', '.incomplete', '.body', '.bin', '.partial'):
        return False
    if path.name.endswith(('.npz.assembling', '.npz.from_local.incoming', '.tar.gz', '.tmp')):
        return False
    if path.suffix == '.npz':
        allowed = rel.parts[0] in ('results', 'review_evidence_actual_v1')
    else:
        allowed = path.suffix in ('.py', '.sh', '.md', '.json', '.jsonl', '.log', '.txt', '.yaml', '.yml', '.toml', '.lock') \
                  or path.name.endswith('exit_code')
    if allowed:
        assert path.stat().st_size <= MAX_SMALL, 'Unexpected large evidence member: ' + str(path)
    return allowed


def verify_local_backup(folder):
    folder = folder.resolve()
    assert read(folder / 'scp_receipt.json')['actual_exit_code'] == 0
    require_zero(folder / 'build_actual_exit_code')
    receipt = read(folder / 'evidence_receipt.json')
    manifest_path = folder / 'evidence_manifest.json'
    assert receipt['actual_exit_code'] == 0 and digest(manifest_path) == receipt['manifest_sha256']
    manifest = read(manifest_path)
    archive_path = folder / receipt['archive']
    assert archive_path.stat().st_size == receipt['bytes'] and digest(archive_path) == receipt['sha256']
    with gzip.open(archive_path, 'rb') as stream:
        for _ in iter(lambda: stream.read(1024 * 1024), b''):
            pass
    extracted = folder / 'extracted'
    extracted.mkdir(exist_ok=False)
    names = []
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            assert member.isfile() and not name.is_absolute() and '..' not in name.parts
            assert not any(':' in part for part in name.parts) and member.name in manifest
            assert member.name not in names
            content = archive.extractfile(member).read()
            expected = manifest[member.name]
            assert len(content) == expected['bytes'] and hashlib.sha256(content).hexdigest() == expected['sha256']
            destination = extracted.joinpath(*name.parts)
            assert extracted in destination.resolve().parents
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(content)
            # Verify the bytes after actual extraction to the local project filesystem.
            assert destination.stat().st_size == expected['bytes'] and digest(destination) == expected['sha256']
            names.append(member.name)
    assert len(names) == len(manifest) == receipt['members']
    result = dict(actual_exit_code=0, verified=time.time(), archive=archive_path.name,
        archive_sha256=receipt['sha256'], members=len(names), actual_scp_exit_code=0,
        local_gzip_CRC=True, every_extracted_member_SHA_and_length=True,
        original_NPZ_copied=False, diagnostic_checkpoints_copied=False)
    (folder / 'LOCAL_BACKUP_VERIFIED.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--closed-signal', help='Root explicitly closes this run before execution.')
    parser.add_argument('--verify-local-backup', type=Path)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'evidence_backup_v1')
    parser.add_argument('--required-final-file', action='append', default=None)
    args = parser.parse_args()
    if args.verify_local_backup:
        return verify_local_backup(args.verify_local_backup)
    assert args.closed_signal, 'Root closed signal is required before packaging'
    out = args.output_dir.resolve()
    assert out.parent == HERE.resolve(), 'Output must be a fresh direct child of H'
    required = args.required_final_file or ['selected_input.json', 'RESULTS.md', 'review_actual_v1.json']
    for name in required:
        path = HERE / name
        assert path.is_file() and path.stat().st_size > 0, str(path)
    for name in ('cpu_actual_exit_code', 'experiment_actual_exit_code', 'literal_actual_exit_code',
                 'data_prepare/finalize_actual_exit_code', 'data_prepare/stat_freeze_actual_exit_code'):
        require_zero(HERE / name)
    cpu = require_zero(HERE / 'cpu_runtime/real_CPU_pytest_exit.json', child=True)
    phases = {name: require_zero(HERE / 'experiment_runtime' / (name + '_exit.json'), child=True) for name in PHASES}
    literal = require_zero(HERE / 'literal_runtime/literal_exit.json', child=True)
    ensure_no_active_owned_jobs()
    data, frozen_data = verify_data()
    protected = verify_protected()
    input_freeze = verify_named_freeze(HERE / 'input_freeze_v1.json')
    run_freeze = verify_named_freeze(HERE / 'run_source_freeze_v1.json')
    assert run_freeze['data_acceptance_sha256'] == data['acceptance_sha256']
    assert run_freeze['data_stat_freeze_sha256'] == data['stat_freeze_sha256']
    report_paths = [HERE / 'results' / name / 'report.json' for name in PHASES]
    report_paths.append(HERE / 'results/literal_baseline/report.json')
    for path in report_paths:
        report = read(path)
        assert report['actual_exit_code'] == 0 and not report['formal_training'] and not report['SR']
        assert report['harness_sha256'] == run_freeze['files']['measure.py']['sha256']
        assert report['adapter_sha256'] == run_freeze['files']['ordered_input.py']['sha256']
        assert report['data_receipt']['sha256'] == data['acceptance_sha256']
        assert report['data_stat_freeze']['sha256'] == data['stat_freeze_sha256']
        assert report['raw_npz_stats_before'] == report['raw_npz_stats_after']
        for name, expected in report['source_sha256'].items():
            assert expected == protected['impls/' + name]['sha256']
    literal_report = read(report_paths[-1])
    assert literal_report['literal_official_sampler_control']
    assert literal_report['control_wrapper_sha256'] == digest(HERE / 'literal_baseline.py')
    # The output directory is created only after every closure gate passes. Old failed attempts are retained.
    out.mkdir(exist_ok=False)
    pointers = {}
    for path in sorted(HERE.glob('results/**/*.pkl')):
        before = stat(path)
        pointers[path.relative_to(HERE).as_posix()] = dict(bytes=before['size'], sha256=digest(path),
            accepted_stat=before, copied_locally=False)
        assert stat(path) == before
    assert pointers, 'Expected diagnostic checkpoint pointer'
    checks = dict(closed_signal=args.closed_signal, checked=time.time(), CPU=cpu, phases=phases,
        literal=literal, data=data, protected_source=protected,
        input_freeze_sha256=digest(HERE / 'input_freeze_v1.json'),
        run_source_freeze_sha256=digest(HERE / 'run_source_freeze_v1.json'),
        final_file_sha256={name: digest(HERE / name) for name in required},
        actual_exit_code=0, formal_training=False, SR=False)
    policy = dict(include='H small evidence, exact official protected source, small diagnostic NPZ',
        original_full_NPZ_excluded=True, diagnostic_checkpoints_pointer_only=True,
        excluded_suffixes=['.part', '.incomplete', '.body', '.bin', '.pkl', '.pyc', '.partial'],
        max_member_bytes=MAX_SMALL, package_algorithm='official_state_v1/collect_evidence.py extended closure gates')
    for name, obj in [('closure_checks.json', checks), ('checkpoint_pointers.json', pointers), ('package_policy.json', policy)]:
        (out / name).write_text(json.dumps(obj, indent=2) + '\n')
    def collect():
        paths = {p.relative_to(HERE).as_posix(): p for p in HERE.rglob('*') if eligible(p, out)}
        for name in protected:
            paths['protected_official_source/' + name] = CASE / 'source/ogbench' / name
        for name in ('closure_checks.json', 'checkpoint_pointers.json', 'package_policy.json'):
            paths['_backup/' + name] = out / name
        return dict(sorted(paths.items()))
    paths = collect()
    before = {name: stat(path) for name, path in paths.items()}
    manifest = {}
    target = out / 'closed_state_workers_evidence_v1.tar.gz'
    partial = target.with_name(target.name + '.partial')
    with tarfile.open(partial, 'w:gz') as archive:
        for name, path in paths.items():
            assert stat(path) == before[name], str(path)
            content = path.read_bytes()
            assert stat(path) == before[name], str(path)
            manifest[name] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            archive.addfile(archive.gettarinfo(str(path), arcname=name), io.BytesIO(content))
    after_paths = collect()
    assert list(paths) == list(after_paths), 'Evidence file membership changed during archive'
    assert {name: stat(path) for name, path in after_paths.items()} == before, 'Evidence source changed during archive'
    assert verify_protected() == protected
    assert verify_data()[0] == data
    for name, pointer in pointers.items():
        assert stat(HERE / name) == pointer['accepted_stat']
    with gzip.open(partial, 'rb') as stream:
        for _ in iter(lambda: stream.read(1024 * 1024), b''):
            pass
    with tarfile.open(partial, 'r:gz') as archive:
        names = []
        for member in archive:
            assert member.isfile() and member.name in manifest
            content = archive.extractfile(member).read()
            assert len(content) == manifest[member.name]['bytes']
            assert hashlib.sha256(content).hexdigest() == manifest[member.name]['sha256']
            names.append(member.name)
        assert len(names) == len(set(names)) == len(manifest)
    os.rename(partial, target)
    (out / 'evidence_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    receipt = dict(actual_exit_code=0, completed=time.time(), archive=target.name,
        bytes=target.stat().st_size, sha256=digest(target), members=len(manifest),
        gzip_CRC=True, all_member_SHA_and_length=True, source_before_after_stat_exact=True,
        official_protected_sources_unchanged=True, accepted_data_stat_unchanged=True,
        original_NPZ_rehashed=False, original_NPZ_copied=False, diagnostic_checkpoints_remote_only=True,
        closure_checks_sha256=digest(out / 'closure_checks.json'), manifest_sha256=digest(out / 'evidence_manifest.json'),
        formal_training=False, SR=False)
    (out / 'evidence_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
