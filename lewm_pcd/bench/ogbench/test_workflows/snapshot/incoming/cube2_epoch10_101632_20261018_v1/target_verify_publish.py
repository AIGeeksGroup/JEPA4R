"""Verify a copied, closed Cube2 bundle and publish to a new, isolated directory."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
FINAL = ROOT / 'checkpoints/cube2'
ARCHIVE_SHA = '0fd39945372a354da5fa42fb8495e4faa90d2806f4f747f5f4204a5b755efc0b'
VERIFY_SHA = 'f26e07e734443befa3930b3abacd52c8698f7e5073ff2f680e56d0418d97ab74'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    with (HERE / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def protect(baseline):
    for path, expected in baseline.items():
        path = Path(path)
        assert path.is_file() and not path.is_symlink(), str(path)
        assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], str(path)


report = {'started': time.time(), 'actual_exit_code': 1, 'target_final_root': str(FINAL),
          'training_encoding_SR_performed': False, 'profiles_modified': False}
try:
    assert HERE.parent == ROOT / 'incoming' and HERE.name == 'cube2_epoch10_101632_20261018_v1'
    assert not FINAL.exists() and not FINAL.is_symlink(), 'destination already exists'
    baseline = json.loads((HERE / 'target_preflight_protection.json').read_text())['protected_cube3_and_profiles']
    protect(baseline)
    archive = HERE / 'cube2_epoch10_bundle.tar'
    assert archive.stat().st_size == 578140160 and sha(archive) == ARCHIVE_SHA
    assert sha(HERE / 'verify_checkpoint_cpu.py') == VERIFY_SHA
    stage = HERE / 'verified_payload'
    stage.mkdir(exist_ok=False)
    with tarfile.open(archive, 'r:') as source:
        members = source.getmembers()
        assert all(m.isfile() for m in members) and len(members) == 22
        names = [m.name for m in members]
        assert len(set(names)) == len(names), 'duplicate tar members'
        manifest = json.load(source.extractfile('cube2/origin_manifest.json'))
        assert manifest['cube_count'] == 2 and manifest['source_host'] == '101632'
        expected = {m['member']: m for m in manifest['source_file_manifest']}
        assert len(expected) == 21 and set(names) == set(expected) | {'cube2/origin_manifest.json'}
        for member in members:
            path = (stage / member.name).resolve()
            assert path.is_relative_to(stage.resolve()), 'unsafe member path'
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as src, path.open('xb') as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            if member.name in expected:
                entry = expected[member.name]
                assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], member.name
    audit_argv = [sys.executable, '-B', str(HERE / 'verify_checkpoint_cpu.py'),
                  '--root', str(stage), '--output', str(HERE / 'CPU_checkpoint_integrity.json')]
    write('CPU_actual_argv.json', audit_argv)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (HERE / 'CPU_checkpoint_integrity.stdout.log').open('xb') as log:
        result = subprocess.run(audit_argv, env=env, stdout=log, stderr=subprocess.STDOUT)
    write('CPU_actual_exit.json', {'actual_exit_code': result.returncode, 'finished': time.time()})
    assert result.returncode == 0, 'CPU checkpoint audit failed'
    audit = json.loads((HERE / 'CPU_checkpoint_integrity.json').read_text())
    assert audit['actual_exit_code'] == 0 and all(v['CPU_tensor_integrity_passed'] for v in audit['modes'].values())
    protect(baseline)
    shutil.copyfile(HERE / 'CPU_checkpoint_integrity.json', stage / 'cube2/CPU_checkpoint_integrity.json')
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage / 'cube2'), -100, os.fsencode(FINAL), 1) != 0:
        raise OSError(ctypes.get_errno(), 'RENAME_NOREPLACE failed')
    protect(baseline)
    for member, entry in expected.items():
        path = FINAL / Path(member).relative_to('cube2')
        assert path.is_file() and path.stat().st_size == entry['bytes']
    report.update(actual_exit_code=0, stage='published_and_verified', source_members=21,
                  archive_sha256=ARCHIVE_SHA, archive_bytes=archive.stat().st_size,
                  protected_cube3_and_profiles_exact=len(baseline), CPU_audit_sha256=sha(HERE / 'CPU_checkpoint_integrity.json'),
                  manifest_sha256=sha(FINAL / 'origin_manifest.json'), no_overwrite_publish=True)
except Exception:
    report['error'] = traceback.format_exc()
report['finished'] = time.time()
write('TARGET_MIGRATION_RECEIPT.json', report)
write('target_actual_exit.json', {'actual_exit_code': report['actual_exit_code']})
print(json.dumps(report, ensure_ascii=False))
raise SystemExit(report['actual_exit_code'])
