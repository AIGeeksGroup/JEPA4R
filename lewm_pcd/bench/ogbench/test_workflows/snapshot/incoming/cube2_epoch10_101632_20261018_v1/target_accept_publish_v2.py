"""Adopt the already byte-verified extraction; preserve the v1 audit failure."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
FINAL = ROOT / 'checkpoints/cube2'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    with (HERE / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def protect(baseline):
    for path, expected in baseline.items():
        path = Path(path)
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], str(path)


report = {'started': time.time(), 'actual_exit_code': 1, 'target_final_root': str(FINAL),
          'training_encoding_SR_performed': False, 'profiles_modified': False,
          'reused_already_transferred_and_extracted_payload': True,
          'prior_auxiliary_failure_preserved': 'v1 validator incorrectly treated trainer SIGReg buffers as model weights'}
try:
    assert HERE.parent == ROOT / 'incoming' and HERE.name == 'cube2_epoch10_101632_20261018_v1'
    assert not FINAL.exists() and not FINAL.is_symlink()
    prior = json.loads((HERE / 'TARGET_MIGRATION_RECEIPT.json').read_text())
    assert prior['actual_exit_code'] == 1 and 'CPU checkpoint audit failed' in prior['error']
    baseline = json.loads((HERE / 'target_preflight_protection.json').read_text())['protected_cube3_and_profiles']
    protect(baseline)
    stage = HERE / 'verified_payload'
    manifest = json.loads((stage / 'cube2/origin_manifest.json').read_text())
    assert manifest['cube_count'] == 2 and manifest['source_host'] == '101632'
    assert sha(stage / 'cube2/origin_manifest.json') == '19e545642cdbfbdbe44e2abd1c05677677b2d68178c68cc0fefd935b71ab5b6d'
    for entry in manifest['source_file_manifest']:
        path = (stage / entry['member']).resolve()
        assert path.is_relative_to(stage.resolve()) and path.is_file() and not path.is_symlink()
        assert path.stat().st_size == entry['bytes']
        if not entry['member'].endswith(('.pt', '.ckpt')):
            assert sha(path) == entry['sha256'], str(path)
    verifier = HERE / 'verify_checkpoint_cpu_v2.py'
    assert sha(verifier) == json.loads((HERE / 'v2_helper_freeze.json').read_text())['verifier_sha256']
    argv = [sys.executable, '-B', str(verifier), '--root', str(stage), '--output', str(HERE / 'CPU_checkpoint_integrity_v2.json')]
    write('CPU_v2_actual_argv.json', argv)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (HERE / 'CPU_checkpoint_integrity_v2.stdout.log').open('xb') as log:
        result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
    write('CPU_v2_actual_exit.json', {'actual_exit_code': result.returncode, 'finished': time.time()})
    assert result.returncode == 0, 'CPU v2 audit failed'
    audit = json.loads((HERE / 'CPU_checkpoint_integrity_v2.json').read_text())
    assert audit['actual_exit_code'] == 0 and all(v['CPU_tensor_integrity_passed'] for v in audit['modes'].values())
    protect(baseline)
    destination = stage / 'cube2/CPU_checkpoint_integrity.json'
    assert not destination.exists()
    shutil.copyfile(HERE / 'CPU_checkpoint_integrity_v2.json', destination)
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage / 'cube2'), -100, os.fsencode(FINAL), 1) != 0:
        raise OSError(ctypes.get_errno(), 'RENAME_NOREPLACE failed')
    protect(baseline)
    report.update(actual_exit_code=0, stage='published_and_verified', source_members=21,
                  archive_sha256='0fd39945372a354da5fa42fb8495e4faa90d2806f4f747f5f4204a5b755efc0b',
                  archive_bytes=578140160, protected_cube3_and_profiles_exact=len(baseline),
                  CPU_audit_sha256=sha(HERE / 'CPU_checkpoint_integrity_v2.json'),
                  manifest_sha256=sha(FINAL / 'origin_manifest.json'), no_overwrite_publish=True)
except Exception:
    report['error'] = traceback.format_exc()
report['finished'] = time.time()
write('TARGET_MIGRATION_RECEIPT_v2.json', report)
write('target_v2_actual_exit.json', {'actual_exit_code': report['actual_exit_code']})
print(json.dumps(report, ensure_ascii=False))
raise SystemExit(report['actual_exit_code'])
