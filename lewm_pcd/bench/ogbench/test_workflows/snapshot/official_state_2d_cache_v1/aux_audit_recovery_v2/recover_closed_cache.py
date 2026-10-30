"""Audit the already-closed producer once; preserve all original failed exits."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
H = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    os.nice(10)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16][-2:])
    code = 1
    runtime = HERE / 'runtime'
    runtime.mkdir(exist_ok=False)
    try:
        freeze = json.loads((HERE / 'freeze.json').read_text())
        for relative, digest in freeze['pins'].items():
            assert sha(H / relative) == digest, relative
        producer = json.loads((H / 'runtime/full_exit.json').read_text())
        assert producer['actual_exit_code'] == 0 and producer['residual_members'] == []
        old_exit = (H / 'supervisor_actual_exit_code').read_text().strip()
        assert old_exit == '1'
        failure = json.loads((H / 'runtime/cache_audit_v1/failure.json').read_text())
        assert failure['error'] == "AssertionError('Unapproved numeric/timing thresholds')"
        assert not (H / 'cache/READY.json').exists()
        argv = [sys.executable, '-B', str(HERE / 'audit_gate_metadata.py'),
                '--cache-dir', str(H / 'cache'), '--plan', str(H / 'producer_plan.json'),
                '--output-dir', str(runtime / 'audit')]
        save(runtime / 'actual_argv.json', dict(argv=argv, started=time.time()))
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (runtime / 'audit.log').open('x') as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=environment)
            actual = child.wait()
        save(runtime / 'audit_exit.json', dict(actual_exit_code=actual, completed=time.time()))
        assert actual == 0
        report_path = runtime / 'audit/report.json'
        report = json.loads(report_path.read_text())
        assert report['actual_exit_code'] == 0 and report['episodes_verified'] == 3300
        assert report['raw_rows_verified'] == 3303300
        assert (runtime / 'audit/actual_exit_code').read_text().strip() == '0'
        assert (H / 'supervisor_actual_exit_code').read_text().strip() == old_exit
        ready = dict(ready_for_downstream=True, producer_actual_exit_code=0,
                     producer_owned_residual_members=[], audit_actual_exit_code=actual,
                     original_supervisor_actual_exit_code=1, original_failure_preserved=True,
                     recovery_source_freeze_sha256=sha(HERE / 'freeze.json'),
                     audit_report_path=str(report_path), audit_report_sha256=sha(report_path),
                     manifest_sha256=report['manifest_sha256'],
                     producer_seal_sha256=report['producer_seal_sha256'],
                     producer_plan_sha256=sha(H / 'producer_plan.json'),
                     checkpoint_sha256=json.loads((H / 'producer_plan.json').read_text())['checkpoint_sha256'],
                     features=3303300, latent_dim=192, dtype='float32', training_updates=0,
                     SR_episodes=0, completed=time.time(),
                     outer_exit_file_required=str(HERE / 'recovery_actual_exit_code'),
                     final_local_backup_claimed=False)
        save(runtime / 'closure.json', ready)
        save(H / 'cache/READY.json', ready)
        code = 0
    except BaseException as error:
        save(runtime / 'failure.json', dict(error=repr(error), actual_exit_code=1, at=time.time()))
        raise
    finally:
        (HERE / 'recovery_actual_exit_code').write_text(str(code) + '\n')
