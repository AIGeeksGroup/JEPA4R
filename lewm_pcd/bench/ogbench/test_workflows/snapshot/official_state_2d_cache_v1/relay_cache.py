"""One pilot -> same-contract full resume -> independent new-feature audit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import prepare_runtime as guard

RUN = HERE / 'runtime'


def write(name, value):
    path = RUN / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase(name, argv, environment=None):
    write('pipeline_status.json', dict(stage=name, started=time.time(), actual_exit_code=None))
    with (RUN / (name + '.log')).open('x') as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                 env=environment)  # Inherit our owned PGID for the resource guard.
        write(name + '_argv.json', dict(argv=argv, pid=child.pid, started=time.time()))
        code = child.wait()
    residual = []
    for _ in range(20):
        residual = [p for p in guard.members(os.getpgrp()) if p['pid'] != os.getpid()]
        if not residual:
            break
        time.sleep(.25)
    write(name + '_exit.json', dict(actual_exit_code=code, residual_members=residual,
                                   completed=time.time()))
    assert code == 0 and not residual, f'{name}: exit={code}, residual={residual}'


def main():
    plan_path = HERE / 'producer_plan.json'
    plan = json.loads(plan_path.read_text())
    cache = Path(plan['cache_dir'])
    python = str(ROOT / 'env_runtime/bin/python')
    common = [python, '-B', str(HERE / 'run_producer_bounded.py'), '--plan', str(plan_path), '--execute']
    phase('pilot', common + ['--pilot'])
    pilot = json.loads((cache / 'pilot_receipt.json').read_text())
    assert pilot['actual_exit_code'] == 0 and pilot['not_full_cache']
    assert pilot['completed_episodes'] == 2 and pilot['completed_rows'] == 2002
    phase('full', common)
    manifest, seal = cache / 'cache_manifest.json', cache / 'cache_seal.json'
    assert json.loads(seal.read_text())['manifest_sha256'] == sha(manifest)
    audit_out = RUN / 'cache_audit_v1'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
    phase('independent_audit', [python, '-B', str(HERE / 'audit_cache.py'),
        '--cache-dir', str(cache), '--plan', str(plan_path), '--output-dir', str(audit_out)], env)
    report_path = audit_out / 'report.json'
    report = json.loads(report_path.read_text())
    assert report['actual_exit_code'] == 0 and report['episodes_verified'] == 3300
    assert report['raw_rows_verified'] == 3303300
    assert (audit_out / 'actual_exit_code').read_text().strip() == '0'
    closure = dict(cache_dir=str(cache), producer_actual_exit_code=0, audit_actual_exit_code=0,
        manifest_sha256=sha(manifest), producer_seal_sha256=sha(seal),
        audit_report_path=str(report_path), audit_report_sha256=sha(report_path),
        producer_plan_sha256=sha(plan_path), features=3303300, latent_dim=192,
        dtype='float32', checkpoint_sha256=plan['checkpoint_sha256'],
        training_updates=0, SR_episodes=0, vectors_storage='remote approved Temp',
        final_local_backup_claimed=False, completed=time.time())
    write('pipeline_closure.json', closure)
    write('pipeline_status.json', dict(stage='producer_and_audit_completed',
        actual_exit_code=0, completed=time.time(), ready_requires_outer_exit=True))


if __name__ == '__main__':
    code = 1
    try:
        main()
        code = 0
    except BaseException as exc:
        write('pipeline_failure.json', dict(error=repr(exc), actual_exit_code=1, at=time.time()))
        raise
    finally:
        (RUN / 'pipeline_actual_exit_code').write_text(str(code) + '\n')
