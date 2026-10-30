"""Reuse the existing owned-stage runner for the frozen PCD producer."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('preflight_v1', 'full_v1'), required=True)
    args = parser.parse_args()
    source = ROOT / 'official_state_2d_cache_v1/relay_cache.py'
    assert digest(source) == json.loads((HERE / args.profile / 'launch_plan.json').read_text())['reused_phase_source_sha256']
    spec = importlib.util.spec_from_file_location('_existing_cache_phase_runner', source)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.HERE, old.RUN = HERE, HERE / args.profile / 'runtime'
    code = 1
    try:
        plan = HERE / 'producer_plan.json'
        python = str(ROOT / 'env_runtime/bin/python')
        if args.profile == 'preflight_v1':
            for stage in ('reference', 'fresh'):
                old.phase(stage, [python, '-B', str(HERE / 'gate.py'), '--plan', str(plan),
                                 '--stage', stage, '--output', str(HERE / 'gate_v1')])
            fresh = json.loads((HERE / 'gate_v1/fresh_report.json').read_text())
            assert fresh['actual_exit_code'] == 0 and fresh['fresh_reload_bitexact'] and fresh['finite']
            old.phase('pilot', [python, '-B', str(HERE / 'produce.py'), '--plan', str(plan), '--execute', '--pilot'])
            pilot = json.loads((HERE / 'cache/pilot_receipt.json').read_text())
            assert pilot['actual_exit_code'] == 0 and pilot['completed_rows'] == 2002
            old.write('closure.json', dict(actual_exit_code=0, plan_sha256=digest(plan),
                pilot_rows=2002, fresh_report_sha256=digest(HERE / 'gate_v1/fresh_report.json'),
                pilot_receipt_sha256=digest(HERE / 'cache/pilot_receipt.json'), full_started=False))
        else:
            preflight = HERE / 'preflight_v1'
            assert (preflight / 'supervisor_actual_exit_code').read_text().strip() == '0'
            before = json.loads((preflight / 'runtime/closure.json').read_text())
            assert before['actual_exit_code'] == 0 and before['plan_sha256'] == digest(plan)
            old.phase('full', [python, '-B', str(HERE / 'produce.py'), '--plan', str(plan), '--execute'])
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
            audit_dir = old.RUN / 'cache_audit_v1'
            old.phase('independent_audit', [python, '-B', str(HERE / 'audit_cache.py'),
                '--cache-dir', str(HERE / 'cache'), '--plan', str(plan), '--output-dir', str(audit_dir)], environment)
            report_path = audit_dir / 'report.json'
            report = json.loads(report_path.read_text())
            assert report['actual_exit_code'] == 0 and report['episodes_verified'] == 3300
            assert report['raw_rows_verified'] == 3303300
            assert (audit_dir / 'actual_exit_code').read_text().strip() == '0'
            old.write('closure.json', dict(actual_exit_code=0, producer_actual_exit_code=0,
                audit_actual_exit_code=0, audit_report_path=str(report_path),
                audit_report_sha256=digest(report_path), plan_sha256=digest(plan),
                manifest_sha256=digest(HERE / 'cache/cache_manifest.json'),
                producer_seal_sha256=digest(HERE / 'cache/cache_seal.json'),
                checkpoint_sha256=json.loads(plan.read_text())['checkpoint_sha256'],
                features=3303300, latent_dim=192, dtype='float32', training_updates=0, SR_episodes=0,
                final_local_backup_claimed=False))
        code = 0
    except BaseException as error:
        old.write('failure.json', dict(error=repr(error), actual_exit_code=1))
        raise
    finally:
        (old.RUN / 'pipeline_actual_exit_code').write_text(str(code) + '\n')
