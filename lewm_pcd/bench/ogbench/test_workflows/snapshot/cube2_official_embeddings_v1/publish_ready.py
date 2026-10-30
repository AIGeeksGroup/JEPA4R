"""Publish readiness only after the serial driver and both full CPU audits exit0."""
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


if __name__ == '__main__':
    plan = json.loads((HERE/'launch_plan.json').read_text())
    assert (HERE/'runtime/outer_exit_code').read_text().strip() == '0'
    assert json.loads((HERE/'runtime/closure.json').read_text())['actual_exit_code'] == 0
    for phase in plan['phases']:
        record = json.loads((HERE/'runtime'/ (phase['name']+'_exit.json')).read_text())
        assert record['actual_exit_code'] == 0 and not record['residual_owned_members']
    for p, expected in plan['source_pins'].items():
        assert sha(p) == expected, p
    for mode in ('2d','3d'):
        report_path = HERE/mode/'independent_audit/report.json'
        report = json.loads(report_path.read_text())
        assert report['actual_exit_code'] == 0 and report['episodes_verified'] == 1100 and report['raw_rows_verified'] == 1101100
        assert (HERE/mode/'independent_audit/actual_exit_code').read_text().strip() == '0'
        manifest = HERE/mode/'cache/cache_manifest.json'
        out = dict(ready_for_downstream=True,actual_exit_code=0,cube_count=2,data_id='cube-double-play-v0',
                   modality=mode,features=1101100,latent_dim=192,dtype='float32',
                   manifest_sha256=sha(manifest),audit_report=str(report_path),audit_sha256=sha(report_path),
                   serial_driver_outer_exit_code=0,final_wrapper_exit_required=str(HERE/'wrapper_exit_code'),
                   protected_cube3_assets_unchanged=True,training_updates=0,SR_episodes=0,
                   full_vectors_local_backup_claimed=False,completed=time.time())
        with (HERE/mode/'cache/READY.json').open('x') as f:json.dump(out,f,indent=2)
    with (HERE/'BOTH_CACHES_READY.json').open('x') as f:
        json.dump(dict(actual_exit_code=0,cube_count=2,modalities=['2d','3d'],features_per_modality=1101100,completed=time.time()),f,indent=2)
