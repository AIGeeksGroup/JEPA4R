"""Combine accepted 2D with newly completed 3D, preserving the performance-gate failure."""
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def accepted_phase(runtime, name):
    record = json.loads((runtime/(name+'_exit.json')).read_text())
    assert record['actual_exit_code'] == 0 and not record['residual_owned_members']


if __name__ == '__main__':
    plan = json.loads((HERE/'control_v3/launch_plan.json').read_text())
    assert (HERE/'runtime_v3/outer_exit_code').read_text().strip() == '0'
    assert json.loads((HERE/'runtime_v3/closure.json').read_text())['actual_exit_code'] == 0
    for name in ('2d_pilot','2d_full','2d_independent_audit','3d_reference_gate','3d_fresh_gate','3d_gpu_preprocessing_gate'):
        accepted_phase(HERE/'runtime_v2',name)
    for phase in plan['phases']:
        accepted_phase(HERE/'runtime_v3',phase['name'])
    for p, expected in plan['source_pins'].items():
        assert sha(p) == expected, p
    original = json.loads((HERE/'3d/GPU_preprocess_report.json').read_text())
    assert original['accepted'] is False and original['prepared_bitexact'] and original['latent_bitexact']
    assert original['candidate_median_seconds'] < original['baseline_median_seconds']
    assert (HERE/'wrapper_v2_exit_code').read_text().strip() == '1'
    for mode in ('2d','3d'):
        report_path = HERE/mode/'independent_audit/report.json'
        report = json.loads(report_path.read_text())
        assert report['actual_exit_code'] == 0 and report['episodes_verified'] == 1100 and report['raw_rows_verified'] == 1101100
        contract_key = 'all_raw_ID_and_true_terminal_next_contracts' if mode == '2d' else 'all_canonical_ID_and_terminal_next_contracts'
        assert report['all_new_latent_SHA_shape_dtype_finite'] and report[contract_key]
        assert (HERE/mode/'independent_audit/actual_exit_code').read_text().strip() == '0'
        out = dict(ready_for_downstream=True,actual_exit_code=0,cube_count=2,data_id='cube-double-play-v0',
            modality=mode,features=1101100,latent_dim=192,dtype='float32',
            manifest_sha256=sha(HERE/mode/'cache/cache_manifest.json'),audit_report=str(report_path),audit_sha256=sha(report_path),
            producer_runtime='runtime_v2' if mode=='2d' else 'runtime_v3',final_driver_outer_exit_code=0,
            original_v2_wrapper_exit_code_preserved=1,original_GPU_performance_gate_accepted=False,
            performance_policy_release=str(HERE/'performance_release_v1/release_decision.json'),
            final_wrapper_exit_required=str(HERE/'wrapper_v3_exit_code'),
            protected_cube3_assets_unchanged=True,training_updates=0,SR_episodes=0,
            full_vectors_local_backup_claimed=False,completed=time.time())
        with (HERE/mode/'cache/READY.json').open('x') as f:json.dump(out,f,indent=2)
    with (HERE/'BOTH_CACHES_READY.json').open('x') as f:
        json.dump(dict(actual_exit_code=0,cube_count=2,modalities=['2d','3d'],features_per_modality=1101100,
            completed=time.time(),original_failed_performance_policy_preserved=True,
            final_wrapper_exit_required=str(HERE/'wrapper_v3_exit_code')),f,indent=2)
