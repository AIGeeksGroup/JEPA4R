"""Continue the immutable cache through the selected resource/GPU-input overlay."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
H = HERE.parent
ROOT = H.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    spec = importlib.util.spec_from_file_location('_existing_cache_phase', ROOT / 'official_state_2d_cache_v1/relay_cache.py')
    phase = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase)
    phase.RUN = HERE / 'resume_v1/runtime'
    code = 1
    try:
        stopped = json.loads((HERE / 'controlled_stop_v1/receipt.json').read_text())
        assert stopped['actual_stop_helper_exit_code'] == 0 and stopped['preserved_episode_receipts'] == 199
        assert stopped['uncommitted_paths'] == stopped['owned_residual_members'] == []
        plan = H / 'producer_plan.json'
        profile = HERE / 'chosen_profile_v1.json'
        runtime = HERE / 'resume_v1/profile_runtime'
        py = str(ROOT / 'env_runtime/bin/python')
        phase.phase('full', [py, '-B', str(HERE / 'v4/run_profile.py'), '--profile', str(profile),
            '--runtime-dir', str(runtime), '--plan', str(plan), '--execute'])
        audit = phase.RUN / 'audit_v1'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
        phase.phase('independent_audit', [py, '-B', str(HERE / 'audit_profile_v2.py'),
            '--cache-dir', str(H / 'cache'), '--plan', str(plan), '--runtime-dir', str(runtime),
            '--stop-dir', str(HERE / 'controlled_stop_v1'), '--output-dir', str(audit)], env)
        feature_path = audit / 'features/report.json'
        runtime_path = audit / 'runtime_report.json'
        feature = json.loads(feature_path.read_text())
        execution = json.loads(runtime_path.read_text())
        assert feature['actual_exit_code'] == execution['actual_exit_code'] == 0
        assert feature['episodes_verified'] == 3300 and feature['raw_rows_verified'] == 3303300
        for path in (audit / 'actual_exit_code', audit / 'runtime_audit_actual_exit_code',
                     audit / 'features/actual_exit_code'):
            assert path.read_text().strip() == '0'
        phase.write('closure.json', dict(actual_exit_code=0, producer_actual_exit_code=0,
            audit_actual_exit_code=0, features=3303300, latent_dim=192, dtype='float32',
            checkpoint_sha256=json.loads(plan.read_text())['checkpoint_sha256'],
            base_plan_sha256=sha(plan), actual_profile_sha256=sha(profile),
            feature_report_path=str(feature_path), feature_report_sha256=sha(feature_path),
            runtime_report_path=str(runtime_path), runtime_report_sha256=sha(runtime_path),
            manifest_sha256=sha(H / 'cache/cache_manifest.json'),
            producer_seal_sha256=sha(H / 'cache/cache_seal.json'),
            runtime_profiles_manifest_sha256=sha(H / 'cache/runtime_profiles_manifest.json'),
            legacy_episodes_preserved=199, actual_renderer_workers=execution['actual_renderer_workers'],
            actual_prefetch_jobs=execution['actual_prefetch_jobs'], preprocessing_device='cuda',
            same_saved_PCD_preprocess_and_latent_bitexact=True,
            cross_rerender_color_bitexact_guaranteed=False,
            original_native_rendering_protocol_unchanged=True,
            old_controlled_stop_exit_codes_preserved=True, training_updates=0, SR_episodes=0,
            final_local_backup_claimed=False))
        code = 0
    except BaseException as error:
        phase.write('failure.json', dict(actual_exit_code=1, error=repr(error)))
        raise
    finally:
        (phase.RUN / 'pipeline_actual_exit_code').write_text(str(code) + '\n')
