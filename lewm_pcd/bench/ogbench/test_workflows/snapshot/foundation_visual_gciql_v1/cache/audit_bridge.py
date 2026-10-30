"""Generalize only identities/dimensions/camera checks of the original full CPU audit."""
import argparse
import json
from pathlib import Path
import types
import numpy as np
from common import sha,replace,COUNTS
from producer_bridge import EXTRA

BASE=Path(__file__).parent/'base/audit_cache.py'
BASE_SHA='a97a55ce945c12b16d6fb473c71b289cdf8b2a0164b697ae0f8d4c5d7f9b5006'

def check_camera(actual,planned):
    for k in ('source_name','render_name','FOV','width','height','offsamples','ob_type',
              'visualize_info','pixel_transparent_arm','pullback','crop'):
        assert actual[k]==planned[k],k
    for k in ('source_position','position','actual_quaternion'):
        np.testing.assert_allclose(actual[k],planned[k],atol=1e-12,rtol=0)
    np.testing.assert_allclose(actual['source_quaternion'],planned['base_quaternion'],atol=1e-12,rtol=0)
    assert actual['arm_material_alpha'] and all(x==1. for x in actual['arm_material_alpha'].values())

def source(plan):
    assert sha(BASE)==BASE_SHA
    n,d=plan['cube_count'],plan['latent_dim'];assert n in COUNTS and d in (768,1024)
    t=BASE.read_text()
    edits=[("EXPECTED_EPISODES = {'train': 3000, 'val': 300}",f'EXPECTED_EPISODES = {COUNTS[n]!r}'),
      ("assert policy in ('cpu', 'cuda', 'auto_validated'), 'Unsupported preprocessing policy'", "assert policy == 'shared_gpu', 'Shared encoder GPU preprocessing required'"),
      ("selected_device = 'cpu'","selected_device = 'cuda'"),
      ("if policy != 'cpu':","if policy not in ('cpu', 'shared_gpu'):"),
      ('ROWS, DIM = 1001, 192',f'ROWS, DIM = 1001, {d}'),
      ("PRODUCER = 'official_state_2d_cache_v1/produce.py'","PRODUCER = 'foundation_visual_gciql_v1/cache/producer_bridge.py'"),
      ("ADAPTER = 'official_state_2d_cache_v1/render_adapter.py'","ADAPTER = 'foundation_visual_gciql_v1/cache/render_adapter.py'"),
      ('qpos_width=35, qvel_width=32',f'qpos_width={14+7*n}, qvel_width={14+6*n}'),
      ("extractor_source_sha256=plan['source_pins'][PRODUCER]","extractor_source_sha256=plan['source_arrays_extractor_sha256']"),
      ("plan['source_pins']['source/lewm_pcd/utils.py']","plan['encoder_adapter_sha256']"),
      ("plan['source_pins']['checkpoint_encoder.py']","plan['encoder_adapter_sha256']"),
      ("(('qpos', 35), ('qvel', 32))",f"(('qpos', {14+7*n}), ('qvel', {14+6*n}))"),
      ('nq=35, nv=32, nu=7',f'nq={14+7*n}, nv={14+6*n}, nu=7')]
    for old,new in edits:t=replace(t,old,new)
    return t

def load(plan):
    m=types.ModuleType('_foundation_full_cache_audit');m.__file__=str(Path(__file__).resolve())
    exec(compile(source(plan),str(BASE),'exec'),m.__dict__)
    m.PROTOCOL_KEYS=m.PROTOCOL_KEYS+EXTRA;m.camera_check=check_camera
    original=m.audit
    def audit(cache_dir,plan_path,progress=None):
        report=original(cache_dir,plan_path,progress)
        proof_path=Path(cache_dir)/'batch1_alignment.json'
        proof=json.loads(proof_path.read_text())
        manifest=json.loads((Path(cache_dir)/'cache_manifest.json').read_text())
        assert proof['actual_exit_code']==0 and proof['passed'] is True
        assert proof['encoder_spec_sha256']==plan['encoder_spec_sha256']
        assert proof['frozen_encoder_state_sha256']==manifest['source_encoder_state_sha256']
        assert proof['atol']==proof['rtol']==1e-5
        report['batch1_alignment']=dict(path=str(proof_path),sha256=sha(proof_path))
        return report
    m.audit=audit
    return m

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path)
    a=p.parse_args();plan=json.loads(a.plan.read_text());load(plan).main()
