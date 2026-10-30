"""CPU metadata factory: current camera inventory and family-specific immutable plans."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import re
import numpy as np
from common import ROOT,HERE,NAMES,COUNTS,sha,digest,save

def camera(n):
    # Match existing Cube1 M5's unmodified front_pixels; Cube2/3 original front+5;
    # Cube4 approved training_camera_v2 front+5 with 0.10m optical pullback.
    sys.path.insert(0,str(ROOT/'source/ogbench'))
    import gymnasium as gym
    import mujoco
    import ogbench
    env=gym.make(f'cube-{NAMES[n]}-v0',ob_type='states',render_mode='rgb_array',
                 width=224,height=224,visualize_info=False,pixel_transparent_arm=False)
    try:
        env.reset(seed=1732001);m=env.unwrapped._model
        assert (m.nq,m.nv,m.nu)==(14+7*n,14+6*n,7)
        name='front_pixels' if n==1 else 'front'
        c=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,name)
        pos=m.cam_pos[c].copy();base=m.cam_quat[c].copy();actual=base.copy()
        angle=0. if n==1 else 5.
        if angle:
            rot=np.empty(4);mujoco.mju_axisAngle2Quat(rot,np.array([1.,0.,0.]),np.deg2rad(angle))
            mujoco.mju_mulQuat(actual,base,rot)
        pullback=.10 if n==4 else 0.
        target=pos.copy()
        if pullback:
            mat=np.empty(9);mujoco.mju_quat2Mat(mat,actual);target+=pullback*mat.reshape(3,3)[:,2]
        assert float(m.cam_fovy[c])==45.
        if n in (2,3):np.testing.assert_allclose(target,[1.287,0.,.509],atol=1e-12,rtol=0)
        if n==4:
            prior=json.loads((ROOT/'cube4_official_embeddings_v1/training_camera_v2/workspace/official_state_2d_cache_v1/producer_plan.json').read_text())['camera']
            np.testing.assert_allclose(target,prior['position'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(actual,prior['actual_quaternion'],atol=1e-12,rtol=0)
        return dict(cube_count=n,env_id=f'cube-{NAMES[n]}-v0',source_name=name,render_name='front_pixels',
            source_position=pos.tolist(),position=target.tolist(),base_quaternion=base.tolist(),actual_quaternion=actual.tolist(),
            FOV=45.,width=224,height=224,offsamples=4,aa_samples=4,local_x_up_degrees=angle,pullback=pullback,
            ob_type='states',visualize_info=False,pixel_transparent_arm=False,crop=None,
            provenance='Cube1 unchanged pinned official front_pixels (prior M5 RGB recipe); Cube2/3 approved front+5; Cube4 training_camera_v2')
    finally:env.close()

def namespace(family,spec,group_prefix='',pilot_only=False):
    assert re.fullmatch(r'[A-Za-z0-9_-]*',group_prefix) is not None,'Unsafe group prefix'
    random=spec.get('initialization',spec.get('init'))=='random'
    diagnostic=spec.get('diagnostic_only',False)
    if random or diagnostic:
        assert random and diagnostic is True,'Random diagnostics require both init=random and diagnostic_only=true'
        assert group_prefix and 'random' in group_prefix.lower(),'Random diagnostics require an explicit random prefix'
        assert pilot_only,'Random diagnostic plans must be pilot-only'
    else:
        assert not group_prefix and not pilot_only,'Official default namespace must remain unchanged'
    assert spec['family']==family
    return group_prefix+family,random

def prepare(family,spec_path=None,group_prefix='',pilot_only=False):
    d={'visreg_vitb768':768,'vjepa21_vitl1024':1024}[family]
    spec=(Path(spec_path) if spec_path else HERE.parent/'encoder/specs'/f'{family}.json').resolve()
    assert spec.is_file(),'Encoder owner must finalize real asset spec first'
    s=json.loads(spec.read_text());print(json.dumps(dict(encoder_spec=s)),flush=True)
    group_namespace,random=namespace(family,s,group_prefix,pilot_only)
    checkpoint=Path(s.get('checkpoint',s.get('checkpoint_path','')))
    assert checkpoint.is_file()
    checkpoint_sha=s['checkpoint_sha256'];assert sha(checkpoint)==checkpoint_sha
    adapter=HERE.parent/'encoder/encoder_adapter.py'
    source_pins={}
    for path in [HERE/'producer_bridge.py',HERE/'render_adapter.py',HERE/'audit_bridge.py',HERE/'common.py',
                 HERE/'prepare_data.py',HERE/'prepare_plans.py',HERE/'deliver.py',
                 HERE/'base/produce.py',HERE/'base/render_adapter.py',HERE/'base/audit_cache.py',adapter,spec]:
        source_pins[str(path.relative_to(ROOT))]=sha(path)
    # Pin renderer source chain, not unrelated train/model modules.
    for rel in ('source/ogbench/ogbench/manipspace/envs/env.py','source/ogbench/ogbench/manipspace/envs/cube_env.py',
                'source/ogbench/ogbench/manipspace/envs/manipspace_env.py'):
        source_pins[rel]=sha(ROOT/rel)
    for n in COUNTS:
        inventory=json.loads((HERE/'data'/f'cube{n}'/'inventory.json').read_text())
        arrays=HERE/'data'/f'cube{n}'/'source_arrays.json'
        record=json.loads(arrays.read_text())
        group=HERE/'groups'/f'{group_namespace}_cube{n}'
        plan=dict(workspace=str(ROOT),root_workspace=str(ROOT),cube_count=n,latent_dim=d,encoder_family=family,
            shared_state_root=str(arrays.parent),source_arrays_receipt_sha256=sha(arrays),
            source_arrays_extractor_sha256=record['contract']['extractor_source_sha256'],
            checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
            encoder_spec=str(spec),encoder_spec_sha256=sha(spec),encoder_adapter=str(adapter),encoder_adapter_sha256=sha(adapter),
            data_acceptance=str(arrays.parent/'data_acceptance.json'),data_acceptance_sha256=sha(arrays.parent/'data_acceptance.json'),
            source_stat_freeze=str(arrays.parent/'stat-freeze.json'),source_stat_freeze_sha256=sha(arrays.parent/'stat-freeze.json'),
            source_pins=source_pins,camera=camera(n),renderer_workers=2,encoder_batch=32,torch_cpu_threads=1,
            split_episodes=COUNTS[n],raw_rows_per_episode=1001,output_dtype='float32',cache_dir=str(group/'cache'),
            pixel_preprocess_device='shared_gpu',preprocess_gate={},
            encoder_preprocessing_device='cuda',host_tensor_dtype='uint8',
            pixel_preprocess_policy='cpu raw uint8 transport; shared encoder API owns GPU preprocessing',
            training_updates=0,SR_episodes=0)
        if random:plan.update(diagnostic_only=True,pilot_only=True,init='random')
        save(group/'plan.json',plan)
        save(group/'data_binding.json',inventory['data_binding'])
        save(group/'camera.json',plan['camera'])
    available=shutil.disk_usage(ROOT).free
    episodes=8 if random else sum(sum(x.values()) for x in COUNTS.values())
    total=episodes*1001*d*4
    budget=dict(family=family,feature_payload_bytes=total,feature_NPY_header_bytes=episodes*128,
        Temp_free_bytes=available,reserve_weights_checkpoints_logs_bytes=8<<30,
        enough=available>=total+episodes*128+(8<<30),scope='four pilot-only 2EP groups' if random else 'before full feature generation; existing downloaded weights already consume free disk')
    assert budget['enough'],budget
    save(HERE/f'space_budget_{group_namespace}.json',budget)
    print(json.dumps(budget))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['vjepa21_vitl1024','visreg_vitb768'],default='vjepa21_vitl1024')
    p.add_argument('--spec',type=Path);p.add_argument('--group-prefix',default='');p.add_argument('--pilot-only',action='store_true')
    p.add_argument('--camera-only',action='store_true');a=p.parse_args()
    if a.camera_only:
        for n in COUNTS:save(HERE/'data'/f'cube{n}'/'camera.json',camera(n))
    else:prepare(a.family,a.spec,a.group_prefix,a.pilot_only)
