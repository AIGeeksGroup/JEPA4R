"""Shared repaired Native evaluator with foundation encoder and exact cache camera."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['MUJOCO_GL']='egl'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
from prepare import CASE, ROOT, sha

NATIVE_SHA='e40a2ff52bd0915b7953d9011c65e02c75f9e0b2d0e454b613d18c275f6c13f0'


def install_core(cfg,config_path,*,tiny=False,gate_smoke=False):
    from runtime import install
    runtime,boundary=install(cfg)
    path=ROOT/'cube23_distance_gciql_v1/native_sr_visual_v1/visual_sr.py'
    assert sha(path)==NATIVE_SHA
    source=path.read_text()
    old='{2: "cube-double-play-v0", 3: "cube-triple-play-v0"}'
    new='{1:"cube-single-play-v0",2:"cube-double-play-v0",3:"cube-triple-play-v0",4:"cube-quadruple-play-v0"}'
    assert source.count(old)==1
    source=source.replace(old,new)
    core=types.ModuleType('_foundation_repaired_native');core.__file__=str(path)
    sys.modules[core.__name__]=core
    exec(compile(source,str(path),'exec'),core.__dict__)
    group=cfg['group_id'];d=cfg['input_contract']['latent_dim'];n=cfg['cube_count']
    core.ROOT=ROOT
    home=Path(cfg['execution_root'])
    if cfg['purpose']=='pilot':assert tiny and gate_smoke
    accept=home/'acceptance'/f'{group}.json'
    final=Path(cfg['output'])/'checkpoints/update_001000000'
    core.GROUPS={group:dict(cube_count=n,state_dim=19+9*n,current_dim=d+19,goal_dim=d+3*n,
        max_episode_steps=4 if tiny else cfg['native_max_episode_steps'],mode='rgb',
        training=Path(config_path),training_sha256=sha(config_path),acceptance=accept,
        gate=(home/'gates'/group/'next_batch.npz') if gate_smoke else Path(cfg['output'])/'final_val_probe.npz',final=final,
        manifest_sha256=sha(final/'manifest.json') if final.exists() else None)}
    def validate(value,label):
        value=np.asarray(value)
        if value.shape!=(d,) or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError(f'{label}: expected finite FP32[{d}]')
        return value
    core.validate_latent=validate
    encoder=cfg['encoder_binding']
    class OnlineEncoder:
        def __init__(self,_group):
            adapter=Path(encoder['adapter']['path']);assert sha(adapter)==encoder['adapter']['sha256']
            specpath=Path(encoder['spec_path']);assert sha(specpath)==encoder['spec_sha256']
            spec=importlib.util.spec_from_file_location('_foundation_encoder_adapter',adapter)
            self.adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.adapter)
            self.model=self.adapter.load_encoder(specpath,device='cuda');self.calls=0
        def encode(self,value):
            if value.shape!=(224,224,3) or value.dtype!=np.uint8:raise ValueError('Expected uint8 HWC224')
            import torch
            with torch.inference_mode():
                out=self.adapter.encode_uint8(self.model,torch.from_numpy(np.ascontiguousarray(value[None])))
            self.calls+=1
            return validate(out[0].detach().cpu().numpy(),'online')
    core.OnlineEncoder=OnlineEncoder
    if gate_smoke:
        assert tiny
        def load_gate(_group):
            gate=home/'gates'/group
            expected=json.loads((gate/'report.json').read_text())
            reload_report=json.loads((home/'gates'/f'{group}_reload.json').read_text())
            assert expected['actual_exit_code']==reload_report['actual_exit_code']==0
            directory,_=boundary.restore_saved(cfg,gate,2)
            with np.load(gate/'next_batch.npz',allow_pickle=False) as saved:
                example={key:saved[key][:1].copy() for key in ('observations','actions','actor_goals')}
            agent=boundary.create_agent(boundary.model.GCIQLAgent,0,example,boundary.agent_config(cfg))
            agent=boundary.restore_agent(agent,str(directory),2)
            assert boundary.jax.default_backend()=='cpu' and boundary.fingerprint(agent)==expected['saved_state']==reload_report['state']
            return agent
        core.load_agent=load_gate
    camera=cfg['camera']
    def configure_camera(base,_group):
        import mujoco
        model=base._model
        target=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,'front_pixels')
        assert target>=0
        model.cam_pos[target]=camera['position']
        model.cam_quat[target]=camera['actual_quaternion']
        model.cam_fovy[target]=camera.get('FOV',camera.get('fov',45.))
        model.vis.quality.offsamples=4
        return dict(camera,encoder_spec_sha256=encoder['spec_sha256'])
    core.configure_camera=configure_camera
    helper=ROOT/'assets/imports/single_cube_rl3_101632_20261018_v1/source_layout/coordination/cached_lewm_state_gciql_xyz_109_20261012_v1/native_v1/locked_dependencies/native_runtime_helpers.py'
    assert sha(helper)=='a83bcf181b00d57fe5b41ef4e61c42dded0561b1ec9d4dc2c7a1c64dfdb8bc3a'
    helper_spec=importlib.util.spec_from_file_location('_foundation_native_physics',helper)
    module=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(module)
    import torch
    torch.set_num_threads(1)
    module.configure_numpy_physics_runtime()
    return core


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--mode',choices=('tiny','full'),required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate-smoke',action='store_true')
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    from audit_native import source_binding
    binding=source_binding(cfg,a.config)
    core=install_core(cfg,a.config,tiny=a.mode=='tiny',gate_smoke=a.gate_smoke)
    core.main(cfg['group_id'],'diagnostic' if a.mode=='tiny' else 'full',a.output)
    assert source_binding(cfg,a.config)==binding
    binding.update(mode=a.mode,gate_smoke=a.gate_smoke,diagnostic_excluded_from_formal_SR=a.mode=='tiny')
    with (a.output/'native_source_binding.json').open('x') as f:json.dump(binding,f,indent=2)


if __name__=='__main__':main()
