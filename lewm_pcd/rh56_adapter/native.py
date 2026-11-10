"""Native RH56 interface. Policy/physics and legacy point sampling remain unchanged."""
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
import mujoco

ROOT=Path(__file__).resolve().parents[1]
EXPERT=(Path(os.environ['RH56_EXPERT_ROOT']).expanduser().resolve()
        if os.environ.get('RH56_EXPERT_ROOT') else None)
BASELINE=Path(os.environ.get('RH56_BASELINE_ROOT',ROOT)).expanduser().resolve()

def expert_files():
    """Require the external native bundle rather than silently selecting another body."""
    if EXPERT is None:
        raise ValueError('provide RH56_EXPERT_ROOT for the native state56/action12 expert bundle')
    paths={name:EXPERT/name for name in ('env.py','ppo.py','experts/relocate_expert.pt')}
    for path in paths.values():
        if not path.is_file():raise FileNotFoundError(f'missing RH56 expert artifact: {path}')
    return paths

def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def physical_hash(state):
    # Exclude seed labels; hash actual randomised scene, physics, and drive state.
    keys=('integration','body_pos','body_quat','site_pos','site_quat','p','v','q_cmd')
    return hashlib.sha256(json.dumps({k:state[k] for k in keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()

class NativeBridge:
    def __init__(self,seed,render=True):
        if mujoco.__version__!='3.12.0':raise RuntimeError('frozen expert requires MuJoCo 3.12.0')
        expert_files()
        sys.path.insert(0,str(EXPERT))
        bundle=importlib.import_module('env')
        if Path(bundle.__file__).resolve()!= (EXPERT/'env.py').resolve():raise RuntimeError('wrong expert env module')
        policy=importlib.import_module('ppo')
        if Path(policy.__file__).resolve()!= (EXPERT/'ppo.py').resolve():raise RuntimeError('wrong expert policy module')
        Agent=policy.Agent
        self.env=bundle.AdroitRM75Env('relocate')
        self.spec_state=mujoco.mjtState.mjSTATE_INTEGRATION
        self.render_enabled=render
        self.renderers={};self.render_models={};self.render_data={}
        self.sampler_module=load_file('_rh56_original_adroit_sampler',BASELINE/'envs/adroit/pcd.py')
        self.geometry=load_file('_rh56_original_depth_projection',BASELINE/'bench/cortexbench/surface_geometry.py')
        self.sampler=self.sampler_module.AdroitPointClouds.__new__(self.sampler_module.AdroitPointClouds)
        s=self.sampler
        s.task='relocate';s.relocate_target_as_object=True;s.separate_object_instances=True
        s.empty_robot_background_fill=False;s.green_object_cap=300;s.num_points=1200
        s.voxel_size=.005;s.obj_fraction=.25;s.crop=np.array([[-.6,-.8,-.25],[.6,.6,1.1]])
        torch.set_num_threads(1)
        allowed=[np.core.multiarray._reconstruct,np.ndarray,np.dtype,type(np.dtype('float64')),
                 type(np.dtype('float32')),np.core.multiarray.scalar]
        with torch.serialization.safe_globals(allowed):
            ck=torch.load(EXPERT/'experts/relocate_expert.pt',map_location='cpu',weights_only=True)
        self.actor=Agent(56,12);self.actor.load_state_dict(ck['agent']);self.actor.eval()
        self.mean,self.var=ck['obs_mean'],ck['obs_var']
        assert all(torch.isfinite(t).all() for t in self.actor.state_dict().values())
        self.reset(seed)

    def reset(self,seed=None):
        if seed is None:seed=self.seed+1
        mujoco.mj_resetData(self.env.model,self.env.data)
        self.seed=int(seed);obs,info=self.env.reset(seed=self.seed)
        return obs,info

    def observation(self):return self.env._obs()

    def proprio(self):
        e=self.env
        return np.r_[e.data.qpos[e.aq],e.data.qpos[e.hq]].astype(np.float32)

    def expert_action(self,obs):
        x=np.clip((obs-self.mean)/np.sqrt(self.var+1e-8),-10,10)
        with torch.no_grad():return self.actor.actor(torch.as_tensor(x,dtype=torch.float32)[None])[0].numpy()

    def step(self,action):
        a=np.asarray(action)
        if a.shape!=(12,) or not np.isfinite(a).all():raise ValueError('finite action12 required')
        return self.env.step(a)

    def get_state(self):
        e=self.env;m,d=e.model,e.data
        state=np.empty(mujoco.mj_stateSize(m,self.spec_state));mujoco.mj_getState(m,d,state,self.spec_state)
        return dict(seed=self.seed,integration=state.tolist(),p=e.p.tolist(),v=e.v.tolist(),q_cmd=e.q_cmd.tolist(),
                    t=e.t,n_goal=e.n_goal,overloads=e.overloads.tolist(),overloaded=bool(e.overloaded),
                    body_pos=m.body_pos.tolist(),body_quat=m.body_quat.tolist(),site_pos=m.site_pos.tolist(),
                    site_quat=m.site_quat.tolist(),rng=e.np_random.bit_generator.state,
                    cached={k:getattr(d,k).tolist() for k in ('xpos','xmat','site_xpos','site_xmat','actuator_force')})

    def set_state(self,state):
        if set(state)=={'seed'}:self.reset(int(state['seed']));return
        e=self.env;m,d=e.model,e.data;self.seed=int(state['seed'])
        for k in ('body_pos','body_quat','site_pos','site_quat'):getattr(m,k)[:]=state[k]
        mujoco.mj_setState(m,d,np.asarray(state['integration']),self.spec_state)
        mujoco.mj_forward(m,d)
        # Keep solver warmstart/control fields and original observed caches from the real snapshot.
        mujoco.mj_setState(m,d,np.asarray(state['integration']),self.spec_state)
        for k,val in state['cached'].items():getattr(d,k)[:]=val
        e.p=np.array(state['p']);e.v=np.array(state['v']);e.q_cmd=np.array(state['q_cmd'])
        e.t=int(state['t']);e.n_goal=int(state['n_goal']);e.overloads[:]=state['overloads']
        e.overloaded=bool(state['overloaded']);e.np_random.bit_generator.state=state['rng']

    def _render_state(self,kind):
        e=self.env
        if kind not in self.renderers:
            m=copy.copy(e.model)
            m.vis.quality.offsamples=8 if kind=='rgb' else 0
            self.render_models[kind]=m;self.render_data[kind]=mujoco.MjData(m)
            self.renderers[kind]=mujoco.Renderer(m,256,256)
        m,d=self.render_models[kind],self.render_data[kind]
        for k in ('body_pos','body_quat','site_pos','site_quat'):getattr(m,k)[:]=getattr(e.model,k)
        integration=np.empty(mujoco.mj_stateSize(e.model,self.spec_state))
        mujoco.mj_getState(e.model,e.data,integration,self.spec_state)
        mujoco.mj_setState(m,d,integration,self.spec_state);mujoco.mj_forward(m,d)
        renderer=self.renderers[kind];renderer.update_scene(d,camera='vil_camera')
        return renderer,m,d

    def capture(self,modality='paired'):
        if modality not in ('paired','rgb','pcd'):raise ValueError(modality)
        result={}
        if modality!='pcd':
            r,_,_=self._render_state('rgb');result['pixels']=r.render().copy()
        if modality=='rgb':return result
        r,m,d=self._render_state('pcd')
        # Modern Renderer depth and segmentation both use the same ID surface pass.
        colors=r.render().copy()
        r.enable_depth_rendering();depth=r.render().copy();r.disable_depth_rendering()
        r.enable_segmentation_rendering();seg=r.render().copy();r.disable_segmentation_rendering()
        cid=m.camera('vil_camera').id
        xyz=self.geometry.depth_to_world(depth,d.cam_xpos[cid],d.cam_xmat[cid].reshape(3,3),m.cam_fovy[cid])
        ids,kinds=seg[:,:,0],seg[:,:,1]
        geom=kinds==int(mujoco.mjtObj.mjOBJ_GEOM)
        safe=np.clip(ids,0,m.ngeom-1)
        robot_body=np.array([(m.body(i).name or '').startswith('rm75/') for i in range(m.nbody)])
        robot=geom & robot_body[m.geom_bodyid[safe]]
        obj=geom & (m.geom_bodyid[safe]==self.env.obj_bid)
        target=(kinds==int(mujoco.mjtObj.mjOBJ_SITE)) & (ids==self.env.target_sid)
        valid=(robot|obj|target)&np.isfinite(depth)&(depth>0)&np.isfinite(xyz).all(axis=-1)
        labels=np.where(target,2,np.where(obj,1,0)).astype(np.int8)
        pixels=np.flatnonzero(valid.reshape(-1))
        s=self.sampler;s.rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.env.t,731]))
        s.last_info={}
        points,cols=s._sample_pen(xyz.reshape(-1,3)[pixels],colors.reshape(-1,3)[pixels],
                                 labels.reshape(-1)[pixels],source_indices=pixels)
        p=points.astype(np.float64)-d.cam_xpos[cid]
        p=p @ d.cam_xmat[cid].reshape(3,3)
        focal=128/np.tan(np.deg2rad(m.cam_fovy[cid])/2)
        uv=np.c_[127.5+p[:,0]/(-p[:,2])*focal,127.5-p[:,1]/(-p[:,2])*focal]
        expected=np.c_[s.last_source_indices%256,s.last_source_indices//256]
        error=float(np.max(np.abs(uv-expected)))
        if error>1e-3:raise RuntimeError(f'point reprojection mismatch {error}')
        stats=dict(s.last_info,robot_visible_pixels=int(robot.sum()),object_visible_pixels=int(obj.sum()),
                   target_visible_pixels=int(target.sum()),max_reprojection_error_px=error,
                   rgb_msaa=8,pcd_msaa=0,depth_units='metres',camera='vil_camera')
        result.update(pcd_xyz=points,pcd_rgb=cols,labels=s.last_labels.copy(),frame_stats=stats,
                      camera_xpos=d.cam_xpos[cid].copy(),camera_xmat=d.cam_xmat[cid].reshape(3,3).copy(),
                      camera_fovy=float(m.cam_fovy[cid]),object_xyz=d.xpos[self.env.obj_bid].copy(),
                      target_xyz=d.site_xpos[self.env.target_sid].copy())
        return result

    def close(self):
        for r in self.renderers.values():r.close()
        self.renderers.clear()
