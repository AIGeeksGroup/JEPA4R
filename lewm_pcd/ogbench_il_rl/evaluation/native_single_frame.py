"""Single-frame mean+official-clip native interface; CPU preparation by default.

Actual goal rendering/rollouts require separate root-scheduled gates. Importing
this module constructs neither an environment nor a renderer/model.
"""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np

ACTION_DOMAIN='official_npz_normalized_box_identity'
IDENTITY_PROTOCOL={'action_domain':ACTION_DOMAIN,'action_normalization':'identity','policy_history':1,'policy_warmup_steps':0,
                   'evaluation':{'temperature':0,'clip':[-1,1],'gaussian_noise':None}}

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def array_record(value):
    x=np.asarray(value)
    if not np.isfinite(x).all():raise ValueError('Nonfinite physical/action array')
    return {'dtype':str(x.dtype),'shape':list(x.shape),'values':x.tolist(),'sha256':hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()}

def assert_identity_protocol(protocol):
    if any(protocol.get(k)!=v for k,v in IDENTITY_PROTOCOL.items()):raise ValueError('Only explicit single-frame, no-warmup, raw normalized-Box identity action protocol is supported')

def default_snapshot(u):
    from .official_state_sr import physical_snapshot
    return physical_snapshot(u)

def default_model_snapshot(u):
    from .official_state_sr import model_snapshot
    return model_snapshot(u)

class SingleFrameAgent:
    """Original evaluate.sample_actions API: one real image every step, no cache."""
    def __init__(self,policy,current_rgb,preprocess_rgb,device='cpu',precision='fp32',on_action=None,state_fusion=False):
        self.policy=policy.eval().requires_grad_(False);self.current_rgb=current_rgb;self.preprocess=preprocess_rgb
        self.state_fusion=bool(state_fusion)
        self.device,self.precision,self.on_action=device,precision,on_action;self.policy_warmup_steps=0;self.reset()
    def reset(self):self.goal=None;self.model_calls=0;self.last_mean=None
    def set_goal_rgb(self,image):
        if np.asarray(image).shape!=(224,224,3) or np.asarray(image).dtype!=np.uint8:raise ValueError('Actual224 AA4 goal image required')
        self.goal=self.preprocess(np.array(image,copy=True)).detach().clone()
    def sample_actions(self,observations,goals,temperature=0,seed=None):
        import torch
        if temperature!=0 or self.goal is None:raise ValueError('Deterministic mean and true rendered goal required')
        rgb=np.asarray(self.current_rgb())
        if rgb.shape!=(224,224,3) or rgb.dtype!=np.uint8:raise ValueError('Actual current224 AA4 RGB required')
        current=self.preprocess(rgb.copy())
        if current.shape!=(1,3,224,224) or self.goal.shape!=(1,3,224,224) or not torch.isfinite(current).all() or not torch.isfinite(self.goal).all():raise ValueError('Single-frame current/goal input contract differs')
        inputs={'pixels':current.to(self.device),'goal':self.goal.to(self.device)}
        if self.state_fusion:
            for key,value in [('current_state',observations),('goal_state',goals)]:
                state=torch.as_tensor(value,dtype=torch.float32,device=self.device)
                if state.shape==(28,):state=state[None]
                if state.shape!=(1,28) or not torch.isfinite(state).all():raise ValueError('Real paired current/goal observation28 required')
                inputs[key]=state
        with torch.inference_mode(),torch.autocast(device_type=self.device,dtype=torch.bfloat16,enabled=self.precision=='bf16'):
            mean=self.policy.get_action(inputs).float().cpu()
        if mean.shape!=(1,5) or not torch.isfinite(mean).all():raise ValueError('Original single-frame head returned invalid mean5')
        # Identity domain: no inverse scaler and no clipping here. The pinned
        # official evaluate clips exactly once before calling env.step.
        result=mean.numpy()[0].astype(np.float32);self.last_mean=result.copy();self.model_calls+=1
        if self.on_action:self.on_action({'model_call':self.model_calls,'action_domain':ACTION_DOMAIN,'unclipped_mean':result.tolist(),'warmup':False})
        return result

class ReplayPairWrapper:
    """Replay only the two archived internal reset actions; original steps stay original."""
    def __init__(self,env,reference,actor,goal_renderer,*,snapshot_fn=default_snapshot,model_snapshot_fn=default_model_snapshot,on_reset=None,on_step=None):
        self.env,self.reference,self.actor,self.goal_renderer=env,reference,actor,goal_renderer
        self.snapshot,self.model_snapshot=snapshot_fn,model_snapshot_fn;self.on_reset,self.on_step=on_reset,on_step
        self.count=0;self.policy_warmup_steps=0;self.last_reset=None
    def __getattr__(self,key):return getattr(self.env,key)
    def reset(self,*,seed=None,options=None):
        ref=self.reference;actual_seed=int(ref['requested_seed']);options=dict(options or {})
        if seed is not None and seed!=actual_seed:raise ValueError('Reference seed changed')
        if options.get('render_goal',False):raise ValueError('Do not change original state baseline render_goal=False reset')
        if self.env.spec.max_episode_steps!=200 or ref['extra_policy_warmup_steps']!=0 or ref['environment_max_episode_steps']!=200:raise ValueError('Official200-step/no-warmup reference differs')
        replay=ref['official_internal_goal_initialization']
        if len(replay)!=2:raise ValueError('Exactly two actual original Box reset actions required')
        self.actor.reset();self.count=0;u=self.env.unwrapped;old_step,old_observe=u.step,u.compute_observation
        calls=[];observed=[]
        def observe():
            ob=old_observe();observed.append((np.array(ob,copy=True),self.snapshot(u)));return ob
        def reset_step(proposed):
            index=len(calls)
            if index>=2:raise ValueError('Original reset performed additional internal step')
            record=replay[index]['action'];action=np.asarray(record['values'],dtype=np.dtype(record['dtype']))
            if action.shape!=(5,) or not np.isfinite(action).all() or np.any(np.abs(action)>1):raise ValueError('Archived normalizedBox action5 invalid')
            answer=old_step(action);calls.append({'index':index,'proposed_independent_Box_action':array_record(proposed),'actual_replayed_action':array_record(action),'post_state':self.snapshot(u)});return answer
        u.step,u.compute_observation=reset_step,observe
        try:ob,info=self.env.reset(seed=actual_seed,options=options)
        finally:u.step,u.compute_observation=old_step,old_observe
        if len(calls)!=2 or u.np_random_seed!=actual_seed:raise ValueError('Original Box replay/reset seed/count differs')
        goal=np.asarray(info['goal']);candidates=[s for x,s in observed if np.array_equal(x,goal)]
        if not candidates:raise ValueError('True original goal physical state was not captured')
        current=self.snapshot(u);goal_state=candidates[-1];model=self.model_snapshot(u)
        for actual,expected,label in [(array_record(ob),ref['observation'],'initial_observation28'),(array_record(goal),ref['goal28'],'goal28'),(current,ref['initial_physical_state'],'initial physical integration/private'),(goal_state,ref['goal_physical_state'],'true goal physical integration/private'),(model,ref['model_colors_camera_lights'],'camera/material/light/body')]:
            if actual!=expected:raise ValueError('Strict original state pairing failed: '+label)
        if u.np_random.bit_generator.state!=ref['rng_state']:raise ValueError('Original environment RNG state differs')
        before=(self.snapshot(u),self.model_snapshot(u),json.loads(json.dumps(u.np_random.bit_generator.state)))
        image=self.goal_renderer(u,goal_state)
        after=(self.snapshot(u),self.model_snapshot(u),u.np_random.bit_generator.state)
        if before!=after:raise ValueError('Rendering goal mutated current physics/private/RNG/model state')
        self.actor.set_goal_rgb(image)
        self.last_reset={'reference_seed':actual_seed,'reference_sha256':digest(ref),'strict_initial_and_goal_match':True,'reset_Box_replay':calls,
          'initial_physical_sha256':digest(current),'goal_physical_sha256':digest(goal_state),'goal_rgb_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
          'policy_history':1,'policy_warmup_steps':0,'max_policy_steps':200,'goal_render_current_state_unchanged':True}
        if self.on_reset:self.on_reset(self.last_reset)
        return ob,info
    def step(self,action):
        action=np.asarray(action)
        if self.count>=200:raise ValueError('Policy step201 forbidden')
        if action.shape!=(5,) or not np.isfinite(action).all() or np.any(np.abs(action)>1):raise ValueError('Pinned official evaluate must send clipped normalizedBox5')
        answer=self.env.step(action);self.count+=1;state=self.snapshot(self.env.unwrapped)
        if self.count==200 and not (answer[2] or answer[3]):raise ValueError('Original TimeLimit failed200-step contract')
        if self.on_step:self.on_step({'step':self.count,'sent_action':array_record(action),'post_state':state,'warmup':False,'model_calls':self.actor.model_calls})
        return answer
    def close(self):return self.env.close()

class SafeGoalRenderer:
    """Separate MjData and AA4 renderer; never swaps the current data pointer."""
    def __init__(self,*,data_factory=None,forward=None,renderer_factory=None,snapshot_fn=default_snapshot,model_snapshot_fn=default_model_snapshot):
        self.data_factory,self.forward,self.renderer_factory=data_factory,forward,renderer_factory;self.snapshot,self.model_snapshot=snapshot_fn,model_snapshot_fn
    def __call__(self,u,goal_state):
        import mujoco
        if int(u._model.vis.quality.offsamples)!=4:raise ValueError('Goal must use original AA4 model')
        before=(self.snapshot(u),self.model_snapshot(u));current_data=u._data
        data=(self.data_factory or mujoco.MjData)(u._model)
        if data is current_data:raise ValueError('Goal data must be independent from current physics')
        required=['qpos','qvel','ctrl','act','mocap_pos','mocap_quat','qacc_warmstart','eq_active','qfrc_applied','xfrc_applied','userdata','plugin_state']
        for key in required:
            if hasattr(data,key):
                if key not in goal_state:raise ValueError('Missing original goal integration field:'+key)
                record=goal_state[key];value=np.asarray(record['values'],dtype=np.dtype(record['dtype']))
                if value.shape!=getattr(data,key).shape:raise ValueError('Goal integration field shape differs:'+key)
                getattr(data,key)[:]=value
        data.time=float(goal_state['time']);(self.forward or mujoco.mj_forward)(u._model,data)
        for key in ['qpos','qvel','ctrl','act','mocap_pos','mocap_quat','eq_active']:
            if key in goal_state and array_record(getattr(data,key))!=goal_state[key]:raise ValueError('True goal pose/control changed by forward:'+key)
        renderer=(self.renderer_factory or mujoco.Renderer)(u._model,height=224,width=224)
        try:renderer.update_scene(data,camera='front_pixels');image=renderer.render().copy()
        finally:renderer.close()
        if image.shape!=(224,224,3) or image.dtype!=np.uint8 or u._data is not current_data or before!=(self.snapshot(u),self.model_snapshot(u)):raise ValueError('Goal image/nonmutation/model contract failed')
        return image

def validate_references(folder):
    folder=Path(folder);records=[];seeds=set()
    for task in range(1,6):
        for index in range(50):
            path=folder/f'task{task}'/f'episode{index:03d}'/'reset.json';ref=json.loads(path.read_text());seed=970000000+task*100000+index
            if ref['requested_seed']!=seed or ref['actual_seed']!=seed or seed in seeds or len(ref['official_internal_goal_initialization'])!=2 or ref['extra_policy_warmup_steps']!=0 or ref['environment_max_episode_steps']!=200:raise ValueError('Original250 state reference manifest differs')
            for phase in ['initial_physical_state','goal_physical_state']:
                for key,size in [('qpos',21),('qvel',20),('ctrl',7)]:
                    if ref[phase][key]['shape']!=[size]:raise ValueError('Original state snapshot shape differs')
            seeds.add(seed);records.append({'task':task,'index':index,'seed':seed,'reset_path':str(path.resolve()),'reset_file_sha256':file_sha(path)})
    return records

def seal_original_source(source,baseline_protocol):
    source=Path(source).resolve();expected='572788d97c2a6225b8edb7c420e304c04d1b3459'
    if baseline_protocol.get('source_commit')!=expected or subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=expected:
        raise ValueError('Different original official source commit')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():raise ValueError('Pinned original source checkout has modifications')
    files={**baseline_protocol['source_sha256']}
    for relative,expected_sha in files.items():
        if file_sha(source/relative)!=expected_sha:raise ValueError('Original baseline source file changed:'+relative)
    for relative in ['ogbench/manipspace/envs/cube_env.py','ogbench/manipspace/envs/manipspace_env.py','ogbench/manipspace/envs/env.py']:
        if (source/relative).is_file():files[relative]=file_sha(source/relative)
    return {'source_root':str(source),'source_commit':expected,'source_file_sha256':files,'original_evaluate_sha256':files['impls/utils/evaluation.py']}

def main():
    p=argparse.ArgumentParser(description='CPU reference/protocol preparation only; never performs native250 evaluation')
    p.add_argument('--references',type=Path,required=True);p.add_argument('--official-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    state={'status':'running','stage':'validate_original250_references','exit_code':None,'GPU_used':False,'simulator_constructed':False};(a.output/'status.json').write_text(json.dumps(state,indent=2))
    try:
        records=validate_references(a.references);baseline_protocol=json.loads((a.references/'protocol.json').read_text());source=seal_original_source(a.official_source,baseline_protocol)
        protocol={**IDENTITY_PROTOCOL,'source_env':'pinned original official Cube states/single/task opaque224AA4/front_pixels','original_source_seal':source,'baseline_protocol_sha256':file_sha(a.references/'protocol.json'),'baseline_data_sha256':baseline_protocol['data_sha256'],'reset_pairing':'replay recorded two internal Box actions; strict initial/goal/controls/mocap/private/RNG',
          'goal_render':'independent MjData from true recorded goal integration snapshot; no current physics swaps or steps','policy_input':'one current RGB and one real goal RGB; Frozen192+goal192 MLP mean5; no history warmup',
          'native_loop':'original official impls/utils/evaluation.py mean+clip[-1,1]','required_next_gates':'new policy/checkpoint gate -> actual reset/goal-render gate -> short native rollout -> only after2D10ep best/last reload, full250SR',
          'adapter_sha256':file_sha(__file__),'reference_manifest':records,'model_checkpoint_binding':'pending actual full2D10epochs best/last + independent strict reload; no2batch checkpoint substitutes',
          'metrics_output':'none in CPU prepare; no new SR/physical rollout claimed'}
        (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2));state.update(status='prepared_CPU_only',stage='complete',exit_code=0,reference_count=250,unique_seeds=250,protocol_sha256=digest(protocol),actual_native_gates_verified=False)
    except Exception:
        import traceback
        state.update(status='failed',exit_code=1,error=traceback.format_exc())
    finally:(a.output/'status.json').write_text(json.dumps(state,indent=2));(a.output/'exit_code').write_text(str(state['exit_code'])+'\n')
    return state['exit_code']

if __name__=='__main__':raise SystemExit(main())
