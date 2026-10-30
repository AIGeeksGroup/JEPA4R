"""CPU final SR wrapper around pinned original utils.evaluation.evaluate.

No policy warmup. Original two random goal-initialization steps are observed,
not changed: env seed alone does not guarantee strict cross-modality pairing.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

BASE=Path('/root/autodl-tmp/workspace/users/lewm/ogbench-official-single-play-20261006')
SOURCE=Path('/root/autodl-tmp/workspace/users/lewm/downstream-smoke-20261005-2133/state/official-source')
CPU=SOURCE.parent/'cpu-deps'
TRAIN=BASE/'official_state_seed0_1m_20261006_1315'


def configure_cpu():
    os.environ['JAX_PLATFORMS']='cpu'
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[key]='4'
    for path in [SOURCE/'impls',SOURCE,CPU]:
        sys.path.insert(0,str(path))


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write(path,value):
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)); tmp.replace(path)


def episode_seed(task,index,base=970000000):
    assert task in range(1,6) and index in range(50)
    return base+task*100000+index


def array_record(value):
    import numpy as np
    x=np.asarray(value)
    assert np.isfinite(x).all()
    payload=str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()
    return {'dtype':str(x.dtype),'shape':list(x.shape),'values':x.tolist(),'sha256':hashlib.sha256(payload).hexdigest()}


def physical_snapshot(u):
    fields=['qpos','qvel','ctrl','act','mocap_pos','mocap_quat','qacc_warmstart','eq_active','qfrc_applied','xfrc_applied','userdata','plugin_state']
    result={key:array_record(getattr(u._data,key)) for key in fields if hasattr(u._data,key)}
    result['time']=float(u._data.time)
    private={}
    for key in ['_mode','_ob_type','_success','_reset_next_step','_target_task','_target_block','_target_pos','_target_yaw','_cur_target','_effector_target_pos','_effector_target_yaw','cur_task_id','_num_cubes','_cube_target_mocap_ids','_terminate_at_goal','_success_timing','_render_goal','_permute_blocks','_use_oracle_rep']:
        if hasattr(u,key):
            value=getattr(u,key)
            if isinstance(value,(str,bool,int,float)):
                private[key]=value
            elif value is not None:
                try:
                    private[key]=array_record(value)
                except (TypeError,ValueError):
                    raise AssertionError(f'unhandled private state {key}')
    result['private_state']=private
    if hasattr(u,'cur_task_info'):
        result['task_info']={key:value if isinstance(value,(str,bool,int,float)) else array_record(value) for key,value in u.cur_task_info.items()}
    result['sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return result


def model_snapshot(u):
    names=['cam_pos','cam_quat','cam_fovy','light_pos','light_dir','light_diffuse','light_ambient','geom_rgba','mat_rgba','body_pos','body_quat']
    return {key:array_record(getattr(u._model,key)) for key in names if hasattr(u._model,key)}


def recorder_class():
    import gymnasium as gym
    import numpy as np

    class Recorder(gym.Wrapper):
        def __init__(self,env,seed,attempt,diagnostic_limit=None):
            super().__init__(env)
            self.seed=seed; self.attempt=attempt; self.limit=diagnostic_limit; self.count=0
            self.trace=(attempt/'steps.jsonl').open('x')

        def reset(self,*,seed=None,options=None):
            assert seed is None or seed==self.seed
            u=self.unwrapped
            old_step=u.step; old_observation=u.compute_observation
            internal_actions=[]; observations=[]
            def observe():
                ob=old_observation()
                observations.append((np.array(ob,copy=True),physical_snapshot(u)))
                return ob
            def internal_step(action):
                answer=old_step(action)
                internal_actions.append({'action':array_record(action),'post_state':physical_snapshot(u)})
                return answer
            u.compute_observation=observe; u.step=internal_step
            try:
                observation,info=self.env.reset(seed=self.seed,options=options)
            finally:
                u.step=old_step; u.compute_observation=old_observation
            assert u.np_random_seed==self.seed
            assert len(internal_actions)==2,'official goal-initialization contract changed'
            goal=np.asarray(info['goal'])
            candidates=[snapshot for ob,snapshot in observations if np.array_equal(ob,goal)]
            assert candidates,'physical goal snapshot missing'
            assert np.asarray(observation).shape==goal.shape==(28,)
            receipt={'requested_seed':self.seed,'actual_seed':int(u.np_random_seed),
                'observation':array_record(observation),'goal28':array_record(goal),
                'initial_physical_state':physical_snapshot(u),'goal_physical_state':candidates[-1],
                'official_internal_goal_initialization':internal_actions,'model_colors_camera_lights':model_snapshot(u),
                'rng_state':u.np_random.bit_generator.state,
                'pairing_limitation':'action_space property creates independent unseeded Box; same env seed alone does not guarantee goal/ctrl equality',
                'extra_policy_warmup_steps':0,'environment_max_episode_steps':self.env.spec.max_episode_steps}
            write(self.attempt/'reset.json',receipt)
            return observation,info

        def step(self,action):
            action=np.asarray(action)
            assert action.shape==(5,) and np.isfinite(action).all()
            assert (action>=-1).all() and (action<=1).all()
            ob,reward,terminated,truncated,info=self.env.step(action)
            self.count+=1
            snapshot=physical_snapshot(self.unwrapped)
            assert snapshot['ctrl']['shape']==[7]
            assert np.isfinite(ob).all() and np.isfinite(reward)
            original_flags={'terminated':bool(terminated),'truncated':bool(truncated)}
            if self.limit is not None and self.count>=self.limit and not (terminated or truncated):
                truncated=True; info=dict(info,diagnostic_stop=True)
            row={'step':self.count,'sent_action':array_record(action),'post_state':snapshot,
                 'reward':float(reward),'original_environment_flags':original_flags,
                 'returned_flags':{'terminated':bool(terminated),'truncated':bool(truncated)}}
            self.trace.write(json.dumps(row,allow_nan=False)+'\n'); self.trace.flush()
            return ob,reward,terminated,truncated,info

        def close(self):
            self.trace.close(); super().close()
    return Recorder


def validate_training_flags(flags,algorithm='gcbc'):
    evaluation_flags={'eval_episodes':20,'video_episodes':0} if algorithm=='gcbc' else {'eval_episodes':50,'video_episodes':1}
    for key,value in dict(seed=0,env_name='cube-single-play-v0',train_steps=1000000,**evaluation_flags).items():
        assert flags[key]==value,key
    for key,value in {'batch_size':1024,'lr':0.0003,'const_std':True,'encoder':None,'frame_stack':None,'actor_hidden_dims':[512,512,512]}.items():
        assert flags['agent'][key]==value,key
    assert flags['agent']['agent_name']==algorithm
    if algorithm=='gciql':
        for key,value in {'actor_loss':'ddpgbc','alpha':1.0,'value_hidden_dims':[512,512,512],
                          'layer_norm':True,'discount':0.99,'tau':0.005,'expectile':0.9}.items():
            assert flags['agent'][key]==value,key
        for key,value in {'log_interval':5000,'eval_interval':100000,'save_interval':100000,'eval_on_cpu':1,
                          'eval_tasks':None,'eval_temperature':0,'eval_gaussian':None,'restore_path':None}.items():
            assert flags[key]==value,key
    else:
        assert algorithm=='gcbc'


def require_final(train,algorithm='gcbc'):
    import subprocess
    assert (train/'exit_code').read_text().strip()=='0'
    if algorithm=='gciql': assert (train/'original_main_exit_code').read_text().strip()=='0'
    result=json.loads((train/'result.json').read_text())
    assert result['status']=='training_completed_pending_independent_final_reload_and_SR_review'
    plan=json.loads((train/'plan.json').read_text())
    assert plan['source_commit']=='572788d97c2a6225b8edb7c420e304c04d1b3459'
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==plan['source_commit']
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True).strip()
    for relative,expected in plan['source_sha256'].items():
        assert sha(SOURCE/relative)==expected
    for name,expected in plan['train_dataset'].items():
        assert sha(BASE/'raw'/name)==expected
    checkpoint=Path(result['final_checkpoint'])
    assert checkpoint.resolve().is_relative_to(train.resolve())
    assert checkpoint.name=='params_1000000.pkl' and sha(checkpoint)==result['final_checkpoint_sha256']
    after=json.loads((train/'final_cpu_restore.json').read_text())
    assert (train/'final_cpu_restore_exit_code').read_text().strip()=='0'
    assert after['status']=='passed' and after['checkpoint_sha256']==sha(checkpoint) and after['network_step']==1000001
    flags=json.loads((checkpoint.parent/'flags.json').read_text())
    validate_training_flags(flags,algorithm)
    if algorithm=='gciql':
        assert plan['algorithm']=='gciql'
        assert after['algorithm']=='gciql' and after['adam_count']==1000000 and after['real_state_action_finite']
    return checkpoint,flags,plan


def load_agent(checkpoint,flags,algorithm='gcbc'):
    import jax,numpy as np,flax.serialization,pickle
    if algorithm=='gciql':
        from agents.gciql import GCIQLAgent as Agent,get_config
    else:
        assert algorithm=='gcbc'
        from agents.gcbc import GCBCAgent as Agent,get_config
    from utils.flax_utils import restore_agent
    assert jax.default_backend()=='cpu'
    cfg=get_config().to_dict(); cfg.update(flags['agent']); cfg['actor_hidden_dims']=tuple(cfg['actor_hidden_dims'])
    assert cfg['agent_name']==algorithm
    if algorithm=='gciql': cfg['value_hidden_dims']=tuple(cfg['value_hidden_dims'])
    template=Agent.create(flags['seed'],np.zeros((1,28),np.float32),np.zeros((1,5),np.float32),cfg)
    epoch=int(checkpoint.stem.split('_')[-1])
    agent=restore_agent(template,str(checkpoint.parent),epoch)
    expected=pickle.loads(checkpoint.read_bytes())['agent']
    a,tree=jax.tree_util.tree_flatten(expected); b,other=jax.tree_util.tree_flatten(flax.serialization.to_state_dict(agent))
    assert tree==other
    assert all(np.asarray(x).dtype==np.asarray(y).dtype and np.asarray(x).shape==np.asarray(y).shape and np.isfinite(np.asarray(y)).all() and np.array_equal(x,y) for x,y in zip(a,b))
    assert int(agent.network.step)==epoch+1
    if algorithm=='gciql': assert int(agent.network.opt_state[0].count)==epoch
    return agent,cfg


@contextmanager
def task_lock(path):
    import fcntl
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def run_episode(agent,cfg,task,index,seed,folder,protocol_sha,diagnostic=False,reference_replay=None,env_kwargs=None,replay_module=None):
    import ogbench,numpy as np,inspect,time
    from utils.evaluation import evaluate
    result_file=folder/'result.json'
    folder.mkdir(parents=True,exist_ok=True)
    if result_file.exists():
        previous=json.loads(result_file.read_text())
        assert previous['status']=='passed' and previous['protocol_sha256']==protocol_sha and previous['seed']==seed
        assert sha(folder/previous['attempt']/'steps.jsonl')==previous['steps_sha256']
        assert sha(folder/previous['attempt']/'reset.json')==previous['reset_sha256']
        assert previous['task']==task and previous['index']==index and previous['diagnostic_only']==diagnostic
        if reference_replay is not None:
            assert sha(folder/previous['attempt']/'pairing.json')==previous['pairing_sha256']
        return previous
    attempt=folder/f'attempt_{len(list(folder.glob("attempt_*")))+1:04d}'
    attempt.mkdir(exist_ok=False)
    status={'status':'running','task':task,'index':index,'seed':seed,'pid':os.getpid(),'exit_code':None}
    write(attempt/'status.json',status)
    start=time.monotonic(); wrapped=None
    try:
        env=ogbench.make_env_and_datasets('cube-single-play-v0',env_only=True,**(env_kwargs or {}))
        assert Path(inspect.getfile(evaluate)).resolve()==(SOURCE/'impls/utils/evaluation.py').resolve()
        assert Path(inspect.getfile(type(env.unwrapped))).resolve().is_relative_to(SOURCE.resolve())
        assert type(env.unwrapped).__module__=='ogbench.manipspace.envs.cube_env'
        assert env.spec.id=='cube-single-v0' and env.spec.max_episode_steps==200
        episode_agent=agent
        if reference_replay is not None:
            if replay_module is None:
                from state_replay_adapter import make_replay_environment
            else:
                make_replay_environment=replay_module.make_replay_environment
            try:
                env,episode_agent=make_replay_environment(env,agent,reference_replay,task,index,attempt,BASE)
            except BaseException:
                env.close(); raise
        wrapped=recorder_class()(env,seed,attempt,4 if diagnostic else None)
        np.random.seed(seed)
        stats,trajs,_=evaluate(episode_agent,wrapped,task_id=task,config=cfg,num_eval_episodes=1,
                              num_video_episodes=0,eval_temperature=0,eval_gaussian=None)
        assert wrapped.count<= (4 if diagnostic else 200) and wrapped.count>0
        assert float(stats['success']) in [0.0,1.0]
        wrapped.close(); wrapped=None
        if reference_replay is not None:
            if replay_module is None:
                from state_replay_adapter import annotate_reset
            else:
                annotate_reset=replay_module.annotate_reset
            pairing_sha=annotate_reset(attempt)
        result={'status':'passed','exit_code':0,'protocol_sha256':protocol_sha,'task':task,'index':index,'seed':seed,
                'attempt':attempt.name,'steps':len(trajs[0]['action']),'success':float(stats['success']),
                'seconds':time.monotonic()-start,'diagnostic_only':diagnostic,
                'steps_sha256':sha(attempt/'steps.jsonl'),'reset_sha256':sha(attempt/'reset.json')}
        if reference_replay is not None: result['pairing_sha256']=pairing_sha
        write(attempt/'result.json',result); write(result_file,result)
        status.update(status='passed',exit_code=0)
        return result
    except BaseException:
        status.update(status='failed',exit_code=1,error=traceback.format_exc()); raise
    finally:
        if wrapped is not None: wrapped.close()
        write(attempt/'status.json',status); (attempt/'exit_code').write_text(str(status['exit_code']))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['describe','gate','task','aggregate'],default='describe')
    parser.add_argument('--train-root',default=str(TRAIN))
    parser.add_argument('--output',default=str(BASE/'official_state_final_sr_seed0_v1'))
    parser.add_argument('--task',type=int,choices=range(1,6))
    parser.add_argument('--algorithm',choices=['gcbc','gciql'],default='gcbc')
    parser.add_argument('--reference-replay',type=Path)
    parser.add_argument('--policy-kind',choices=['state','frozen-pcd-gciql','cached-state-xyz','cached-visual-state-xyz'],default='state')
    parser.add_argument('--config',type=Path)
    args=parser.parse_args()
    proposed={'tasks':[1,2,3,4,5],'episodes_per_task':50,'max_environment_steps':200,'seed_base':970000000,
        'seed_formula':'base + task_id*100000 + zero_based_episode_index','action_path':'original evaluate: sample_actions temperature0 and clip[-1,1]',
        'extra_policy_warmup':0,'gate_max_policy_steps':4,'gate_seed':970900000,
        'environment_lifetime':'fresh original CubeEnv per episode; explicit seed supplied by outer reset wrapper',
        'pairing_limitation':'original reset goal uses independently sampled new Box RNG; record internal actions and complete goal/initial state, no reset modification'}
    if args.algorithm=='gciql': proposed['algorithm']='gciql'
    if args.policy_kind=='frozen-pcd-gciql':
        assert args.algorithm=='gciql' and args.config is not None and args.reference_replay is not None
        proposed.update(policy_kind=args.policy_kind,policy_input='real main1200 XYZ/uint8-valued RGB current+goal -> frozen FP32 encoder+projector192 -> original Flax GCIQL',
            action_path='original GCIQL sample_actions internal clip, then original evaluate clip[-1,1]',
            whole_model_visual_exact_claim=False,appearance_exception='sealed target_object_0 alpha0 versus state reference.2; collision-disabled only')
    if args.policy_kind=='cached-state-xyz':
        assert args.algorithm=='gciql' and args.config is not None and args.reference_replay is not None
        proposed.update(policy_kind=args.policy_kind,policy_input='current[online visual192 or zero192,official state28]; goalXYZ3 only',
            action_path='original GCIQL sample_actions internal clip, then original evaluate clip[-1,1]',
            whole_model_visual_exact_claim=False,appearance_exception='sealed target_object_0 alpha0 versus state reference.2; collision-disabled only')
    if args.policy_kind=='cached-visual-state-xyz':
        assert args.algorithm=='gciql' and args.config is not None and args.reference_replay is not None
        proposed.update(policy_kind=args.policy_kind,policy_input='current[online visual192,official state28]; goal[real paired visual192,objectXYZ3]',
            action_path='original GCIQL sample_actions internal clip, then original evaluate clip[-1,1]',
            goal_encoding='one real forward per policy step from fixed raw paired goal; no encoder-output reuse',current_encoding='one real forward per policy step',
            whole_model_visual_exact_claim=False,appearance_exception='sealed target_object_0 alpha0 versus state reference.2; collision-disabled only')
    if args.reference_replay is not None:
        proposed.update(reference_replay=str(args.reference_replay.resolve()),gate_seed=episode_seed(1,0),
            pairing_limitation='explicit replay of existing reference; whole initial/goal/ctrl/mocap/private/RNG/model equality')
        if args.policy_kind in ('frozen-pcd-gciql','cached-state-xyz','cached-visual-state-xyz'):
            proposed['pairing_limitation']='exact reference initial/goal/ctrl/mocap/private/RNG; sealed nonphysical marker-alpha visual exception, all other model components exact'
    if args.stage=='describe': print(json.dumps(proposed,indent=2)); return
    configure_cpu()
    native=None; env_kwargs=None; reference_module=None
    train=Path(args.train_root)
    if args.policy_kind=='frozen-pcd-gciql':
        import pcd_gciql_native as native
        checkpoint,flags,plan=native.bind_completed(args.config,train)
        env_kwargs=plan['environment_kwargs']; reference_module=native
    elif args.policy_kind in ('cached-state-xyz','cached-visual-state-xyz'):
        import cached_state_xyz_native as native
        checkpoint,flags,plan=native.bind_completed(args.config,train)
        env_kwargs=plan['environment_kwargs']; reference_module=native
    else:
        checkpoint,flags,plan=require_final(train,args.algorithm)
    if native is not None and plan['config'].get('goal_mode')=='latent_object_xyz' and plan['config']['visual_mode']=='zero192':
        proposed.update(policy_input='current[zeros192,official state28]; goal[zeros192,objectXYZ3]',
            goal_encoding='zero visual channels; no goal renderer or encoder',current_encoding='zero visual channels; no current renderer or encoder',zero_visual_channels=True)
    reference_replay=None
    if args.reference_replay is not None:
        if native is None:
            from state_replay_adapter import prepare_reference_replay
        else:
            prepare_reference_replay=native.prepare_reference_replay
        reference_replay=(prepare_reference_replay(args.reference_replay,BASE,SOURCE,plan['config'])
                          if args.policy_kind in ('cached-state-xyz','cached-visual-state-xyz') else prepare_reference_replay(args.reference_replay,BASE,SOURCE))
    if native is None:
        protocol=dict(proposed,checkpoint_sha256=sha(checkpoint),flags_sha256=sha(checkpoint.parent/'flags.json'),
                  final_afterhook_sha256=sha(train/'final_cpu_restore.json'),source_commit=plan['source_commit'],
                  source_sha256=plan['source_sha256'],data_sha256=plan['train_dataset'],runner_sha256=sha(__file__))
    else:
        protocol=dict(proposed,checkpoint_sha256=sha(checkpoint),actual_training_binding=plan,
            agent_config=flags['agent'],agent_config_source=str(train/'resolved.json'),
            agent_config_source_sha256=sha(train/'resolved.json'),runner_sha256=sha(__file__),
            native_adapter_sha256=sha(Path(native.__file__)))
    if reference_replay is not None:
        adapter_path=Path(native.__file__) if native is not None else Path(prepare_reference_replay.__globals__['__file__'])
        protocol.update(reference_replay_binding=reference_replay,state_replay_adapter_sha256=sha(adapter_path))
    protocol_sha=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    output=Path(args.output)
    if args.stage=='gate':
        if output.exists():
            assert json.loads((output/'protocol.json').read_text())==protocol
        else:
            output.mkdir(exist_ok=False)
            write(output/'protocol.json',protocol)
            write(output/'binding.json',{'train_root':str(train),'final_checkpoint':str(checkpoint),'protocol_sha256':protocol_sha})
    else:
        assert json.loads((output/'protocol.json').read_text())==protocol
    if args.stage=='aggregate':
        results=[]; seeds=set()
        for task in range(1,6):
            for index in range(50):
                result=json.loads((output/f'task{task}'/f'episode{index:03d}'/'result.json').read_text())
                assert result['status']=='passed' and not result['diagnostic_only'] and result['protocol_sha256']==protocol_sha
                assert result['task']==task and result['index']==index and result['seed']==episode_seed(task,index)
                assert result['seed'] not in seeds; seeds.add(result['seed']); results.append(result)
                episode=output/f'task{task}'/f'episode{index:03d}'/result['attempt']
                assert 1<=result['steps']<=200 and result['success'] in [0.0,1.0]
                assert sha(episode/'reset.json')==result['reset_sha256']
                assert sha(episode/'steps.jsonl')==result['steps_sha256']
                if reference_replay is not None:
                    assert sha(episode/'pairing.json')==result['pairing_sha256']
        write(output/'aggregate.json',{'status':'passed','exit_code':0,'protocol_sha256':protocol_sha,'episodes':250,
            'success_rate':sum(r['success'] for r in results)/250,
            'per_task':{str(t):sum(r['success'] for r in results if r['task']==t)/50 for t in range(1,6)},
            'gate_excluded':True,'single_seed_run_not_eight_seed_paper_aggregate':True})
        return
    agent,cfg=native.load_factory(checkpoint,flags,plan) if native is not None else load_agent(checkpoint,flags,args.algorithm)
    if args.stage=='gate':
        with task_lock(output/'gate.lock'):
            result=run_episode(agent,cfg,1,0,proposed['gate_seed'],output/'gate',protocol_sha,True,reference_replay,env_kwargs,reference_module)
            if not (output/'gate_receipt.json').exists(): write(output/'gate_receipt.json',result)
            else: assert json.loads((output/'gate_receipt.json').read_text())==result
        return
    assert args.task is not None
    gate=json.loads((output/'gate_receipt.json').read_text())
    assert gate['status']=='passed' and gate['protocol_sha256']==protocol_sha and gate['diagnostic_only']
    taskroot=output/f'task{args.task}'; taskroot.mkdir(exist_ok=True)
    with task_lock(taskroot/'task.lock'):
        status={'status':'running','pid':os.getpid(),'task':args.task,'exit_code':None,'completed':0}
        try:
            for index in range(50):
                run_episode(agent,cfg,args.task,index,episode_seed(args.task,index),taskroot/f'episode{index:03d}',protocol_sha,reference_replay=reference_replay,env_kwargs=env_kwargs,replay_module=reference_module)
                status['completed']=index+1; status['updated_at']=datetime.now(timezone.utc).isoformat(); write(taskroot/'status.json',status)
            status.update(status='passed',exit_code=0)
        except BaseException:
            status.update(status='failed',exit_code=1,error=traceback.format_exc()); raise
        finally:
            write(taskroot/'status.json',status); (taskroot/'exit_code').write_text(str(status['exit_code']))


if __name__=='__main__': main()
