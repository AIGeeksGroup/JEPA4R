"""Final1M state/XYZ evaluation through unchanged official evaluate and physics."""
import argparse
from contextlib import closing
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
PRIORITY=HERE.parent
CASE=PRIORITY.parent
ROOT=CASE.parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def array_sha(value):
    value=np.asarray(value)
    return hashlib.sha256(str((value.shape,value.dtype.str)).encode()+value.tobytes()).hexdigest()


def project_goal(goal,cubes):
    value=np.asarray(goal)
    assert value.shape==(19+9*cubes,) and np.isfinite(value).all()
    return np.concatenate([value[19+9*i:22+9*i] for i in range(cubes)])


class GoalXYZPolicy:
    def __init__(self,agent,cubes):
        self.agent,self.cubes=agent,cubes
        self.observation_hashes=[];self.actor_keys=[];self.goal=None

    def sample_actions(self,observations,goals=None,**kwargs):
        observation=np.asarray(observations);goal=np.asarray(goals)
        assert observation.shape==(19+9*self.cubes,) and np.isfinite(observation).all()
        projected=project_goal(goal,self.cubes)
        if self.goal is None:
            self.goal=goal.copy();self.projected=projected.copy()
            self.observation_dtype=str(observation.dtype)
        else:
            assert np.array_equal(goal,self.goal) and np.array_equal(projected,self.projected)
        self.observation_hashes.append(array_sha(observation))
        self.actor_keys.append(np.asarray(kwargs['seed']).copy())
        # Original restored GCIQL sampler, temperature and clipping remain intact.
        return self.agent.sample_actions(observations=observations,goals=projected,**kwargs)


class SeedReset:
    def __init__(self,env,seed,cubes):
        self.env,self.seed,self.cubes=env,seed,cubes
        self.reset_record=None

    def reset(self,*args,**kwargs):
        assert self.reset_record is None and 'seed' not in kwargs
        observation,info=self.env.reset(*args,seed=self.seed,**kwargs)
        actual_seed=int(self.env.unwrapped.np_random_seed)
        assert actual_seed==self.seed
        goal=np.asarray(info['goal']);xyz=project_goal(goal,self.cubes)
        self.reset_record=dict(seed=self.seed,actual_env_seed=actual_seed,options=kwargs.get('options'),
            current_shape=list(np.asarray(observation).shape),current_sha256=array_sha(observation),
            full_goal=goal.tolist(),full_goal_dtype=str(goal.dtype),full_goal_sha256=array_sha(goal),
            goal_XYZ=xyz.tolist(),goal_XYZ_shape=list(xyz.shape),goal_XYZ_sha256=array_sha(xyz),
            goal_XYZ_offsets=[19+9*i for i in range(self.cubes)],
            goal_XYZ_is_official_scaled10=True,extra_policy_warmup=0,
            action_space_seed_explicitly_changed=False,
            legacy_visual_scene_exact_pairing_claimed=False)
        return observation,info

    def __getattr__(self,name):
        return getattr(self.env,name)


def load_agent(config):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('','-1')
    for path,expected in config['source_pins'].items():
        assert sha(path)==expected,path
    assert sha(PRIORITY/'release_freeze.json')==config['training_source_release_sha256']
    training=json.loads(Path(config['training_config']).read_text())
    assert sha(config['training_config'])==config['training_config_sha256']
    acceptance=json.loads(Path(config['training_acceptance']).read_text())
    assert sha(config['training_acceptance'])==config['training_acceptance_sha256']
    assert acceptance['actual_exit_code']==0 and acceptance['full153_state_exact']
    assert acceptance['network_step']==1000001 and acceptance['Adam_count']==1000000
    assert acceptance['actual_current_dim']==config['current_dim'] and acceptance['actual_goal_dim']==config['goal_dim']
    assert acceptance['synthetic_zero_padding'] is False
    sys.path[:0]=[str(CASE),str(ROOT),str(ROOT/'source/ogbench/impls'),str(ROOT/'source/ogbench')]
    import gciql_boundary as boundary
    assert boundary.jax.default_backend()=='cpu'
    assert boundary.jax.config.jax_threefry_partitionable is False
    assert boundary.jax.config.jax_default_matmul_precision is None and boundary.jax.config.jax_enable_x64 is False
    directory,sidecar=boundary.restore_saved(training,Path(training['output']),1000000)
    assert directory.resolve()==Path(config['final_checkpoint']).resolve()
    assert sha(directory/'manifest.json')==config['final_manifest_sha256']
    assert sha(config['template_batch'])==config['template_batch_sha256']
    with np.load(config['template_batch'],allow_pickle=False) as saved:
        example={key:saved[key][:1].copy() for key in ('observations','actions','actor_goals')}
    assert example['observations'].shape==(1,config['current_dim']) and example['actor_goals'].shape==(1,config['goal_dim'])
    agent=boundary.create_agent(boundary.model.GCIQLAgent,0,example,boundary.agent_config(training))
    agent=boundary.restore_agent(agent,str(directory),1000000)
    fingerprint=boundary.fingerprint(agent)
    assert fingerprint==acceptance['state'] and fingerprint['leaves']==153
    assert int(agent.network.step)==1000001 and int(agent.network.opt_state[0].count)==1000000
    return agent,dict(actual_exit_code=0,checkpoint=str(directory/'params_1000000.pkl'),
        checkpoint_sha256=sha(directory/'params_1000000.pkl'),state=fingerprint,
        current_dim=config['current_dim'],goal_dim=config['goal_dim'],synthetic_zero_padding=False,
        network_step=1000001,Adam_count=1000000,backend='cpu',new_optimizer_updates=0,
        dataset_loaded=False,encoder_calls=0,WM_dynamics_calls=0),training['agent_config']


def episode(config,agent,agent_config,evaluate,ogbench,gym,out,task,index,diagnostic):
    out.mkdir(parents=True,exist_ok=False)
    seed=config['seed_base']+task*100000+index
    code=1
    try:
        random.seed(seed);np.random.seed(seed)
        env=ogbench.make_env_and_datasets(config['dataset_id'],env_only=True)
        with closing(env):
            assert env.spec.max_episode_steps==config['max_episode_steps']
            assert len(env.unwrapped.task_infos)==5
            assert env.unwrapped._num_cubes==config['cube_count']
            assert env.unwrapped._permute_blocks is True and env.unwrapped._use_oracle_rep is False
            wrapped=gym.wrappers.TimeLimit(env,max_episode_steps=4) if diagnostic else env
            seeded=SeedReset(wrapped,seed,config['cube_count'])
            policy=GoalXYZPolicy(agent,config['cube_count'])
            stats,trajs,videos=evaluate(policy,seeded,task_id=task,config=agent_config,
                num_eval_episodes=1,num_video_episodes=0,eval_temperature=0,eval_gaussian=None)
            assert len(trajs)==1 and not videos
            traj=trajs[0];steps=len(traj['action'])
            assert 1<=steps<=(4 if diagnostic else config['max_episode_steps'])
            assert steps==len(policy.observation_hashes)==len(policy.actor_keys)
            actions=np.stack(traj['action']);rewards=np.asarray(traj['reward']);dones=np.asarray(traj['done'],bool)
            successes=np.asarray([row['success'] for row in traj['info']],bool)
            assert actions.shape==(steps,5) and np.isfinite(actions).all() and np.all(np.abs(actions)<=1)
            assert dones[-1] and np.isfinite(rewards).all() and float(stats['success']) in (0.,1.)
            write(out/'reset.json',dict(seeded.reset_record,task_id=task,episode=index,cube_count=config['cube_count']))
            write(out/'policy_inputs.json',dict(current_dim=config['current_dim'],goal_dim=config['goal_dim'],
                current_dtype=policy.observation_dtype,goal_dtype=str(policy.projected.dtype),
                observations_sha256=policy.observation_hashes,goal_XYZ_sha256=array_sha(policy.projected),
                all_inputs_finite=True,synthetic_zero_padding=False,policy_calls=steps,
                perstep_raw_observations_persisted=False,encoder_calls=0))
            with (out/'steps.npz').open('xb') as stream:
                np.savez_compressed(stream,actions=actions,rewards=rewards,dones=dones,
                                    successes=successes,actor_keys=np.stack(policy.actor_keys))
        result=dict(actual_exit_code=0,group_id=config['group_id'],task_id=task,episode=index,seed=seed,
            steps=steps,success=int(stats['success']),policy_calls=steps,diagnostic=diagnostic,
            environment_closed=True,extra_policy_warmup=0,original_evaluate_reused=True,
            files={name:dict(bytes=(out/name).stat().st_size,sha256=sha(out/name))
                   for name in ('reset.json','policy_inputs.json','steps.npz')})
        write(out/'result.json',result);code=0
        return result
    except BaseException as error:
        write(out/'failure.json',dict(actual_exit_code=1,error=repr(error)))
        raise
    finally:
        (out/'attempt_exit_code').write_text(str(code)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--mode',choices=('diagnostic','full'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    out=args.output.resolve();assert out.is_relative_to(HERE) and out!=HERE
    out.mkdir(parents=True,exist_ok=False)
    code=1;started=time.time()
    try:
        write(out/'status.json',dict(stage='loading_final_checkpoint',started=started,completed_episodes=0))
        agent,binding,agent_config=load_agent(config)
        import gymnasium as gym
        import ogbench
        from utils.evaluation import evaluate
        actual_evaluator=Path(inspect.getfile(evaluate)).resolve()
        assert actual_evaluator==(ROOT/'source/ogbench/impls/utils/evaluation.py').resolve()
        assert sha(actual_evaluator)==config['source_pins'][str(actual_evaluator)]
        write(out/'binding.json',dict(binding,config_path=str(args.config),config_sha256=sha(args.config),
            adapter_sha256=sha(__file__),original_evaluate_sha256=config['source_pins'][str(ROOT/'source/ogbench/impls/utils/evaluation.py')]))
        write(out/'protocol.json',dict(config,mode=args.mode,num_video_episodes=0,eval_temperature=0,eval_gaussian=None,
            extra_policy_warmup=0,diagnostic_TimeLimit_steps=4 if args.mode=='diagnostic' else None,
            Box_sampling_limitation='Original reset internally samples an unseeded Box; only original env/reset seed fixed. No legacy visual per-scene exact pairing claim.',
            original_physics_reset_goal_success_controller_unmodified=True))
        results=[]
        tasks=[1] if args.mode=='diagnostic' else [1,2,3,4,5]
        for task in tasks:
            task_results=[]
            for index in range(1 if args.mode=='diagnostic' else 50):
                item=episode(config,agent,agent_config,evaluate,ogbench,gym,
                    out/f'task{task}'/f'episode_{index:03d}',task,index,args.mode=='diagnostic')
                results.append(item);task_results.append(item)
                temp=out/'status.tmp';temp.write_text(json.dumps(dict(stage='evaluating',completed_episodes=len(results),
                    current_task=task,last_episode=index,policy_steps=sum(r['steps'] for r in results),updated=time.time())))
                temp.replace(out/'status.json')
            write(out/f'task{task}'/'summary.json',dict(actual_exit_code=0,task_id=task,episodes=len(task_results),
                successes=sum(r['success'] for r in task_results),policy_steps=sum(r['steps'] for r in task_results)))
        expected=1 if args.mode=='diagnostic' else 250
        assert len(results)==expected and len({r['seed'] for r in results})==expected
        report=dict(actual_exit_code=0,group_id=config['group_id'],cube_count=config['cube_count'],mode=args.mode,
            episodes=expected,successes=sum(r['success'] for r in results),success_rate=sum(r['success'] for r in results)/expected,
            policy_steps=sum(r['steps'] for r in results),unique_seeds=expected,all_attempts_exit0=True,
            diagnostic_excluded_from_SR=args.mode=='diagnostic',new_training_updates=0,
            completed=time.time(),elapsed_seconds=time.time()-started)
        write(out/'report.json',report);code=0
    except BaseException as error:
        write(out/'failure.json',dict(actual_exit_code=1,error=repr(error)));raise
    finally:
        (out/'actual_exit_code').write_text(str(code)+'\n')


if __name__=='__main__':
    main()
