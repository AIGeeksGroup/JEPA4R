"""Fresh feedback-expert mixed trajectories with the final visible-surface sampler."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from .adroit_data import receipt,sha


def load_pinned_expert(path):
    # Reuse the existing class without importing simulator registrations.
    import importlib.util
    source=Path(__file__).resolve().parents[2]/'envs/adroit/policy.py'
    spec=importlib.util.spec_from_file_location('_adroit_saved_state_expert',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    dependencies=Path(__file__).resolve().parents[3]/'DEPENDENCIES.json'
    expected=json.loads(dependencies.read_text(encoding='utf-8'))['adroit_state_expert']['sha256']
    return module.SavedStateExpert.load(path,expected,env_name='relocate-v0')


def episode_plan(rng,length=200):
    """Original mixed probabilities and branch/noise distributions."""
    kind=int(rng.choice(3,p=[.5,.3,.2]))
    noise=float(rng.uniform(0,.2)) if kind<2 else 0.
    if kind!=1:return kind,noise,-1,0
    start=int(rng.integers(15,length-40))
    span=length if rng.uniform()<.5 else int(rng.integers(10,31))
    return kind,noise,start,span


def mixed_action(policy,env,rng,noise,random_action):
    """Match AdroitFeedbackPolicy: raw expert mean + noise, then one clip."""
    import numpy as np
    if random_action:action=rng.uniform(-1,1,30)
    else:
        result=policy.get_action(np.asarray(env.get_obs(),np.float32))
        action=np.asarray(result[1]['evaluation'],np.float32)
        if action.shape!=(30,) or not np.isfinite(action).all():raise ValueError('invalid expert evaluation mean')
        action=action+rng.normal(0,noise,30)
    return np.clip(action,-1,1).astype(np.float32)


def fingerprint(state):
    import numpy as np
    text=json.dumps(state,sort_keys=True,allow_nan=False,default=lambda value:np.asarray(value).tolist())
    return hashlib.sha256(text.encode()).hexdigest()


def collect_split(root,split,count,seed,policy):
    import h5py
    import numpy as np
    import torch
    from bench.cortexbench.run_adroit_bc3d import build_point_clouds
    from mj_envs.envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0
    paths=[root/(split+'_'+route+'.h5') for route in ('rgb','pcd')]
    env=RelocateEnvV0(); sampler=build_point_clouds(env.sim,'relocate-v0')
    plans=np.random.default_rng(seed); actions_rng=np.random.default_rng(seed+10000)
    files=[];initials=[];counts=[0,0,0]
    try:
        # Establish the GL context before recording the first native reset.
        env.seed(seed);env.reset();sampler.seed(seed);sampler()
        files=[h5py.File(path,'x') for path in paths]
        for f in files:
            f.attrs.update(task='relocate-v0',schema='pre_action_rows_with_outgoing_action',
                camera='vil_camera',msaa=0,points=1200,synthetic_target=False,expert_feedback=True)
            f.create_dataset('ep_len',shape=(0,),maxshape=(None,),dtype='int64')
            f.create_dataset('ep_offset',shape=(0,),maxshape=(None,),dtype='int64')
        seen=set()
        for episode in range(count):
            episode_seed=seed+episode;np.random.seed(episode_seed)
            env.seed(episode_seed);obs=np.asarray(env.reset(),np.float32).copy();sampler.seed(episode_seed)
            initial=env.get_env_state(); digest=fingerprint(initial)
            if digest in seen:raise ValueError('duplicate native initial state')
            seen.add(digest);initials.append(dict(initial,seed=episode_seed,reset_observation=obs.copy(),
                initial_state_sha256=digest.encode(),original_episode_index=episode))
            kind,noise,start,span=episode_plan(plans);counts[kind]+=1
            rows=[]
            for step in range(200):
                before=env.get_env_state();xyz,rgb=sampler()
                pixels=env.sim.render(width=256,height=256,camera_name='vil_camera',device_id=0)[::-1].copy()
                for key,value in before.items():np.testing.assert_array_equal(value,env.get_env_state()[key])
                random_action=kind==2 or (kind==1 and start<=step<start+span)
                with torch.no_grad():action=mixed_action(policy,env,actions_rng,noise,random_action)
                action=np.asarray(action,np.float32).copy()
                if action.shape!=(30,) or not np.isfinite(action).all() or np.max(np.abs(action))>1:
                    raise ValueError('mixed executed action violates normalized 30D contract')
                next_obs,reward,done,info=env.step(action.copy())
                rows.append(dict(action=action,observation=obs.copy(),qpos=np.asarray(before['qpos']).copy(),
                    qvel=np.asarray(before['qvel']).copy(),proprio=obs[:30].copy(),seed=np.int64(episode_seed),
                    step_idx=np.int64(step),initial_state_sha256=digest.encode(),sample_type=np.uint8(kind),
                    rand_action=np.uint8(random_action),noise_std=np.float32(noise),branch_t=np.int32(start),
                    branch_len=np.int32(span),reward=np.float64(reward),goal_achieved=np.uint8(info.get('goal_achieved',False)),
                    next_observation=np.asarray(next_obs,np.float32).copy(),done=np.uint8(done),
                    time_limit=np.uint8(step==199 and not done),target_pos=np.asarray(before['target_pos']).copy(),
                    pixels=pixels,pcd_xyz=np.asarray(xyz,np.float32),pcd_rgb=np.asarray(rgb,np.uint8)))
                obs=np.asarray(next_obs,np.float32).copy()
            for route,f in zip(('rgb','pcd'),files):
                for key in rows[0]:
                    if (route=='rgb' and key.startswith('pcd_')) or (route=='pcd' and key=='pixels'):continue
                    array=np.asarray([row[key] for row in rows])
                    if array.dtype.kind in 'fc' and not np.isfinite(array).all():raise ValueError('nonfinite '+key)
                    if key not in f:
                        f.create_dataset(key,shape=(0,*array.shape[1:]),maxshape=(None,*array.shape[1:]),
                            dtype=array.dtype,chunks=True,compression='lzf' if array.ndim>2 else None)
                    f[key].resize((episode*200+200,*array.shape[1:]));f[key][episode*200:]=array
                for key,value in (('ep_len',200),('ep_offset',episode*200)):
                    f[key].resize((episode+1,));f[key][episode]=value
                f.flush()
            state={'stage':split,'completed':episode+1,'failed':0,'updated_at':time.time(),'actual_exit_code':None}
            temporary=root/'collection-state.tmp';temporary.write_text(json.dumps(state));temporary.replace(root/'collection-state.json')
        initial_path=root/('initial_states_'+split+'.npz')
        np.savez(initial_path,**{key:np.asarray([item[key] for item in initials]) for key in initials[0]})
        nonvisual=root/(split+'_nonvisual.h5')
        with h5py.File(nonvisual,'x') as header:
            for key in files[1]:
                if not key.startswith('pcd_'):header[key]=h5py.ExternalLink(paths[1].name,'/'+key)
            header.attrs.update(task='relocate-v0',schema='pre_action_rows_with_outgoing_action',
                backing=paths[1].name)
        return paths+[initial_path,nonvisual],counts
    finally:
        for f in files:f.close()
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--expert',type=Path,required=True)
    parser.add_argument('--train-episodes',type=int,required=True);parser.add_argument('--test-episodes',type=int,required=True)
    parser.add_argument('--seed',type=int,default=42);args=parser.parse_args()
    if min(args.train_episodes,args.test_episodes)<1:raise ValueError('positive split sizes required')
    if args.train_episodes>=2000000:raise ValueError('train/test seed ranges would overlap')
    args.output.mkdir(parents=True,exist_ok=False)
    expert=load_pinned_expert(args.expert);policy=expert.policy
    artifacts=[];counts={}
    try:
        for split,count,seed in (('train',args.train_episodes,args.seed),('test',args.test_episodes,args.seed+2000000)):
            paths,composition=collect_split(args.output,split,count,seed,policy);artifacts.extend(paths);counts[split]=composition
        receipt(args.output/'collection.json',passed=True,actual_exit_code=0,train_episodes=args.train_episodes,
            test_episodes=args.test_episodes,episode_length=200,mixed_counts=counts,
            expert_sha256=expert.provenance['policy_sha256'],
            producer_sha256=sha(__file__),synthetic_target=False,artifacts=artifacts+[
                args.expert,Path(__file__).resolve().parents[3]/'DEPENDENCIES.json',
                Path(__file__).resolve().parents[2]/'config/adroit_relocate_sampling.json'])
    except Exception as error:
        receipt(args.output/'collection.json',passed=False,actual_exit_code=1,error=repr(error),artifacts=artifacts)
        raise


if __name__=='__main__':main()
