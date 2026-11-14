"""Plan-driven paired collection using the original feedback mix and HDF writer.

No simulator, model or sampling algorithm is changed here. External plans select
unique episodes; ``protocol`` delegates unforced 50/30/20 draws to AdroitEnv.
"""
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from jepa4r.envs.twins.rh56._runtime import RUNTIME_HELPERS_SHA256, atomic_json, file_sha256 as sha

def validate_plan(payload):
    rows=payload.get('episodes') if isinstance(payload,dict) else None
    if not isinstance(rows,list) or not rows:raise ValueError('plan episodes must be a nonempty list')
    result=[];seeds=set()
    for row in rows:
        if not isinstance(row,dict):raise ValueError('plan episode must be an object')
        split,seed,coverage=row.get('split'),row.get('seed'),row.get('coverage')
        if split not in ('train','val','test','bc_demo','preview'):raise ValueError('unsupported split')
        if type(seed) is not int or not 0<=seed<2**32:raise ValueError('seed must be a uint32 integer')
        if seed in seeds:raise ValueError('duplicate episode seed in plan')
        if coverage not in ('protocol','expert'):raise ValueError('external coverage must be protocol or expert')
        if split in ('bc_demo', 'preview') and coverage != 'expert':
            raise ValueError('auxiliary demos require expert coverage')
        seeds.add(seed);result.append((split,seed,coverage))
    return result

def plan_config(coverage):
    force=(None if coverage=='protocol' else 2 if coverage=='random' else 1 if coverage in ('tail','burst') else 0)
    return SimpleNamespace(force_type=force,mix=(.5,.3,.2),noise_max=.2,episode_len=200,
                           branch_tail_prob=.5,burst_len=(10,30))

def draw_collection_plan(original_draw,coverage,rng):
    kind,noise,branch_t,branch_len=original_draw(plan_config(coverage),rng)
    if coverage=='expert':noise=0.
    if coverage=='tail':branch_len=200
    if coverage=='burst':branch_len=20
    actual=('expert' if coverage=='expert' else 'backbone' if kind==0 else 'random' if kind==2
            else 'tail' if branch_len==200 else 'burst')
    return kind,noise,branch_t,branch_len,actual

def reject_expert_failure(split,success,allow_failure):
    return split in ('bc_demo','preview') and not success and not allow_failure

def goal_receipt(geometry_flags,native_flags,native_success,overloaded):
    geometry=np.asarray(geometry_flags,dtype=bool);native=np.asarray(native_flags,dtype=bool)
    if geometry.shape!=native.shape:raise ValueError('native and geometry goal arrays differ in length')
    success=bool(geometry.sum()>25 and not overloaded)
    return dict(success=success,native_success=bool(native_success),success_consistent=success==bool(native_success),
                goal_steps=int(geometry.sum()),native_goal_steps=int(native.sum()),
                goal_flag_mismatch_steps=np.flatnonzero(geometry!=native).tolist())

def make_parser():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True);ap.add_argument('--preview',action='store_true')
    ap.add_argument('--plan',type=Path,help='Frozen episode plan; external rows use protocol or expert coverage')
    ap.add_argument('--camera',choices=('native','opposite'),default='native')
    ap.add_argument(
        '--allow-expert-failure',
        action='store_true',
        help='Retain failed auxiliary expert attempts for later audited selection',
    )
    return ap

def main(argv=None):
    ap=make_parser();args=ap.parse_args(argv)
    if args.plan and args.preview:ap.error('--plan and --preview are mutually exclusive')
    plan_payload=json.loads(args.plan.read_text(encoding='utf-8')) if args.plan else None
    plan=(validate_plan(plan_payload) if args.plan else
          [('preview',60005000,'expert')] if args.preview else
          [('train',60000000,'backbone'),('train',60000001,'tail'),('train',60000002,'burst'),
           ('train',60000003,'random'),('val',60001000,'backbone'),('test',60002000,'backbone'),
           ('bc_demo',60004000,'expert')])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    state=dict(phase='collecting',started=time.time(),completed=0,failed_episodes=0,exit_code=None)
    def save():state['updated']=time.time();atomic_json(out/'state.json',state)
    save()
    native=None
    try:
        import h5py
        from . import native as native_module
        from .native import NativeBridge,BASELINE,load_file,physical_hash
        factory=NativeBridge
        camera_module=None
        if args.camera=='opposite':
            from . import camera_preview as camera_module
            factory=camera_module.OppositeCameraBridge
        sys.path.insert(0,str(BASELINE.parent))
        from jepa4r.data.relocate.paired_writer import AdroitPairedWriter
        legacy_env=load_file('_rh56_original_plan',BASELINE/'envs/adroit/env.py')
        legacy_policy=load_file('_rh56_original_feedback',BASELINE/'envs/adroit/policy.py')
        config = dict(
            camera=args.camera,
            episodes=[dict(split=s, seed=seed, coverage=c) for s, seed, c in plan],
            mix=[0.5, 0.3, 0.2],
            noise_max=0.2,
            branch_tail_prob=0.5,
            burst_len=[10, 30],
            episode_len=200,
            dt=0.01,
            allow_expert_failure=args.allow_expert_failure,
            action_alignment='outgoing; last row NaN; NPZ T actions/T+1 states',
            goal_columns='goal_achieved/geometry_goal_achieved use synchronized current geometry; '
                         'native_goal_achieved/native_success preserve original step info',
        )
        config_hash=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        provenance=dict(producer_sha256=sha(__file__),runtime_helpers_sha256=RUNTIME_HELPERS_SHA256,
                        native_adapter_sha256=sha(native_module.__file__),
                        expert_artifacts={name:dict(path=str(path),sha256=sha(path))
                                          for name,path in native_module.expert_files().items()},
                        trusted_native_bundle=getattr(native_module,'LAST_BUNDLE_IDENTITY',None),
                        camera_adapter_sha256=sha(camera_module.__file__) if camera_module else None,
                        baseline_plan_sha256=sha(BASELINE/'envs/adroit/env.py'),
                        baseline_policy_sha256=sha(BASELINE/'envs/adroit/policy.py'),
                        paired_writer_sha256=sha(BASELINE/'data/relocate/paired_writer.py'),
                        plan_sha256=sha(args.plan) if args.plan else None,config_sha256=config_hash)
        atomic_json(out/'config.json',dict(config=config,provenance=provenance))
        manifest=dict(scope='frozen externally planned collection' if args.plan else 'bounded coverage smoke',
                      action_alignment=config['action_alignment'],camera=args.camera,provenance=provenance,
                      episodes=[],auxiliary=[])
        all_seeds=set();all_initials=set()
        with AdroitPairedWriter(out/'paired_rgb.h5',out/'paired_pcd.h5') as writer:
            for split,seed,requested_coverage in plan:
                if native is None:native=factory(seed)
                obs,_=native.reset(seed);initial=native.get_state();initial_hash=physical_hash(initial)
                if seed in all_seeds or initial_hash in all_initials:
                    raise ValueError('seed or physical initial state reused')
                all_seeds.add(seed);all_initials.add(initial_hash)
                rng=np.random.default_rng(np.random.SeedSequence([seed,919]))
                kind,noise,branch_t,branch_len,coverage=draw_collection_plan(
                    legacy_env.AdroitEnv._draw_plan,requested_coverage,rng)
                policy=legacy_policy.AdroitFeedbackPolicy.__new__(legacy_policy.AdroitFeedbackPolicy)
                policy.expert=SimpleNamespace(action_dim=12,action=lambda env:env.expert_action(env.observation()))
                policy.rng=np.random.default_rng(np.random.SeedSequence([seed,313]))
                policy.env = SimpleNamespace(
                    num_envs=1, envs=[SimpleNamespace(unwrapped=SimpleNamespace(native_env=native))]
                )
                frames = []
                observations = []
                props = []
                actions = []
                rewards = []
                traces = []
                states = []
                infos = []
                qpositions = []
                qvelocities = []
                for t in range(201):
                    frame = native.capture()
                    frames.append(frame)
                    observations.append(obs.copy())
                    props.append(native.proprio())
                    for view_kind in ('rgb','pcd'):
                        render_model=native.render_models[view_kind];render_data=native.render_data[view_kind]
                        camera_id=render_model.camera('vil_camera').id
                        if (not np.array_equal(render_data.cam_xpos[camera_id],frame['camera_xpos']) or
                                not np.array_equal(render_data.cam_xmat[camera_id].reshape(3,3),frame['camera_xmat']) or
                                render_model.cam_fovy[camera_id]!=frame['camera_fovy']):
                            raise RuntimeError('RGB/PCD actual camera pose mismatch')
                    states.append(native.get_state())
                    qpositions.append(native.env.data.qpos.copy());qvelocities.append(native.env.data.qvel.copy())
                    if t==200 or (infos and (infos[-1]['terminated'] or infos[-1]['truncated'])):break
                    rand=kind==2 or kind==1 and branch_t<=t<branch_t+branch_len
                    info_for_policy={'rand_action':np.array([[rand]]),'noise_std':np.array([[noise]])}
                    action=policy.get_action(info_for_policy)[0]
                    obs,reward,te,tr,info=native.step(action)
                    actions.append(action.copy());rewards.append(reward)
                    infos.append(dict(info,terminated=bool(te),truncated=bool(tr),rand_action=bool(rand)))
                    if (t+1)%40==0:print(f'{split} seed={seed} step={t+1}',flush=True)
                T=len(actions);F=T+1
                if len(frames)!=F:raise RuntimeError('T+1 frame contract violated')
                for i in range(T):
                    obj,tgt=frames[i+1]['object_xyz'],frames[i+1]['target_xyz']
                    dist=float(np.linalg.norm(obj-tgt));traces.append(np.r_[i,obj,tgt,dist,int(dist<.1)])
                tr=np.asarray(traces);goals=tr[:,8].astype(bool)
                native_goals=np.array([i['goal_achieved'] for i in infos],dtype=bool)
                goal_audit=goal_receipt(goals,native_goals,infos[-1]['success'],native.env.overloaded)
                success=goal_audit['success']
                if int(native_goals.sum())!=int(native.env.n_goal):raise RuntimeError('native goal counter mismatch')
                camera=dict(mode=args.camera,position=frames[0]['camera_xpos'].tolist(),
                            rotation=frames[0]['camera_xmat'].tolist(),fovy=float(frames[0]['camera_fovy']),
                            resolution=[256,256],rgb_msaa=8,pcd_msaa=0)
                if args.camera=='opposite':
                    if (not np.array_equal(frames[0]['camera_xpos'],[0.,1.2,1.2]) or camera['fovy']!=45. or
                            frames[0]['camera_xmat'][1,2]<=0 or frames[0]['camera_xmat'][2,2]<=0):
                        raise RuntimeError('opposite camera contract mismatch')
                if any(not np.array_equal(f['camera_xpos'],frames[0]['camera_xpos']) or
                       not np.array_equal(f['camera_xmat'],frames[0]['camera_xmat']) or
                       f['camera_fovy']!=frames[0]['camera_fovy'] for f in frames):
                    raise RuntimeError('fixed episode camera changed between frames')
                episode_name=f'{split}_{seed}'
                full_initial=json.dumps(initial,sort_keys=True)
                arrays = dict(
                    pixels=np.stack([f['pixels'] for f in frames]),
                    xyz=np.stack([f['pcd_xyz'] for f in frames]),
                    rgb=np.stack([f['pcd_rgb'] for f in frames]),
                    label=np.stack([f['labels'] for f in frames]),
                    observations=np.asarray(observations),
                    proprio=np.asarray(props),
                    actions=np.asarray(actions),
                    rewards=np.asarray(rewards),
                    trace=tr,
                    integration=np.array([s['integration'] for s in states]),
                    qpos=np.asarray(qpositions),
                    qvel=np.asarray(qvelocities),
                    initial_state_json=np.array(full_initial),
                    initial_state_sha256=np.array(initial_hash),
                    frame_stats=np.array(json.dumps([f['frame_stats'] for f in frames])),
                    seed=np.int64(seed),
                    dt=np.float64(0.01),
                    success=np.bool_(success),
                    native_goal_achieved=native_goals,
                    native_success=np.bool_(goal_audit['native_success']),
                    native_success_history=np.array([i['success'] for i in infos], dtype=bool),
                    rand_action_mask=np.array([i['rand_action'] for i in infos], dtype=bool),
                    sample_type=np.int8(kind),
                    noise_std=np.float64(noise),
                    branch_t=np.int32(branch_t),
                    branch_len=np.int32(branch_len),
                    producer_sha256=np.array(provenance['producer_sha256']),
                    config_sha256=np.array(config_hash),
                    camera_json=np.array(json.dumps(camera)),
                    goal_audit_json=np.array(json.dumps(goal_audit)),
                    type=np.array(coverage),
                    camera_xpos=frames[0]['camera_xpos'],
                    camera_xmat=frames[0]['camera_xmat'],
                    camera_fovy=frames[0]['camera_fovy'],
                )
                integration=arrays['integration']
                path=out/f'{episode_name}.npz';np.savez_compressed(path,**arrays)
                with np.load(path,allow_pickle=False) as check:
                    for key in ('observations','proprio','actions','rewards','trace','xyz','qpos','qvel'):
                        if not np.isfinite(check[key]).all():raise ValueError(f'nonfinite {key}')
                    assert check['observations'].shape==(F,56) and check['actions'].shape==(T,12)
                record = dict(
                    seed=seed,
                    split=split,
                    initial_state_sha256=initial_hash,
                    steps=T,
                    rows=F,
                    **goal_audit,
                    final_distance=float(tr[-1, 7]),
                    overload=bool(native.env.overloaded),
                    npz=path.name,
                    npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    sample_type=kind,
                    coverage=coverage,
                    requested_coverage=requested_coverage,
                    noise_std=noise,
                    branch_t=branch_t,
                    branch_len=branch_len,
                    camera=camera,
                    producer_sha256=provenance['producer_sha256'],
                    config_sha256=config_hash,
                )
                if not goal_audit['success_consistent']:
                    manifest.setdefault('rejected_episodes',[]).append(record)
                    atomic_json(out/'manifest.json',manifest)
                    raise RuntimeError(f'independent success mismatch; retained complete diagnostic {path}')
                if split in ('train','val','test'):
                    record['episode_index']=len(manifest['episodes']);manifest['episodes'].append(record)
                    outgoing=np.r_[np.asarray(actions),np.full((1,12),np.nan)].astype(np.float32)
                    episode = dict(
                        pixels=arrays['pixels'],
                        pcd_xyz=arrays['xyz'],
                        pcd_rgb=arrays['rgb'],
                        action=outgoing,
                        observation=arrays['observations'],
                        native_observation=arrays['observations'],
                        proprio=arrays['proprio'],
                        qpos=arrays['qpos'],
                        qvel=arrays['qvel'],
                        episode_seed=np.full(F, seed, np.int64),
                        step_idx=np.arange(F, dtype=np.int32),
                        sample_type=np.full(F, kind, np.int8),
                        noise_std=np.full(F, noise),
                        branch_t=np.full(F, branch_t, np.int32),
                        branch_len=np.full(F, branch_len, np.int32),
                        goal_achieved=np.r_[False, goals],
                        geometry_goal_achieved=np.r_[False, goals],
                        native_goal_achieved=np.r_[False, native_goals],
                        native_success=np.r_[False, [i['success'] for i in infos]],
                        reward=np.r_[0.0, rewards],
                        rand_action=np.r_[[i['rand_action'] for i in infos], False].astype(np.int8),
                        executed_action_valid=np.r_[np.ones(T), 0].astype(np.int8),
                        native_integration=integration,
                        drive_p=np.array([s['p'] for s in states]),
                        drive_v=np.array([s['v'] for s in states]),
                        q_cmd=np.array([s['q_cmd'] for s in states]),
                    )
                    writer.write_episode(episode)
                else:
                    manifest['auxiliary'].append(record)
                    if reject_expert_failure(split,success,args.allow_expert_failure):
                        atomic_json(out/'manifest.json',manifest)
                        raise RuntimeError(f'{split} expert sample failed; retained {path}')
                state['completed']+=1;state['failed_episodes']+=not success;save()
                atomic_json(out/'manifest.json',manifest)
                print(json.dumps(record),flush=True)
        assert len(all_seeds)==len(plan) and len(all_initials)==len(plan)
        if manifest['episodes']:
            with h5py.File(out/'paired_rgb.h5') as a,h5py.File(out/'paired_pcd.h5') as b:
                keys=set(a)-{'pixels'}
                assert keys==set(b)-{'pcd_xyz','pcd_rgb'}
                for key in keys:assert np.array_equal(a[key][:],b[key][:],equal_nan=True),key
                for rec,off,ln in zip(manifest['episodes'],a['ep_offset'][:],a['ep_len'][:]):
                    assert ln==rec['rows'] and np.all(a['episode_seed'][off:off+ln]==rec['seed'])
        atomic_json(
            out / 'seed_audit.json',
            dict(
                unique_seeds=sorted(all_seeds),
                unique_initial_hashes=sorted(all_initials),
                pairwise_disjoint=True,
                record_count=len(plan),
                source='actual initial states; seed labels excluded from hash',
            ),
        )
        state.update(phase='complete',exit_code=0);save()
    except BaseException:
        state.update(phase='failed',exit_code=1,error=traceback.format_exc());save();raise
    finally:
        if native is not None:native.close()

if __name__=='__main__':main()
