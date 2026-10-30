"""CPU metadata auditor. No model/renderer/simulator imports or rawlatent invention."""
import argparse,ast,datetime,hashlib,json,os,shlex,traceback
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import numpy as np
W=Path('/root/autodl-tmp/workspace/users/lewm');B=W/'ogbench-official-single-play-20261006'
UTIL=B/'official_lewm_gcbc_20261006/audit_native_committed.py'
UTIL_SHA='0e7756be72e850fbba2cdd110662b4ab584cf39e9b8f5e8e5d1292651dc4caeb'
PCD=B/'official_lewm_gcbc_20261006/native_sr_v1/audit_final_pcd_sr.py'
PCD_SHA='b397fe6943ee66c1948a64e9e38fd27c0e6ba19af16faf6bbd4e524a8083dc4f'
N=W/'coordination/rl3_m0_m5_m6_visual_proprio_smoke_109_20261013_v1/formal_start_v1/native1m_v1'
FREEZE_SHA='d8f4adfa41aaeb002198dc1f134503e27d2033f0250c847baee2ac9cb91283dc'

def load_functions(path,expected,names,scope):
    payload=path.read_bytes();assert hashlib.sha256(payload).hexdigest()==expected,str(path)
    nodes={n.name:n for n in ast.parse(payload).body if isinstance(n,ast.FunctionDef)}
    exec(compile(ast.Module(body=[nodes[k] for k in names],type_ignores=[]),str(path),'exec'),scope)
    return {k:hashlib.sha256(ast.dump(nodes[k],include_attributes=False).encode()).hexdigest() for k in names}

def validate_goal_source(source,goal,reference,config,seed,ns):
    check,raw_hash,frame=ns['check'],ns['raw_hash'],ns['frame']
    check(source['goal_XYZ']==goal[0].tolist() and source['goal_XYZ_sha256']==hashlib.sha256(goal.tobytes()).hexdigest(),'Same true referencegoal28 XYZ3')
    row=source['goal_frame'];physical=reference['goal_physical_state']
    if config['modality']=='pcd':
        check(source['raw_goal_shape']==[1200,6] and source['raw_goal_dtype']=='float32' and source['raw_goal_sha256']==row['cloud_sha256'],'Rawgoal PCD source shape/dtype/hash')
        initial=ns['digest'](np.random.default_rng(seed).bit_generator.state)
        frame(row,'goal',0,seed,physical,initial)
    else:
        check(source['raw_goal_shape']==[224,224,3] and source['raw_goal_dtype']=='uint8' and row['phase']=='goal' and row['shape']==[224,224,3] and source['raw_goal_sha256']==row['RGB_AA4_sha256'],'Rawgoal RGB224AA4 source shape/dtype/hash')
        check(row['source_time']==physical['time'],'True goal RGB scene time')
        for key in ['qpos','qvel','ctrl','mocap_pos','mocap_quat']:check(row['source_pose_sha256'][key]==raw_hash(physical[key]),'True goalarm/object sourcepose '+key)
    check(isinstance(source['raw_goal_sha256'],str) and len(source['raw_goal_sha256'])==64 and all(c in '0123456789abcdef' for c in source['raw_goal_sha256']),'Rawgoal hash syntax')
    return row


def validate_step_goal(pc,source,goal_frame,index,ns):
    check=ns['check'];identity=pc['goal_identity']
    check(pc['current_encoder_calls']==pc['goal_encoder_calls']==index and pc['encoder_forward_calls']==2*index,'Actual2N per-step forward counters')
    check(pc['goal_raw_sha256']==identity['raw_goal_sha256']==source['raw_goal_sha256'] and identity['goal_frame']==goal_frame,'Stable rawgoal frame identity; eachstep freshgoal forward')
    check(identity['goal_input_shape']==[1,195] and all(pc[k] is False for k in ['offline_feature_cache','current_feature_reuse','goal_feature_reuse']),'Visual195 no encoderfeature reuse')
    for key in ['goal_latent_sha256','goal_input_sha256']:
        h=identity[key];check(isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h),'Recorded actualgoal hash '+key)


def validate_zero_goal(pair,case,goal,ns):
    check=ns['check']
    check(pair['goal_mode']=='latent_object_xyz' and pair['goal_visual_rendered'] is False and pair['zero_visual_channels'] is True,'Zero195 numeric goal only')
    check(pair['goal_XYZ']==goal[0].tolist() and pair['goal_XYZ_bytes_sha256']==hashlib.sha256(goal.tobytes()).hexdigest(),'True paired goalXYZ3 numeric bytes')
    check(case['zero_visual_channels'] is True and case['goal_visual_rendered'] is False,'Zero195 close-time goal receipt')


def validate_zero_step(pc,goal,ns):
    check=ns['check'];expected=np.concatenate((np.zeros((1,192),np.float32),goal),axis=-1)
    check(pc['encoder_forward_calls']==pc['current_encoder_calls']==pc['goal_encoder_calls']==0,'Zero195 all actual encoder counts zero')
    check(pc['zero_visual_channels'] is True and pc['goal_visual_rendered'] is False and pc['current_frame']=={'visual_mode':'zero192','rendered':False},'Zero195 no current or goal visual frame')
    check(pc['policy_input_shapes']==[[1,220],[1,195]] and pc['precision']=='float32','Zero195 retained policy capacity')
    check(all(pc[k] is False for k in ['runtime_feature_cache','offline_feature_cache','current_feature_reuse','goal_feature_reuse']),'Zero195 no feature reuse')
    check(pc['goal_input_sha256']==hashlib.sha256(expected.tobytes()).hexdigest(),'Exactly zeros192 plus trueXYZ3 goal input bytes')


def current_dims(config):
    profile=config.get('state_profile','full28')
    if profile not in ('full28','robot_proprio'):raise ValueError('Unknown explicit state profile')
    return 192+(28 if profile=='full28' else 19)


def validate_profile_metadata(row,config,ns):
    if config.get('state_profile','full28')=='robot_proprio':
        ns['check'](row.get('state_profile')=='robot_proprio' and row.get('network_state_indices')==list(range(19)) and row.get('excluded_object_state_indices')==list(range(19,28)),'Recorded no privileged object-state9 numeric network channels')


def episode(path,refs,protocol_sha,config,diagnostic,ns):
    sha,digest,read,check,arrays,raw_hash,frame,appearance=[ns[k] for k in ['sha','digest','read','check','arrays','raw_hash','frame','appearance']]
    original_sha=sha(path);r=read(path);task,index=r['task'],r['index'];seed=970000000+task*100000+index
    check(r['status']=='passed' and r['exit_code']==0 and r['protocol_sha256']==protocol_sha and r['seed']==seed and bool(r['diagnostic_only'])==diagnostic and r['success'] in [0.,1.],'Actual result/protocol/case')
    check(1<=r['steps']<=(4 if diagnostic else 200),'Real episode step limit')
    folder=(path.parent/r['attempt']).resolve();check(folder.parent==path.parent.resolve() and (folder/'exit_code').read_text().strip()=='0' and read(folder/'result.json')==r,'Actual attempt exit/result identity')
    status=read(folder/'status.json');check(status['status']=='passed' and status['exit_code']==0 and status['task']==task and status['index']==index,'Actual attempt status')
    for name,key in [('steps.jsonl','steps_sha256'),('reset.json','reset_sha256'),('pairing.json','pairing_sha256')]:check(sha(folder/name)==r[key],'Artifact identity '+name)
    reset,pair,case,app=read(folder/'reset.json'),read(folder/'pairing.json'),read(folder/'input_case.json'),read(folder/'appearance.json');arrays(reset);arrays(pair)
    entry=refs['reference_manifest'][f'{task}:{index}'];ref=read(entry['reset_path']);arrays(ref)
    check(sha(entry['reset_path'])==entry['reset_file_sha256']==pair['reference_reset_file_sha256'],'Original reference payload')
    check(reset['requested_seed']==reset['actual_seed']==ref['requested_seed']==ref['actual_seed']==seed==pair['reference_seed'],'Actual paired seed')
    for key in ['observation','goal28','initial_physical_state','goal_physical_state','rng_state','official_internal_goal_initialization']:check(reset[key]==ref[key],'Reference equality '+key)
    appearance(app,ref,{'source_RGB_recipe_sha256':sha(config['rgb_recipe'])})
    check(reset['model_colors_camera_lights']==app['actual_snapshot'] and reset['whole_model_visual_exact_claim'] is False,'Declared appearance difference only')
    check(reset['reference_replay']==pair and reset['input_case']==case and pair['strict_initial_and_goal_match'] and pair['goal_render_current_state_unchanged'] and pair['policy_warmup_steps']==0 and pair['max_policy_steps']==200 and pair['whole_model_visual_exact_claim'] is False,'Paired physical lifecycle')
    check(pair['initial_physical_sha256']==digest(ref['initial_physical_state']) and pair['goal_physical_sha256']==digest(ref['goal_physical_state']) and pair['reference_sha256']==digest(ref),'Physical/full reference hashes')
    check(len(pair['reset_Box_replay'])==len(ref['official_internal_goal_initialization'])==2,'Two actual archived reset actions')
    for n,rr in enumerate(pair['reset_Box_replay']):check(rr['index']==n and rr['actual_replayed_action']==ref['official_internal_goal_initialization'][n]['action'] and rr['post_state']==ref['official_internal_goal_initialization'][n]['post_state'],'Actual reset replay')
    goal=np.asarray(ref['goal28']['values'],dtype=ref['goal28']['dtype'])[19:22].astype(np.float32)[None]
    goal_sha=hashlib.sha256(goal.tobytes()).hexdigest();encoded=config['visual_mode']=='cache';expected_calls=r['steps'] if encoded else 0;dim=current_dims(config)
    if encoded:
        goal_source=read(folder/'goal_source.json');check(sha(folder/'goal_source.json')==case['goal_source_sha256'] and pair['goal_source']==goal_source,'Actual reset goal source binding')
        check(pair['goal_mode']=='latent_object_xyz' and pair['goal_visual_rendered'] is True and pair['goal_XYZ']==goal[0].tolist() and pair['goal_visual_raw_sha256']==goal_source['raw_goal_sha256'],'Paired real visual goal with officialXYZ3')
        goal_frame=validate_goal_source(goal_source,goal,ref,config,seed,ns)
    else:
        check(not (folder/'goal_source.json').exists(),'Zero195 does not fabricate rendered goal source')
        validate_zero_goal(pair,case,goal,ns)
    validate_profile_metadata(case,config,ns)
    check(case['policy_steps']==case['model_calls']==r['steps'] and case['current_encoder_calls']==expected_calls and case['goal_encoder_calls']==expected_calls and case['visual_mode']==config['visual_mode'],'Actual currentN andgoalN forward counts')
    check(case['total_encoder_calls']==(2*r['steps'] if encoded else 0) and case['policy_input_shapes']==[[1,dim],[1,195]] and all(case[k] is False for k in ['offline_feature_cache','current_feature_reuse','goal_feature_reuse']),'Explicit2N no outputfeature reuse')
    check(case['frozen_state_check_applicable']==encoded and case['frozen_full_state_exact']==encoded and case['whole_model_visual_exact_claim'] is False,'Frozen checks applicability')
    check(sha(folder/'perception.jsonl')==case['perception_sha256'] and sha(folder/'appearance.json')==case['appearance_sha256'],'Close-time artifact binding')
    steps=(folder/'steps.jsonl').read_text().splitlines();perception=(folder/'perception.jsonl').read_text().splitlines();check(len(steps)==len(perception)==r['steps'],'One actual perception/action receipt per policy step')
    rng=digest(np.random.default_rng(seed).bit_generator.state);previous=ref['initial_physical_state'];fallback=0
    for n,(line,pline) in enumerate(zip(steps,perception),1):
        step,pc=json.loads(line),json.loads(pline);arrays(step)
        check(step['step']==pc['model_call']==n and pc['encoder_forward_calls']==(2*n if encoded else 0) and pc['current_encoder_calls']==pc['goal_encoder_calls']==(n if encoded else 0),'Sequential actual model/encoder calls')
        check(pc['policy_input_shapes']==[[1,dim],[1,195]] and pc['precision']=='float32' and pc['visual_mode']==config['visual_mode'] and pc['runtime_feature_cache'] is False,'State-profile current / visual195 goal FP32 contract')
        validate_profile_metadata(pc,config,ns)
        check(pc['goal_XYZ']==goal[0].tolist() and pc['goal_XYZ_sha256']==goal_sha,'Per-step original goal carrier projection')
        if encoded:validate_step_goal(pc,goal_source,goal_frame,n,ns)
        else:validate_zero_step(pc,goal,ns)
        h=pc['current_input_sha256'];check(isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h),'Recorded actual current-input hash')
        current=pc['current_frame']
        if encoded and config['modality']=='pcd':
            rng=frame(current,'current',n-1,seed,previous,rng);fallback+=int(current['sampling']['fallback_used'])
        elif encoded:
            check(current['phase']=='current' and current['shape']==[224,224,3] and current['source_time']==previous['time'],'RGB real current source shape/time')
            h=current['RGB_AA4_sha256'];check(len(h)==64 and all(c in '0123456789abcdef' for c in h),'Recorded real RGB hash')
            for k in ['qpos','qvel','ctrl','mocap_pos','mocap_quat']:check(current['source_pose_sha256'][k]==raw_hash(previous[k]),'RGB true pre-action physical state '+k)
        else:check(current=={'visual_mode':'zero192','rendered':False},'Zero bypass frame/render/encoder')
        check(pc['sampled_action_already_GCIQL_clipped'] and pc['official_evaluate_clip_follows'],'Explicit original clipping labels')
        sampled=np.asarray(pc['sampled_action'],dtype=np.float32);sent=np.asarray(step['sent_action']['values'],dtype=np.float32)
        check(sampled.shape==sent.shape==(5,) and np.isfinite(sampled).all() and np.abs(sampled).max()<=1 and np.array_equal(sent,np.clip(sampled,-1,1)),'GCIQL sample/evaluate clip/action5')
        post=step['post_state'];check(post['sha256']==digest({k:v for k,v in post.items() if k!='sha256'}),'Poststate content hash')
        for key,size in [('ctrl',7),('qpos',21),('qvel',20)]:check(post[key]['shape']==[size],'Physical dimensions '+key)
        check(np.isfinite(step['reward']),'Finite original reward')
        original,returned=step['original_environment_flags'],step['returned_flags']
        if diagnostic and n==4 and not (original['terminated'] or original['truncated']):
            check(returned=={'terminated':False,'truncated':True},'Only original diagnostic fourstep cap')
        else:check(original==returned,'Original termination flags unchanged')
        if not diagnostic:check(bool(step['original_environment_flags']['terminated'] or step['original_environment_flags']['truncated'])==(n==len(steps)),'Legitimate final-step environment termination')
        previous=post
    check(step['reward']==r['success'],'Actual final original success')
    if not diagnostic:check(r['steps']==200 or (r['success']==1. and step['original_environment_flags']['terminated']),'Original success or time limit')
    check(sha(path)==original_sha,'Result stable during CPU audit')
    return {'path':str(path),'sha256':original_sha,'task':task,'index':index,'seed':seed,'steps':r['steps'],'success':r['success'],'fallback':fallback,'result_before_attempt_exit':path.stat().st_mtime_ns<(folder/'exit_code').stat().st_mtime_ns}

def selected_budget_metadata(config,resolved,audit,probe,manifest,parent_manifest,check):
    dim=current_dims(config);profile=config.get('state_profile','full28');encoded=config['visual_mode']=='cache'
    check(config.get('selected_updates')==1000000 and config.get('goal_mode')=='latent_object_xyz' and config['visual_mode'] in ('cache','zero192'),'Only actual1M visual-or-zero195 selected checkpoint')
    check(profile==('robot_proprio' if encoded else 'full28') and resolved['binding']['config'].get('state_profile','full28')==profile,'Actual RL3 cached-visual proprio19 or zero-visual oracle28 profile')
    check(resolved['total_updates']==resolved['binding']['config']['total_updates']==1000000,'Actual1M training budget')
    check(resolved['initial_counter']==0 and resolved['resumed_from'] is None and resolved['binding']['config']['batch_size']==1024 and resolved['binding']['config']['seed']==0,'Freshseed0 B1024 lineage')
    check(manifest['counter']==audit['counter']==audit['adam_count']==1000000 and audit['network_step']==1000001,'Selected checkpoint/CPU audit counters')
    check(audit[f'original_GCIQL_fresh_factory_{dim}_goal195_full153_leaves_exact'],'Actual state-profile/195 full153 proof')
    if encoded:check(probe['all_four_roles_match_source_cached_latents_or_zero'] is True and probe['all_four_roles_visual192_exact_zero'] is False,'Actual finalCPU four-role same-ID cached visual interface')
    else:check(probe['all_four_roles_visual192_exact_zero'] is True,'Actual finalCPU four-role zero195 interface')
    check(probe['restored_network_step']==1000001 and probe['adam_count']==1000000 and probe['goal_shape']==[4,195] and probe['current_next_shape']==[4,dim],'Actual restored195 interface/counters')
    check(manifest['agent_sha256']==config['agent_sha256']==probe['checkpoint_sha256'] and manifest['sidecar_sha256']==config['sidecar_sha256']==probe['sidecar_sha256'],'Selected checkpoint/sidecar identity')
    return 1000000

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['tiny','full'],required=True)
    for key in ['config','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expected-config-sha256',required=True);p.add_argument('--driver',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--expected-plan-sha256')
    p.add_argument('--required-zero-exit',type=Path,action='append',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    ns=dict(Path=Path,np=np,hashlib=hashlib,json=json,os=os)
    helpers=load_functions(UTIL,UTIL_SHA,['sha','digest','read','save','check','arrays'],ns)
    sha,digest,read,save,check=[ns[n] for n in ['sha','digest','read','save','check']]
    try:
        config=read(a.config);check(sha(a.config)==a.expected_config_sha256,'Exact accepted config')
        check(sha(N/'aux/freeze_v1/manifest.json')==FREEZE_SHA,'Unique frozen source manifest')
        for name,h in read(N/'aux/freeze_v1/manifest.json')['files'].items():check(sha(N/name)==h,'Frozen entry source '+name)
        for path,h in config['source_sha256'].items():check(Path(path).suffix not in ['.pt','.pkl','.h5','.npz','.npy'] and sha(path)==h,'Bound small source '+path)
        for path in list(config['required_zero_exits'])+a.required_zero_exit:check(Path(path).read_text().strip()=='0','Actual required exit '+str(path))
        audit,probe=read(config['final_audit']),read(config['final_probe']);check(audit['exit_code']==probe['exit_code']==0 and audit[f'original_GCIQL_fresh_factory_{current_dims(config)}_goal195_full153_leaves_exact'] and probe['full153_saved_state_exact'],'Accepted real final153 restore proof')
        resolved=read(Path(config['train_root'])/'resolved.json');selected=config.get('selected_updates',100000)
        manifest=read(Path(config['train_root'])/f'checkpoints/update_{selected:09d}/manifest.json')
        parent_manifest=read(Path(config['parent_transaction']['path'])/'manifest.json') if selected==500000 else None
        selected_budget_metadata(config,resolved,audit,probe,manifest,parent_manifest,check)
        R=Path(config['native_output']);protocol=read(R/'protocol.json');ph=digest(protocol);binding=read(R/'binding.json')
        check(ph==binding['protocol_sha256'] and protocol['policy_kind']=='cached-visual-state-xyz' and protocol['algorithm']=='gciql' and protocol['episodes_per_task']==50 and protocol['max_environment_steps']==200 and protocol['extra_policy_warmup']==0,'Actual native protocol')
        check(protocol['checkpoint_sha256']==config['agent_sha256']==probe['checkpoint_sha256'],'Selected exact budget checkpoint identity through accepted witness')
        actual=protocol['actual_training_binding'];check(actual['config']==config and actual['config_sha256']==a.expected_config_sha256,'Actual complete config binding')
        check(protocol['agent_config']==read(Path(config['train_root'])/'resolved.json')['binding']['agent_config'],'Original agent factory config')
        refs=protocol['reference_replay_binding'];check(len(refs['reference_manifest'])==250,'Original250 paired references')
        for key,name in [('reference_protocol_sha256','protocol.json'),('reference_episode_manifest_sha256','episode_manifest.json'),('reference_independent_audit_sha256','independent_full_audit.json')]:check(sha(Path(refs['reference_root'])/name)==refs[key],'Reference header '+name)
        ns.update(CORE=config['source_sha256'][config['pcd_core']],PROVIDER=config['source_sha256'][config['pcd_provider']]);helpers.update(load_functions(PCD,PCD_SHA,['raw_hash','frame','appearance'],ns))
        gate=read(R/'gate_receipt.json');check(gate==read(R/'gate/result.json') and gate['diagnostic_only'],'Actual tiny excluded from SR')
        gate_record=episode(R/'gate/result.json',refs,ph,config,True,ns)
        records=[]
        if a.mode=='full':
            check(a.driver is not None and a.plan is not None and a.expected_plan_sha256 is not None,'Full driver/plan required')
            driver,plan=read(a.driver/'status.json'),read(a.plan)
            check(sha(a.plan)==a.expected_plan_sha256==driver['plan_sha256'],'Approved full plan')
            check((a.driver/'exit_code').read_text().strip()=='0' and driver['exit_code']==0 and driver['status']=='completed_pending_independent_audit' and driver['completed_tasks']==5,'Actual full driver completion')
            check([s['name'] for s in driver['stages']]==['task1','task2','task3','task4','task5','aggregate'] and all(s['exit_code']==0 for s in driver['stages']),'Six actual child exits')
            for path,h in plan['sealed_inputs'].items():check(Path(path).suffix not in ['.pt','.pkl','.h5','.npz','.npy'] and sha(path)==h,'Sealed small input '+path)
            start=datetime.datetime.fromisoformat(driver['started_at']).timestamp();end=datetime.datetime.fromisoformat(driver['updated_at']).timestamp();wrappers=[]
            for folder in (N/'aux/stage_launches').iterdir():
                if not all((folder/n).is_file() for n in ['argv.txt','exit_code','finished_at']):continue
                argv=shlex.split((folder/'argv.txt').read_text());stage=argv[argv.index('--stage')+1] if '--stage' in argv else None
                if '--output' not in argv or argv[argv.index('--output')+1]!=str(R) or stage not in ['task','aggregate'] or not start<=(folder/'finished_at').stat().st_mtime<=end+1:continue
                check((folder/'exit_code').read_text().strip()=='0','Actual stage shell exit')
                check('--policy-kind=cached-visual-state-xyz' in argv and '--algorithm=gciql' in argv and argv[argv.index('--config')+1]==str(a.config) and argv[argv.index('--reference-replay')+1]==refs['reference_root'],'Actual stage argv')
                check(not any(t in (folder/'run.log').read_text() for t in ['Traceback (most recent call last)','Exception ignored in:','EGLError']),'Runtime/cleanup exceptions')
                wrappers.append('task'+argv[argv.index('--task')+1] if stage=='task' else stage)
            check(sorted(wrappers)==['aggregate','task1','task2','task3','task4','task5'],'Six actual wrappers, tiny excluded')
            paths=[R/f'task{t}'/f'episode{i:03d}'/'result.json' for t in range(1,6) for i in range(50)];check(set(R.glob('task*/episode*/result.json'))==set(paths),'Exactly250 root records')
            records=[episode(path,refs,ph,config,False,ns) for path in paths];check(len({r['seed'] for r in records})==250,'Unique formal seeds')
            success=sum(r['success'] for r in records);per={str(t):sum(r['success'] for r in records if r['task']==t)/50 for t in range(1,6)};aggregate=read(R/'aggregate.json')
            check(aggregate['status']=='passed' and aggregate['exit_code']==0 and aggregate['episodes']==250 and aggregate['success_rate']==success/250 and aggregate['per_task']==per and aggregate['protocol_sha256']==ph and aggregate['gate_excluded'],'Independent full250 aggregate recomputation')
            check(sha(R/'aggregate.json')==driver['aggregate_sha256'],'Completed aggregate identity')
        report={'status':'passed_tiny_metadata_source_contract_only' if a.mode=='tiny' else 'passed_full250_metadata_and_source_with_explicit_limits','exit_code':0,'mode':a.mode,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'episodes':len(records),'unique_seeds':len({r['seed'] for r in records}),'steps_audited':sum(r['steps'] for r in records),'successes':sum(r['success'] for r in records),'success_rate':sum(r['success'] for r in records)/250 if records else None,'per_task':per if records else None,'gate_record':gate_record,'records':records,'protocol_sha256':ph,'config_sha256':a.expected_config_sha256,'plan_sha256':a.expected_plan_sha256,'checkpoint_sha256':config['agent_sha256'],'source_freeze_sha256':FREEZE_SHA,'reused_helper_source_SHA':{str(UTIL):UTIL_SHA,str(PCD):PCD_SHA},'exact_helper_AST':helpers,'required_zero_exits':[str(x) for x in a.required_zero_exit],'limits':['CPU artifacts/source only; no policy/encoder/render/physics rerun.','Recorded current hash bound to frozen visual192+proprio19 or zero192+oracle28 assembly; per-step rawlatent/state input not stored, so no independent numerical reconstruction claim.','True reference goalXYZ3 verified; zero195 goal bytes independently reconstructed or visual195 goal scene/source2N receipts verified; no unrecorded encoder numerics reconstructed.','Actual original termination/success and action5/ctrl7 receipts verified; raw unclipped actor means not available.','Results precede attempt finally exits; acceptance only after required real exits0; no commit-after-exit claim.','Unavailablec050 entire helper source not claimed exact; declared a83 two-helper scope with preservedfcbc/572.','Full250 one seed per route is not eight-seed paper aggregate; tiny is excluded.']}
        save(a.output/'report.json',report);save(a.output/'status.json',{k:v for k,v in report.items() if k!='records'});(a.output/'inner_exit_code').write_text('0\n');print(json.dumps({k:v for k,v in report.items() if k not in ['records','gate_record']},indent=2))
    except BaseException:
        save(a.output/'status.json',{'status':'failed','exit_code':1,'error':traceback.format_exc()});(a.output/'inner_exit_code').write_text('1\n');raise
if __name__=='__main__':main()
