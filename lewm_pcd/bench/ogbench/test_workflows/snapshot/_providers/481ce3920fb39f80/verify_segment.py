"""Thin CPU acceptance for cached220/visualXYZ195 runs; original sampler and Flax restore reused."""
import argparse,csv,hashlib,importlib.util,json,os,pickle,subprocess,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['JAX_PLATFORMS']='cpu'
W=Path('/root/autodl-tmp/workspace/users/lewm')
ROOT=W/'coordination/rl3_m0_m5_m6_visual_proprio_smoke_109_20261013_v1/formal_start_v1';C=ROOT/'core_v1'
previous=W/'coordination/pcd_frozen_lewm_gciql_100k_b128_109_20261010_v1/post_start_review_v1/audit.py'
assert hashlib.sha256(previous.read_bytes()).hexdigest()=='6e475f772534c5c9798a55fb81dc066d9ab53608574140bacb247f851c5a3714'
spec=importlib.util.spec_from_file_location('accepted_audit_helpers',previous);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
load=helper.load;sha=helper.sha

def validate_segment_identity(config,resolved,run,start,parent):
    assert config==resolved['binding']['config'] and config['batch_size']==1024 and config['seed']==0
    assert resolved['initial_counter']==start and resolved['initial_network_step']==start+1
    if start:
        assert parent is not None and Path(resolved['resumed_from']).resolve()==Path(parent).resolve()
        assert Path(run).resolve()!=Path(config['output']).resolve()
    else:
        assert parent is None and resolved['resumed_from'] is None


def validate_fresh_profile(config,resolved,run,start,parent,count):
    validate_segment_identity(config,resolved,run,start,parent)
    assert start==0 and parent is None and Path(run).resolve()==Path(config['output']).resolve()
    assert config['goal_mode']=='latent_object_xyz' and config['reward_mode']=='current_object_xyz'
    assert config['input_profile']=='cached' and config['visual_mode'] in ('cache','zero192') and config['workers'] in (0,2,4)
    assert config.get('state_profile','full28') in ('full28','robot_proprio')
    assert config['total_updates']==resolved['total_updates']==1000000
    assert config['save_interval']==100000 and config['validation_interval']==5000
    assert config['evaluation_episodes']==0 and 0<count<=1000000 and count%100000==0


def current_dims(config):
    profile=config.get('state_profile','full28');assert profile in ('full28','robot_proprio')
    return 192+(28 if profile=='full28' else 19)


def validate_input_kernels(params,config=None):
    dim=current_dims(config or {})+195
    expected={'modules_actor':[dim,512],'modules_value':[dim,512],
              'modules_critic':[2,dim+5,512],'modules_target_critic':[2,dim+5,512]}
    actual={name:list(params[name]['actor_net' if name=='modules_actor' else 'value_net']['Dense_0']['kernel'].shape)
            for name in expected}
    assert actual==expected, 'Restored network input dimensions do not match state_profile/goal195/action5'
    return actual


def main(a):
    import numpy as np
    cfg=load(a.config);R=a.run or Path(cfg['output']);resolved=load(R/'resolved.json');binding=resolved['binding'];count=a.counter
    start=a.start_counter;target=resolved['total_updates']
    validate_fresh_profile(cfg,resolved,R,start,a.parent_transaction,count)
    freeze_path=ROOT/'adapter_v1/freeze_v1/manifest.json';freeze=load(freeze_path)
    assert sha(freeze_path)=='03f3c3551a46c80bc1f82f4472781369d806d1cdb3f773c7b8da6c53b9dbe75e'
    pins={Path(x['path']).name:x['sha256'] for x in freeze['files']}
    assert {k:sha(C/k) for k in pins}==pins==binding['code']
    assert cfg==binding['config'] and cfg['input_profile']=='cached' and cfg['batch_size']==1024 and cfg['seed']==0
    assert cfg['goal_mode']=='latent_object_xyz' and cfg['reward_mode']=='current_object_xyz' and cfg['workers'] in (0,2,4)
    assert binding['runtime']['runtime_encoder_forwards']==0 and binding['runtime']['cached_feature_loading']=='readonly_RAM'
    assert binding['frozen_full_state_semantics']=='cached_source_binding_not_runtime_WM'
    assert hashlib.sha256(json.dumps(binding['cache_sources'],sort_keys=True).encode()).hexdigest()==binding['frozen_full_state']
    validate_segment_identity(cfg,resolved,R,start,a.parent_transaction)
    expected_argv=[str(C/'train.py'),'--config',str(a.config),'--output',str(R),'--total-updates',str(target)]
    if start:expected_argv+=['--resume',str(a.parent_transaction)]
    assert resolved['argv']==expected_argv
    plan=load(a.launcher/'plan.json');assert plan['argv'][1:]==expected_argv and plan['config_sha256']==sha(a.config)
    state=load(R/'status.json');assert state['completed_updates']>=count and state['failed_stages']==0
    exits={}
    if count==target:
        exits={k:int(p.read_text()) for k,p in {'trainer':R/'trainer_exit_code','child':a.launcher/'child_exit_code','driver':a.launcher/'driver_exit_code','outer':a.outer_exit}.items()}
        assert all(v==0 for v in exits.values())
        assert state['exit_code']==0 and state['completed_updates']==state['adam_count']==target
        assert state['network_step']==target+1 and state['validation_count']==target//5000
    with (R/'train.csv').open(newline='') as f:csvrows=[r for r in csv.DictReader(f) if int(r['step'])<=count]
    assert [int(r['step']) for r in csvrows]==list(range((start//5000+1)*5000,count+1,5000))
    assert all(np.isfinite(float(v)) for row in csvrows for v in row.values())
    assert all(float(row['training/grad/norm'])>0 for row in csvrows)
    cache=cfg['cache']
    for field in ('manifest','seal','terminal_manifest'):assert sha(Path(cache[field]))==cache[field+'_sha256']
    cm,seal,tm=(load(Path(cache[k])) for k in ('manifest','seal','terminal_manifest'))
    for m in (cm,seal,tm):assert m['status']=='passed' and m['exit_code']==0
    assert tm['cache_manifest_sha256']==cache['manifest_sha256'] and tm['cache_seal_sha256']==cache['seal_sha256']
    assert tm['source_modality']==cache['modality'] and tm['raw_id_mapping']=='episode*1001+1000'
    for key in ('source_checkpoint_sha256','preprocess_source_sha256','encoder_hashes','precision'):assert tm[key]==cm[key]
    stats_checked=0
    for path,expected in seal['cache_stats'].items():
        s=Path(path).stat();actual=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
        assert actual==expected,path;stats_checked+=1
    for split in ('train','val'):assert sha(Path(tm['splits'][split]['path']))==tm['splits'][split]['sha256']
    sides={};records=[];parent_record=None
    if start:
        parent=Path(a.parent_transaction);pm=load(parent/'manifest.json')
        assert parent.name==f'update_{start:09d}' and pm['counter']==start
        assert sha(parent/f'params_{start}.pkl')==pm['agent_sha256'] and sha(parent/'run_state.pkl')==pm['sidecar_sha256']
        with (parent/'run_state.pkl').open('rb') as f:ps=pickle.load(f)
        assert helper.normalize(ps['binding'])==binding and ps['budget_total']<=target
        assert ps['network_step']==start+1 and ps['adam_count']==start
        assert ps['events']==dict(committed_updates=start,validation_count=start//5000,next_stage='sample_train',last_validation=start-start%5000,checkpoint_event_complete=True,next_update=start+1)
        sides[start]=ps;parent_record=dict(path=str(parent),**pm)
    for k in range(start+cfg['save_interval'],count+1,cfg['save_interval']):
        p=R/f'checkpoints/update_{k:09d}';manifest=load(p/'manifest.json');assert manifest['counter']==k
        assert sha(p/f'params_{k}.pkl')==manifest['agent_sha256'] and sha(p/'run_state.pkl')==manifest['sidecar_sha256']
        with (p/'run_state.pkl').open('rb') as f:side=pickle.load(f)
        assert helper.normalize(side['binding'])==binding and side['budget_total']==target
        assert side['network_step']==k+1 and side['adam_count']==k
        assert side['events']==dict(committed_updates=k,validation_count=k//5000,next_stage='sample_train',last_validation=k-k%5000,checkpoint_event_complete=True,next_update=k+1)
        sides[k]=side;records.append(manifest)
    if count==target:assert len(list((R/'checkpoints').glob('update_*')))==(target-start)//cfg['save_interval']
    assert sha(Path(cfg['source_config']))==cfg['source_config_sha256']
    source=load(Path(cfg['source_config']));official=W/source['official']
    for rel,h in binding['source_locks'].items():
        if rel.endswith(('.py','.json','.yaml','.sh')):assert sha(W/rel)==h,rel
    assert subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()=='572788d97c2a6225b8edb7c420e304c04d1b3459'
    assert not subprocess.check_output(['git','-C',str(official),'status','--porcelain'],text=True).strip()
    sys.path[:0]=[str(official/'impls'),str(official)]
    from ogbench.utils import load_dataset
    from utils.datasets import Dataset,GCDataset
    samplers={}
    for split,suffix in [('train',''),('val','-val')]:
        compact=load_dataset(str(W/source['raw']/f'cube-single-play-v0{suffix}.npz'),compact_dataset=True)
        assert helper.arr_digest(compact['observations'])==binding['original_state'][split]
        compact['observations']=np.arange(len(compact['observations']),dtype=np.int64)[:,None]
        sampler=GCDataset(Dataset.create(**compact),binding['agent_config']);samplers[split]=sampler
        for key in ('initial_locs','terminal_locs'):assert helper.arr_digest(getattr(sampler,key))==binding['sampler'][split]['arrays'][key]
        assert helper.arr_digest(sampler.dataset.valid_idxs)==binding['sampler'][split]['valid_idxs']
    np.random.seed(0);samplers['train'].sample(1)
    for k in range(1,count+1):
        samplers['train'].sample(cfg['batch_size'])
        if k%5000==0:samplers['val'].sample(cfg['batch_size'])
        if k in sides:helper.equal_rng(np.random.get_state(),sides[k]['rng']['numpy'])
    expected=samplers['train'].sample(cfg['batch_size']);np.random.set_state(sides[count]['rng']['numpy']);actual=samplers['train'].sample(cfg['batch_size'])
    for key in expected:np.testing.assert_array_equal(actual[key],expected[key])
    import jax,flax,torch,random
    assert jax.default_backend()=='cpu'
    spec=importlib.util.spec_from_file_location('frozen_cached_gciql',C/'gciql.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    template=module.GCIQLAgent.create(0,np.zeros((1,current_dims(cfg)),np.float32),np.zeros((1,5),np.float32),binding['agent_config'],ex_goals=np.zeros((1,195),np.float32))
    from utils.flax_utils import restore_agent
    p=R/f'checkpoints/update_{count:09d}';restored=restore_agent(template,str(p),count)
    with (p/f'params_{count}.pkl').open('rb') as f:saved=pickle.load(f)['agent']
    aa,ta=jax.tree_util.tree_flatten(saved);bb,tb=jax.tree_util.tree_flatten(flax.serialization.to_state_dict(restored))
    assert ta==tb and len(aa)==153
    for x,y in zip(aa,bb):
        assert np.asarray(x).dtype==np.asarray(y).dtype and np.isfinite(np.asarray(y)).all()
        np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
    actual_input_kernels=validate_input_kernels(restored.network.params,cfg)
    assert int(restored.network.step)==count+1 and int(restored.network.opt_state[0].count)==count
    key=jax.random.split(jax.random.PRNGKey(0))[0];key=jax.lax.fori_loop(0,count,lambda i,key:jax.random.split(key)[0],key)
    np.testing.assert_array_equal(np.asarray(restored.rng),np.asarray(key))
    rng=sides[count]['rng'];assert rng['python']==random.Random(0).getstate()
    assert torch.equal(rng['torch'],torch.Generator().manual_seed(0).get_state())
    assert all(x.dtype==torch.uint8 for x in rng['cuda'])
    report=dict(status='passed_CPU_final_acceptance' if count==target else 'passed_CPU_segment_checkpoint',exit_code=0,
        counter=count,network_step=count+1,actual_input_kernel_shapes=actual_input_kernels,floating_leaf_dtypes=sorted({str(np.asarray(v).dtype) for v in bb if np.issubdtype(np.asarray(v).dtype,np.floating)}),current_dims=current_dims(cfg),goal_dims=195,save_interval=cfg['save_interval'],jax_default_matmul_precision_actual=jax.config.jax_default_matmul_precision,precision_label='float32 arrays/unchanged official JAX precision defaults',adam_count=count,batch_size=cfg['batch_size'],train_examples=count*cfg['batch_size'],validation_events=count//5000,
        segment_start_counter=start,segment_train_examples=(count-start)*cfg['batch_size'],segment_validation_batches=len(csvrows),parent_transaction=parent_record,
        all_saved_agents_and_sidecars_SHA_checked=len(records),
        all_saved_counter_event_binding_numpy_rng_exact=True,final_JAX_rng_exact=True,python_torch_cpu_seed0_exact=True,
        cuda_rng_structure_present=True,next_raw_batch_exact=True,next_raw_batch_digests={k:helper.arr_digest(v) for k,v in actual.items()},
        cache_stats_unchanged=stats_checked,terminal_payload_SHA_exact=True,source_binding_hash=binding['frozen_full_state'],
        source_binding_semantics=binding['frozen_full_state_semantics'],runtime_encoder_forward_count=0,
        data_labels='current objectXYZ distance<=.4scaled;rewardsuccess-1;mask1-success;originalsampler/rewardadapter consumesnoRNG',
        code=binding['code'],config_sha256=sha(a.config),final_checkpoint=records[-1],checkpoint_records=records,
        actual_training_exit_codes=exits,csv_rows=len(csvrows),csv_sha256=sha(R/'train.csv'),last_validation=csvrows[-1] if csvrows else None,
        evidence_limits=['Original full agent restored on CPU; no loss/backward/optimizerstep/modelencoder/actionforward performed.',
            'Frozen fingerprint is cache/source-binding identity; no runtime WM exists in this training route and no WM hash/feature re-extraction is claimed.',
            'Source cached payload content hashes reuse accepted producer+seal evidence with currentstats; terminal smallNPZ contents are rehashed.',
            'Sampler replay validates every100000 saved RNG/event boundary and next rawbatch; no per-update productiontrace or re-executed update is claimed.',
            'Newdistance labels/inputmapping rely on accepted narrowcoretests and sameversion actualGPUgate; freshCPUcachedactionprobe verifies interface separately; nativeSR is separate evidence.'])
    report[f'original_GCIQL_fresh_factory_{current_dims(cfg)}_goal195_full153_leaves_exact']=True
    with (a.output/'report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps({'status':report['status'],'counter':count,'leaf_count':153,'checkpoints':len(records),'report_sha256':sha(a.output/'report.json')},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--launcher',type=Path,required=True)
    parser.add_argument('--outer-exit',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--counter',type=int,default=1000000)
    parser.add_argument('--run',type=Path);parser.add_argument('--start-counter',type=int,default=0)
    parser.add_argument('--parent-transaction',type=Path);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);main(args)
