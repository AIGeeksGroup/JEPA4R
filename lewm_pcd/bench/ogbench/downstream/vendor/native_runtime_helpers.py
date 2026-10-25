"""Thin committed-best/last native SR entry. No stage dispatch is automatic."""
import argparse,hashlib,importlib.util,json,os,sys,time,traceback,contextlib
from pathlib import Path
import numpy as np

POLICY_SHA='d54e578d331f15a4f1bbf6e4335b27834fe3f120b8341da0c429dd54e9613ef4'
DATASET_SHA='5c922fad193fbd6cffad58992504ce3107fbd4230d99741d15c4a5c77af07181'
TRAINER_SHA='55b3fadf54023e23ac56a87b41eb333bcb24e54490ae200db6a458420fd65f00'
FROZEN_SHA='9c498fafa1f68778ccf116f6a811672e584607a4e5f34e8010970591c6ac43ee'
HASH_HELPER_SHA='5bc65e983040045817d49ff8b715bb9dcc57e13fa8d7192ed0a5e5d164b6eb8a'
WM_CONFIG_JSON_SHA='c23f8d59c185b7e8953649eb3e18de94315bfab99e0cfc624032abc08753e536'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def validate_pcd_resume_counts(resume):
    if resume.get('global_optimizer_steps')!=30000 or resume.get('global_micro_steps')!=479790 or resume.get('sampler',{}).get('committed')!=706560:raise ValueError('Actual PCD snapshot must be exact30000/479790/706560')
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,path)
def contained(root,relative):
    p=(Path(root)/relative).resolve()
    if not p.is_relative_to(Path(root).resolve()):raise ValueError('Checkpoint path escapes owner run')
    return p

def validate_completed_run(train,launch,code,data_index,data_seal,selection,observation_modality='pixels',precision='fp32',optimizer_updates=None,trainer_source=None):
    if observation_modality=='pcd':return validate_completed_pcd_run(train,launch,code,data_index,data_seal,selection,precision,optimizer_updates,trainer_source)
    if observation_modality!='pixels' or precision!='fp32' or optimizer_updates is not None:raise ValueError('Default RGB evaluation contract must stay FP32 completed10ep')
    train,launch,code=Path(train).resolve(),Path(launch).resolve(),Path(code).resolve()
    status=json.loads((train/'status.json').read_text());protocol=json.loads((train/'protocol.json').read_text());manifest=json.loads((train/'checkpoint_manifest.json').read_text());ph=canonical_sha(protocol)
    if selection not in ['best','last']:raise ValueError('Selected completed model must be best or last')
    if (train/'exit_code').read_text().strip()!='0' or any((launch/n).read_text().strip()!='0' for n in ['driver_exit_code','child_exit_code']):raise ValueError('Real trainer/child/driver exits must all be0')
    if any(status.get(k)!=v for k,v in {'status':'passed','exit_code':0,'completed_epochs':10,'global_micro_steps':156250,'global_optimizer_steps':9770,'protocol_sha256':ph}.items()):raise ValueError('Only completed10ep/9770update/156250micro run may enter native evaluation')
    if manifest.get('protocol_sha256')!=ph or len(manifest.get('epochs',[]))!=10 or [x['epoch'] for x in manifest['epochs']]!=list(range(1,11)):raise ValueError('Ten completed committed epoch entries required')
    factory=protocol['factory'];expected={'_target_':'policy.create_policy','modality':'pixels','history_size':1,'latent_dim':192,'action_dim':5,'seed':0}
    if any(factory.get(k)!=v for k,v in expected.items()) or set(factory)!=set(expected)|{'source_checkpoint'}:raise ValueError('Wrong or old history3/Transformer/source factory')
    if protocol.get('precision')!='fp32' or protocol.get('loss')!='unit std Gaussian NLL meanB sumA5' or protocol.get('optimizer')!='Adam' or protocol.get('lr')!=.0003 or protocol.get('weight_decay')!=0:raise ValueError('Wrong final rawBox/unitStd/AdamFP32 training method')
    source=Path(factory['source_checkpoint']).resolve()
    checks=[(source,protocol['source_checkpoint_sha256']),(source.parent/'config.yaml',protocol['source_config_sha256']),(code/'policy.py',protocol['policy_source_sha256']),(code/'dataset.py',protocol['dataset_source_sha256']),(code/'trainer.py',protocol['trainer_source_sha256']),(Path(data_index),protocol['data_index_sha256']),(Path(data_seal),protocol['data_seal_sha256'])]
    for p,expected_sha in checks:
        if sha(p)!=expected_sha:raise ValueError('Sealed training/data/source file changed:'+str(p))
    best=min(manifest['epochs'],key=lambda entry:entry['metrics']['val_nll'])
    if manifest['best']!=best or manifest['last']!=manifest['epochs'][-1]:raise ValueError('Best/last selection not actual full-validation completed epochs')
    entry=manifest[selection]
    if entry.get('train_samples')!=1000000 or entry.get('validation_samples')!=100000 or entry.get('global_optimizer_steps')!=entry['epoch']*977 or entry.get('global_micro_steps')!=entry['epoch']*15625:raise ValueError('Selected model is not a full epoch checkpoint')
    model=contained(train,entry['modelonly']);resume=contained(train,entry['resume']);config=model.parent/'config.json'
    if sha(model)!=entry['modelonly_sha256'] or sha(resume)!=entry['resume_sha256'] or json.loads(config.read_text())!=factory:raise ValueError('Committed model/resume/factory SHA differs')
    return {'train_root':str(train),'launch_root':str(launch),'code_root':str(code),'selection':selection,'epoch':entry['epoch'],'protocol_sha256':ph,'training_protocol':protocol,'factory':factory,'entry':entry,
       'modelonly':str(model),'modelonly_sha256':entry['modelonly_sha256'],'resume':str(resume),'resume_sha256':entry['resume_sha256'],'factory_config_sha256':sha(config),
       'data_index':str(Path(data_index).resolve()),'data_seal':str(Path(data_seal).resolve()),'manifest_sha256':sha(train/'checkpoint_manifest.json'),'model_loaded_or_native_run':False,
       'best_last_same_model':manifest['best']['modelonly_sha256']==manifest['last']['modelonly_sha256']}


def validate_completed_pcd_run(train,launch,code,data_index,data_seal,selection,precision,optimizer_updates,trainer_source):
    train,launch,code=Path(train).resolve(),Path(launch).resolve(),Path(code).resolve()
    if selection!='update30000' or optimizer_updates!=30000 or precision!='bf16' or trainer_source is None:raise ValueError('PCD selection must explicitly be BF16 exact update30000, with actual trainer source')
    status=json.loads((train/'status.json').read_text());protocol=json.loads((train/'protocol.json').read_text());manifest=json.loads((train/'checkpoint_manifest.json').read_text());ph=canonical_sha(protocol)
    parent=protocol['continuation_parent'];continuation=json.loads((train/'continuation_receipt.json').read_text())
    if continuation.get('status')!='accepted_parent_continuation' or continuation.get('exit_code')!=0 or continuation.get('parent')!=parent or continuation.get('child_protocol_sha256')!=ph or continuation.get('child_trainer_source_sha256')!=protocol['trainer_source_sha256'] or continuation.get('restored_head_optimizer_rng_sampler') is not True or parent.get('start_optimizer_updates')!=27000:raise ValueError('Actual accepted27000 continuation lineage required')
    if (train/'exit_code').read_text().strip()!='0' or any((launch/n).read_text().strip()!='0' for n in ['driver_exit_code','child_exit_code']):raise ValueError('Real completed continuation trainer/child/driver exits must be0')
    required={'status':'passed','exit_code':0,'global_optimizer_steps':30000,'global_micro_steps':479790,'completed_epochs':30,'partial_epoch_number':31,'partial_epoch_train_samples':706560,'validation_consumed_samples':100000,'protocol_sha256':ph}
    if any(status.get(k)!=v for k,v in required.items()) or manifest.get('protocol_sha256')!=ph:raise ValueError('Exact completed30000 full+partial counters required')
    factory=protocol['factory'];expected={'_target_':'policy.create_policy','modality':'pcd','history_size':1,'latent_dim':192,'action_dim':5,'seed':0}
    if any(factory.get(k)!=v for k,v in expected.items()) or set(factory)!=set(expected)|{'source_checkpoint'}:raise ValueError('Wrong PCD singleframe factory')
    if protocol.get('precision')!='bf16' or protocol.get('loss')!='unit std Gaussian NLL meanB sumA5' or protocol.get('optimizer')!='Adam' or protocol.get('lr')!=.0003 or protocol.get('weight_decay')!=0:raise ValueError('Wrong BF16/rawBox/unitStd/Adam training method')
    source=Path(factory['source_checkpoint']).resolve();trainer_source=Path(trainer_source).resolve()
    checks=[(source,protocol['source_checkpoint_sha256']),(source.parent/'config.yaml',protocol['source_config_sha256']),(source.parent/'config.json',protocol['source_config_json_sha256']),
        (code/'policy.py',protocol['policy_source_sha256']),(code/'dataset_pcd.py',protocol['dataset_source_sha256']),(trainer_source,protocol['trainer_source_sha256']),
        (Path(data_index),protocol['data_index_sha256']),(Path(data_seal),protocol['data_seal_sha256'])]
    for path,expected_sha in checks:
        if sha(path)!=expected_sha:raise ValueError('Sealed PCD source/data changed:'+str(path))
    entry=manifest['last']
    expected_entry={'global_optimizer_steps':30000,'global_micro_steps':479790,'epoch':31,'partial_epoch':True,'epoch_completed':False,'completed_full_epochs':30,'train_samples':706560,'validation_samples':100000}
    if any(entry.get(k)!=v for k,v in expected_entry.items()):raise ValueError('Last must be exact30000 partial export, not best/old checkpoint')
    model=contained(train,entry['modelonly']);resume=contained(train,entry['resume']);config=model.parent/'config.json'
    if sha(model)!=entry['modelonly_sha256'] or sha(resume)!=entry['resume_sha256'] or json.loads(config.read_text())!=factory:raise ValueError('Exact30000 model/resume/factory differs')
    return {'train_root':str(train),'launch_root':str(launch),'code_root':str(code),'selection':selection,'epoch':31,'resume_epoch':30,'protocol_sha256':ph,'training_protocol':protocol,'factory':factory,'entry':entry,
        'modelonly':str(model),'modelonly_sha256':entry['modelonly_sha256'],'resume':str(resume),'resume_sha256':entry['resume_sha256'],'factory_config_sha256':sha(config),'data_index':str(Path(data_index).resolve()),
        'data_seal':str(Path(data_seal).resolve()),'manifest_sha256':sha(train/'checkpoint_manifest.json'),'model_loaded_or_native_run':False,'best_last_same_model':False,
        'observation_modality':'pcd','observation_precision':'bf16','optimizer_updates':30000,'trainer_source':str(trainer_source),'prefix_lineage':parent,'continuation_receipt_sha256':sha(train/'continuation_receipt.json')}

def validate_episode_set(rows,expected_count=250):
    if len(rows)!=expected_count:raise ValueError('Missing complete real episode records')
    seeds=set();keys=set()
    for row in rows:
        task,index=row['task'],row['index'];expected=970000000+task*100000+index
        if not 1<=task<=5 or not 0<=index<50 or row['seed']!=expected or expected in seeds or (task,index) in keys:raise ValueError('Repeated/wrong task or paired reference seed')
        if not 1<=row['steps']<=200 or row['policy_warmup_steps']!=0 or row['model_calls']!=row['steps'] or len(row['last_ctrl7'])!=7 or row['success'] not in [0.,1.]:raise ValueError('Actual step/control/no-warmup contract failed')
        seeds.add(expected);keys.add((task,index))
    return True

def configure(a):
    os.environ['MUJOCO_GL']='egl';os.environ['JAX_PLATFORMS']='cpu'
    repo=a.repo_root/'lewm_pcd';sys.path[:0]=[str(a.code_root),str(a.official_source),str(a.smoke_utils_root),str(repo),str(repo/'third_party/stable-worldmodel'),str(a.official_root)]
    # Explicit pinned file import avoids generic utils namespace collisions.
    import torch,policy
    if Path(policy.__file__).resolve()!=(a.code_root/'policy.py').resolve() or sha(policy.__file__)!=POLICY_SHA:raise ValueError('Wrong new source-factory policy module')
    torch.set_num_threads(1)
    configure_numpy_physics_runtime()
    return torch,policy

_BLAS_LIMITER=None
def numpy_blas_paths(pools,numpy_file=None):
    owner=(Path(numpy_file or np.__file__).resolve().parent.parent/'numpy.libs').resolve()
    paths=[p['filepath'] for p in pools if p['user_api']=='blas' and Path(p['filepath']).resolve().parent==owner]
    if len(paths)!=1:raise ValueError('Exactly one actual NumPy wheel-linked BLAS library required')
    return paths

def configure_numpy_physics_runtime():
    """Match original state evaluator IK math; keep Torch threads separately1."""
    global _BLAS_LIMITER
    from threadpoolctl import threadpool_info,ThreadpoolController
    before=threadpool_info();paths=numpy_blas_paths(before)
    _BLAS_LIMITER=ThreadpoolController().select(filepath=paths).limit(limits=4,user_api='blas')
    after=threadpool_info();pools=[p for p in after if p['filepath'] in paths]
    if len(pools)!=1 or pools[0]['num_threads']!=4:raise ValueError('Original state NumPy-linked BLAS physics runtime requires4 threads')
    others=[p for p in after if p['filepath'] not in paths]
    if {p['filepath']:p['num_threads'] for p in before if p['filepath'] not in paths}!={p['filepath']:p['num_threads'] for p in others}:raise ValueError('Targeted NumPy limit unexpectedly changed other runtime pools')
    torch_threads=sys.modules['torch'].get_num_threads() if 'torch' in sys.modules else None
    if torch_threads is not None and torch_threads!=1:raise ValueError('Torch inference must remain1 thread')
    return {'numpy_version':np.__version__,'numpy_path':np.__file__,'BLAS_threads':4,'threadpools':pools,'other_threadpools_record_only':others,'library_sha256':{p['filepath']:sha(p['filepath']) for p in pools},'torch_num_threads':torch_threads,'reference':'official_state_sr NumPy BLAS4 physics; other BLAS/OpenMP pools are recorded, not constrained to4'}

def cpu_overlay_path(a):return Path(a.cpu_deps if a.cpu_deps is not None else a.official_source.parent/'cpu-deps').resolve()

def seal_runtime_dependencies(a,protocol):
    overlay=cpu_overlay_path(a);wm=Path(protocol['factory']['source_checkpoint'])
    files=[wm.parent/'config.json',a.smoke_utils_root/'lewm_gcbc_adapter.py',a.smoke_utils_root/'gcbc_smoke.py',a.official_root/'official_state_sr.py',
           overlay.parent/'cpu-requirements.txt',overlay.parent/'cpu-deps.freeze.txt']
    if sha(files[0])!=(protocol['source_config_json_sha256'] if protocol['factory'].get('modality','pixels')=='pcd' else WM_CONFIG_JSON_SHA) or sha(files[1])!=protocol['frozen_adapter_sha256'] or sha(files[2])!=HASH_HELPER_SHA:raise ValueError('Actual pretrained/encoder/hash dependency differs from training source seal')
    for name in ['jax','jaxlib','numpy','scipy','ml_dtypes','opt_einsum']:
        candidates=list(overlay.glob(name+'-*.dist-info/METADATA'))
        if len(candidates)!=1:raise ValueError('CPU overlay metadata missing or ambiguous:'+name)
        files.append(candidates[0])
    if protocol['factory'].get('modality','pixels')=='pcd':
        files += [a.repo_root/'lewm_pcd/envs/pcd_wrapper.py',a.official_root/'pcd_full_v1/pcd_geometry.py',Path(__file__).with_name('native_pcd_frame.py')]
        from native_pcd_frame import CORE_SHA,ADAPTER_SHA
        if sha(files[-3])!=CORE_SHA or sha(files[-2])!=ADAPTER_SHA:raise ValueError('Frozen main/provider differs')
        files.append(a.train_root/'continuation_receipt.json')
    return {str(p.resolve()):sha(p) for p in files}

def load_original_evaluation(source,cpu_deps):
    os.environ['JAX_PLATFORMS']='cpu';cpu_deps=Path(cpu_deps).resolve()
    if str(cpu_deps) not in sys.path:sys.path.append(str(cpu_deps))
    import jax,jaxlib
    if not Path(jax.__file__).resolve().is_relative_to(cpu_deps) or not Path(jaxlib.__file__).resolve().is_relative_to(cpu_deps) or any(x.platform!='cpu' for x in jax.devices()):raise ValueError('Original evaluate requires sealed CPU-only JAX overlay')
    spec=importlib.util.spec_from_file_location('original_official_evaluate',Path(source)/'impls/utils/evaluation.py');evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
    return evaluation

def load_bound_model(binding,a,device='cpu'):
    torch,policy=configure(a)
    from gcbc_smoke import state_hash
    pcd=binding.get('observation_modality','pixels')=='pcd'
    if sha(a.code_root/'dataset.py')!=DATASET_SHA or sha(a.code_root/'trainer.py')!=TRAINER_SHA:raise ValueError('Frozen base training source changed')
    if pcd and (sha(a.code_root/'dataset_pcd.py')!=binding['training_protocol']['dataset_source_sha256'] or sha(binding['trainer_source'])!=binding['training_protocol']['trainer_source_sha256']):raise ValueError('Frozen PCD continuation source changed')
    source=binding['factory']['source_checkpoint']
    if pcd and binding['training_protocol']['source_checkpoint_sha256']!='ab9f2e69d58eea4f4d8f469a7dd259c5944f45d0d54ec20061cf63c7a9630d03':raise ValueError('Different frozen DexArt source checkpoint')
    if sha(source)!=binding['training_protocol']['source_checkpoint_sha256'] or sha(binding['modelonly'])!=binding['modelonly_sha256'] or sha(binding['resume'])!=binding['resume_sha256']:raise ValueError('Bound selected/source checkpoint changed')
    model=policy.create_policy(**{k:v for k,v in binding['factory'].items() if k!='_target_'}).to(device);initial_frozen=state_hash(model.encoder);full=torch.load(binding['modelonly'],map_location='cpu',weights_only=True);missing=model.load_state_dict(full,strict=True)
    resume=torch.load(binding['resume'],map_location='cpu',weights_only=False)
    if pcd:validate_pcd_resume_counts(resume)
    import trainer
    if Path(trainer.__file__).resolve()!=(a.code_root/'trainer.py').resolve():raise ValueError('Wrong source trainer hash implementation')
    training_hash=trainer.module_hash(model.encoder)
    if missing.missing_keys or missing.unexpected_keys or resume['protocol_sha256']!=binding['protocol_sha256'] or resume['epoch']!=binding.get('resume_epoch',binding['epoch']) or state_hash(model.encoder)!=(initial_frozen if pcd else FROZEN_SHA) or resume['frozen_state_sha256']!=training_hash or binding['entry']['frozen_state_sha256']!=training_hash:raise ValueError('Fresh full model/frozen/protocol mismatch')
    actual=model.action_head.state_dict()
    if set(actual)!=set(resume['head']):raise ValueError('Resume/full-model head keys differ')
    for key,value in resume['head'].items():torch.testing.assert_close(actual[key].detach().cpu(),value,rtol=0,atol=0)
    if len(resume['optimizer']['param_groups'])!=1 or resume['optimizer']['param_groups'][0]['lr']!=.0003 or resume['optimizer']['param_groups'][0]['weight_decay']!=0:raise ValueError('Selected resume optimizer storage is not original Adam contract')
    model.eval().requires_grad_(False)
    physics_runtime=configure_numpy_physics_runtime()
    if 'physics_numpy_runtime' in binding and {k:v for k,v in physics_runtime.items() if k!='other_threadpools_record_only'}!={k:v for k,v in binding['physics_numpy_runtime'].items() if k!='other_threadpools_record_only'}:raise ValueError('Bound exact physics NumPy/BLAS/Torch runtime changed')
    receipt={'fresh_source_factory_full_state_strict':True,'frozen_all_state_sha256':state_hash(model.encoder),'training_frozen_all_state_sha256':training_hash,'hash_algorithm_labels':{'frozen_all_state_sha256':'gcbc_smoke.state_hash','training_frozen_all_state_sha256':'trainer.module_hash'},'head_sha256':state_hash(model.action_head),'full_policy_sha256':state_hash(model),'physics_numpy_runtime':physics_runtime,
      'resume_head_exact':True,'resume_storage_sha256':binding['resume_sha256'],'model_storage_sha256':binding['modelonly_sha256'],'device':device,'model_loaded_or_native_run':True}
    return torch,model,receipt

def validate_binding_seal(binding,code_root,official_root,runner_source=__file__):
    payload={k:v for k,v in binding.items() if k!='evaluation_protocol_sha256'}
    if canonical_sha(payload)!=binding.get('evaluation_protocol_sha256'):raise ValueError('Evaluation binding content/hash changed')
    checks=[(Path(runner_source),'native_runner_source_sha256'),
            (Path(official_root)/'native_single_frame_adapter.py','adapter_source_file_sha256'),
            (Path(code_root)/'native_appearance_allowlist_v3.py','appearance_allowlist_source_file_sha256'),
            (Path(official_root)/'cube_state_rgb_validation_v1/config.json','source_RGB_recipe_sha256')]
    for path,key in checks:
        if sha(path)!=binding.get(key):raise ValueError('Bound evaluation source/recipe changed:'+str(path))
    for path,expected in binding.get('runtime_dependency_files',{}).items():
        if sha(path)!=expected:raise ValueError('Bound evaluation runtime dependency changed:'+path)
    for path,expected in binding.get('physics_numpy_runtime',{}).get('library_sha256',{}).items():
        if sha(path)!=expected:raise ValueError('Bound BLAS binary changed:'+path)

def check_binding(path,a):
    binding=json.loads(Path(path).read_text())
    validate_binding_seal(binding,a.code_root,a.official_root)
    if seal_runtime_dependencies(a,binding['training_protocol'])!=binding['runtime_dependency_files'] or str(cpu_overlay_path(a))!=binding['cpu_deps']:raise ValueError('Evaluation runtime dependency paths differ after binding')
    current=validate_completed_run(binding['train_root'],binding['launch_root'],a.code_root,a.data_index,a.data_seal,binding['selection'],binding.get('observation_modality','pixels'),binding.get('observation_precision','fp32'),binding.get('optimizer_updates'),binding.get('trainer_source'))
    for key in ['modelonly_sha256','resume_sha256','protocol_sha256','manifest_sha256']:
        if current[key]!=binding[key]:raise ValueError('Committed owner changed after bind:'+key)
    if binding.get('bound_success') is not True:raise ValueError('Fresh source-factory reload has not passed')
    return binding

def validate_previous_episode(row,binding,task,index,diagnostic):
    expected={'status':'passed','exit_code':0,'task':task,'index':index,'seed':970000000+task*100000+index,'diagnostic_only':diagnostic,'score_eligible':not diagnostic,
              'modelonly_sha256':binding['modelonly_sha256'],'evaluation_protocol_sha256':binding['evaluation_protocol_sha256']}
    if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('Previous episode case/model identity differs')
    validate_episode_set([row],expected_count=1)
    if diagnostic and row['steps']>4:raise ValueError('Tiny diagnostic exceeded4 steps')

def finish_episode(attempt,result_path,result,status,close):
    try:
        close()
        write(attempt/'result.json',result);status.update(status='passed',exit_code=0);write(attempt/'status.json',status);(attempt/'exit_code').write_text('0\n')
        write(result_path,result)
    except BaseException:
        status.update(status='failed',exit_code=1,error=traceback.format_exc());write(attempt/'status.json',status);(attempt/'exit_code').write_text('1\n');raise

def close_owned_environment(target,extra_close=None):
    """Close only this episode's lazy current renderer, then its environment."""
    u=target.unwrapped;renderer=getattr(u,'_renderer',None);errors=[]
    if extra_close is not None:
        try:extra_close()
        except BaseException:errors.append({'stage':'owned_NoAA_renderer.close','error':traceback.format_exc()})
    if renderer is not None:
        try:renderer.close()
        except BaseException:errors.append({'stage':'owned_current_renderer.close','error':traceback.format_exc()})
        finally:u._renderer=None
    try:target.close()
    except BaseException:errors.append({'stage':'owned_environment.close','error':traceback.format_exc()})
    if errors:raise RuntimeError('Owned episode cleanup failed: '+json.dumps(errors))

def run_episode(binding,a,model,torch,task,index,diagnostic):
    from gcbc_smoke import state_hash
    from native_single_frame_adapter import SingleFrameAgent,ReplayPairWrapper,SafeGoalRenderer,seal_original_source
    from native_appearance_allowlist_v3 import AppearanceAllowlist
    from official_state_sr import physical_snapshot,model_snapshot
    import ogbench,mujoco,hydra
    from omegaconf import OmegaConf
    root=a.output/('tiny' if diagnostic else f'task{task}')/f'episode{index:03d}';root.mkdir(parents=True,exist_ok=True);result_path=root/'result.json'
    if result_path.exists():
        previous=json.loads(result_path.read_text())
        validate_previous_episode(previous,binding,task,index,diagnostic)
        previous_folder=root/previous['attempt']
        if previous['status']!='passed' or previous['modelonly_sha256']!=binding['modelonly_sha256'] or previous['evaluation_protocol_sha256']!=binding['evaluation_protocol_sha256'] or sha(previous_folder/'steps.jsonl')!=previous['steps_sha256'] or sha(previous_folder/'reset.json')!=previous['reset_sha256'] or sha(previous_folder/'appearance.json')!=previous['appearance_receipt_sha256'] or (previous_folder/'exit_code').read_text().strip()!='0':raise ValueError('Prior completed result binding differs')
        return previous
    attempt=root/f'attempt_{len(list(root.glob("attempt_*")))+1:04d}';attempt.mkdir(exist_ok=False);status={'status':'running','exit_code':None,'task':task,'index':index,'diagnostic_only':diagnostic};write(attempt/'status.json',status);env=wrapper=None;pcd_frame=None;committed=False
    try:
        source=a.official_root/'official_state_final_sr_seed0_v1/metadata_snapshot_v1';reference_path=source/f'task{task}'/f'episode{index:03d}'/'reset.json';reference=json.loads(reference_path.read_text());seed=970000000+task*100000+index
        if sha(reference_path)!=binding['reference_manifest'][str(task)+':'+str(index)]['reset_file_sha256']:raise ValueError('Paired reference changed after bind')
        seal_original_source(a.official_source,json.loads((source/'protocol.json').read_text()))
        cfg=OmegaConf.load(Path(binding['factory']['source_checkpoint']).parent/'config.yaml');preproc=hydra.utils.instantiate(cfg.data.preprocessor)
        def transform(rgb):return preproc({'pixels':torch.from_numpy(rgb.copy()).permute(2,0,1)[None]})['pixels']
        env=ogbench.make_env_and_datasets('cube-single-play-v0',env_only=True,ob_type='states',width=224,height=224,mode='task',visualize_info=False,terminate_at_goal=True,max_episode_steps=200)
        modality=binding.get('observation_modality','pixels');precision=binding.get('observation_precision','fp32');goal_renderer=SafeGoalRenderer()
        current=lambda:env.unwrapped.render(camera='front_pixels')
        if modality=='pcd':
            from native_pcd_frame import NativePCDFrame
            pcd_frame=NativePCDFrame(a.repo_root/'lewm_pcd/envs/pcd_wrapper.py',a.official_root/'pcd_full_v1/pcd_geometry.py',seed)
            def transform(cloud):return preproc({'pcd_xyz':torch.from_numpy(cloud[:,:3].copy())[None],'pcd_rgb':torch.from_numpy(cloud[:,3:].astype(np.uint8))[None]})['pcd']
            def current():
                u=env.unwrapped;before=(physical_snapshot(u),model_snapshot(u),json.loads(json.dumps(u.np_random.bit_generator.state)))
                cloud=pcd_frame.capture(u,u._data,u.render(camera='front_pixels'),'current')
                if before!=(physical_snapshot(u),model_snapshot(u),u.np_random.bit_generator.state):raise ValueError('Current PCD capture mutated physical/private/model/RNG state')
                return cloud
            goal_renderer=SafeGoalRenderer(frame_transform=lambda u,data,rgb:pcd_frame.capture(u,data,rgb,'goal'))
        actor=SingleFrameAgent(model,current,transform,device=a.device,precision=precision,observation_modality=modality)
        appearance=AppearanceAllowlist(reference['model_colors_camera_lights'],a.official_root/'cube_state_rgb_validation_v1/config.json',model_snapshot,lambda m,g:mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g))
        with (attempt/'steps.jsonl').open('x',buffering=1) as stepfile:
            def on_reset(receipt):
                if pcd_frame is not None:receipt['PCD_goal_frame']=pcd_frame.last_receipts['goal']
                write(attempt/'reset.json',receipt);write(attempt/'appearance.json',appearance.receipt)
            def on_step(row):
                row['mean_unclipped_normalized_box']=actor.last_mean.tolist()
                if pcd_frame is not None:row['PCD_current_frame']=pcd_frame.last_receipts['current']
                stepfile.write(json.dumps(row,allow_nan=False)+'\n');stepfile.flush()
            wrapped=ReplayPairWrapper(env,reference,actor,goal_renderer,model_snapshot_fn=appearance,on_reset=on_reset,on_step=on_step)
            class Cutoff:
                def __getattr__(self,key):return getattr(wrapped,key)
                def reset(self,**kw):return wrapped.reset(**kw)
                def step(self,action):
                    ob,reward,term,trunc,info=wrapped.step(action)
                    if diagnostic and wrapped.count>=4 and not(term or trunc):trunc=True;info=dict(info,diagnostic_only_stop=True)
                    return ob,reward,term,trunc,info
            wrapper=wrapped;before=state_hash(model)
            evaluation=load_original_evaluation(a.official_source,cpu_overlay_path(a))
            with (attempt/'run.log').open('x',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                np.random.seed(seed);stats,trajs,_=evaluation.evaluate(actor,Cutoff(),task_id=task,config={'discrete':False},num_eval_episodes=1,num_video_episodes=0,eval_temperature=0,eval_gaussian=None)
            if wrapped.count!=actor.model_calls or not 1<=wrapped.count<=(4 if diagnostic else 200) or state_hash(model)!=before or state_hash(model.encoder)!=(binding['fresh_reload']['frozen_all_state_sha256'] if modality=='pcd' else FROZEN_SHA):raise ValueError('Real policy steps/frozen model gate failed')
        last=physical_snapshot(env.unwrapped);result={'status':'passed','exit_code':0,'task':task,'index':index,'seed':seed,'steps':wrapped.count,'model_calls':actor.model_calls,'policy_warmup_steps':0,
          'success':float(stats['success']),'diagnostic_only':diagnostic,'score_eligible':not diagnostic,'last_ctrl7':last['ctrl']['values'],'modelonly_sha256':binding['modelonly_sha256'],
          'evaluation_protocol_sha256':binding['evaluation_protocol_sha256'],'attempt':attempt.name,'steps_sha256':sha(attempt/'steps.jsonl'),'reset_sha256':sha(attempt/'reset.json'),'appearance_receipt_sha256':sha(attempt/'appearance.json'),
          'appearance_allowlist_source_file_sha256':sha(a.code_root/'native_appearance_allowlist_v3.py'),'whole_model_visual_exact_claim':False,'frozen_all_state_exact':True}
        closing=wrapper if wrapper is not None else env;wrapper=env=None
        finish_episode(attempt,result_path,result,status,lambda:close_owned_environment(closing,pcd_frame.close if pcd_frame is not None else None));committed=True;return result
    except BaseException:status.update(status='failed',exit_code=1,error=traceback.format_exc());raise
    finally:
        if not committed:
            try:
                if wrapper is not None:close_owned_environment(wrapper,pcd_frame.close if pcd_frame is not None else None)
                elif env is not None:close_owned_environment(env,pcd_frame.close if pcd_frame is not None else None)
            except BaseException:status.update(status='failed',exit_code=1,close_error=traceback.format_exc())
            write(attempt/'status.json',status);(attempt/'exit_code').write_text(str(status['exit_code'])+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['bind','tiny','task','aggregate'],required=True)
    for key in ['train-root','launch-root','code-root','data-index','data-seal','repo-root','smoke-utils-root','official-source','official-root','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--selection',choices=['best','last','update30000'],required=True);p.add_argument('--binding',type=Path);p.add_argument('--task',type=int,choices=range(1,6));p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--observation-modality',choices=['pixels','pcd'],default='pixels');p.add_argument('--precision',choices=['fp32','bf16'],default='fp32');p.add_argument('--optimizer-updates',type=int);p.add_argument('--trainer-source',type=Path)
    p.add_argument('--cpu-deps',type=Path,help='Existing CPU JAX overlay; defaults to official-source parent/cpu-deps')
    p.add_argument('--allow-native',action='store_true',help='Only after completed2D/model reload and explicit root simulator/GPU handoff')
    a=p.parse_args();a.code_root=a.code_root.resolve();a.output=a.output.resolve()
    if a.stage=='bind':
        a.output.mkdir(exist_ok=False);binding=validate_completed_run(a.train_root,a.launch_root,a.code_root,a.data_index,a.data_seal,a.selection,a.observation_modality,a.precision,a.optimizer_updates,a.trainer_source)
        binding['runtime_dependency_files']=seal_runtime_dependencies(a,binding['training_protocol']);binding['cpu_deps']=str(cpu_overlay_path(a))
        torch,model,reload=load_bound_model(binding,a,'cpu')
        from native_single_frame_adapter import IDENTITY_PROTOCOL,validate_references,seal_original_source
        refs=a.official_root/'official_state_final_sr_seed0_v1/metadata_snapshot_v1';source=seal_original_source(a.official_source,json.loads((refs/'protocol.json').read_text()))
        record=validate_references(refs);binding['reference_manifest']={str(r['task'])+':'+str(r['index']):r for r in record}
        binding.update(bound_success=True,fresh_reload=reload,physics_numpy_runtime=reload['physics_numpy_runtime'],evaluation_action_contract=IDENTITY_PROTOCOL,original_source_seal=source,
          appearance_allowlist_source_file_sha256=sha(a.code_root/'native_appearance_allowlist_v3.py'),adapter_source_file_sha256=sha(a.official_root/'native_single_frame_adapter.py'),native_runner_source_sha256=sha(__file__),
          source_RGB_recipe_sha256=sha(a.official_root/'cube_state_rgb_validation_v1/config.json'),native_episodes_run=0)
        binding['evaluation_protocol_sha256']=canonical_sha(binding);write(a.output/'binding.json',binding);write(a.output/'status.json',{'status':'bound_completed_policy_CPU','exit_code':0,'native_episodes_run':0});(a.output/'exit_code').write_text('0\n');return 0
    if a.binding is None:raise ValueError('Completed/fresh bound selection required')
    binding=check_binding(a.binding,a)
    if binding.get('observation_modality','pixels')!=a.observation_modality or binding.get('observation_precision','fp32')!=a.precision:raise ValueError('Native input modality/precision changed after binding')
    if binding['selection']!=a.selection or binding['native_runner_source_sha256']!=sha(__file__):raise ValueError('Selected/source eval protocol changed')
    if a.stage=='aggregate':
        rows=[json.loads((a.output/f'task{task}'/f'episode{index:03d}'/'result.json').read_text()) for task in range(1,6) for index in range(50)]
        validate_episode_set(rows)
        for row in rows:
            if row['diagnostic_only'] or row['status']!='passed' or row['exit_code']!=0 or row['evaluation_protocol_sha256']!=binding['evaluation_protocol_sha256'] or row['modelonly_sha256']!=binding['modelonly_sha256']:raise ValueError('Unaccepted/incomplete/mixed-checkpoint episode')
            folder=a.output/f"task{row['task']}"/f"episode{row['index']:03d}"/row['attempt']
            if (folder/'exit_code').read_text().strip()!='0' or sha(folder/'steps.jsonl')!=row['steps_sha256'] or sha(folder/'reset.json')!=row['reset_sha256'] or sha(folder/'appearance.json')!=row['appearance_receipt_sha256']:raise ValueError('Saved episode artifacts changed')
        write(a.output/'aggregate.json',{'status':'passed','exit_code':0,'selection':a.selection,'episodes':250,'modelonly_sha256':binding['modelonly_sha256'],'evaluation_protocol_sha256':binding['evaluation_protocol_sha256'],'success_rate':sum(r['success'] for r in rows)/250,
          'per_task':{str(t):sum(r['success'] for r in rows if r['task']==t)/50 for t in range(1,6)},'best_last_same_model':binding['best_last_same_model'],'tiny_gates_excluded':True});return 0
    if not a.allow_native:raise ValueError('Native stages require explicit approved handoff flag; do not auto-evaluate')
    import fcntl
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'native.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch,model,reload=load_bound_model(binding,a,a.device)
        if a.stage=='tiny':
            result=run_episode(binding,a,model,torch,1,0,True);write(a.output/'tiny_gate.json',result);return 0
        gate=json.loads((a.output/'tiny_gate.json').read_text())
        if not gate['diagnostic_only'] or gate['status']!='passed' or gate['modelonly_sha256']!=binding['modelonly_sha256'] or gate['evaluation_protocol_sha256']!=binding['evaluation_protocol_sha256']:raise ValueError('Matching tiny selected-checkpoint native gate required')
        if a.task is None:raise ValueError('Task id required')
        for index in range(50):run_episode(binding,a,model,torch,a.task,index,False)
    return 0

if __name__=='__main__':raise SystemExit(main())
