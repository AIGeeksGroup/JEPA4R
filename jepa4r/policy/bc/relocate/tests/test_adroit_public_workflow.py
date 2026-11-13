# Moved from OGBench-cube@4fbb0c6:lewm_pcd/scripts/checks/test_adroit_public_workflow.py (verbatim except: module names / paths of the moved code adapted)
"""Focused contract checks for the public native Relocate workflow."""
import importlib.util
import importlib
import ast
from contextlib import nullcontext
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
import h5py
import pytest

ROOT = Path(__file__).resolve().parents[5]


def load(name):
    path = ROOT / 'jepa4r/policy/bc/relocate' / (name + '.py')
    assert path.is_file(), 'the public Native Relocate adapter is not implemented'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(tmp_path, model='jepa4r', mode='smoke'):
    return SimpleNamespace(env='adroit', model=model, mode=mode, output=tmp_path/'new',
                           wm_python='wm-python', sim_python='sim-python', seed=42)


def test_smoke_connects_real_collection_training_reload_and_online_bc(tmp_path):
    workflow = load('adroit_workflow')
    stages = workflow.plan(args(tmp_path), {'adroit_expert':'expert.pkl',
        'adroit_official_demos':'official', 'adroit_official_sha256':'a'*64})
    assert [s['name'] for s in stages] == ['collect', 'data-audit', 'wm-train',
        'wm-reload', 'official-demos', 'bc-seed100', 'policy-reload-seed100']
    assert '--train-episodes' in stages[0]['command']
    assert stages[0]['command'][stages[0]['command'].index('--train-episodes')+1] == '2'
    assert stages[0]['command'][stages[0]['command'].index('--test-episodes')+1] == '1'
    assert all(isinstance(s['command'],list) and s['cwd'] for s in stages)
    assert not (tmp_path/'new').exists(), 'planning must be read-only'


def test_full_keeps_three_paper_seeds_and_25_online_trajectories(tmp_path):
    workflow = load('adroit_workflow')
    stages = workflow.plan(args(tmp_path, 'lewm', 'full'), {})
    bc = [s for s in stages if s['name'].startswith('bc-seed')]
    assert [s['name'] for s in bc] == ['bc-seed100','bc-seed200','bc-seed300']
    assert all('eval_num_traj=25' in s['command'] and 'epochs=100' in s['command'] for s in bc)
    assert '--train-episodes' in stages[0]['command']
    assert stages[0]['command'][stages[0]['command'].index('--train-episodes')+1] == '10000'


def make_pair(root):
    root.mkdir()
    rows=np.tile(np.arange(20,dtype=np.float32),2)
    states=np.repeat(rows[:,None],39,axis=1)
    image=np.tile(np.arange(256,dtype=np.uint8)[None,:,None],(256,1,3))
    cloud=np.zeros((1200,3),np.float32);cloud[:,0]=np.linspace(0,1,1200)
    for route in ('rgb','pcd'):
        with h5py.File(root/(route+'.h5'),'w') as f:
            f['ep_len']=[20,20]; f['ep_offset']=[0,20]
            f['action']=np.zeros((40,30),np.float32)
            f['observation']=states;f['next_observation']=states+1;f['proprio']=states[:,:30]
            f['qpos']=np.zeros((40,36)); f['qvel']=np.zeros((40,36))
            f['seed']=np.repeat([42,43],20)
            f['step_idx']=np.tile(np.arange(20,dtype=np.int64),2)
            f['initial_state_sha256']=np.repeat([b'a'*64,b'b'*64],20)
            for key in ('sample_type','rand_action','goal_achieved','done','time_limit'):
                f[key]=np.zeros(40,np.uint8)
            f['noise_std']=np.zeros(40,np.float32);f['reward']=np.zeros(40)
            f['branch_t']=np.full(40,-1,np.int32);f['branch_len']=np.zeros(40,np.int32)
            f['target_pos']=np.zeros((40,3))
            if route=='rgb':
                f.create_dataset('pixels',shape=(40,256,256,3),dtype='uint8',chunks=True,compression='lzf')
                f['pixels'][:]=image
            else:
                f['pcd_xyz']=np.broadcast_to(cloud,(40,1200,3))
                f['pcd_rgb']=np.zeros((40,1200,3),np.uint8)
    return root/'rgb.h5',root/'pcd.h5'


def test_data_gate_checks_content_and_rejects_duplicate_reset(tmp_path):
    data=load('adroit_data')
    rgb,pcd=make_pair(tmp_path/'data')
    report=data.audit_pair(rgb,pcd,expected_episodes=2,episode_length=20)
    assert report['passed'] and report['frames']==40
    with h5py.File(pcd,'r+') as f: f['action'][19,0]=1
    with pytest.raises(ValueError,match='action'): data.audit_pair(rgb,pcd,2,20)
    with h5py.File(pcd,'r+') as f: f['action'][19,0]=0
    for p in (rgb,pcd):
        with h5py.File(p,'r+') as f: f['initial_state_sha256'][20:]=b'a'*64
    with pytest.raises(ValueError,match='initial'): data.audit_pair(rgb,pcd,2,20)


def test_verifier_never_accepts_only_a_touched_receipt(tmp_path):
    workflow=load('adroit_workflow')
    stage=workflow.plan(args(tmp_path),{})[0]
    assert workflow.verify_stage(stage,args(tmp_path),{})['passed'] is False


def test_bc_gate_requires_finite_losses_and_actual_eval_rounds():
    learning=load('adroit_learning')
    metrics=[{'epoch_loss':1.2},{'eval/epoch':0,'eval/success':0.0}]
    assert learning.validate_metrics(metrics,1)['evaluation_rounds']==1
    with pytest.raises(ValueError,match='evaluation'):
        learning.validate_metrics(metrics[:1],1)
    with pytest.raises(ValueError,match='finite'):
        learning.validate_metrics([{'epoch_loss':float('nan')},metrics[1]],1)


def test_collector_records_the_actual_executed_actions_and_pre_action_states(tmp_path,monkeypatch):
    """Exercise the real HDF writer around a small deterministic simulator double."""
    monkeypatch.syspath_prepend(str(ROOT))
    collector=importlib.import_module('jepa4r.policy.bc.relocate.adroit_collect')
    instances=[]
    class Native:
        def __init__(self):
            self.t=0;self.actions=[];self.sim=self;instances.append(self)
        def seed(self,seed):self.seed_value=seed
        def reset(self):self.t=0;return self.get_obs()
        def get_obs(self):return np.full(39,self.t,np.float32)
        def get_env_state(self):
            return {k:np.full(n,self.t,dtype=np.float64) for k,n in
                [('qpos',36),('qvel',36),('hand_qpos',30),('obj_pos',3),('palm_pos',3),('target_pos',3)]}
        def step(self,action):
            self.actions.append(action.copy());self.t+=1
            return self.get_obs(),float(self.t),False,{'goal_achieved':False}
        def render(self,**kwargs):return np.tile(np.arange(256,dtype=np.uint8)[None,:,None],(256,1,3))
        def close(self):pass
    class Sampler:
        def __init__(self,sim):self.sim=sim
        def seed(self,seed):pass
        def __call__(self):
            xyz=np.full((1200,3),self.sim.t,np.float32);xyz[:,0]+=np.linspace(0,1,1200)
            return xyz,np.zeros((1200,3),np.uint8)
    fake=types.ModuleType('mj_envs.envs.hand_manipulation_suite.relocate_v0');fake.RelocateEnvV0=Native
    point=types.ModuleType('jepa4r.policy.bc.relocate.cortexbench.run_adroit_bc3d');point.build_point_clouds=lambda sim,task:Sampler(sim)
    monkeypatch.setitem(sys.modules,fake.__name__,fake);monkeypatch.setitem(sys.modules,point.__name__,point)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(no_grad=nullcontext))
    policy=SimpleNamespace(get_action=lambda obs:(None,{'evaluation':np.zeros(30,np.float32)}))
    paths,_=collector.collect_split(tmp_path,'train',1,42,policy)
    with h5py.File(paths[0],'r') as f:
        np.testing.assert_array_equal(f['action'][:],np.asarray(instances[0].actions))
        np.testing.assert_array_equal(f['observation'][:,0],np.arange(200))
        np.testing.assert_array_equal(f['next_observation'][:,0],np.arange(1,201))
    with np.load(tmp_path/'initial_states_train.npz') as initial:
        assert initial['original_episode_index'].tolist()==[0]
    assert load('adroit_data').audit_pair(paths[0],paths[1],1)['passed']


def test_mixed_plan_matches_the_final_native_feedback_contract(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    collector=importlib.import_module('jepa4r.policy.bc.relocate.adroit_collect')
    tree=ast.parse((ROOT/'jepa4r/envs/adroit/env.py').read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AdroitEnv')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_draw_plan')
    namespace={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'final-native-plan','exec'),namespace)
    settings=SimpleNamespace(force_type=None,mix=np.array([.5,.3,.2]),noise_max=.2,
        episode_len=200,branch_tail_prob=.5,burst_len=(10,30))
    left=np.random.default_rng(42);right=np.random.default_rng(42)
    for _ in range(200):assert collector.episode_plan(left)==namespace['_draw_plan'](settings,right)


def test_expert_noise_precedes_the_single_final_clip(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    collector=importlib.import_module('jepa4r.policy.bc.relocate.adroit_collect')
    assert hasattr(collector,'mixed_action'), 'explicit feedback action bridge is missing'
    policy=SimpleNamespace(get_action=lambda obs:(None,{'evaluation':np.full(30,1.2,np.float32)}))
    env=SimpleNamespace(get_obs=lambda:np.zeros(39,np.float32))
    rng=SimpleNamespace(normal=lambda mean,std,size:np.full(size,-.4))
    action=collector.mixed_action(policy,env,rng,noise=.2,random_action=False)
    np.testing.assert_allclose(action,np.full(30,.8,np.float32),rtol=0,atol=1e-7)


@pytest.mark.parametrize('field,value,match',[
    ('gradient_steps',0,'gradient'),('positive_lr_steps',0,'learning rate'),
    ('after_sha256','before','parameter'),('adam_state_steps',[1.0],'Adam')])
def test_optimizer_gate_rejects_calls_without_real_updates(field,value,match):
    learning=load('adroit_learning')
    assert hasattr(learning,'validate_optimizer_evidence'), 'optimizer integrity gate is missing'
    evidence=dict(before_sha256='before',after_sha256='after',gradient_steps=2,
        positive_lr_steps=1,adam_state_steps=[2.0],hook_count=2)
    evidence[field]=value
    with pytest.raises(ValueError,match=match):learning.validate_optimizer_evidence(evidence)


def test_optimizer_pre_hook_rejects_all_none_gradients(monkeypatch):
    learning=load('adroit_learning')
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace())
    observer=SimpleNamespace(parameters=[SimpleNamespace(grad=None)])
    with pytest.raises(ValueError,match='nonempty'):
        learning.OptimizerObserver.before_step(observer,None,(),{})


def test_trainer_observer_keeps_manager_isinstance_contract_and_arguments():
    learning=load('adroit_learning')
    assert hasattr(learning,'observed_trainer_type'), 'type-preserving Trainer boundary is missing'
    class OriginalTrainer:
        def __init__(self,*args,**kwargs):self.received=(args,kwargs)
    existing=SimpleNamespace(cfg={'model':'original'})
    added=object()
    chosen=learning.observed_trainer_type(OriginalTrainer,lambda config:added)
    trainer=chosen('unchanged',callbacks=[existing],max_epochs=10,precision='bf16')
    assert isinstance(chosen,type)
    assert isinstance(trainer,chosen) and isinstance(trainer,OriginalTrainer)
    assert trainer.received==(('unchanged',),{'callbacks':[existing,added],'max_epochs':10,'precision':'bf16'})


def test_zero_weight_pldm_auxiliary_is_not_representation_learning_evidence():
    learning=load('adroit_learning')
    evidence=dict(before_sha256='same',after_sha256='same',gradient_steps=2,
        nonzero_gradient_steps=0,zero_gradient_steps=2,positive_lr_steps=1,
        adam_state_steps=[2.0],hook_count=2,role='pldm_aux_idm_zero_weight',
        require_nonzero_gradient=False,auxiliary_loss_weight=0.0)
    assert learning.validate_optimizer_evidence(evidence)['zero_gradient_steps']==2


def test_zero_gradient_exception_cannot_be_claimed_by_the_main_optimizer():
    learning=load('adroit_learning')
    evidence=dict(before_sha256='before',after_sha256='after',gradient_steps=2,
        nonzero_gradient_steps=0,zero_gradient_steps=2,positive_lr_steps=1,
        adam_state_steps=[2.0],hook_count=2,role='representation',
        require_nonzero_gradient=False,auxiliary_loss_weight=0.0)
    with pytest.raises(ValueError,match='auxiliary'):
        learning.validate_optimizer_evidence(evidence)


def test_pldm_auxiliary_role_requires_actual_parameter_ownership_and_zero_weight():
    learning=load('adroit_learning')
    assert hasattr(learning,'optimizer_role'), 'parameter ownership role check is missing'
    main=SimpleNamespace(requires_grad=True);aux=SimpleNamespace(requires_grad=True)
    model=SimpleNamespace(named_parameters=lambda:[('weight',main)])
    idm=SimpleNamespace(named_parameters=lambda:[('weight',aux)])
    module=SimpleNamespace(model=model,idm=idm)
    opt=SimpleNamespace(param_groups=[{'params':[aux]}])
    role=learning.optimizer_role(module,opt,'pldm',idm_enabled=True,idm_weight=0.)
    assert role['role']=='pldm_aux_idm_zero_weight' and role['parameter_names']==['idm.weight']
    assert role['require_nonzero_gradient'] is False
    for method,weight in [('jepa4r',0.),('pldm',1.)]:
        with pytest.raises(ValueError):learning.optimizer_role(module,opt,method,idm_enabled=True,idm_weight=weight)
    opt.param_groups=[{'params':[main]}]
    assert learning.optimizer_role(module,opt,'pldm',idm_enabled=True,idm_weight=0.)['require_nonzero_gradient'] is True


def test_observer_accepts_finite_zero_auxiliary_gradients_but_rejects_main_zeros(monkeypatch):
    learning=load('adroit_learning')
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(isfinite=np.isfinite,count_nonzero=np.count_nonzero))
    observer=SimpleNamespace(parameters=[SimpleNamespace(grad=np.zeros(3))],require_nonzero_gradient=False,
        gradient_steps=0,nonzero_gradient_steps=0,zero_gradient_steps=0,positive_lr_steps=0,learning_rates=[])
    optimizer=SimpleNamespace(param_groups=[{'lr':5e-5}])
    learning.OptimizerObserver.before_step(observer,optimizer,(),{})
    assert observer.gradient_steps==observer.zero_gradient_steps==observer.positive_lr_steps==1
    assert observer.nonzero_gradient_steps==0
    observer.require_nonzero_gradient=True
    with pytest.raises(ValueError,match='representation'):
        learning.OptimizerObserver.before_step(observer,optimizer,(),{})


@pytest.mark.parametrize('corruption,match',[
    ('bounds','action'),('seed','seed'),('step','step_idx'),('next','next_observation'),
    ('proprio','proprio'),('missing','next_observation'),('dtype','dtype'),
    ('blank_rgb','blank'),('empty_cloud','cloud')])
def test_audit_rejects_invalid_native_wm_content(tmp_path,corruption,match):
    data=load('adroit_data');rgb,pcd=make_pair(tmp_path/'pair')
    if corruption in ('blank_rgb','empty_cloud'):
        with h5py.File(rgb if corruption=='blank_rgb' else pcd,'r+') as f:
            f['pixels' if corruption=='blank_rgb' else 'pcd_xyz'][:20]=0
    elif corruption=='dtype':
        with h5py.File(pcd,'r+') as f:
            values=f['action'][:];del f['action'];f['action']=values.astype(np.float64)
    else:
        for path in (rgb,pcd):
            with h5py.File(path,'r+') as f:
                if corruption=='bounds':f['action'][0,0]=123
                elif corruption=='seed':f['seed'][5]=99
                elif corruption=='step':f['step_idx'][5]=99
                elif corruption=='next':f['next_observation'][5,0]=999
                elif corruption=='proprio':f['proprio'][5,0]=999
                else:del f['next_observation']
    with pytest.raises(ValueError,match=match):data.audit_pair(rgb,pcd,2,20)


def test_audit_records_individual_blank_frames_without_filtering(tmp_path):
    data=load('adroit_data');rgb,pcd=make_pair(tmp_path/'pair')
    with h5py.File(rgb,'r+') as f:f['pixels'][0]=0
    with h5py.File(pcd,'r+') as f:f['pcd_xyz'][1]=0
    result=data.audit_pair(rgb,pcd,2,20)
    assert result['frames']==40 and result['blank_rgb_frames']==1 and result['empty_cloud_frames']==1


def test_pldm_reload_uses_the_original_native_preprocessor():
    learning=load('adroit_learning')
    assert hasattr(learning,'reload_preprocessor'), 'PLDM reload preprocessor boundary is missing'
    calls=[];sentinel=object()
    native=SimpleNamespace(get_img_preprocessor=lambda *args:(calls.append(args) or sentinel))
    cfg=SimpleNamespace(img_size=224,data=SimpleNamespace())
    def forbidden(_):raise AssertionError('PLDM has no cfg.data.preprocessor')
    assert learning.reload_preprocessor('pldm',cfg,forbidden,native)==sentinel
    assert calls==[('pixels','pixels',224)]


def test_pinned_expert_rejects_sha_mismatch_before_unpickle(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    collector=importlib.import_module('jepa4r.policy.bc.relocate.adroit_collect')
    assert hasattr(collector,'load_pinned_expert'), 'trusted expert SHA boundary is missing'
    spec=importlib.util.spec_from_file_location('reviewed_state_expert',ROOT/'jepa4r/envs/adroit/policy.py')
    policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy);calls=[]
    monkeypatch.setattr(policy.pickle,'loads',lambda payload:calls.append(payload))
    source=tmp_path/'wrong-expert.pickle';source.write_bytes(b'untrusted bytes')
    with pytest.raises(ValueError,match='SHA-256 mismatch'):collector.load_pinned_expert(source)
    assert calls==[]


def test_plan_env_preserves_explicit_dependencies_and_fixed_protocol(tmp_path):
    import copy,os
    workflow=load('adroit_workflow')
    assets={'stable_worldmodel_root':'/swm','mjrl_root':'/mjrl','mj_envs_root':'/mjenv',
        'eai_vc_root':'/eai','mujoco_path':'/mujoco','pythonpath':['/dmc-overlay','/swm'],
        'environment':{'OMP_NUM_THREADS':'1','LD_LIBRARY_PATH':'/driver:/custom',
            'PYTHONPATH':'/explicit-overlay','ADROIT_FRESH_EVAL':'0','ADROIT_PCD_CONFIG':'/old.json'}}
    before=copy.deepcopy(assets);stages=workflow.plan(args(tmp_path),assets)
    env=stages[0]['environment'];paths=env['PYTHONPATH'].split(os.pathsep)
    assert assets==before and not (tmp_path/'new').exists()
    assert '/dmc-overlay' in paths and '/explicit-overlay' in paths and paths.count('/swm')==1
    assert paths.index('/swm')<paths.index('/dmc-overlay')
    assert env['OMP_NUM_THREADS']=='1' and env['LD_LIBRARY_PATH']=='/driver:/custom'
    assert env['ADROIT_FRESH_EVAL']=='1' and env['ADROIT_PCD_CONFIG']==str(ROOT/'jepa4r/envs/adroit/adroit_relocate_sampling.json')


def test_frozen_encoder_sha_is_verified_without_constructing_a_model(tmp_path):
    learning=load('adroit_learning')
    assert hasattr(learning,'verify_frozen_checkpoint'), 'fixed released-weight source gate is missing'
    path=tmp_path/'same-size-random.pth';path.write_bytes(b'abc')
    with pytest.raises(ValueError,match='SHA-256'):
        learning.verify_frozen_checkpoint('visreg',path,{'visreg':{'bytes':3,'sha256':'0'*64}})
    assert learning.verify_frozen_checkpoint('pldm',tmp_path/'not-a-fixed-weight') is None


def test_supplemental_audit_never_rewrites_the_original_receipt(tmp_path):
    data=load('adroit_data');root=tmp_path/'data';root.mkdir()
    original=root/'audit.json';original.write_text('original accepted evidence')
    config=SimpleNamespace(data_root=root,train_episodes=2,test_episodes=1,report=None)
    with pytest.raises(FileExistsError):data.audit(config)
    config.report=tmp_path/'supplement.json'
    with pytest.raises(OSError):data.audit(config)
    assert original.read_text()=='original accepted evidence'
    import json
    failure=json.loads(config.report.read_text())
    assert failure['passed'] is False and failure['actual_exit_code']==1
    assert failure['validator_sha256']==data.sha(ROOT/'jepa4r/policy/bc/relocate/adroit_data.py')


def test_pldm_two_batch_smoke_keeps_two_steps_for_each_native_optimizer(tmp_path,monkeypatch):
    learning=load('adroit_learning')
    assert hasattr(learning,'pldm_training_plan'), 'pure native PLDM plan boundary is missing'
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.setenv('ADROIT_FRAMEWORK_ROOT',str(ROOT/'third_party/stable-worldmodel'))
    config=SimpleNamespace(data_root=tmp_path/'data',output=tmp_path/'wm',mode='smoke')
    plan=learning.pldm_training_plan(config)
    # The observed SPT global_step counts batches; both optimizers step per batch.
    assert '+trainer.max_steps=2' in plan['hydra_overrides']
    assert '+trainer.max_steps=4' not in plan['hydra_overrides']
    assert '+trainer.limit_train_batches=2' in plan['hydra_overrides']
    assert not config.output.exists()


@pytest.mark.parametrize('missing_weight',[False,True])
def test_stage_receipt_cannot_substitute_unrelated_or_missing_weight_artifacts(tmp_path,missing_weight):
    import json,hashlib
    workflow=load('adroit_workflow');config=args(tmp_path)
    stage=next(s for s in workflow.plan(config,{}) if s['name']=='wm-reload')
    required=[Path(p) for p in stage.get('required_artifacts',[])]
    if not required:required=[tmp_path/'new/wm/checkpoints/relocate/weights_epoch_1.pt',tmp_path/'new/wm/checkpoints/relocate/config.json']
    artifacts=[]
    for path in required:
        if missing_weight and path.name.startswith('weights_epoch_'):continue
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'bound input')
        artifacts.append(dict(path=str(path),bytes=11,sha256=hashlib.sha256(b'bound input').hexdigest()))
    if not missing_weight:
        unrelated=tmp_path/'irrelevant.txt';unrelated.write_bytes(b'bound input')
        artifacts=[dict(path=str(unrelated),bytes=11,sha256=hashlib.sha256(b'bound input').hexdigest())]
    report=Path(stage['receipt']);report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(dict(passed=True,actual_exit_code=0,strict_reload=True,artifacts=artifacts)))
    assert workflow.verify_stage(stage,config,{})['passed'] is False


@pytest.mark.parametrize('count',[1,99])
def test_official_consumer_requires_all_100_sources_before_smoke_render(tmp_path,count,monkeypatch):
    import json
    monkeypatch.syspath_prepend(str(ROOT));data=load('adroit_data')
    assert hasattr(data,'validated_official_source'), 'actual official source-content boundary is missing'
    canonical=json.loads((ROOT/'jepa4r/policy/bc/relocate/DEPENDENCIES.json').read_text())['official_relocate']['raw_sha256']
    with pytest.raises(ValueError,match='100'):
        data.validated_official_source(tmp_path/'raw.pickle',canonical,lambda *a:(count,None,canonical))


def test_official_consumer_rejects_noncanonical_raw_sha_before_loading(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT));data=load('adroit_data')
    assert hasattr(data,'validated_official_source'), 'actual official source-content boundary is missing'
    def forbidden(*a):raise AssertionError('noncanonical source must not be loaded')
    with pytest.raises(ValueError,match='raw SHA'):
        data.validated_official_source(tmp_path/'raw.pickle','0'*64,forbidden)


def test_official_consumer_rejects_100_repeated_valid_demo_records(tmp_path,monkeypatch):
    import json
    monkeypatch.syspath_prepend(str(ROOT));data=load('adroit_data')
    assert hasattr(data,'validated_official_source'), 'actual official source-content boundary is missing'
    canonical=json.loads((ROOT/'jepa4r/policy/bc/relocate/DEPENDENCIES.json').read_text())['official_relocate']['raw_sha256']
    demo={'init_state_dict':{'qpos':np.zeros(36)},'actions':np.zeros((200,30)),
        'state_obs':np.zeros((200,39)),'env_infos':{'proprio':np.zeros((200,30))},'rewards':np.zeros(200)}
    source=tmp_path/'raw.pickle';source.write_bytes(b'test loader fixture')
    loader=lambda *a:(100,lambda i:(demo,{'path':str(source)}),canonical)
    with pytest.raises(ValueError,match='duplicate'):
        data.validated_official_source(source,canonical,loader)


@pytest.mark.parametrize('change',['clip_actions','state_obs','proprio','rewards','dtype'])
def test_official_consumer_hashes_actual_decoded_values_not_manifest_claims(tmp_path,monkeypatch,change):
    import json
    monkeypatch.syspath_prepend(str(ROOT));data=load('adroit_data')
    contract=importlib.import_module('jepa4r.policy.bc.relocate.official_contract')
    canonical=json.loads((ROOT/'jepa4r/policy/bc/relocate/DEPENDENCIES.json').read_text())['official_relocate']['raw_sha256']
    source=tmp_path/'demos';source.mkdir();payload=source/'fixture.pickle';payload.write_bytes(b'pure loader fixture')
    manifest=source.parent/'manifest.json';manifest.write_text(json.dumps({'source_sha256':canonical}))
    demos=[]
    for index in range(100):
        actions=np.zeros((200,30));actions[0,0]=1.6
        demos.append({'init_state_dict':{'qpos':np.full(36,float(index))},'actions':actions,
            'state_obs':np.zeros((200,39)),'env_infos':{'proprio':np.zeros((200,30))},'rewards':np.zeros(200)})
    records=[contract.demo_content_record(i,demo) for i,demo in enumerate(demos)]
    trusted=tmp_path/'test-contract.json'
    trusted.write_text(json.dumps({'official_relocate':{'policy_critical_content_sha256':contract.content_digest(records)}}))
    monkeypatch.setattr(contract,'TRUSTED_CONTRACT_PATH',trusted)
    reads=[]
    def read(index):reads.append(index);return demos[index],{'path':str(payload)}
    loader=lambda *a:(100,read,data.sha(manifest))
    result=data.validated_official_source(source,data.sha(manifest),loader)
    assert result['content']['count']==100 and reads==list(range(100))
    assert demos[7]['actions'][0,0]==1.6
    if change=='clip_actions':demos[7]['actions']=np.clip(demos[7]['actions'],-1,1)
    elif change=='dtype':demos[7]['actions']=demos[7]['actions'].astype(np.float32)
    elif change=='proprio':demos[7]['env_infos']['proprio'][0,0]=1
    else:demos[7][change].flat[0]=1
    with pytest.raises(ValueError,match='digest mismatch'):
        data.validated_official_source(source,data.sha(manifest),loader)


def test_stage_receipt_covers_exact_expected_paths_and_accepts_extra_audits(tmp_path):
    import json,hashlib
    workflow=load('adroit_workflow');config=args(tmp_path)
    stage=next(s for s in workflow.plan(config,{}) if s['name']=='wm-reload')
    assert stage['required_artifacts']
    paths=[Path(path) for path in stage['required_artifacts']]+[tmp_path/'extra-audit.json']
    artifacts=[]
    for path in paths:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'bound input')
        artifacts.append(dict(path=str(path),bytes=11,sha256=hashlib.sha256(b'bound input').hexdigest()))
    Path(stage['receipt']).write_text(json.dumps(dict(passed=True,actual_exit_code=0,strict_reload=True,artifacts=artifacts)))
    assert workflow.verify_stage(stage,config,{})['passed'] is True
