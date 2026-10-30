"""Metadata contracts and original command selection; no NN imports."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

ROOT=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
F=ROOT/'foundation_visual_gciql_v1'; CASE=ROOT/'visreg_cube34_gciql_v1'
STORE=Path('/root/autodl-tmp/visreg_cube34_gciql_20261027_v1_outputs')
H=CASE/'release_v1'; PY=ROOT/'env_runtime/bin/python'
TEMPLATE=ROOT/'cube23_distance_gciql_v1/five_group_extension_v1/configs/G08_cube3_rgb.json'
SMOKE=F/'control/smoke/launch_plan.json'; FORMAL=F/'control/formal_w8/launch_plan.json'
EXPECTED_SHA='2b2426e7136381c8654524ab34391744caec3c4ea2b1afdbf3377b25bc6da83c'
COUNTERS=[1]+list(range(100000,1000001,100000))
STAGES=('cube3_model_gates','cube3_full','cube4_model_gates','cube4_full')
IO_KEYS={'execution_root','output','commands'}

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_new(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def atomic(p,value):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,p)
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def stat(p):
    s=Path(p).stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino,device=s.st_dev)
def stat_contract(p,want):
    actual=stat(p);assert all(actual[k]==v for k,v in want.items()),str(p)
def pins_contract(pins):
    for p,h in pins.items():assert sha(p)==h,p

def owned_bytes(root):
    total=0;seen=set();root=Path(root)
    if not root.exists():return 0
    assert root.is_dir() and not root.is_symlink()
    for parent,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(parent)/d).is_symlink()]
        for n in files:
            p=Path(parent)/n
            if p.is_symlink():continue
            s=p.stat();key=(s.st_dev,s.st_ino)
            if key not in seen:total+=s.st_size;seen.add(key)
    return total

def space_contract(free,paid,budget=3100000000):
    reserve=512*(1<<20);required=max(0,budget-paid)+reserve
    assert free>=required,f'Temp insufficient: free={free}, required={required}, paid_once={paid}'
    return dict(free_bytes=free,owned_STORE_paid_bytes=paid,artifact_budget_bytes=budget,
                reserve_bytes=reserve,required_free_bytes=required,remaining_headroom_bytes=free-required,
                paid_deducted_exactly_once=True)

def space_admission():
    sizes=[]
    for n in (3,4):
        for p in (STORE/'pilot/gates'/f'visreg_vitb768_cube{n}'/'checkpoints').glob('update_*'):
            if p.is_dir():sizes.append(owned_bytes(p))
    # Whole-agent params already include optimizer/target state; never multiply by Adam again.
    measured=max(sizes,default=0)
    budget=max(3100000000,26*(measured+(1<<20))+400000000) if sizes else 3100000000
    out=space_contract(shutil.disk_usage('/root/autodl-tmp').free,owned_bytes(STORE),budget)
    out.update(measured_pilot_checkpoint_bytes=sizes,projected_checkpoint_count=26,
               NN_weights_or_tensor_contents_read=False,at=time.time())
    return out

def remap_config(original,config_path,home,commands_factory):
    cfg=copy.deepcopy(original);cfg['execution_root']=str(home)
    cfg['output']=str(Path(home)/'runs'/cfg['group_id'])
    cfg['commands']=commands_factory(Path(config_path),cfg)
    assert {k:v for k,v in original.items() if k not in IO_KEYS}=={k:v for k,v in cfg.items() if k not in IO_KEYS}
    return cfg

def cfgpath(n,purpose):return CASE/'configs'/purpose/f'visreg_vitb768_cube{n}.json'
def home(purpose):return CASE/'outputs'/purpose
def stage_dir(stage):assert stage in STAGES;return CASE/'control/stages'/stage
def delivery_path(n):return F/'cache/w8_groups'/f'visreg_vitb768_cube{n}'/'delivery.json'

def original_config(n,purpose,prep):
    dp=delivery_path(n);delivery=read(dp);delivery['purpose']=purpose
    cfg=prep.make_config(read(TEMPLATE),delivery)
    cfg['delivery_binding']=dict(path=str(dp),sha256=sha(dp))
    cfg['parent_template']=dict(path=str(TEMPLATE),sha256=sha(TEMPLATE))
    return cfg

def config_contract(cfg,n,purpose,prep):
    path=cfgpath(n,purpose)
    assert cfg==remap_config(original_config(n,purpose,prep),path,home(purpose),prep.commands)
    assert cfg['input_contract']==dict(current_dim=787,goal_dim=768+3*n,latent_dim=768,zero_padding=False)
    assert cfg['encoder_binding']['family']=='visreg_vitb768' and cfg['fresh'] and cfg['resume_from'] is None
    assert (cfg['seed'],cfg['batch_size'],cfg['total_updates'],cfg['native_max_episode_steps'])==(0,1024,1000000,1000)

def selected_phases(n,full,configs):
    g=f'visreg_vitb768_cube{n}';purpose='formal' if full else 'pilot';cfg=configs[purpose]
    base=read(FORMAL if full else SMOKE)
    labels=('05_train','06_accept','07_native','08_native_audit') if full else ('04_hooks','05_gate','06_reload','07_native','08_native_audit')
    commands=cfg['commands'];outputs=[]
    for label in labels:
        candidates=[p for p in base['phases'] if p['name']==g+'_'+label];assert len(candidates)==1
        p=copy.deepcopy(candidates[0]);old=list(p['argv'])
        if label.endswith('hooks'):tail=[str(F/'functional/runtime.py'),'verify-hooks','--config',str(cfgpath(n,purpose))]
        elif label.endswith('native_audit'):tail=[str(F/'functional/audit_native.py'),'--config',str(cfgpath(n,purpose)),'--mode','full' if full else 'tiny','--output',str(home(purpose)/'native_sr'/g)]
        else:
            action=label.split('_',1)[1]
            tail=commands[action]
        index=old.index(tail[0]);p['argv']=old[:index]+tail
        assert p['cwd']==str(ROOT) and not p['checks']
        assert old[:index] in ([str(PY),'-B'],[str(PY),'-B',str(ROOT/'cube4_visual_gciql_v1/control_v1/gpu_lease.py'),'--',str(PY),'-B'])
        p['name']=f'c{n}_'+('full_' if full else 'gate_')+label;outputs.append(p)
    assert [p['environment']['CUDA_VISIBLE_DEVICES'] for p in outputs]==(['0','','0',''] if full else ['','0','','0',''])
    assert [p['environment']['JAX_PLATFORMS'] for p in outputs]==(['cuda','cpu','cpu','cpu'] if full else ['cpu','cuda','cpu','cpu','cpu'])
    return outputs

def pilot_contract(cfg,gate,reload):
    g=cfg['group_id'];assert gate['actual_exit_code']==reload['actual_exit_code']==0
    assert gate['role']==reload['role']==g and (gate['updates'],gate['batch_size'],gate['network_step'],gate['adam_count'])==(2,1024,3,2)
    assert gate['initial_state']['leaves']==gate['saved_state']['leaves']==153
    assert gate['formal_training_started'] is False and gate['encoder_calls']==gate['WM_dynamics_calls']==0
    width=cfg['input_contract']['current_dim']+cfg['input_contract']['goal_dim']
    shapes=gate['first_layer_kernels'].values();assert {s[-2] for s in shapes}=={width,width+5} and all(s[-1]==512 for s in shapes)
    assert reload['state']==gate['saved_state'] and reload['full153_state_exact'] and reload['next_batch_and_rawIDs_exact']
    assert reload['applied_host_RNG']==['numpy','python','torchCPU'] and reload['no_updates']
    assert reload['finite_actions'] and reload['CPU_action_forward'] and reload['backend']=='cpu' and reload['fresh_process']

def formal_contract(cfg,accept):
    assert accept['actual_exit_code']==0 and accept['role']==cfg['group_id'] and accept['full153_state_exact']
    assert accept['state']['leaves']==153 and (accept['Adam_count'],accept['network_step'],accept['CSV_validation_records'])==(1000000,1000001,200)
    assert [r['counter'] for r in accept['checkpoint_transactions']]==COUNTERS
    assert (accept['actual_current_dim'],accept['actual_goal_dim'])==(cfg['input_contract']['current_dim'],cfg['input_contract']['goal_dim'])
    assert accept['synthetic_zero_padding'] is False and accept['finite_actions'] and accept['CPU_action_forward'] and accept['no_optimizer_updates']
    assert accept['entire_1M_host_sampler_replay_performed'] is False and accept['SR_performed'] is False

def native_contract(g,cfgsha,bindingsha,binding,report,audit,full):
    for x in (binding,report,audit):assert x['actual_exit_code']==0 and x['group_id']==g
    assert binding['config_sha256']==audit['config_sha256']==cfgsha and audit['source_binding_sha256']==bindingsha
    assert binding['encoder_family']=='visreg_vitb768' and binding['latent_dim']==768
    assert binding['mode']==audit['mode']==('full' if full else 'tiny') and binding['gate_smoke'] is (not full)
    assert report['mode']==('full' if full else 'diagnostic')
    expected=250 if full else 1
    assert report['episodes']==report['unique_seeds']==audit['episodes']==audit['unique_seeds']==expected
    assert audit['actual_step_arrays_checked'] and audit['diagnostic_excluded_from_formal_SR'] is (not full)

def closure_contract(plan,closure,exits,argv,outer,wrapper,status,dispatchsha,plansha):
    assert outer==wrapper==0 and status['actual_exit_code']==closure['actual_exit_code']==0
    assert closure['completed_phases']==len(plan['phases']) and dispatchsha==plansha
    for k in ('cube_counts','families','training_updates','SR_episodes'):assert closure[k]==plan[k],k
    assert set(exits)==set(argv)=={p['name'] for p in plan['phases']}
    for p in plan['phases']:
        x=exits[p['name']];assert x['actual_exit_code']==0 and x['residual_owned_members']==[]
        assert argv[p['name']]['argv']==p['argv'] and argv[p['name']]['cwd']==p['cwd']

def stage_closed(stage,wrapper=None):
    t=stage_dir(stage);r=t/'runtime';plan=read(t/'launch_plan.json')
    assert not (r/'failure.json').exists(),str(r)
    closure_contract(plan,read(r/'closure.json'),{p['name']:read(r/(p['name']+'_exit.json')) for p in plan['phases']},
        {p['name']:read(r/(p['name']+'_argv.json')) for p in plan['phases']},int((r/'outer_exit_code').read_text()),
        int((t/'wrapper_exit_code').read_text()) if wrapper is None else wrapper,read(r/'status.json'),read(r/'actual_dispatch.json')['launch_plan_sha256'],sha(t/'launch_plan.json'))
    n=plan['cube_counts'][0];full=plan['mode']=='full';purpose='formal' if full else 'pilot';cfg=read(cfgpath(n,purpose));g=cfg['group_id'];h=home(purpose)
    if full:
        accept=read(h/'acceptance'/f'{g}.json');formal_contract(cfg,accept)
        final=Path(cfg['output'])/'checkpoints/update_001000000'
        assert accept['final_checkpoint']==str(final) and accept['final_manifest_sha256']==sha(final/'manifest.json')
    else:pilot_contract(cfg,read(h/'gates'/g/'report.json'),read(h/'gates'/f'{g}_reload.json'))
    output=h/'native_sr'/g;bp=output/'native_source_binding.json'
    native_contract(g,sha(cfgpath(n,purpose)),sha(bp),read(bp),read(output/'report.json'),read(output/'independent_audit.json'),full)
    return dict(actual_exit_code=0,stage=stage,completed_phases=len(plan['phases']),reports_verified=True)

def require_predecessor(stage):
    i=STAGES.index(stage)
    if i:return stage_closed(STAGES[i-1])
    return None

def fresh_outputs(stage):
    n=int(stage[4]);full=stage.endswith('_full');purpose='formal' if full else 'pilot';g=f'visreg_vitb768_cube{n}';h=home(purpose)
    targets=[h/'runs'/g,h/'acceptance'/f'{g}.json',h/'native_sr'/g] if full else [h/'gates'/g,h/'gates'/f'{g}_reload.json',h/'native_sr'/g,h/'runs'/g]
    targets.extend([stage_dir(stage)/'runtime',stage_dir(stage)/'wrapper_exit_code'])
    for p in targets:assert not p.exists() and not p.is_symlink(),'Existing artifact must be inspected, not repeated: '+str(p)

def weight_contract(receipt,expected):
    want=expected['shared_encoder']['checkpoint']
    assert receipt['actual_exit_code']==0 and receipt['path']==want['remote_path'] and receipt['sha256']==want['accepted_sha256']
    assert receipt['source_before_after_stat_exact'] and receipt['stat']['size']==want['accepted_bytes']
    stat_contract(receipt['path'],receipt['stat'])

def cache_contract(expected):
    assert set(expected['groups'])=={'visreg_vitb768_cube3','visreg_vitb768_cube4'}
    for group,e in expected['groups'].items():
        for f in e['small_files']:
            assert Path(f['remote_path']).stat().st_size==f['bytes'] and sha(f['remote_path'])==f['sha256'],f['remote_path']
        for x in e['phase_exits']:
            assert sha(x['remote_path'])==x['sha256'];r=read(x['remote_path'])
            assert all(r[k]==v for k,v in x['expected_fields'].items())
        a=e['field_assertions'];assert (a['latent_dim'],a['dtype'],a['encoder_family'])==(768,'float32','visreg_vitb768')
        assert a['B1_proof']['atol']==a['B1_proof']['rtol']==1e-5 and a['B1_proof']['passed']
        assert a['full_cpu_audit']['actual_exit_code']==0
        for d in a['source_NPZ'].values():stat_contract(d['path'],d['accepted_stat'])
        for d in a['source_arrays']['splits'].values():
            for key in ('qpos','qvel'):stat_contract(d[key]['path'],d[key]['stat'])
        pins_contract({str(ROOT/p):h for p,h in a['plan']['source_pins'].items()})
        for key in ('data_acceptance','source_stat_freeze'):assert sha(a['plan'][key])==a['plan'][key+'_sha256']
        delivery=read(delivery_path(a['cube_count']))
        assert delivery['encoder_binding']==a['encoder_binding'] and delivery['feature_loader_arguments']==a['feature_loader_arguments'] and delivery['camera']==a['camera']
        for split in ('train','val'):assert Path(delivery['feature_loader_arguments'][split]['cache_dir']).is_dir()
    for key in ('spec','adapter'):
        x=expected['shared_encoder'][key];assert sha(x['remote_path'])==x['sha256']
    assert read(expected['shared_encoder']['spec']['remote_path'])==expected['shared_encoder']['field_assertions']
    return dict(actual_exit_code=0,original_cache_acceptance_adopted=True,small_files_checked=expected['small_files_total'],
                source_NPZ_qpos_qvel_stat_verified=True,vectors_rehashed=False,NN_started=False)

def validate_plan(plan,stage,predecessor=True):
    rel=read(CASE/'RELEASE.json');assert sha(CASE/'RELEASE.json')==plan['release_sha256']
    pins_contract(plan['source_pins']);assert plan['source_pins']==rel['source_pins']
    n=int(stage[4]);full=stage.endswith('_full');prep=module('_original_metadata_prepare',F/'functional/prepare.py')
    configs={p:read(cfgpath(n,p)) for p in ('pilot','formal')}
    for purpose,cfg in configs.items():config_contract(cfg,n,purpose,prep)
    assert plan['phases']==selected_phases(n,full,configs) and plan['cube_counts']==[n] and plan['families']==['visreg_vitb768']
    assert plan['queue']=='visreg34_'+stage and plan['mode']==('full' if full else 'model_gates')
    assert (plan['training_updates'],plan['SR_episodes'])==((1000000,250) if full else (0,0))
    assert (plan['diagnostic_optimizer_updates'],plan['diagnostic_tiny_episodes'])==((0,0) if full else (2,1))
    assert plan['process_guard']==str(ROOT/'prepare_runtime.py') and plan['own_RSS_limit_bytes']==60*(1<<30)
    assert CASE.joinpath('outputs').is_symlink() and CASE.joinpath('outputs').resolve()==STORE.resolve()
    weight_contract(read(CASE/'references/weight_identity.json'),read(H/'expected_cache_bindings_v1.json'))
    if predecessor:require_predecessor(stage)
