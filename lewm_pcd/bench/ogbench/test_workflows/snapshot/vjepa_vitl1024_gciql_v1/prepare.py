"""Fresh VJ-only metadata and original-CLI queues; no model construction."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
BASE = ROOT/'foundation_visual_gciql_v1'
CASE = ROOT/'vjepa_vitl1024_gciql_v1'
STORE = Path('/root/autodl-tmp/vjepa_vitl1024_gciql_20261026_v1_outputs')
LOCAL = 'D:/LeWM_Artifacts/coordination/vjepa_vitl1024_gciql_31435_20261026_v1'
PY = ROOT/'env_runtime/bin/python'
BASE_ENV = ROOT/'pldm_cube14_gciql_v1/baseline_environments.json'
FAMILY = 'vjepa21_vitl1024'
COUNTS = {1:{'train':1000,'val':100},2:{'train':1000,'val':100},3:{'train':3000,'val':300},4:{'train':5000,'val':500}}
PREPARE_SHA = '17141ea3dffd485455f9a54dbcb52a3b569c583b2f1326d7074d2d96b4a78464'
SPEC_SHA = '319355585e08d686337a6d84a43fab75f64cf4c5e4dd33f041f0201d6f25d951'
ADAPTER_SHA = '8ac34a03554671ccdbe5cdd63d6a7f8881850a25e31afeb7fbc3321f18766463'
CHECKPOINT_SHA = '7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6'
POLICY = 'same_batch_official_fp16_reference_v1'
BUDGET, MARGIN = 6800000000, 512 << 20


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def stat(path):
    s=Path(path).stat()
    return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,device=s.st_dev,inode=s.st_ino)


def cache_admission():
    runtime=BASE/'control/fp16_bulk_w8_v1/runtime'
    assert (runtime/'outer_exit_code').read_text().strip() == (runtime.parent/'wrapper_exit_code').read_text().strip() == '0'
    closure=read(runtime/'closure.json')
    assert closure['actual_exit_code']==0 and closure['completed_phases']==12
    assert closure['cube_counts']==[1,2,3,4] and closure['families']==[FAMILY]
    assert closure['training_updates']==closure['SR_episodes']==0
    accepted={}
    for n,counts in COUNTS.items():
        group=f'{FAMILY}_cube{n}';home=BASE/'cache/fp16_bulk_w8_groups'/group
        plan_path=home/'plan.json';plan=read(plan_path);cache=Path(plan['cache_dir'])
        assert (plan['cube_count'],plan['encoder_family'],plan['latent_dim'])==(n,FAMILY,1024)
        assert plan['split_episodes']==counts and plan['raw_rows_per_episode']==1001
        assert plan['inference_precision']=='fp16' and plan['output_dtype']=='float32'
        assert plan['batch_reference_policy']==POLICY and plan['encoder_batch']==32
        assert plan['checkpoint_sha256']==CHECKPOINT_SHA
        spec=Path(plan['encoder_spec']);adapter=Path(plan['encoder_adapter'])
        assert sha(spec)==plan['encoder_spec_sha256']==SPEC_SHA
        assert adapter.name=='precision_bridge.py' and sha(adapter)==plan['encoder_adapter_sha256']==ADAPTER_SHA
        declaration=read(spec)
        assert declaration['family']==FAMILY and declaration['output_dim']==1024
        assert declaration['inference_precision']=='fp16' and declaration['batch_reference_policy']==POLICY
        assert declaration['checkpoint_sha256']==CHECKPOINT_SHA and declaration['checkpoint']==plan['checkpoint']
        arrays=Path(plan['shared_state_root'])/'source_arrays.json'
        assert sha(arrays)==plan['source_arrays_receipt_sha256']
        source=read(arrays);assert source['contract']['extractor_source_sha256']==plan['source_arrays_extractor_sha256']
        for item in source['splits'].values():
            assert stat(item['source_npz'])==item['accepted_stat']
            for key in ('qpos','qvel'):assert stat(item[key]['path'])==item[key]['stat']
        audit_path=home/'full_audit_v1/report.json';audit=read(audit_path);episodes=sum(counts.values())
        assert (audit_path.parent/'actual_exit_code').read_text().strip()=='0'
        assert (audit['actual_exit_code'],audit['episodes_verified'],audit['raw_rows_verified'],audit['main_rows_verified'])==(0,episodes,episodes*1001,episodes*1000)
        assert (audit['latent_dim'],audit['dtype'])==(1024,'float32')
        assert all(audit[k] for k in ('all_new_latent_SHA_shape_dtype_finite','all_raw_ID_and_true_terminal_next_contracts','tracked_stats_stable'))
        assert audit['plan_sha256']==sha(plan_path) and audit['manifest_sha256']==sha(cache/'cache_manifest.json')
        assert audit['producer_seal_sha256']==sha(cache/'cache_seal.json')
        proof=read(cache/'precision_reference_alignment.json')
        assert proof['actual_exit_code']==0 and proof['batch_reference_policy']==POLICY
        assert proof['inference_precision']=='fp16' and proof['encoder_spec_sha256']==SPEC_SHA
        assert (proof['offline_encoder_batch'],proof['online_reference_batch'])==(32,1)
        for key,batch in [('B32',32),('B1',1)]:
            p=proof['same_batch_reference'][key]
            assert p['wrapper_official_bitwise_equal'] and p['finite'] and p['max_abs_difference']==0 and p['embedding_shape']==[batch,1024]
        assert proof['legacy_diagnostic']['used_for_acceptance'] is False
        delivery_path=home/'delivery.json';delivery=read(delivery_path);binding=delivery['encoder_binding']
        assert (delivery['purpose'],delivery['group_id'],delivery['cube_count'],delivery['latent_dim'])==('formal',group,n,1024)
        assert binding==dict(family=FAMILY,latent_dim=1024,checkpoint_sha256=CHECKPOINT_SHA,spec_path=str(spec),spec_sha256=SPEC_SHA,adapter=dict(path=str(adapter),sha256=ADAPTER_SHA))
        assert delivery['cache_audit']==dict(path=str(audit_path),sha256=sha(audit_path))
        for split,count in counts.items():
            args=delivery['feature_loader_arguments'][split]
            assert args['cache_dir']==str(cache) and args['manifest_sha256']==audit['manifest_sha256']
            assert args['checkpoint_sha256']==CHECKPOINT_SHA and args['expected_rows']==count*1001
            assert args['source_npz']==source['splits'][split]['source_npz'] and delivery['data_binding']['splits'][split]['episodes']==count
        for suffix in ('01_encode_full','02_cache_audit','03_deliver'):
            phase=read(runtime/(group+'_'+suffix+'_exit.json'))
            assert phase['actual_exit_code']==0 and phase['residual_owned_members']==[]
        accepted[n]=dict(plan_path=str(plan_path),plan_sha256=sha(plan_path),delivery_path=str(delivery_path),delivery_sha256=sha(delivery_path),plan=plan,delivery=delivery)
    return accepted


def make_config(template,item,purpose,path):
    source=BASE/'functional/prepare.py';assert sha(source)==PREPARE_SHA
    spec=importlib.util.spec_from_file_location('_VJ_original_metadata_factory',source)
    factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
    factory.ROOT,factory.CASE=ROOT,CASE
    delivery=copy.deepcopy(item['delivery']);delivery['purpose']=purpose
    cfg=factory.make_config(template,delivery)
    cfg['delivery_binding']=dict(path=item['delivery_path'],sha256=item['delivery_sha256'])
    cfg['cache_plan_binding']=dict(path=item['plan_path'],sha256=item['plan_sha256'])
    group=cfg['group_id']
    cfg['durability']=dict(group_id=group,pilot_output=str(CASE/'pilot/gates'/group),formal_output=str(CASE/'runs'/group),ack_root=str(CASE/'control/backup_acks'),failure_root=str(CASE/'control/backup_failures'))
    if purpose=='pilot':
        cfg['diagnostic_scope']='bounded downstream gate on accepted full cache/full official NPZ; no new embedding pilot'
    factory.CASE=BASE
    cfg['commands']=factory.commands(path,cfg)
    cfg['commands']['native_audit']=[str(BASE/'functional/audit_native.py'),'--config',str(path),'--mode','tiny' if purpose=='pilot' else 'full','--output',str(Path(cfg['execution_root'])/'native_sr'/group)]
    return cfg


def environments():
    phases=read(BASE_ENV)['phases']
    return {key:dict(next(p['environment'] for p in phases if p['name'].endswith(suffix))) for key,suffix in [('cpu','_06_accept'),('gpu','_05_train'),('native','_07_native')]}


def build_queue(configs,mode,pins):
    env=environments();phases=[]
    for n in COUNTS:
        path,cfg=configs[n,'pilot' if mode=='smoke' else 'formal'];g=cfg['group_id'];cmd=cfg['commands']
        backup=[str(CASE/'backup_gate.py'),'--config',str(path),'--stage','gate' if mode=='smoke' else 'train','--report',str(CASE/'control'/mode/'durability_reports'/(g+'.json'))]
        steps=([('verify_hooks',[str(BASE/'functional/runtime.py'),'verify-hooks','--config',str(path)],'cpu'),('gate',cmd['gate'],'gpu'),('backup_gate',backup,'cpu'),('reload',cmd['reload'],'cpu'),('native_tiny',cmd['native'],'native'),('tiny_audit',cmd['native_audit'],'cpu')] if mode=='smoke' else [('train',cmd['train'],'gpu'),('backup_gate',backup,'cpu'),('accept',cmd['accept'],'cpu'),('native_full',cmd['native'],'native'),('native_audit',cmd['native_audit'],'cpu')])
        for i,(label,command,kind) in enumerate(steps,1):
            argv=[str(PY),'-B']+command
            if env[kind]['CUDA_VISIBLE_DEVICES']=='0':argv=[str(PY),'-B',str(ROOT/'cube4_visual_gciql_v1/control_v1/gpu_lease.py'),'--']+argv
            phases.append(dict(name=f'{g}_{i:02d}_{label}',argv=argv,cwd=str(ROOT),environment=env[kind]))
    return dict(schema_version=1,queue='vj_downstream',mode=mode,families=[FAMILY],cube_counts=[1,2,3,4],expected_features=11011000,training_updates=0 if mode=='smoke' else 4000000,SR_episodes=0 if mode=='smoke' else 1000,diagnostic_optimizer_updates=8 if mode=='smoke' else 0,process_guard=str(ROOT/'prepare_runtime.py'),own_RSS_limit_bytes=60<<30,phases=phases,source_pins=pins)


def space_admission():
    owned={}
    if STORE.exists():
        for p in STORE.rglob('*'):
            if p.is_file() and p.resolve().is_relative_to(STORE.resolve()):
                s=p.stat();owned[s.st_dev,s.st_ino]=s.st_size
    consumed=sum(owned.values());free=shutil.disk_usage(STORE if STORE.exists() else STORE.parent).free
    remaining=max(0,BUDGET-consumed);assert free>=remaining+MARGIN,(free,remaining+MARGIN)
    return dict(actual_exit_code=0,at=time.time(),artifact_budget_bytes=BUDGET,credited_owned_artifact_bytes=consumed,remaining_artifact_budget_bytes=remaining,margin_bytes=MARGIN,required_now=remaining+MARGIN,free_bytes=free,excess_bytes=free-remaining-MARGIN,filesystem_free_not_user_quota=True)


def prepare():
    accepted=cache_admission()
    assert not STORE.exists() and not (CASE/'configs').exists(),'Fresh namespaces required'
    admission=space_admission();STORE.mkdir()
    for name in ('runs','gates','acceptance','native_sr','pilot','control'):
        (STORE/name).mkdir();(CASE/name).symlink_to(STORE/name,target_is_directory=True)
    save(CASE/'control/STORAGE_ADMISSION.json',admission)
    template_path=ROOT/'cube23_distance_gciql_v1/five_group_extension_v1/configs/G08_cube3_rgb.json';template=read(template_path);configs={}
    for n,item in accepted.items():
        for purpose in ('pilot','formal'):
            path=CASE/'configs'/purpose/f'{FAMILY}_cube{n}.json';cfg=make_config(template,item,purpose,path)
            cfg['parent_template']=dict(path=str(template_path),sha256=sha(template_path));save(path,cfg);configs[n,purpose]=(path,cfg)
    backup_path=CASE/'control/backup_config.json'
    save(backup_path,dict(case_root=str(CASE),config_paths=[str(configs[n,p][0]) for p in ('pilot','formal') for n in COUNTS],local_root=LOCAL,ssh_host='root@connect.westd.seetacloud.com',ssh_port=31435,remote_python=str(PY),remote_script=str(CASE/'backup_watch.py'),remote_config=str(backup_path)))
    original=BASE/'control/fp16_bulk_w8_v1/launch_plan.json';pins={str(original):sha(original),str(BASE_ENV):sha(BASE_ENV),str(template_path):sha(template_path),str(backup_path):sha(backup_path)}
    runtime=original.parent/'runtime'
    for p in [runtime/'closure.json',runtime/'outer_exit_code',runtime.parent/'wrapper_exit_code']+[runtime/f'{FAMILY}_cube{n}_{suffix}_exit.json' for n in COUNTS for suffix in ('01_encode_full','02_cache_audit','03_deliver')]:
        pins[str(p)]=sha(p)
    for path,digest in read(original)['source_pins'].items():
        if Path(path).suffix=='.py':pins[str(ROOT/path)]=digest
    for item in accepted.values():
        for rel,digest in item['plan']['source_pins'].items():pins[str(ROOT/rel)]=digest
        for key in ('plan_path','delivery_path'):pins[item[key]]=item[key.replace('_path','_sha256')]
        plan=item['plan'];home=Path(item['plan_path']).parent;cache=Path(plan['cache_dir'])
        metadata=[home/'full_audit_v1/report.json',home/'full_audit_v1/actual_exit_code',cache/'cache_manifest.json',cache/'cache_seal.json',cache/'precision_reference_alignment.json',Path(plan['shared_state_root'])/'source_arrays.json',Path(plan['encoder_spec']),Path(plan['encoder_adapter'])]
        metadata += [Path(plan[key]) for key in ('data_acceptance','source_stat_freeze') if key in plan]
        for p in metadata:pins[str(p)]=sha(p)
        declaration=read(plan['encoder_spec'])
        for rel,digest in declaration.get('source_files_sha256',{}).items():pins[str(Path(declaration['source_root'])/rel)]=digest
    for directory in (CASE,BASE/'functional'):
        for p in directory.glob('*.py'):pins[str(p)]=sha(p)
    for p in CASE.glob('*.sh'):pins[str(p)]=sha(p)
    for path,_ in configs.values():pins[str(path)]=sha(path)
    for path,digest in pins.items():assert sha(path)==digest,path
    for mode in ('smoke','full'):save(CASE/'control'/mode/'launch_plan.json',build_queue(configs,mode,pins))
    save(CASE/'PREPARED.json',dict(actual_exit_code=0,at=time.time(),groups=[f'{FAMILY}_cube{n}' for n in COUNTS],source_pins=len(pins),smoke_phases=24,formal_phases=20,original_sources_modified=False,embedding_rerun=False,NN_started=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true')
    if parser.parse_args().execute:prepare()
    else:print(json.dumps(dict(prepared_only=True,NN_started=False)))
