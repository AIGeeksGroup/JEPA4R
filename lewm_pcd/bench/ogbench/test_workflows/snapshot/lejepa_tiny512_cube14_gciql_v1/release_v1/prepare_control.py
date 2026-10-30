"""Build/publish fixed Tiny512 metadata; no learner/controller/model implementation."""
import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys

ROOT=PurePosixPath('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE=ROOT/'lejepa_tiny512_cube14_gciql_v1'
STORE=PurePosixPath('/root/autodl-tmp/lejepa_tiny512_cube14_20261029_v1_outputs')
PY=ROOT/'env_runtime/bin/python';F=ROOT/'foundation_visual_gciql_v1';H=CASE/'release_v1'
TEMPLATE=ROOT/'cube23_distance_gciql_v1/five_group_extension_v1/configs/G08_cube3_rgb.json'
LEASE=ROOT/'cube4_visual_gciql_v1/control_v1/gpu_lease.py'
BIG_ROLES={'tiny_epoch10_weights','official_NPZ','official_qpos_qvel_NPY'}
EXTRA_PINS={str(ROOT/'prepare_runtime.py'):'0331473db0b0566d3097c8cd9cf7362a820badc8c8537134ea39388ab79e6063',str(TEMPLATE):'6d717c2123217020fec593367f3eecb42e01dfe203e9aae8d5a38733045f55b5'}
COUNTS={1:dict(train=1000,val=100),2:dict(train=1000,val=100),3:dict(train=3000,val=300),4:dict(train=5000,val=500)}
SUFFIXES=('01_encode_full','02_cache_audit','03_deliver','04_hooks','05_gate','06_reload','07_native_tiny','08_tiny_audit','09_train','10_accept','11_native_full','12_native_audit')
BASE_ENV_SHA='ce36a0f67e573fe8c19a52fc2c5d6e74eb7c0af87bea81429d067315c4ccdbd4'
ROLE_ENV_SHA='c09c2589b11c866a137be705432eeaf08ae75db06e53d1417b7c2144b697b5c5'
EXPECTED_INPUT_SHA='ae9f818f53c48344eb224a60f8ec7efa15ac676484ceb77c708845cec434d7f1'
SOURCE_SERIAL_SHA='dac39378e01c4c76d0a78dce472c9bb6e937d9f909f4ae12eabe2b7362cf6a38'


def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def encode(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def canonical(value):return digest(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def group(n):assert type(n) is int and n in COUNTS;return f'lejepa_tiny512_cube{n}'
def gp(n):return CASE/'cache/groups'/group(n)
def cfg(n,purpose):return CASE/'configs'/purpose/(group(n)+'.json')
def st(path):
    x=Path(path).stat();return dict(size=x.st_size,mtime_ns=x.st_mtime_ns,ctime_ns=x.st_ctime_ns,inode=x.st_ino,device=x.st_dev)


def environment_roles(raw):
    assert digest(raw)==BASE_ENV_SHA
    items={x['name'].split('_cube1_')[1]:x['environment'] for x in json.loads(raw)['phases']}
    roles={'encode':items['01_encode_full'],'cpu':items['06_accept'],'train':items['05_train'],'native':items['07_native'],'audit':items['08_native_audit']}
    assert canonical(roles)==ROLE_ENV_SHA
    return roles


def budget(measured_cp_bytes,static_bytes):
    # One whole-agent checkpoint already contains Adam and targets; do not multiply it again.
    slot=max(128<<20,int(measured_cp_bytes)+(1<<20))
    parts=dict(raw_FP32_D512=11011000*512*4,NPY_headers=11000*128,
        original_receipt_rawID_map_metadata_allowance=11000*(16<<10),
        whole_agent_52checkpoint_slots=52*slot,atomic_transaction_margin=256<<20,
        Native_arrays_reports_margin=4<<30,probe_margin=512<<20,logs_margin=512<<20,
        static_metadata_margin=max(128<<20,int(static_bytes)))
    amount=max(36000000000,sum(parts.values()))
    return dict(artifact_budget_bytes=amount,reserve_bytes=8<<30,components=parts,checkpoint_slot_bytes=slot,
        checkpoint_slot_is_whole_agent_including_Adam_once=True,actual_new_Tiny_CP_size_not_yet_measured=True,
        map_format='Original perEP raw start/count/terminal/IDs SHA + manifest receipt references; no new dense disk maps or raw RGB',
        source_state_arrays_readonly_reused_no_new_extraction=True,estimate_not_quota=True,budget_may_increase_not_decrease=True)


def space_contract(free,paid,contract):
    assert free>=0 and paid>=0
    required=max(0,contract['artifact_budget_bytes']-paid)+contract['reserve_bytes']
    return dict(free_bytes=free,owned_STORE_paid_bytes=paid,required_free_bytes=required,
        remaining_headroom_bytes=free-required,enough=free>=required,paid_deducted_exactly_once=True,
        artifact_budget_bytes=contract['artifact_budget_bytes'],reserve_bytes=contract['reserve_bytes'])


def owned_bytes(root):
    root=Path(root);seen=set();total=0
    if not root.exists():return 0
    assert root.is_dir() and not root.is_symlink()
    for parent,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[n for n in dirs if not (Path(parent)/n).is_symlink()]
        for name in files:
            p=Path(parent)/name
            if p.is_symlink():continue
            x=p.stat();key=(x.st_dev,x.st_ino)
            if key not in seen:seen.add(key);total+=x.st_size
    return total


def _command(script,*args,lease=False):
    command=[str(PY),'-B',str(script),*[str(x) for x in args]]
    return [str(PY),'-B',str(LEASE),'--',*command] if lease else command


def phases(n,roles):
    g=group(n);home=gp(n);plan=home/'plan.json';run=CASE/'outputs/formal/runs'/g
    pilot=CASE/'outputs/pilot';formal=CASE/'outputs/formal';gate=pilot/'gates'/g;nt=pilot/'native_sr'/g;nf=formal/'native_sr'/g
    bridge=H/'tiny_bridge.py';runtime=CASE/'control/full/runtime'
    rows=[
      ('encode',_command(bridge,'cache-producer','--plan',plan,'--execute',lease=True),[]),
      ('cpu',_command(bridge,'cache-audit','--plan',plan,'--cache-dir',CASE/'outputs/cache'/g/'cache','--output-dir',home/'full_audit_v1'),[]),
      ('cpu',_command(F/'cache/deliver.py','--plan',plan,'--audit-report',home/'full_audit_v1/report.json','--producer-exit-file',runtime/(g+'_01_encode_full_exit.json')),[]),
      ('cpu',_command(bridge,'runtime','verify-hooks','--config',cfg(n,'pilot'),'--delivery',home/'delivery.json','--formal-config',cfg(n,'formal'),'--cache-control',runtime),[]),
      ('train',_command(bridge,'runtime','gate','--config',cfg(n,'pilot'),'--output',gate,lease=True),[]),
      ('cpu',_command(bridge,'runtime','reload','--config',cfg(n,'pilot'),'--output',gate,'--report',pilot/'gates'/(g+'_reload.json')),[]),
      ('native',_command(bridge,'native','--config',cfg(n,'pilot'),'--mode','tiny','--gate-smoke','--output',nt,lease=True),[]),
      ('audit',_command(bridge,'audit','--config',cfg(n,'pilot'),'--mode','tiny','--output',nt),[]),
      ('train',_command(bridge,'runtime','train','--config',cfg(n,'formal'),lease=True),[]),
      ('cpu',_command(bridge,'runtime','accept','--config',cfg(n,'formal'),'--output',run,'--report',formal/'acceptance'/(g+'.json')),[]),
      ('native',_command(bridge,'native','--config',cfg(n,'formal'),'--mode','full','--output',nf,lease=True),[]),
      ('audit',_command(bridge,'audit','--config',cfg(n,'formal'),'--mode','full','--output',nf),[])]
    checks=[
      [(CASE/'outputs/cache'/g/'cache/cache_seal.json',dict(actual_exit_code=0,full_raw_ID_coverage=True,source_encoder_frozen=True,per_episode_readback_exact=True))],
      [(home/'full_audit_v1/report.json',dict(actual_exit_code=0,episodes_verified=sum(COUNTS[n].values()),raw_rows_verified=sum(COUNTS[n].values())*1001,main_rows_verified=sum(COUNTS[n].values())*1000,latent_dim=512,dtype='float32'))],
      [(home/'delivery.json',dict(purpose='formal',group_id=g,cube_count=n,latent_dim=512))],
      [(CASE/'configs/seals'/(g+'.json'),dict(actual_exit_code=0,group_id=g,NN_performed=False))],
      [(gate/'report.json',dict(actual_exit_code=0,batch_size=1024,updates=2,actual_backend='gpu',network_step=3,adam_count=2,encoder_calls=0,WM_dynamics_calls=0))],
      [(pilot/'gates'/(g+'_reload.json'),dict(actual_exit_code=0,full153_state_exact=True,next_batch_and_rawIDs_exact=True,finite_actions=True,no_updates=True))],
      [(nt/'report.json',dict(actual_exit_code=0,episodes=1,unique_seeds=1,online_encoder=True,encoding_protocol='fresh_current_and_goal_each_policy_call_2N',temperature=0.0,gaussian_noise=None,extra_policy_warmup=0))],
      [(nt/'independent_audit.json',dict(actual_exit_code=0,episodes=1,actual_step_arrays_checked=True,mode='tiny',diagnostic_excluded_from_formal_SR=True))],
      [(run/'status.json',dict(exit_code=0,completed_updates=1000000,adam_count=1000000,network_step=1000001,validation_count=200,failed_stages=0))],
      [(formal/'acceptance'/(g+'.json'),dict(actual_exit_code=0,full153_state_exact=True,Adam_count=1000000,network_step=1000001,CSV_validation_records=200,finite_actions=True,actual_current_dim=531,actual_goal_dim=512+3*n,entire_1M_host_sampler_replay_performed=False))],
      [(nf/'report.json',dict(actual_exit_code=0,episodes=250,unique_seeds=250,max_episode_steps={1:200,2:500,3:1000,4:1000}[n],online_encoder=True,encoding_protocol='fresh_current_and_goal_each_policy_call_2N',temperature=0.0,gaussian_noise=None,extra_policy_warmup=0))],
      [(nf/'independent_audit.json',dict(actual_exit_code=0,episodes=250,unique_seeds=250,actual_step_arrays_checked=True,mode='full',diagnostic_excluded_from_formal_SR=False))]]
    return [dict(name=g+'_'+suffix,argv=argv,environment=copy.deepcopy(roles[role]),cwd=str(ROOT),checks=[dict(path=str(path),equals=values) for path,values in check]) for suffix,(role,argv,_),check in zip(SUFFIXES,rows,checks)]


def validate_launch(plan):
    assert plan['queue']=='tiny_serial' and plan['families']==['lejepa_tiny512'] and plan['cube_counts']==[1,2,3,4]
    assert (plan['training_updates'],plan['SR_episodes'],plan['expected_features'])==(4000000,1000,11011000)
    assert plan['own_RSS_limit_bytes']==60<<30 and plan['process_guard']==str(ROOT/'prepare_runtime.py')
    assert plan['groups']==[dict(group_id=group(n),family='lejepa_tiny512',cube_count=n,latent_dim=512) for n in range(1,5)]
    assert canonical(plan['environment_roles'])==ROLE_ENV_SHA
    assert plan['phases']==sum([phases(n,plan['environment_roles']) for n in range(1,5)],[])
    assert len(plan['phases'])==len({x['name'] for x in plan['phases']})==48
    for path in plan['source_pins']:
        assert not any(s in path for s in ('/configs/pilot/','/configs/formal/','/full_audit_v1/','/delivery.json','weights_epoch_10.pt'))
    return True


def build_pack(rebind_dir,adapter_dir,expected_inputs,baseline_env,input_admission,checkpoint_reference,accepted_existing=None):
    rebind_dir,adapter_dir=Path(rebind_dir),Path(adapter_dir)
    raw_expected=Path(expected_inputs).read_bytes();assert digest(raw_expected)==EXPECTED_INPUT_SHA
    expected=json.loads(raw_expected);origin=read(rebind_dir/'PACK_MANIFEST.json');admission=read(input_admission)
    assert origin['actual_exit_code']==admission['actual_exit_code']==0 and origin['case_root']==str(CASE)
    assert admission['source_actual_SHA256']==origin['input_integrity_binding']['sha256']
    assert admission['manifest_SHA256']==origin['input_integrity_binding']['expected_manifest_sha256']==EXPECTED_INPUT_SHA
    raw_actual=(rebind_dir/'clone_rebind/INPUT_INTEGRITY_ACTUAL.json').read_bytes();actual=json.loads(raw_actual)
    assert digest(raw_actual)==origin['input_integrity_binding']['sha256'] and actual['actual_exit_code']==0 and actual['file_count']==186 and actual['split_count']==8
    af=read(adapter_dir/'SOURCE_FREEZE.json');assert af['actual_exit_code']==0 and af['component5_unchanged']
    for rel,record in af['files'].items():assert sha(adapter_dir/rel)==record['sha256'] and (adapter_dir/rel).stat().st_size==record['bytes']
    payload={}
    def add(rel,raw):
        assert not PurePosixPath(rel).is_absolute() and '..' not in PurePosixPath(rel).parts and rel not in payload
        payload[rel]=raw
    for rel,record in origin['files'].items():
        raw=(rebind_dir/rel).read_bytes();assert digest(raw)==record['sha256'] and len(raw)==record['bytes'] and record['path']==str(CASE/rel)
        add(rel,raw)
    assert len(payload)==26
    assert set(origin['groups'])=={'1','2','3','4'}
    for n in range(1,5):
        g=origin['groups'][str(n)];prefix=f'clone_rebind/cube{n}/'
        arrays=json.loads(payload[prefix+'source_arrays.json']);accept=json.loads(payload[prefix+'data_acceptance.json']);frozen=json.loads(payload[prefix+'stat-freeze.json'])
        assert g['cube_count']==n and g['data_binding']['cube_count']==n and accept['actual_exit_code']==frozen['actual_exit_code']==0
        assert g['data_binding']['splits']==accept['datasets']
        assert g['source_arrays_receipt_sha256']==digest(payload[prefix+'source_arrays.json'])
        assert g['data_acceptance_sha256']==digest(payload[prefix+'data_acceptance.json'])
        assert g['source_stat_freeze_sha256']==digest(payload[prefix+'stat-freeze.json'])
        assert frozen['acceptance_sha256']==g['data_acceptance_sha256']
        assert g['arrays_contract']==arrays['contract'] and len(g['arrays_contract'])==7
        assert g['arrays_protocol_sha256']==arrays['protocol_sha256']==canonical(arrays['contract'])
        assert g['source_arrays_extractor_sha256']==arrays['contract']['extractor_source_sha256']=='a05b712d49026a0d52f43e4e43b3106879b0e965af2abb61a8f017bae523f595'
        assert arrays['contract']['data_acceptance_sha256']==g['data_acceptance_sha256'] and arrays['contract']['source_stat_freeze_sha256']==g['source_stat_freeze_sha256']
        assert arrays['contract']['split_episodes']==COUNTS[n] and arrays['extraction_performed_by_this_case'] is False
    for rel in ('tiny_encoder.py','tiny_bridge.py','ENCODER_COMPONENT_FREEZE.json'):
        add('release_v1/'+rel,(adapter_dir/rel).read_bytes())
    add('release_v1/ADAPTER_SOURCE_FREEZE.json',(adapter_dir/'SOURCE_FREEZE.json').read_bytes())
    add('release_v1/prepare_control.py',Path(__file__).read_bytes())
    add('release_v1/run_dispatch.py',Path(__file__).with_name('run_dispatch.py').read_bytes())
    add('provenance/INPUT_CONTENT_ADMISSION_ACCEPTED.json',Path(input_admission).read_bytes())
    add('provenance/input_integrity_actual.json',raw_actual)
    add('provenance/baseline_environments.json',Path(baseline_env).read_bytes())
    add('provenance/CLONE_REBIND_PACK_MANIFEST.json',(rebind_dir/'PACK_MANIFEST.json').read_bytes())
    accepted_path=Path(accepted_existing) if accepted_existing else Path(input_admission).parent/'encoder_gate_v1/publish_component_actual.json'
    accepted=read(accepted_path);assert accepted['actual_exit_code']==0 and accepted['CASE']==str(CASE) and accepted['file_count']==8 and len(accepted['files'])==8
    assert accepted['NN_performed'] is False
    add('provenance/ENCODER_COMPONENT_PUBLICATION_ACCEPTED.json',accepted_path.read_bytes())
    roles=environment_roles(payload['provenance/baseline_environments.json'])
    big=[x for x in expected['files'] if x['role'] in BIG_ROLES];assert len(big)==28
    basepins={x['path']:x['expected_sha256'] for x in expected['files'] if x['role'] not in BIG_ROLES};assert len(basepins)==158
    assert set(basepins).isdisjoint(EXTRA_PINS);basepins.update(EXTRA_PINS)
    for n in range(1,5):
        spec_raw=(adapter_dir/'specs'/f'cube{n}.json').read_bytes();spec=json.loads(spec_raw);assert spec['cube_count']==n and spec['family']=='lejepa_tiny512' and spec['output_dim']==512
        add(f'cache/groups/{group(n)}/encoder_spec.json',spec_raw)
        add(f'cache/groups/{group(n)}/data_binding.json',encode(origin['groups'][str(n)]['data_binding']))
        add(f'cache/groups/{group(n)}/camera.json',encode(spec['downstream_camera']))
    static={str(CASE/rel):digest(raw) for rel,raw in payload.items()}
    allbase={**basepins,**static}
    for n in range(1,5):
        spec=json.loads(payload[f'cache/groups/{group(n)}/encoder_spec.json']);g=origin['groups'][str(n)]
        assert g['cube_count']==n and g['shared_state_root']==str(CASE/'clone_rebind'/f'cube{n}') and g['extraction_performed_by_this_case'] is False
        pins={str(PurePosixPath(path).relative_to(ROOT)):digest_ for path,digest_ in allbase.items() if PurePosixPath(path).is_relative_to(ROOT)}
        plan=dict(workspace=str(ROOT),root_workspace=str(ROOT),cube_count=n,latent_dim=512,encoder_family='lejepa_tiny512',
            shared_state_root=g['shared_state_root'],source_arrays_receipt_sha256=g['source_arrays_receipt_sha256'],source_arrays_extractor_sha256=g['source_arrays_extractor_sha256'],
            data_acceptance=g['data_acceptance'],data_acceptance_sha256=g['data_acceptance_sha256'],source_stat_freeze=g['source_stat_freeze'],source_stat_freeze_sha256=g['source_stat_freeze_sha256'],
            checkpoint=spec['checkpoint'],checkpoint_sha256=spec['checkpoint_sha256'],encoder_spec=str(gp(n)/'encoder_spec.json'),encoder_spec_sha256=digest(payload[f'cache/groups/{group(n)}/encoder_spec.json']),
            encoder_adapter=str(H/'tiny_encoder.py'),encoder_adapter_sha256=digest(payload['release_v1/tiny_encoder.py']),source_pins=pins,camera=spec['downstream_camera'],
            renderer_workers=2,prefetch=2,encoder_batch=32,torch_cpu_threads=1,split_episodes=COUNTS[n],raw_rows_per_episode=1001,output_dtype='float32',storage_dtype='float32',inference_precision='fp32',
            cache_dir=str(CASE/'outputs/cache'/group(n)/'cache'),pixel_preprocess_device='shared_cpu',preprocess_gate={},encoder_preprocessing_device='cpu',host_tensor_dtype='uint8',encoder_preprocess=spec['preprocessing'],
            pixel_preprocess_policy='Original Tiny returned PIL224 transforms once on CPU; frozen CUDAFP32 model',training_updates=0,SR_episodes=0)
        add(f'cache/groups/{group(n)}/plan.json',encode(plan))
    reference=read(checkpoint_reference);assert reference['actual_exit_code']==0 and reference['all_local_SHA_size_exact'] and reference['all_source_before_after_stat_exact']
    add('provenance/D512_CP_SIZE_REFERENCE.json',Path(checkpoint_reference).read_bytes())
    contract=budget(reference['bytes'],sum(len(v) for v in payload.values()))
    add('control/full/SPACE_BUDGET.json',encode(dict(**contract,reference_path=str(CASE/'provenance/D512_CP_SIZE_REFERENCE.json'),original_local_reference=str(checkpoint_reference),reference_SHA256=sha(checkpoint_reference),source_CP_scope='Accepted D512 Cube4 final transaction; new52wholeCP slots128MiB are conservative estimates')))
    pins={**basepins,**{str(CASE/rel):digest(raw) for rel,raw in payload.items()}}
    launch=dict(schema_version=1,queue='tiny_serial',families=['lejepa_tiny512'],cube_counts=[1,2,3,4],
        groups=[dict(group_id=group(n),family='lejepa_tiny512',cube_count=n,latent_dim=512) for n in range(1,5)],expected_features=11011000,training_updates=4000000,SR_episodes=1000,
        diagnostic_optimizer_updates=8,diagnostic_SR_episodes=4,process_guard=str(ROOT/'prepare_runtime.py'),own_RSS_limit_bytes=60<<30,source_pins=pins,environment_roles=roles,
        phases=sum([phases(n,roles) for n in range(1,5)],[]),backup_advisory=True,full_budget=contract)
    validate_launch(launch);add('control/full/launch_plan.json',encode(launch))
    records={rel:dict(path=str(CASE/rel),sha256=digest(raw),bytes=len(raw)) for rel,raw in payload.items()}
    accepted_names={'release_v1/tiny_encoder.py','release_v1/ENCODER_COMPONENT_FREEZE.json','provenance/input_integrity_actual.json','provenance/INPUT_CONTENT_ADMISSION_ACCEPTED.json'}|{f'cache/groups/{group(n)}/encoder_spec.json' for n in range(1,5)}
    assert set(accepted['files'])==accepted_names
    accepted_map={rel:{key:record[key] for key in ['sha256','bytes']} for rel,record in accepted['files'].items()}
    for rel,record in accepted_map.items():assert records[rel]['sha256']==record['sha256'] and records[rel]['bytes']==record['bytes']

    manifest=dict(schema_version=1,actual_exit_code=0,case_root=str(CASE),STORE=str(STORE),files=records,
        launch_plan_SHA256=digest(payload['control/full/launch_plan.json']),phase_count=48,excluded_large_input_pins=28,old_small_source_pins=158,supplemental_control_source_pins=2,accepted_existing=accepted_map,accepted_existing_publication_SHA256=sha(accepted_path),
        input_actual_SHA256=digest(raw_actual),target_hostname=actual['hostname'],full_budget=contract,future_cfg_delivery_audit_SHA_prefilled=False,
        clone_rebind_pack_manifest_SHA256=sha(rebind_dir/'PACK_MANIFEST.json'),adapter_freeze_SHA256=sha(adapter_dir/'SOURCE_FREEZE.json'),NN_performed=False)
    return payload,manifest


def write_pack(output,payload,manifest):
    output=Path(output);assert not output.exists(),'Existing pack or partial refused'
    output.mkdir(parents=True)
    for rel,raw in payload.items():
        target=output/'payload'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(raw)
    with (output/'PACK_MANIFEST.json').open('xb') as stream:stream.write(encode(manifest))


def write_owned(root,relative,raw):
    root=Path(root);rel=PurePosixPath(relative);assert not rel.is_absolute() and '..' not in rel.parts
    target=root.joinpath(*rel.parts);assert target.resolve().is_relative_to(root.resolve())
    if target.exists():
        assert target.is_file() and target.stat().st_size==len(raw) and sha(target)==digest(raw),'Existing nonmatching/partial file refused: '+str(target)
        return 'reused_byteexact'
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:stream.write(raw)
    assert target.stat().st_size==len(raw) and sha(target)==digest(raw)
    return 'written'


def existing_admission(root,payload,accepted):
    root=Path(root);reused=[]
    for rel,raw in payload.items():
        path=root.joinpath(*PurePosixPath(rel).parts)
        assert path.resolve().is_relative_to(root.resolve())
        if path.exists():
            assert rel in accepted,'Unknown existing/partial file refused even if same bytes: '+rel
            record=accepted[rel]
            assert path.is_file() and path.stat().st_size==record['bytes']==len(raw) and sha(path)==record['sha256']==digest(raw)
            reused.append(rel)
    for rel in accepted:assert root.joinpath(*PurePosixPath(rel).parts).is_file(),'Previously accepted file missing: '+rel
    return reused


def _actual_stats(actual):
    assert actual['actual_exit_code']==0
    import platform
    assert platform.node()==actual['hostname'],'Target instance changed'
    for item in actual['files']:assert st(item['path'])==item['source_after_all_content_checks'],item['path']
    return dict(actual_exit_code=0,checked_current_stats=len(actual['files']),payloads_rehashed=0)


def input_stat_gate():
    return _actual_stats(read(str(CASE/'clone_rebind/INPUT_INTEGRITY_ACTUAL.json')))


def installed_preflight(expected_plan_sha,require_fresh=True):
    path=Path(str(CASE/'control/full/launch_plan.json'));assert sha(path)==expected_plan_sha
    plan=read(path);validate_launch(plan)
    for name,digest_ in plan['source_pins'].items():assert sha(name)==digest_,name
    inputs=input_stat_gate();store=Path(str(STORE));outputs=Path(str(CASE/'outputs'))
    assert store.is_dir() and not store.is_symlink() and outputs.is_symlink() and outputs.resolve()==store.resolve()
    if require_fresh:
        assert not Path(str(CASE/'control/full/runtime')).exists()
        assert not Path(str(CASE/'control/full/wrapper_exit_code')).exists()
        assert owned_bytes(store)==0,'Existing experiment artifacts require inspection, not duplicate dispatch'
    space=space_contract(shutil.disk_usage('/root/autodl-tmp').free,owned_bytes(store),plan['full_budget'])
    return dict(actual_exit_code=0,ready=space['enough'],space=space,input_stat_gate=inputs,launch_plan_SHA256=expected_plan_sha,NN_performed=False)


def publish_bundle(bundle):
    manifest=bundle['manifest'];assert manifest['actual_exit_code']==0 and manifest['case_root']==str(CASE) and manifest['STORE']==str(STORE)
    store=Path(str(STORE));outputs=Path(str(CASE/'outputs'));runtime=Path(str(CASE/'control/full/runtime'))
    assert not store.exists() and not outputs.exists() and not outputs.is_symlink() and not runtime.exists(),'Fresh STORE/outputs/runtime required; preserve partial'
    payload={rel:base64.b64decode(raw,validate=True) for rel,raw in bundle['payload_b64'].items()};assert set(payload)==set(manifest['files'])
    for rel,raw in payload.items():
        record=manifest['files'][rel];assert record['path']==str(CASE/rel) and record['sha256']==digest(raw) and record['bytes']==len(raw)
    plan=json.loads(payload['control/full/launch_plan.json']);validate_launch(plan)
    for name,digest_ in plan['source_pins'].items():
        if not PurePosixPath(name).is_relative_to(CASE):assert sha(name)==digest_,name
    reused=existing_admission(Path(str(CASE)),payload,manifest['accepted_existing'])
    _actual_stats(json.loads(payload['clone_rebind/INPUT_INTEGRITY_ACTUAL.json']))
    admission=space_contract(shutil.disk_usage('/root/autodl-tmp').free,0,manifest['full_budget'])
    if not admission['enough']:
        print(json.dumps(dict(actual_exit_code=0,status='WAIT_FOR_SPACE',space=admission,published=False,NN_performed=False)));return
    results={rel:write_owned(Path(str(CASE)),rel,raw) for rel,raw in payload.items()}
    store.mkdir();outputs.parent.mkdir(parents=True,exist_ok=True);outputs.symlink_to(store,target_is_directory=True)
    result=installed_preflight(manifest['launch_plan_SHA256'])
    result.update(status='static_published_no_dispatch',files=results,accepted_existing_reused=reused,unknown_existing_files_accepted=False,NN_performed=False)
    print(json.dumps(result))


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    build=sub.add_parser('build')
    for key in ['rebind-pack','adapter-dir','expected-inputs','baseline-env','input-admission','checkpoint-reference','output']:build.add_argument('--'+key,type=Path,required=True)
    build.add_argument('--accepted-existing',type=Path)
    emit=sub.add_parser('emit-publisher');emit.add_argument('--pack',type=Path,required=True);emit.add_argument('--output',type=Path,required=True)
    check=sub.add_parser('preflight');check.add_argument('--expected-plan-sha256',required=True)
    args=parser.parse_args()
    if args.action=='build':
        payload,manifest=build_pack(args.rebind_pack,args.adapter_dir,args.expected_inputs,args.baseline_env,args.input_admission,args.checkpoint_reference,args.accepted_existing)
        write_pack(args.output,payload,manifest);print(json.dumps(dict(actual_exit_code=0,output=str(args.output),files=len(payload),phase_count=48,NN_performed=False)));return
    if args.action=='preflight':print(json.dumps(installed_preflight(args.expected_plan_sha256)));return
    manifest=read(args.pack/'PACK_MANIFEST.json');payload={rel:(args.pack/'payload'/rel).read_bytes() for rel in manifest['files']}
    for rel,raw in payload.items():assert digest(raw)==manifest['files'][rel]['sha256'] and len(raw)==manifest['files'][rel]['bytes']
    bundle=dict(manifest=manifest,payload_b64={rel:base64.b64encode(raw).decode() for rel,raw in payload.items()})
    original=Path(__file__).read_text();marker="\nif __name__=='__main__':main()";assert original.count(marker)==1
    generated=original.replace(marker,'')+'\npublish_bundle(json.loads(base64.b64decode('+repr(base64.b64encode(json.dumps(bundle).encode()).decode())+')))\n'
    with args.output.open('x') as stream:stream.write(generated)
    print(json.dumps(dict(actual_exit_code=0,publisher=str(args.output),NN_performed=False)))


if __name__=='__main__':main()
