"""CASE3 C4-only13 checkpoint metadata adapter over frozen vendor copy/receiver."""
import argparse,copy,hashlib,importlib.util,json,os,sys,time,types
from pathlib import Path,PurePosixPath

ROOT=PurePosixPath('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
OLD_CASE=ROOT/'lejepa_tiny512_cube14_gciql_v1';CASE=ROOT/'lejepa_tiny512_cube4_resource_v3'
STORE=PurePosixPath('/root/autodl-tmp/lejepa_tiny512_cube14_20261029_v1_outputs/resource_c4_v3')
P=Path('D:/LeWM_Artifacts/coordination/lejepa_tiny512_cube14_gciql_test_20261028_v1')
LOCAL=P/'backup_c4_resource_v3';G='lejepa_tiny512_cube4'
ACK=str(CASE/'control/backup_ack/c4_advisory_v1/acks');FAIL=str(CASE/'control/backup_ack/c4_advisory_v1/failures')
STATIC_DIGEST='e18682ff9af043068d0a4f2664ecd8f5dfba0829b1aceff6fdd982e97f295e5a'
BASE_COMPONENT_DIGEST='3f32a9b0c044fdb13d7f94e977fae13bd9725d068a539afeaf0e8091e8d6d542'
OLD39_SHA='d0118500d132448f44f3578193d5957d5d4377a6f712d9952f60a14833aa9c00'
ORIGINALS={'tiny_backup_endpoint.py':'b4d7dd8c104edd84387ead894283ee4888bc3453c4a69905709beb551984fbaf','tiny_backup_receiver.py':'0d2727496610f8d25b69ddbda2d1b5290f8bbea2a2827fcb20feacd0bf594491','vendor/backup_copy.py':'0f88ebfe8a909fe68b21cb1f22a2bc4fb598110f1cff7b528a433a031babb571','vendor/backup_watch.py':'3fdcac5b385f26a4d39d8efd020580e88272ca0a1baf0b1d041cc7cb2cda4541','vendor/backup_gate.py':'8cbbc47218485b3a0206841f01ea2ea8cfd7fe9ae81dddb9cacd2b3f8f10a016','vendor/tests/test_backup.py':'fe02cf89eb5aa11b73006c50110ea34382c5ea7cb07e87565076e1644f706f97'}

def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):
    path=Path(path);require(path.suffix not in ('.pkl','.npy','.npz','.pt') and path.stat().st_size<=16<<20,'Metadata only')
    return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def wire(p):return str(PurePosixPath(str(p).replace('\\','/')))
def checkpoint_keys():return {(s,G,k) for s,cs in {'pilot':[1,2],'formal':[1]+list(range(100000,1000001,100000))}.items() for k in cs}

def expected_keys(cfg):
    require(cfg['case_root']==str(CASE) and cfg['store_root']==str(STORE),'Wrong new CASE3/STORE')
    require(cfg['local_root']==LOCAL.as_posix() and cfg['ack_root']==ACK and cfg['failure_root']==FAIL,'Wrong exclusive backup namespace')
    require(cfg['expected_checkpoint_count']==13 and cfg['expected_file_count']==52 and cfg['poll_seconds']==180 and cfg['copy_timeout_seconds']==1800 and cfg['advisory_only'] is True and cfg['future_config_SHA_prefilled'] is False,'Wrong advisory13/52 policy')
    want=[dict(stage=s,group_id=G,config_path=str(CASE/'configs'/s/(G+'.json'))) for s in ('pilot','formal')]
    require(cfg['config_sources']==want and set(cfg['groups'])=={'4'},'Only two dynamic new C4 config identities')
    group=cfg['groups']['4'];require(group['group_id']==G and group['encoder_spec_path']==str(OLD_CASE/'cache/groups'/G/'encoder_spec.json') and group['encoder_spec_sha256']=='71fc78da02ad2d6b1840491ef98a0dcc5fb228158b943f169566ee34f05151be' and group['checkpoint_sha256']=='66269b172e06cfbd3b50be0460604ef3f46dc27e1e283a67cef31813c3ae244b','Wrong original Cube4 encoder')
    require(digest(cfg['component_sources'])==BASE_COMPONENT_DIGEST and digest(cfg['resource_static_files'])==STATIC_DIGEST,'Frozen base/resource static source set differs')
    require(cfg['ssh_alias_reference']=='test' and cfg['ssh_host']=='root@connect.westc.seetacloud.com' and cfg['ssh_port']==16755 and cfg['expected_remote_hostname']=='autodl-container-1705479f78-1c6b9afc','Wrong target')
    require(cfg['remote_python']==str(ROOT/'env_runtime/bin/python') and cfg['remote_script']==str(CASE/'backup_advisory_v1/c4_checkpoint_backup.py') and cfg['remote_config']==str(CASE/'control/backup_ack/c4_advisory_v1/endpoint_config.json'),'Wrong independent endpoint path')
    require(cfg['module_sha256']==sha(__file__) and cfg['old39_sha256']==OLD39_SHA,'Wrong frozen candidate/old39 ledger')
    return checkpoint_keys()

def originals(local):
    base=P/'backup_candidate_v1' if local else Path(str(OLD_CASE/'backup_candidate_v1'))
    for name in ORIGINALS if local else ('tiny_backup_endpoint.py','vendor/backup_watch.py'):require(sha(base/name)==ORIGINALS[name],'Original frozen module differs: '+name)
    sys.path[:0]=[str(base),str(base/'vendor')]
    import tiny_backup_endpoint as e
    require(sha(e.__file__)==ORIGINALS['tiny_backup_endpoint.py'] and sha(e.b.__file__)==ORIGINALS['vendor/backup_watch.py'],'Wrong imported endpoint/vendor')
    if not local:return e,None,base
    import backup_copy as c
    require(sha(c.__file__)==ORIGINALS['vendor/backup_copy.py'],'Wrong vendor copier');return e,c,base

def resource_module(cfg):
    path=CASE/'release_v1/c4_resource_bridge.py';require(sha(path)==cfg['resource_static_files'][str(path)]['sha256'],'Resource bridge changed')
    spec=importlib.util.spec_from_file_location('_new_c4_backup_resource',Path(path));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def config_view_check(cfg,value,n,stage,original):
    require(n==4 and stage in ('pilot','formal'),'Wrong new C4 source purpose')
    require(value['execution_root']==str(CASE/'outputs'/stage) and value['output']==str(CASE/'outputs'/stage/'runs'/G) and value['config_seal_path']==str(CASE/'configs/seals'/(G+'.json')),'Wrong real new cfg IO')
    view=copy.deepcopy(value);base=copy.deepcopy(cfg);base['case_root']=str(OLD_CASE)
    view['execution_root']=str(OLD_CASE/'outputs'/stage);view['output']=str(OLD_CASE/'outputs'/stage/'runs'/G);view['config_seal_path']=str(OLD_CASE/'configs/seals'/(G+'.json'))
    original(base,view,4,stage)  # Metadata-only view preserves actual cfg bytes; original encoder/science checker.

def group_binding(cfg,n,e,old_group,old_validate):
    require(n==4,'Resource context only C4')
    hook=Path(str(CASE/'control/full/runtime'/(G+'_04_hooks_exit.json')));seal=Path(str(CASE/'configs/seals'/(G+'.json')))
    if not hook.is_file() or not seal.is_file():return None
    value,_=e.metadata(hook);require(type(value['actual_exit_code']) is int and value['actual_exit_code']==0 and value['residual_owned_members']==[],'Real hook04 not0/drained')
    files={}
    for path,binding in cfg['resource_static_files'].items():
        info=e.b.small_file(path);require(info['sha256']==binding['sha256'] and info['stat']['size']==binding['bytes'],'Resource static metadata changed');files[path]=info
    planpath=CASE/'cache/groups'/G/'plan.json';plan=read(planpath);bridge=resource_module(cfg)
    old_load=e.load_bridge;old_check=e.validate_config
    try:
        with bridge.scoped_tiny_bridge(plan,Path(planpath)) as scoped:
            e.load_bridge=lambda unused:scoped;e.validate_config=lambda c,v,count,stage:config_view_check(c,v,count,stage,old_validate)
            context=old_group(cfg,4)
        proof=bridge.final_resource_evidence(plan)  # Closed numeric/pool/stage small evidence only; no audit rerun.
    finally:e.load_bridge=old_load;e.validate_config=old_check
    bindings=[proof['resource_equivalence'],proof['adoption_ledger'],proof['resource_execution']['pool_exit'],*proof['resource_execution']['stages']]
    for binding in bindings:
        info=e.b.small_file(binding['path']);require(info['sha256']==binding['sha256'],'Closed resource binding changed');files[binding['path']]=info
    for path,info in files.items():require(e.b.stat(path)==info['stat'],'Resource metadata changed during witness read')
    context['binding']['resource_context']=dict(schema_version=1,actual_exit_code=0,case_root=str(CASE),resource_plan=dict(path=str(planpath),sha256=files[str(planpath)]['sha256']),files=files,closed_resource_proof=proof)
    return context

def dispatch(cfg,action,payload=None):
    expected_keys(cfg);e,_,_=originals(False)
    old_expected,old_group,old_validate,old_scan=e.expected_keys,e.group_binding,e.validate_config,e.scan
    e.expected_keys=expected_keys;e.group_binding=lambda c,n:group_binding(c,n,e,old_group,old_validate)
    def scan(c):
        result=old_scan(c);require(all(s['group_id']==G for s in result['checkpoints']) and result['logs']==[],'Wrong scan scope')
        result['pending_groups']=[g for g in result.get('pending_groups',[]) if g==G];return result
    e.scan=scan
    try:return e.dispatch(cfg,action,payload)
    finally:e.expected_keys=old_expected;e.group_binding=old_group;e.validate_config=old_validate;e.scan=old_scan

def validate_resource_snapshot(cfg,snapshot,original_selector):
    identity=original_selector(cfg,snapshot);require(identity in checkpoint_keys(),'Outside new13')
    context=snapshot['metadata_binding']['resource_context'];require(context['actual_exit_code']==0 and context['case_root']==str(CASE),'Missing real resource context')
    require(context['resource_plan']==dict(path=str(CASE/'cache/groups'/G/'plan.json'),sha256=cfg['resource_static_files'][str(CASE/'cache/groups'/G/'plan.json')]['sha256']),'Wrong resource plan witness')
    for path,binding in cfg['resource_static_files'].items():require(context['files'][path]['sha256']==binding['sha256'] and context['files'][path]['stat']['size']==binding['bytes'],'Resource static snapshot changed')
    proof=context['closed_resource_proof'];state=CASE/'cache/groups'/G/'resource_state'
    require(proof['resource_equivalence']['path']==str(state/'resource_equivalence.json') and proof['resource_execution']['pool_exit']['path']==str(state/'pool/pool_exit.json'),'Wrong numeric/pool proof path')
    require(proof['adoption_ledger']==dict(path=str(CASE/'provenance/adoption_ledger.json'),sha256=cfg['resource_static_files'][str(CASE/'provenance/adoption_ledger.json')]['sha256']),'Wrong compact adoption proof')
    stages=proof['resource_execution']['stages'];require(len(stages) in (2,4,6),'Wrong closed resource generation count')
    expected_stages=[str(state/'pool/stages'/f'generation_{i:03d}{ending}.json') for i in range(len(stages)//2) for ending in ('','_closed')]
    require([x['path'] for x in stages]==expected_stages,'Resource stage path scope changed')
    bindings=[proof['resource_equivalence'],proof['adoption_ledger'],proof['resource_execution']['pool_exit'],*proof['resource_execution']['stages']]
    require(set(context['files'])==set(cfg['resource_static_files'])|{x['path'] for x in bindings},'Unexpected resource witness files')
    for binding in bindings:
        require(binding['path'] in context['files'] and context['files'][binding['path']]['sha256']==binding['sha256'],'Resource proof snapshot incomplete')
    return identity

def receiver_module(cfg,c,base):
    original_raw=(base/'tiny_backup_receiver.py').read_bytes();require(hashlib.sha256(original_raw).hexdigest()==ORIGINALS['tiny_backup_receiver.py'],'Original receiver bytes changed');raw=original_raw.decode('utf-8')
    edits=[('expected_checkpoint_count=52,expected_payload_file_count=208,','expected_checkpoint_count=13,expected_payload_file_count=52,'),('payload_file_count=208','payload_file_count=52')]
    for before,after in edits:require(raw.count(before)==1,'Counted receiver metadata anchor differs');raw=raw.replace(before,after,1)
    module=types.ModuleType('_new_c4_only13_receiver');module.__file__=str(base/'tiny_backup_receiver.py');exec(compile(raw,module.__file__,'exec'),module.__dict__)
    original_selector=module.selected_identity;module.expected_keys=expected_keys
    module.selected_identity=lambda config,snapshot:validate_resource_snapshot(config,snapshot,original_selector)
    def copy_snapshot(snapshot,root,transport):
        module.selected_identity(cfg,snapshot);require(Path(root).resolve()==LOCAL.resolve(),'Wrong local new13 root')
        require(not Path(root).joinpath(*snapshot['target_parts']).exists(),'Unknown partial/foreign receipt preserved; no adoption')
        return c.copy_snapshot(snapshot,root,transport)
    module.c=types.SimpleNamespace(copy_snapshot=copy_snapshot,accepted_receipt=c.accepted_receipt)
    return module,dict(original_sha256=ORIGINALS['tiny_backup_receiver.py'],effective_sha256=hashlib.sha256(raw.encode()).hexdigest(),only_count_metadata_edits=edits)

def transport(cfg,c,b):
    path=P/'backup_recovery_candidate_v1/tiny_backup_recovery.py';require(sha(path)=='3f4172069838243dfbf521f8238e823c942b41e160bfcdb600447d540287d9f7','Frozen1800 transport source changed')
    spec=importlib.util.spec_from_file_location('_c4_frozen_copy_transport',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.NEW_ACK_ROOT=ACK
    return module.transport_type(c,b)(cfg,LOCAL/'backup_runtime')

def verify_old39(c,b):
    path=Path(__file__).with_name('OLD39_ADOPTION.json');require(sha(path)==OLD39_SHA,'Old39 adoption record changed');value=read(path)
    require(value['actual_exit_code']==0 and len(value['records'])==39,'Old39 proof incomplete');seen=set()
    for record in value['reports']:require(sha(record['path'])==record['sha256'],'Old acceptance proof changed')
    for record in value['records']:
        identity=record['stage'],record['group_id'],record['counter'];require(identity not in seen,'Old duplicate identity');seen.add(identity)
        require(record['group_id'] in {f'lejepa_tiny512_cube{n}' for n in (1,2,3)} and record['counter'] in b.COUNTERS[record['stage']],'Old39 source scope differs')
        require(sha(record['receipt_path'])==record['receipt_sha256'] and sha(record['first_hash_proof']['path'])==record['first_hash_proof']['sha256'],'Accepted old receipt/firstSHA proof changed')
        receipt=read(record['receipt_path']);require(b.digest_json(receipt['snapshot'])==record['snapshot_sha256'],'Old source snapshot changed');c.accepted_receipt(receipt['snapshot'],record['receipt_path'])
        for name,stat in record['payload_stats'].items():require(b.stat(Path(record['receipt_path']).parent/name)==stat,'Accepted old payload stat changed')
    require(len(seen)==39,'Old39 identities incomplete');return value

def run(cfg,config_sha,admission_path,admission_sha):
    expected_keys(cfg);e,c,base=originals(True);proof=verify_old39(c,e.b)
    require(sha(admission_path)==admission_sha,'Root admission changed');admission=read(admission_path)
    require(admission['actual_exit_code']==0 and 0<=time.time()-admission['checked_at']<=600 and admission['ssh_G']==dict(hostname='connect.westc.seetacloud.com',user='root',port=16755) and admission['remote_hostname']==cfg['expected_remote_hostname'] and admission['source_pins_verified'] is True and admission['config_sha256']==config_sha,'Fresh Root target/source admission required')
    require(not LOCAL.exists(),'Existing new13 root preserved; no automatic restart');require(c.shutil.disk_usage(LOCAL.parent).free>13*(128<<20)+(64<<20),'D free insufficient')
    LOCAL.mkdir();runtime=LOCAL/'backup_runtime';runtime.mkdir();called=False;code=1
    try:
        e.b.new_json(LOCAL/'STARTUP.json',dict(actual_exit_code=0,scope='newCASE3 C4 only13/52',old39_proof_sha256=OLD39_SHA,old_payload_hashes=0,copy_timeout_seconds=1800,automatic_retry=False,formal_or_SR_acceptance=False))
        module,overlay=receiver_module(cfg,c,base);e.b.new_json(LOCAL/'RECEIVER_OVERLAY.json',overlay);worker=transport(cfg,c,e.b)
        called=True;result=module.run(cfg,config_sha,transport=worker);require(result==0,'Original receiver returned nonzero');code=0;return 0
    except BaseException as error:
        if isinstance(error,KeyboardInterrupt):code=130
        if not called:e.b.new_json(runtime/'admission_failure.json',dict(actual_exit_code=code,error=repr(error),automatic_retry=False,advisory_only=True))
        raise
    finally:
        e.b.new_json(LOCAL/'worker_outcome.json',dict(actual_exit_code=code,original_receiver_called=called,advisory_only=True,automatic_restart=False))
        if not called:e.b.new_json(runtime/f'worker_exit_{time.time_ns()}.json',dict(actual_exit_code=code,admission_failed_before_receiver=True,advisory_only=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--action',choices=('scan','poststat','ack','failure'));p.add_argument('--run',action='store_true');p.add_argument('--config-sha256');p.add_argument('--admission',type=Path);p.add_argument('--admission-sha256');a=p.parse_args();cfg=read(a.config)
    if a.action:
        require(not a.run and a.admission is None,'Endpoint mode cannot start a worker');payload=None if a.action=='scan' else json.load(sys.stdin);print(json.dumps(dispatch(cfg,a.action,payload),sort_keys=True));return 0
    require(a.run and a.config_sha256==sha(a.config) and a.admission and a.admission_sha256,'Explicit Root single-worker admission required')
    return run(cfg,a.config_sha256,a.admission,a.admission_sha256)

if __name__=='__main__':raise SystemExit(main())
