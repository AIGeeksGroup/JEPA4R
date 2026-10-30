"""Pure metadata pack plus exclusive C4-only runner over the original serial supervisor."""
import argparse
import base64
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import subprocess
import sys
import time
import traceback

ROOT=PurePosixPath('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
OLD_CASE=ROOT/'lejepa_tiny512_cube14_gciql_v1'
NEW_CASE=ROOT/'lejepa_tiny512_cube4_resource_v1'
STORE=PurePosixPath('/root/autodl-tmp/lejepa_tiny512_cube14_20261029_v1_outputs')
OUTPUT_STORE=STORE/'resource_c4_v1'
G='lejepa_tiny512_cube4';OLD_GROUP=OLD_CASE/'cache/groups'/G;GROUP=NEW_CASE/'cache/groups'/G
CACHE=OLD_CASE/'outputs/cache'/G/'cache';T=NEW_CASE/'control/full';R=T/'runtime'
OLD_GLOBAL=OLD_CASE/'control/full/launch_plan.json';OLD_R=OLD_CASE/'control/full/runtime'
OLD_BRIDGE=OLD_CASE/'release_v1/tiny_bridge.py';BRIDGE=NEW_CASE/'release_v1/c4_resource_bridge.py'
PLAN=GROUP/'plan.json';PY=ROOT/'env_runtime/bin/python'
OLD_PLAN_SHA='ba4ca64b31fb60b8f02b4637d0f56c73d1615bfc8641c0e94bd4a9c6baa9d74b'
OLD_BRIDGE_SHA='98d684a1065b4c44f026ada58ac9664d026998fd74cdbe3872bd09913bb51c62'
SERIAL=ROOT/'foundation_visual_gciql_v1/control/serial.py'
SERIAL_SHA='dac39378e01c4c76d0a78dce472c9bb6e937d9f909f4ae12eabe2b7362cf6a38'
POLICY_SHA='6d0ede59a7977f026d3cdae583e8c4014d7cf5feb45ad479ab39a965f50511b6'
STAT_KEYS={'size','mtime_ns','ctime_ns','inode','device'}

def require(value,message):
    if not value:raise ValueError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def encode(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def read(path):return json.loads(Path(path).read_bytes())
def canonical(value):return digest(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def wire(path):return str(PurePosixPath(str(path).replace('\\','/')))
def sha(path):
    path=Path(path);require(path.suffix not in ('.pkl','.npy','.npz','.pt'),'No payload hash in metadata preparation')
    require(path.stat().st_size<=64<<20,'Metadata source too large');return digest(path.read_bytes())
def st(path):
    v=Path(path).stat();return dict(size=v.st_size,mtime_ns=v.st_mtime_ns,ctime_ns=v.st_ctime_ns,inode=v.st_ino,device=v.st_dev)
def new_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(encode(value))
def stat_contract(value):require(set(value)==STAT_KEYS and all(type(x) is int and x>=0 for x in value.values()),'Bad stat contract')

def original_file(ledger,path):
    path=str(path);require(PurePosixPath(path).suffix not in ('.pkl','.npy','.npz','.pt'),'Original payload must never be opened')
    proof=ledger['original_files'].get(path)
    if proof is None:
        before=st(path);raw=Path(path).read_bytes();proof=dict(sha256=digest(raw),bytes=len(raw),source_before=before,source_after=st(path),raw_b64=base64.b64encode(raw).decode())
    stat_contract(proof['source_before']);require(proof['source_before']==proof['source_after'],'Original metadata changed')
    raw=base64.b64decode(proof['raw_b64'],validate=True)
    require(len(raw)==proof['bytes']==proof['source_before']['size'] and digest(raw)==proof['sha256'],'Original metadata raw SHA/size differs')
    return raw

def json_original(ledger,path):return json.loads(original_file(ledger,path))

ENTRY_FIELDS=['split','episode','receipt_sha256','receipt_stat','latent_sha256','latent_stat']
STAT_FIELDS=['size','mtime_ns','ctime_ns','inode','device']

def require_sha(value):require(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value),'Invalid SHA256')

def expand_runtime_ledger(raw):
    """Expand only compact metadata in memory; never read a receipt/latent or import science."""
    require(raw['schema_version']==2 and raw['group_id']==G and raw['cube_count']==4 and raw['cache_dir']==str(CACHE),'Wrong compact runtime ledger scope')
    require(raw['entry_fields']==ENTRY_FIELDS and raw['stat_fields']==STAT_FIELDS,'Unknown compact field order')
    require_sha(raw['source_intake_sha256']);expanded=copy.deepcopy({k:v for k,v in raw.items() if k not in ('entries','entry_fields','stat_fields')});entries=[];seen=set()
    for row in raw['entries']:
        require(isinstance(row,list) and len(row)==6,'Malformed compact entry')
        split,ep,receipt_sha,receipt_stat,latent_sha,latent_stat=row
        require(split in ('train','val') and type(ep) is int and 0<=ep<{'train':5000,'val':500}[split] and (split,ep) not in seen,'Wrong/duplicate compact episode');seen.add((split,ep))
        require_sha(receipt_sha);require_sha(latent_sha)
        require(isinstance(receipt_stat,list) and isinstance(latent_stat,list) and len(receipt_stat)==len(latent_stat)==5,'Malformed compact stat')
        rs=dict(zip(STAT_FIELDS,receipt_stat));ls=dict(zip(STAT_FIELDS,latent_stat));stat_contract(rs);stat_contract(ls)
        directory=CACHE/split/f'episode_{ep:06d}'
        entries.append(dict(split=split,episode=ep,receipt_path=str(directory/'receipt.json'),receipt_sha256=receipt_sha,receipt_stat=rs,latent=dict(path=str(directory/'latent.npy'),sha256=latent_sha,stat=ls)))
    expanded['entries']=entries;return expanded

def compact_adoption(ledger,intake_sha,oldgroup):
    """New derived schema2; original full intake and all original bytes remain unchanged."""
    require(ledger['schema_version']==1,'Prepare requires complete Root input schema1');require_sha(intake_sha)
    fixed={str(OLD_R/(name+'_exit.json')) for name in ledger['old_phase_exits']}
    fixed|={str(OLD_R/(G+'_01_encode_full'+end)) for end in ('_exit.json','_argv.json')}
    fixed|={str(OLD_R/name) for name in ('outer_exit_code','status.json','failure.json')}
    fixed|={str(OLD_CASE/'control/full'/name) for name in ('wrapper_exit_code','worker_outcome.json')}
    fixed|={str(OLD_GROUP/name) for name in ('plan.json','data_binding.json','camera.json','encoder_spec.json')}
    fixed|={str(CACHE/'protocol.json'),str(PurePosixPath(oldgroup['shared_state_root'])/'source_arrays.json'),oldgroup['data_acceptance'],oldgroup['source_stat_freeze']}
    require(fixed<=set(ledger['original_files']),'Root complete intake must contain fixed rawmetadata evidence')
    result=copy.deepcopy({k:v for k,v in ledger.items() if k not in ('entries','original_files')})
    result.update(schema_version=2,source_intake_sha256=intake_sha,cache_dir=str(CACHE),entry_fields=ENTRY_FIELDS,stat_fields=STAT_FIELDS,original_files={k:ledger['original_files'][k] for k in sorted(fixed)},entries=[[x['split'],x['episode'],x['receipt_sha256'],[x['receipt_stat'][k] for k in STAT_FIELDS],x['latent']['sha256'],[x['latent']['stat'][k] for k in STAT_FIELDS]] for x in ledger['entries']])
    require(expand_runtime_ledger(result)['entries']==ledger['entries'],'Compact expansion changed metadata identity')
    return result

def compact_bytes(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def validate_adoption(old,ledger):
    require(ledger['schema_version'] in (1,2) and ledger['group_id']==G and ledger['cube_count']==4,'Wrong adoption scope')
    require(ledger['old_global_plan_sha256']==OLD_PLAN_SHA and len(old['phases'])==48,'Wrong old48 plan')
    expected={p['name'] for p in old['phases'][:36]};require(set(ledger['old_phase_exits'])==expected,'Exactly old36 exits required')
    for name,value in ledger['old_phase_exits'].items():
        actual=json_original(ledger,OLD_R/(name+'_exit.json'))
        require(value==actual and type(value['actual_exit_code']) is int and value['actual_exit_code']==0 and value['residual_owned_members']==[],'Old36 not truly0/drained')
    stop=ledger['post_stop'];require(type(stop['actual_exit_code']) is int and stop['actual_exit_code']!=0 and stop['residual_owned_members']==[],'Actual old interrupted/drained exit required')
    require(stop['old_runtime_root']==str(OLD_R) and type(stop['owned_PGID']) is int and stop['owned_PGID']>1,'Wrong old ownership')
    interrupted=json_original(ledger,OLD_R/(G+'_01_encode_full_exit.json'))
    require(interrupted['actual_exit_code']==stop['actual_exit_code'] and interrupted['residual_owned_members']==[],'Old producer exit differs')
    argv=json_original(ledger,OLD_R/(G+'_01_encode_full_argv.json'))
    require(argv['pid']==stop['owned_PGID'] and argv['argv']==old['phases'][36]['argv'] and argv['cwd']==old['phases'][36]['cwd'],'Old phase identity changed')
    for path in (OLD_R/'outer_exit_code',OLD_CASE/'control/full/wrapper_exit_code'):
        require(int(original_file(ledger,path))!=0,'Old interruption must not be rewritten to0')
    for path in (OLD_R/'status.json',OLD_R/'failure.json',OLD_CASE/'control/full/worker_outcome.json'):
        value=json_original(ledger,path);require(type(value['actual_exit_code']) is int and value['actual_exit_code']!=0,'Old failure/outcome evidence missing')
    require(ledger['old_global_closure_absent'] is True,'Do not adopt a fabricated original48 closure')
    raw=original_file(ledger,OLD_GROUP/'plan.json');require(digest(raw)==old['source_pins'][str(OLD_GROUP/'plan.json')],'Old C4 plan fingerprint differs')
    plan=json.loads(raw);require(plan['cube_count']==4 and plan['encoder_family']=='lejepa_tiny512' and plan['latent_dim']==512,'Wrong old group')
    require((plan['renderer_workers'],plan['prefetch'],plan['encoder_batch'],plan['torch_cpu_threads'])==(2,2,32,1),'Wrong legacy W2/PF2/B32 identity')
    require(plan['inference_precision']=='fp32' and plan['storage_dtype']=='float32' and plan['cache_dir']==str(CACHE),'Wrong old precision/physical cache')
    protocol=json_original(ledger,CACHE/'protocol.json')
    require(canonical(protocol)==ledger['original_protocol_sha256'] and protocol['producer_source_sha256']==ledger['original_producer_source_sha256']==OLD_BRIDGE_SHA,'Old protocol/producer identity differs')
    require(protocol['renderer_workers']==2 and protocol['prefetch']==2,'Old receipts must retain original W2 protocol')
    require(json_original(ledger,OLD_GROUP/'camera.json')==plan['camera'],'Old camera metadata differs')
    spec=json_original(ledger,plan['encoder_spec']);require(digest(original_file(ledger,plan['encoder_spec']))==plan['encoder_spec_sha256'] and spec['checkpoint_sha256']==plan['checkpoint_sha256'] and spec['downstream_camera']==plan['camera'],'Old encoder/camera/spec differs')
    require(digest(original_file(ledger,PurePosixPath(plan['shared_state_root'])/'source_arrays.json'))==plan['source_arrays_receipt_sha256'],'Old source-array receipt content differs')
    for field in ('data_acceptance','source_stat_freeze'):
        require(digest(original_file(ledger,plan[field]))==plan[field+'_sha256'],'Original data/stat metadata changed')
    for name in ('data_binding.json','camera.json','encoder_spec.json'):
        path=str(OLD_GROUP/name);require(digest(original_file(ledger,path))==old['source_pins'][path],'Original group metadata source pin differs')
    identities=set()
    for item in ledger['entries']:
        split,ep=item['split'],item['episode'];require(split in ('train','val') and type(ep) is int and 0<=ep<plan['split_episodes'][split] and (split,ep) not in identities,'Wrong/duplicate adoptedEP')
        identities.add((split,ep));directory=CACHE/split/f'episode_{ep:06d}'
        require(item['receipt_path']==str(directory/'receipt.json') and item['latent']['path']==str(directory/'latent.npy'),'Adoption path escapes own physical cache')
        raw_receipt=original_file(ledger,item['receipt_path']);receipt=json.loads(raw_receipt)
        receipt_proof=ledger['original_files'].get(item['receipt_path']);current_stat=receipt_proof['source_before'] if receipt_proof is not None else st(item['receipt_path'])
        require(digest(raw_receipt)==item['receipt_sha256'] and current_stat==item['receipt_stat'],'Adopted receipt SHA/stat differs')
        stat_contract(item['latent']['stat']);require(item['latent']['stat']['size']==receipt['latent']['bytes'] and item['latent']['sha256']==receipt['latent']['sha256'] and receipt['latent']['path']==item['latent']['path'],'Latent stat/first SHA binding differs')
        if ledger['schema_version']==2:require(st(item['latent']['path'])==item['latent']['stat'],'Current adopted latent stat changed; no payload rehash')
        require(receipt['actual_exit_code']==0 and receipt['split']==split and receipt['episode']==ep and receipt['protocol_sha256']==ledger['original_protocol_sha256'] and receipt['producer_source_sha256']==OLD_BRIDGE_SHA,'Old receipt identity differs')
        require(receipt['latent']['shape']==[1001,512] and receipt['latent']['dtype']=='float32' and receipt['checkpoint_sha256']==plan['checkpoint_sha256'],'Old latent/checkpoint contract differs')
        require(receipt['camera_contract_sha256']==canonical(plan['camera']) and receipt['source_arrays_receipt']==dict(path=str(PurePosixPath(plan['shared_state_root'])/'source_arrays.json'),sha256=plan['source_arrays_receipt_sha256']),'Old camera/source array receipt binding differs')
        require((receipt['raw_row_start'],receipt['raw_row_count'],receipt['main_rows'],receipt['terminal_raw_id'])==(ep*1001,1001,1000,ep*1001+1000),'Old rawID/terminal contract differs')
    for item in ledger['uncommitted_files']:
        require(str(item['path']).startswith(str(CACHE)+'/') and '..' not in PurePosixPath(item['path']).parts and item['path'] not in {v['latent']['path'] for v in ledger['entries']},'Uncommitted path escapes scope or overlaps adopted data')
        stat_contract(item['stat'])
    require({('train',0),('train',1)}<=identities,'Original EP0/EP1 required for the authorized first-newEP numeric gate')
    require(len(identities)<sum(plan['split_episodes'].values()),'No new EP remains; do not start a resource stage for an already complete cache')
    return plan,protocol

def route_phase(phase):
    value=copy.deepcopy(phase);argv=list(value['argv'])
    if str(OLD_BRIDGE) in argv:
        i=argv.index(str(OLD_BRIDGE));mode=argv[i+1];argv[i:i+2]=[str(BRIDGE),'--resource-plan',str(PLAN),'audit-native' if mode=='audit' else mode]
    else:
        script=str(ROOT/'foundation_visual_gciql_v1/cache/deliver.py');require(script in argv,'Unexpected phase script')
        i=argv.index(script);argv[i:i+1]=[str(BRIDGE),'--resource-plan',str(PLAN),'deliver']
    mapping=[(str(OLD_GROUP),str(GROUP)),(str(OLD_R),str(R)),(str(OLD_CASE/'outputs/pilot'),str(NEW_CASE/'outputs/pilot')),(str(OLD_CASE/'outputs/formal'),str(NEW_CASE/'outputs/formal'))]
    mapping += [(str(OLD_CASE/'configs'/purpose/(G+'.json')),str(NEW_CASE/'configs'/purpose/(G+'.json'))) for purpose in ('pilot','formal','seals')]
    def path(s):
        for before,after in mapping:
            if s==before or s.startswith(before+'/'):return after+s[len(before):]
        return s
    value['argv']=[path(s) for s in argv]
    for check in value.get('checks',[]):check['path']=path(check['path'])
    require(value['environment']==phase['environment'] and value['cwd']==phase['cwd'],'Original phase environment/cwd changed')
    return value

def prepare(old_plan_path,ledger_path,policy_path,output_root):
    oldraw=Path(old_plan_path).read_bytes();require(digest(oldraw)==OLD_PLAN_SHA,'Only actual frozen original48 plan allowed');old=json.loads(oldraw)
    ledgerraw=Path(ledger_path).read_bytes();ledger=json.loads(ledgerraw);oldgroup,protocol=validate_adoption(old,ledger)
    runtime_ledger=compact_adoption(ledger,digest(ledgerraw),oldgroup);runtime_raw=compact_bytes(runtime_ledger)
    require(len(runtime_raw)<=2<<20,'Compact runtime ledger exceeds2MiB; preserve input and review instead of widening limits')
    policyraw=Path(policy_path).read_bytes();require(digest(policyraw)==POLICY_SHA,'Wrong authorized resource policy')
    here=Path(__file__).resolve().parent;sys.path.insert(0,str(here));import resource_pool_v2
    defaults=resource_pool_v2.policy_defaults();resource_pool_v2.validate_policy(defaults)
    files={}
    def add(rel,raw):require(rel not in files,'Duplicate payload');files[rel]=raw
    for name in ('resource_pool_v2.py','c4_resource_bridge.py','prepare_continuation.py'):add('release_v1/'+name,(here/name).read_bytes())
    for name in ('data_binding.json','camera.json','encoder_spec.json'):add('cache/groups/'+G+'/'+name,original_file(ledger,OLD_GROUP/name))
    add('provenance/original_global_plan.json',oldraw);add('provenance/adoption_ledger.json',runtime_raw);add('provenance/resource_policy.json',policyraw)
    add('provenance/original_group_plan.json',original_file(ledger,OLD_GROUP/'plan.json'));add('provenance/original_protocol.json',original_file(ledger,CACHE/'protocol.json'))
    for mode in ('producer','audit'):add(f'provenance/effective_original_{mode}.py',(here.parent/'sources'/f'effective_tiny_cube4_{mode}.py').read_bytes())
    recipe=dict(metadata_root=str(NEW_CASE),state_dir=str(GROUP/'resource_state'),original_plan=dict(path=str(OLD_GROUP/'plan.json'),sha256=digest(original_file(ledger,OLD_GROUP/'plan.json'))),original_protocol=dict(path=str(CACHE/'protocol.json'),sha256=digest(original_file(ledger,CACHE/'protocol.json'))),adoption_ledger=dict(path=str(NEW_CASE/'provenance/adoption_ledger.json'),sha256=digest(runtime_raw)),resource_policy=dict(path=str(NEW_CASE/'provenance/resource_policy.json'),sha256=POLICY_SHA),original_bridge=dict(path=str(OLD_BRIDGE),sha256=OLD_BRIDGE_SHA),original_effective_sources={mode:dict(path=str(NEW_CASE/f'provenance/effective_original_{mode}.py'),sha256=digest(files[f'provenance/effective_original_{mode}.py'])) for mode in ('producer','audit')},numeric_gate_proof_path=str(GROUP/'resource_state/resource_equivalence.json'))
    plan=copy.deepcopy(oldgroup);plan.update(renderer_workers=16,prefetch=2,resource_runtime_policy=defaults,renderer_workers_semantics='initial_capacity_and_policy_maximum',resource_continuation=recipe)
    for rel,raw in files.items():plan['source_pins'][str((NEW_CASE/rel).relative_to(ROOT))]=digest(raw)
    add('cache/groups/'+G+'/plan.json',encode(plan))
    phases=[route_phase(p) for p in old['phases'][36:]]
    launch=dict(schema_version=1,queue='tiny_cube4_resource',families=['lejepa_tiny512'],cube_counts=[4],groups=[G],expected_features=5505500,training_updates=1000000,SR_episodes=250,diagnostic_optimizer_updates=2,diagnostic_SR_episodes=1,process_guard=old['process_guard'],own_RSS_limit_bytes=old['own_RSS_limit_bytes'],source_pins=dict(old['source_pins']),phases=phases,full_budget=old['full_budget'],backup_advisory=True,resource_continuation_plan=dict(path=str(PLAN),sha256=digest(files['cache/groups/'+G+'/plan.json'])))
    require(launch['own_RSS_limit_bytes']==60*(1<<30),'Original RSS60 guard differs')
    for rel,raw in files.items():launch['source_pins'][str(NEW_CASE/rel)]=digest(raw)
    add('control/full/launch_plan.json',encode(launch))
    add('control/full/SPACE_BUDGET.json',encode(old['full_budget']))
    add('provenance/UNCOMMITTED_REQUIRING_ROOT_ISOLATION.json',encode(dict(files=ledger['uncommitted_files'],adopted=False,actions_performed=False,execution_blocked=bool(ledger['uncommitted_files']))))
    add('provenance/AGGREGATE_COMPLETION_CONTRACT.json',encode(dict(old_successful_phases=36,new_required_phases=12,total_joined_phases=48,original48_outer_exit_zero_claimed=False,wholeTaskComplete=False,required_resource_provenance=True,required_new_CP_source='newCASE13 checkpoints; old recovery endpoint cannot recognize new C4 automatically',final_D_inventory='39 originalCase plus13 newCase; actual independent receipts required')))
    out=Path(output_root);out.mkdir(parents=True,exist_ok=False)
    for rel,raw in files.items():
        path=out/'payload'/rel;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(raw)
    manifest=dict(schema_version=1,actual_exit_code=0,prepared_only=True,NN_started=False,metadata_root=str(NEW_CASE),files={rel:dict(remote_path=str(NEW_CASE/rel),sha256=digest(raw),bytes=len(raw)) for rel,raw in files.items()},required_exclusive_STORE_subtree=str(OUTPUT_STORE),required_symlink=dict(path=str(NEW_CASE/'outputs'),target=str(OUTPUT_STORE)),legacy_cache_unchanged=str(CACHE),old36_actual_exits_adopted=36,new_phases=12,uncommitted_needing_Root_isolation=len(ledger['uncommitted_files']),wholeTaskComplete=False)
    new_json(out/'PACK_MANIFEST.json',manifest);return manifest

def validate_launch(plan):
    oldpath=Path(NEW_CASE/'provenance/original_global_plan.json');require(sha(oldpath)==OLD_PLAN_SHA,'Original plan anchor changed');old=read(oldpath)
    require(plan['queue']=='tiny_cube4_resource' and plan['cube_counts']==[4] and plan['families']==['lejepa_tiny512'] and plan['groups']==[G],'Wrong continuation scope')
    require(plan['phases']==[route_phase(x) for x in old['phases'][36:]] and len(plan['phases'])==12,'New twelve commands/env differ')
    require((plan['expected_features'],plan['training_updates'],plan['SR_episodes'],plan['own_RSS_limit_bytes'])==(5505500,1000000,250,60*(1<<30)),'Changed scientific budget/guard')
    binding=plan['resource_continuation_plan'];require(binding['path']==str(PLAN) and sha(PLAN)==binding['sha256'],'Resource plan changed')
    resource=read(PLAN)['resource_continuation'];ledger=expand_runtime_ledger(read(resource['adoption_ledger']['path']));require(sha(resource['adoption_ledger']['path'])==resource['adoption_ledger']['sha256'],'Poststop ledger changed')
    validate_adoption(old,ledger);require(not ledger['uncommitted_files'],'Uncommitted files need Root poststop isolation, not adoption')
    require(Path(NEW_CASE/'outputs').is_symlink() and Path(NEW_CASE/'outputs').resolve()==Path(OUTPUT_STORE).resolve(),'Outputs must map exclusive sameSTORE/resource_c4_v1')
    for path,want in plan['source_pins'].items():require(sha(path)==want,'Source pin changed: '+path)

def load_serial():
    require(sha(SERIAL)==SERIAL_SHA,'Original serial source changed');sys.path.insert(0,str(Path(SERIAL).parent))
    spec=importlib.util.spec_from_file_location('_resource_original_serial',SERIAL);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def execute_plan(path,expected,token):
    directory=Path(path).parent;code=1;called=False;claimed=False;error=None;oldargv=list(sys.argv)
    try:
        require(wire(path)==str(T/'launch_plan.json'),'Execution path outside new fixed control')
        lock=read(directory/'dispatch.lock');require(lock['plan_sha256']==expected and lock['script_sha256']==sha(__file__) and isinstance(token,str) and digest(token.encode())==lock['token_sha256'],'No legitimate reserved attempt/token')
        require(len(token)==64 and all(c in '0123456789abcdef' for c in token) and isinstance(lock['attempt_id'],str) and len(lock['attempt_id'])==32 and all(c in '0123456789abcdef' for c in lock['attempt_id']),'Invalid attempt/token shape')
        require(not any((directory/name).exists() for name in ('dispatch_failure.json','worker_claim.json','worker_outcome.json','worker_failure.json','wrapper_exit_code','runtime')),'Attempt already claimed or runtime exists')
        new_json(directory/'worker_claim.json',dict(attempt_id=lock['attempt_id'],pid=os.getpid(),started=time.time(),plan_sha256=expected));claimed=True
        require(sha(path)==expected,'Launch plan differs');plan=read(path);validate_launch(plan)
        serial=load_serial();serial.validate_plan=validate_launch
        sys.argv=[str(SERIAL),'--plan-directory',str(directory),'--execute']
        called=True;require(serial.main()==0,'Original serial returned nonzero')
        require((directory/'runtime/outer_exit_code').read_text().strip()=='0','Original serial outer is not actual0');code=0;return 0
    except BaseException as exc:
        if isinstance(exc,SystemExit):code=exc.code if type(exc.code) is int else (0 if exc.code is None else 1)
        elif isinstance(exc,KeyboardInterrupt):code=130
        error=dict(error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        sys.argv[:]=oldargv
        if claimed:
            runtime=directory/'runtime'
            if code and runtime.is_dir():new_json(runtime/'continuation_failure.json',dict(actual_exit_code=1,**error))
            if runtime.is_dir() and not (runtime/'outer_exit_code').exists():
                with (runtime/'outer_exit_code').open('x') as stream:stream.write(str(code)+'\n')
            if error:new_json(directory/'worker_failure.json',dict(actual_exit_code=code,original_serial_called=called,**error))
            outer=None;outer_error=None
            try:
                if (runtime/'outer_exit_code').exists():outer=int((runtime/'outer_exit_code').read_text())
            except (OSError,ValueError) as exc:outer_error=repr(exc)
            new_json(directory/'worker_outcome.json',dict(actual_exit_code=code,original_serial_called=called,runtime_created=runtime.exists(),original_outer_exit_code=outer,original_outer_read_error=outer_error,automatic_restart=False))
            with (directory/'wrapper_exit_code').open('x') as stream:stream.write(str(code)+'\n')

def dispatch_plan(path,expected):
    require(wire(path)==str(T/'launch_plan.json') and sha(path)==expected,'Wrong new plan/expected SHA');plan=read(path);validate_launch(plan)
    directory=Path(path).parent
    require(not any((directory/n).exists() for n in ('dispatch.lock','dispatch_failure.json','dispatch_receipt.json','worker_claim.json','worker_outcome.json','worker_failure.json','wrapper_exit_code','runtime')),'Existing attempt preserved, never restart')
    token=secrets.token_hex(32);attempt=secrets.token_hex(16)
    new_json(directory/'dispatch.lock',dict(attempt_id=attempt,plan_sha256=expected,script_sha256=sha(__file__),token_sha256=digest(token.encode())))
    argv=[str(PY),'-B',str(Path(__file__).resolve()),'--execute-plan',str(path),'--expected-plan-sha256',expected,'--attempt-token',token]
    try:
        with (directory/'dispatch.stdout.log').open('x') as out,(directory/'dispatch.stderr.log').open('x') as err:
            child=subprocess.Popen(argv,cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    except BaseException:
        new_json(directory/'dispatch_failure.json',dict(actual_exit_code=1,attempt_id=attempt,worker_started=False,error=traceback.format_exc(),automatic_restart=False));raise
    receipt=dict(actual_dispatch_exit_code=0,attempt_id=attempt,pid_locator_only=child.pid,argv=argv,queue_completed=False,automatic_restart=False)
    new_json(directory/'dispatch_receipt.json',receipt);return receipt

def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--old-plan',type=Path);mode.add_argument('--dispatch-plan',type=Path);mode.add_argument('--execute-plan',type=Path)
    parser.add_argument('--adoption-ledger',type=Path);parser.add_argument('--resource-policy',type=Path);parser.add_argument('--output-root',type=Path);parser.add_argument('--expected-plan-sha256');parser.add_argument('--attempt-token');a=parser.parse_args()
    if a.old_plan:
        require(a.adoption_ledger and a.resource_policy and a.output_root and a.expected_plan_sha256 is None and a.attempt_token is None,'Preparation inputs only');result=prepare(a.old_plan,a.adoption_ledger,a.resource_policy,a.output_root)
    else:
        require(a.adoption_ledger is None and a.resource_policy is None and a.output_root is None and a.expected_plan_sha256,'Execution and preparation modes are separate')
        if a.execute_plan:return execute_plan(a.execute_plan,a.expected_plan_sha256,a.attempt_token)
        require(a.attempt_token is None,'Token reserved for claimed worker');result=dispatch_plan(a.dispatch_plan,a.expected_plan_sha256)
    print(json.dumps(result,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
