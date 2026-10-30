"""C4-only resource/metadata overlay over frozen Tiny producer and audit loops."""
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

GROUP='lejepa_tiny512_cube4'
SOURCE_SHA={'producer':'7d2829026fe2a3d6aa008134637579d7458c4a7799d41bc80e977946fabd3839',
            'audit':'67ff13ca692940521585b553902a34e811296bea57e95d1f146ec962a677a408'}
BRIDGE_SHA='98d684a1065b4c44f026ada58ac9664d026998fd74cdbe3872bd09913bb51c62'
POOL_FILENAME='resource_pool_v2.py'
EXTRA_KEYS=('prefetch','inference_precision','storage_dtype','encoder_preprocess')
RESOURCE_KEYS=('resource_runtime_policy','renderer_workers_semantics','resource_continuation')

def require(value,message):
    if not value:raise ValueError(message)
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def raw_stat(path):
    value=Path(path).stat()
    return dict(size=value.st_size,mtime_ns=value.st_mtime_ns,ctime_ns=value.st_ctime_ns,device=value.st_dev,inode=value.st_ino)
def bound_read(binding):
    path=Path(binding['path']);before=raw_stat(path);content=path.read_bytes()
    require(hashlib.sha256(content).hexdigest()==binding['sha256'],'Bound metadata SHA: '+str(path))
    require(raw_stat(path)==before,'Bound metadata changed: '+str(path))
    return json.loads(content)
def new_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def metadata_paths(plan):
    recipe=plan['resource_continuation'];root=Path(recipe['metadata_root'])
    return dict(root=root,home=root/'cache/groups'/GROUP,runtime=root/'control/full/runtime',state=Path(recipe['state_dir']))

def validate_original_protocol(original,old_plan):
    require(old_plan.get('pixel_preprocess_device')=='shared_cpu' and old_plan.get('encoder_preprocessing_device')=='cpu' and original.get('pixel_preprocess_device')=='cpu','Original sharedCPU policy must resolve to the original CPU device')
    for key,value in original.items():
        if key in old_plan and key!='pixel_preprocess_device':require(old_plan[key]==value,'Original protocol/plan mismatch: '+key)

def validate_plan(plan,old_plan,check_bindings=True):
    require(old_plan['cube_count']==plan['cube_count']==4 and plan['latent_dim']==512,'C4 Tiny512 only')
    allowed={'renderer_workers','prefetch','source_pins',*RESOURCE_KEYS}
    require(set(plan)==set(old_plan)|set(RESOURCE_KEYS),'Unexpected/missing plan fields')
    require({k:v for k,v in plan.items() if k not in allowed}=={k:v for k,v in old_plan.items() if k not in allowed},'Scientific/physical-cache plan changed')
    require(plan['renderer_workers']==16 and plan['prefetch']==2,'Authorized initial16/PF2 per worker required')
    require(plan['renderer_workers_semantics']=='initial_capacity_and_policy_maximum','Actual worker semantics must be explicit')
    require(all(plan['source_pins'].get(k)==v for k,v in old_plan['source_pins'].items()),'Original immutable source pins changed')
    root=metadata_paths(plan)['root'];cache=Path(plan['cache_dir'])
    require(root!=cache and root not in cache.parents and cache not in root.parents,'New metadata must be separate from physical legacy cache')
    require(root in metadata_paths(plan)['state'].parents,'Private resource state must remain inside new metadata root')
    if check_bindings:
        recipe=plan['resource_continuation']
        require(bound_read(recipe['original_plan'])==old_plan,'Original plan binding differs')
        require(recipe['original_bridge']['sha256']==BRIDGE_SHA and sha(recipe['original_bridge']['path'])==BRIDGE_SHA,'Original Tiny bridge differs')
        original=bound_read(recipe['original_protocol'])
        require(original['producer_source_sha256']==BRIDGE_SHA,'Original producer source is not the accepted Tiny bridge')
        require(original['render_adapter_source_sha256']==old_plan['source_pins']['foundation_visual_gciql_v1/cache/render_adapter.py'],'Original renderer source differs')
        validate_original_protocol(original,old_plan)
        ledger=bound_read(recipe['adoption_ledger'])
        require(ledger['schema_version']==2 and ledger['group_id']==GROUP and ledger['cube_count']==4,'Wrong compact adoption schema/group')
        stop=ledger['post_stop'];require(type(stop['actual_exit_code']) is int and stop['actual_exit_code']!=0 and stop['residual_owned_members']==[],'Legacy producer must actually interrupt and drain')
        require(ledger['original_protocol_sha256']==digest(original),'Ledger original protocol differs')
        require(ledger['original_producer_source_sha256']==original['producer_source_sha256'],'Ledger original producer differs')
        bound_read(recipe['resource_policy'])
        pool=load_pool();require(recipe['resource_policy']['sha256']==pool.POLICY_SHA,'Authorized resource proposal SHA differs')
        for mode,binding in recipe['original_effective_sources'].items():
            require(mode in SOURCE_SHA and binding['sha256']==SOURCE_SHA[mode] and sha(binding['path'])==SOURCE_SHA[mode],'Effective source binding differs')
        pool.validate_policy(plan['resource_runtime_policy'])
    return plan

def load_pool():
    path=Path(__file__).with_name(POOL_FILENAME);name='resource_pool_v2'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module)
    module=sys.modules[name]
    require(Path(module.__file__).resolve()==path.resolve() and sha(module.__file__)==sha(path),'Foreign resource_pool module alias')
    return module

def load_context(plan):
    recipe=plan['resource_continuation'];old=bound_read(recipe['original_plan']);validate_plan(plan,old)
    ledger=runtime_ledger(plan)
    entries=ledger['entries'];keys=[(x['split'],x['episode']) for x in entries]
    require(len(keys)==len(set(keys)),'Duplicate adopted episode')
    for entry in entries:
        require(entry['split'] in plan['split_episodes'] and type(entry['episode']) is int and 0<=entry['episode']<plan['split_episodes'][entry['split']],'Adopted EP outside fixed scope')
        expected=Path(plan['cache_dir'])/entry['split']/f"episode_{entry['episode']:06d}"
        require(Path(entry['receipt_path']).resolve()==(expected/'receipt.json').resolve() and Path(entry['latent']['path']).resolve()==(expected/'latent.npy').resolve(),'Adoption path outside fixed EP')
    ledger['_entry_index']={key:entry for key,entry in zip(keys,entries)}
    return old,ledger

def runtime_ledger(plan):
    """Expand the one compact schema through its sole metadata-preparation owner."""
    raw=bound_read(plan['resource_continuation']['adoption_ledger'])
    require(raw['schema_version']==2,'Only compact runtime adoption ledger is accepted')
    path=Path(__file__).with_name('prepare_continuation.py')
    spec=importlib.util.spec_from_file_location('_c4_continuation_metadata',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.expand_runtime_ledger(raw)

def validate_receipt(record,path,plan,protocol,arrays_binding,ledger):
    require(record['status']=='episode_committed' and record['actual_exit_code']==0,'Incomplete receipt')
    require(record['checkpoint_sha256']==plan['checkpoint_sha256'] and record['source_arrays_receipt']==arrays_binding,'Receipt model/source arrays differ')
    key=(record['split'],record['episode'])
    entries=([ledger['_entry_index'][key]] if key in ledger['_entry_index'] else []) if '_entry_index' in ledger else [x for x in ledger['entries'] if (x['split'],x['episode'])==key]
    if entries:
        require(len(entries)==1,'Duplicate ledger EP');entry=entries[0]
        require(Path(path).resolve()==Path(entry['receipt_path']).resolve() and sha(path)==entry['receipt_sha256'],'Adopted receipt bytes differ')
        require(raw_stat(path)==entry['receipt_stat'],'Adopted receipt stat differs')
        require(record['protocol_sha256']==ledger['original_protocol_sha256'] and record['producer_source_sha256']==ledger['original_producer_source_sha256'],'Legacy production identity differs')
        require(all(record['latent'][k]==entry['latent'][k] for k in ('path','sha256')),'Legacy latent identity differs')
        require(raw_stat(entry['latent']['path'])==entry['latent']['stat'],'Legacy latent stat changed after cutoff')
        return 'adopted_legacy'
    require(record['protocol_sha256']==protocol and record.get('producer_source_sha256')==sha(__file__),'Unknown receipt production identity')
    stage=record.get('resource_stage');require(isinstance(stage,dict),'New receipt lacks actual resource stage')
    require(type(stage['generation']) is int and 0<=stage['generation']<=2 and stage['actual_workers']==(16,12,8)[stage['generation']],'Actual generation/tier mismatch')
    require(stage['actual_workers'] in (16,12,8) and stage['prefetch_per_worker']==2 and stage['window']==2*stage['actual_workers'],'Wrong actual worker/PF window')
    stage_path=Path(stage['stage_record_path']);home=metadata_paths(plan)['state']/'pool/stages'
    require(stage_path.parent==home and stage_path.name==f"generation_{stage['generation']:03d}.json" and not stage_path.is_symlink(),'Resource stage escaped private generation')
    actual=bound_read(dict(path=str(stage_path),sha256=stage['stage_record_sha256']))
    for key in ('generation','actual_workers','prefetch_per_worker','window','policy_source_sha256'):
        require(actual[key]==stage[key],'Resource stage differs: '+key)
    require(actual['actual_exit_code']==0 and actual['resource_module_sha256']==sha(Path(__file__).with_name(POOL_FILENAME)),'Resource module/stage differs')
    require(stage['policy_source_sha256']==plan['resource_runtime_policy']['policy_source_sha256'],'Resource policy differs')
    return 'new_resource_segment'

def _science_nodes(source,mode):
    tree=ast.parse(source)
    if mode=='producer':
        names={'canonical_ids','source_observation_rows','model_hash','prepared_and_encoded','infer'}
        return [(x.name,ast.dump(x,include_attributes=False)) for x in ast.walk(tree) if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name in names]
    # The complete original per-array SHA/shape/dtype/finite/stat tail stays exact.
    loop=next(x for x in ast.walk(tree) if isinstance(x,ast.For) and isinstance(x.target,ast.Tuple) and ast.unparse(x.target)=='(episode, entry)')
    start=next(i for i,x in enumerate(loop.body) if isinstance(x,ast.Assign) and ast.unparse(x.targets[0])=='latent')
    return [ast.dump(x,include_attributes=False) for x in loop.body[start:]]

def _edits(mode):
    if mode=='producer':return [
      ("selection_path = out / 'preprocess_selection.json'","selection_path = RESOURCE_STATE / 'preprocess_selection.json'"),
      ("protocol_path = out / 'protocol.json'","protocol_path = RESOURCE_STATE / 'protocol.json'"),
      ("assert plan['renderer_workers'] == 2 and plan['encoder_batch'] == 32 and plan['torch_cpu_threads'] == 1","assert plan['renderer_workers'] == 16 and plan['prefetch'] == 2 and plan['encoder_batch'] == 32 and plan['torch_cpu_threads'] == 1"),
      ("os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-8:])","apply_affinity(plan['resource_runtime_policy']['allowed_affinity'])"),
      ("save(out / 'status.json'", "save(RESOURCE_STATE / 'producer_status.json'"),
      ("assert record['protocol_sha256'] == protocol and record['source_arrays_receipt'] == arrays_binding","validate_receipt(record, receipt_path, plan, protocol, arrays_binding, ADOPTION_LEDGER)"),
      ("with ProcessPoolExecutor(max_workers=2, mp_context=context, initializer=init_worker,\n                initargs=(str(workspace), str(arrays_path), plan['camera'])) as pool:","with StagePool(max_workers=16, mp_context=context, initializer=init_worker,\n                initargs=(str(workspace), str(arrays_path), plan['camera']), policy=plan['resource_runtime_policy'], state_dir=RESOURCE_STATE/'pool', expected_new_episodes=len(pending)) as pool:\n            resource_equivalence_gate(pool, plan, infer, render_episode, render_rows)"),
      ('range(min(2, len(pending)))','range(min(pool.window, len(pending)))'),
      ('encoder_rows_valid=1001, padding_rows=padding, encoder_rows_executed=1001 + padding)','encoder_rows_valid=1001, padding_rows=padding, encoder_rows_executed=1001 + padding,\n                    resource_stage=pool.receipt_binding((split, episode)))'),
      ("save(epdir / 'receipt.json', record)","save(epdir / 'receipt.json', record)\n                pool.after_commit((split, episode), epdir / 'receipt.json', len(queue))"),
      ("                try:\n                    job = next(iterator)\n                    queue.append((job, pool.submit(render_episode, job)))\n                except StopIteration:\n                    pass", "                for _ in range(pool.refill_slots(len(queue))):\n                    try:\n                        job = next(iterator)\n                    except StopIteration:\n                        break\n                    queue.append((job, pool.submit(render_episode, job)))"),
      ("save(out / 'cache_manifest.json', manifest)","manifest.update(final_resource_evidence(plan))\n        exclusive_final(out / 'cache_manifest.json', manifest)"),
      ("save(out / 'cache_seal.json', dict(","exclusive_final(out / 'cache_seal.json', dict("),
      ("path = out / 'status.json'","path = RESOURCE_STATE / 'producer_status.json'")]
    require(mode=='audit','Unknown overlay mode')
    return [
      ("manifest, manifest_sha = read(cache_dir / 'cache_manifest.json')","manifest, manifest_sha = read(cache_dir / 'cache_manifest.json')\n    audit_resource_manifest(manifest, plan, read)"),
      ("stored_protocol, protocol_file_sha = read(cache_dir / 'protocol.json')","stored_protocol, protocol_file_sha = read(RESOURCE_STATE / 'protocol.json')"),
      ("assert all(receipt[key] == value for key, value in identities.items()), 'EP identity/provenance'","audit_receipt_identity(receipt, receipt_path, plan, protocol_sha, identities, arrays_path, arrays_sha, track, ADOPTION_LEDGER)")]

def effective_source(original,mode,recipe=None):
    require(hashlib.sha256(original.encode()).hexdigest()==SOURCE_SHA[mode],'Frozen effective source differs')
    source=original;edits=[]
    for before,after in _edits(mode):
        count=source.count(before);expected=2 if before=="save(out / 'status.json'" else 1
        require(count==expected,'Counted resource/I/O edit differs: '+before)
        source=source.replace(before,after);edits.append([before,after,count])
    require(_science_nodes(source,mode)==_science_nodes(original,mode),'Scientific function/array AST changed')
    proof=dict(mode=mode,original_sha256=SOURCE_SHA[mode],effective_sha256=hashlib.sha256(source.encode()).hexdigest(),edits=edits,scientific_AST_exact=True)
    require(inverse_source(source,proof)==original,'Overlay inverse differs')
    return source,proof

def inverse_source(source,proof):
    require(hashlib.sha256(source.encode()).hexdigest()==proof['effective_sha256'],'Effective source changed')
    for before,after,count in reversed(proof['edits']):
        require(source.count(after)==count,'Inverse counted edit differs');source=source.replace(after,before)
    require(hashlib.sha256(source.encode()).hexdigest()==proof['original_sha256'],'Inverse source differs')
    return source

def exclusive_final(path,value):
    new_json(path,value)

def final_resource_evidence(plan):
    recipe=plan['resource_continuation'];state=metadata_paths(plan)['state']
    gate=state/'resource_equivalence.json';require(Path(recipe['numeric_gate_proof_path'])==gate,'Numeric gate proof path differs')
    proof=read(gate);require(proof['actual_exit_code']==0 and proof['plan_digest']==digest(plan),'Numeric gate not accepted for this plan')
    require(proof['bridge_sha256']==sha(__file__) and proof['resource_policy']==recipe['resource_policy'],'Numeric gate source/policy differs')
    pool_exit=state/'pool/pool_exit.json';result=read(pool_exit)
    require(result['actual_exit_code']==0 and result['all_submissions_committed'] is True and result['generation_count'] in (1,2,3),'Resource pool not successfully drained')
    stages=[]
    for generation in range(result['generation_count']):
        path=state/'pool/stages'/f'generation_{generation:03d}.json';closed=path.with_name(f'generation_{generation:03d}_closed.json')
        stage=read(path);closure=read(closed)
        require(stage['generation']==generation and stage['actual_workers']==(16,12,8)[generation],'Resource generation identity differs')
        require(stage['resource_module_sha256']==sha(Path(__file__).with_name(POOL_FILENAME)),'Resource stage source differs')
        exits=closure['owned_worker_exitcodes']
        require(closure['actual_exit_code']==0 and closure['no_next_generation_overlap'] is True and closure['queued_commits_remaining']==0 and 1<=len(exits)<=stage['actual_workers'] and all(x==0 for x in exits.values()),'Actual spawned workers did not exit0/drain')
        stages.extend([dict(path=str(path),sha256=sha(path)),dict(path=str(closed),sha256=sha(closed))])
    return dict(resource_equivalence=dict(path=str(gate),sha256=sha(gate)),resource_execution=dict(pool_exit=dict(path=str(pool_exit),sha256=sha(pool_exit)),stages=stages),adoption_ledger=recipe['adoption_ledger'])

def audit_resource_manifest(manifest,plan,read_metadata):
    expected=final_resource_evidence(plan)
    require(all(manifest[k]==v for k,v in expected.items()),'Manifest resource/adoption binding differs')
    bindings=[expected['resource_equivalence'],expected['adoption_ledger'],expected['resource_execution']['pool_exit'],*expected['resource_execution']['stages']]
    for binding in bindings:read_metadata(binding['path'],binding['sha256'])

def audit_receipt_identity(receipt,path,plan,protocol,identities,arrays_path,arrays_sha,track,ledger):
    kind=validate_receipt(receipt,path,plan,protocol,dict(path=str(arrays_path),sha256=arrays_sha),ledger)
    expected=dict(identities)
    if kind=='adopted_legacy':expected.update(protocol_sha256=ledger['original_protocol_sha256'],producer_source_sha256=ledger['original_producer_source_sha256'])
    require(all(receipt[k]==v for k,v in expected.items()),'EP scientific/provenance identity')
    if kind=='new_resource_segment':track(receipt['resource_stage']['stage_record_path'])

def resource_equivalence_gate(pool,plan,infer,render_episode,render_rows):
    """One old complete EP plus the next EP's first B32; same loaded original model."""
    import numpy as np
    path=metadata_paths(plan)['state']/'resource_equivalence.json'
    require(Path(plan['resource_continuation']['numeric_gate_proof_path'])==path,'Numeric gate proof path differs')
    require(not path.exists(),'Resource numerical proof already exists; preserve it')
    proof=dict(actual_exit_code=1,formal_training_updates=0,formal_SR_episodes=0,atol=1e-5,rtol=1e-5,resource_stage=dict(pool.stage_binding),configured_capacity=16,probe_worker_pids=[],numeric_scope='two bounded probes on actual observed probe workers; all16 numeric validation not claimed; concurrent16/affinity requires Root proc readback after missing-job refill',plan_digest=digest(plan),bridge_sha256=sha(__file__),resource_policy=plan['resource_continuation']['resource_policy'])
    try:
        cache=Path(plan['cache_dir']);full=read(cache/'train/episode_000000/receipt.json');next_record=read(cache/'train/episode_000001/receipt.json')
        rendered=pool.probe(render_episode,('train',0))
        proof['probe_worker_pids'].append(rendered['worker_pid'])
        require(np.array_equal(rendered['raw_ids'],np.arange(1001,dtype=np.int64)),'Resource EP0 raw IDs differ')
        require(rendered['RGB_sha256']==full['RGB_sha256'],'Resource EP0 RGB SHA differs')
        geometry=rendered['render_geometry'];require(geometry['qpos_max_abs_error']==geometry['qvel_max_abs_error']==0,'Resource EP0 geometry differs')
        values,_=infer(rendered['pixels']);old=np.load(full['latent']['path'],mmap_mode='r',allow_pickle=False)
        require(values.shape==old.shape==(1001,512),'Resource EP0 shape differs')
        maximum=float(np.max(np.abs(values-old)));bitwise=bool(np.array_equal(values,old))
        require(np.isfinite(values).all() and np.isfinite(old).all() and np.allclose(values,old,atol=1e-5,rtol=1e-5),'Resource EP0 latent differs')
        rows=np.arange(1001,1033,dtype=np.int64);boundary=pool.probe(render_rows,('train',rows))
        proof['probe_worker_pids'].append(boundary['worker_pid'])
        require(np.array_equal(boundary['raw_ids'],rows),'Resource EP1 raw boundary differs')
        geometry=boundary['render_geometry'];require(geometry['qpos_max_abs_error']==geometry['qvel_max_abs_error']==0,'Resource EP1 geometry differs')
        value2,_=infer(boundary['pixels']);old2=np.load(next_record['latent']['path'],mmap_mode='r',allow_pickle=False)[:32]
        require(value2.shape==old2.shape==(32,512) and np.isfinite(value2).all() and np.isfinite(old2).all() and np.allclose(value2,old2,atol=1e-5,rtol=1e-5),'Resource EP1 B32 latent differs')
        proof.update(actual_exit_code=0,EP0=dict(receipt_sha256=sha(cache/'train/episode_000000/receipt.json'),RGB_SHA_exact=True,max_abs_difference=maximum,bitwise=bitwise,rows=1001,worker_pid=rendered['worker_pid']),
          EP1=dict(receipt_sha256=sha(cache/'train/episode_000001/receipt.json'),max_abs_difference=float(np.max(np.abs(value2-old2))),bitwise=bool(np.array_equal(value2,old2)),raw_ids=rows.tolist(),worker_pid=boundary['worker_pid']))
    except BaseException as exc:proof['error']=repr(exc);raise
    finally:new_json(path,proof)

@contextlib.contextmanager
def cache_module(plan,mode):
    old,ledger=load_context(plan);recipe=plan['resource_continuation'];paths=metadata_paths(plan);paths['state'].mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('_resource_original_tiny',recipe['original_bridge']['path']);bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
    with bridge._cache_scope():
        api=bridge._cache_api(mode)
        source=Path(recipe['original_effective_sources'][mode]['path']).read_text();effective,proof=effective_source(source,mode)
        if mode=='producer':api.effective_source=lambda value:effective
        else:api.source=lambda value:effective
        module=api.load(plan);module.__file__=str(Path(__file__).resolve())
        pool=load_pool();module.__dict__.update(RESOURCE_STATE=paths['state'],ADOPTION_LEDGER=ledger,StagePool=pool.StagePool,apply_affinity=pool.apply_affinity,validate_receipt=validate_receipt,exclusive_final=exclusive_final,resource_equivalence_gate=resource_equivalence_gate,audit_receipt_identity=audit_receipt_identity,final_resource_evidence=final_resource_evidence,audit_resource_manifest=audit_resource_manifest)
        if mode=='producer':module.HERE=bridge.ROOT/'foundation_visual_gciql_v1/cache';module.CONTRACT_KEYS+=EXTRA_KEYS+RESOURCE_KEYS
        else:
            module.PRODUCER=str(Path(__file__).resolve().relative_to(bridge.ROOT));module.PROTOCOL_KEYS+=EXTRA_KEYS+RESOURCE_KEYS;module.camera_check=api.check_camera
        yield module,proof

@contextlib.contextmanager
def scoped_tiny_bridge(plan,plan_path):
    old,ledger=load_context(plan);recipe=plan['resource_continuation'];paths=metadata_paths(plan)
    spec=importlib.util.spec_from_file_location('_resource_original_tiny_runtime',recipe['original_bridge']['path']);bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
    original_case=bridge.case;original_validate=bridge.validate_plan;original_recipe=bridge.recipe;original_bootstrap=bridge.bootstrap;original_pins=bridge.cache_pins
    def validate(value):
        validate_plan(value,old);view=copy.deepcopy(value);view['renderer_workers']=view['prefetch']=2
        bridge.case=original_case
        try:original_validate(view)
        finally:bridge.case=lambda:paths['root']
    def resource_recipe():
        result=original_recipe();result['resource_continuation']=dict(bridge=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),plan=dict(path=str(plan_path),sha256=sha(plan_path)),adoption_ledger=recipe['adoption_ledger'],resource_policy=recipe['resource_policy'],resource_runtime_policy=plan['resource_runtime_policy'])
        return result
    def pins(cfg,require_seal=True):
        result=original_pins(cfg,require_seal);result[str(Path(__file__).resolve())]=sha(__file__);result[str(Path(__file__).with_name(POOL_FILENAME))]=sha(Path(__file__).with_name(POOL_FILENAME))
        for key in ('original_plan','original_protocol','adoption_ledger','resource_policy'):
            binding=recipe[key];require(sha(binding['path'])==binding['sha256'],'Resource evidence changed');result[binding['path']]=binding['sha256']
        return result
    def bootstrap():
        modules=original_bootstrap();prepare=modules['prepare'];commands=prepare.commands
        def routed(path,cfg):
            result=commands(path,cfg)
            for key,argv in result.items():result[key]=[str(Path(__file__).resolve()),'--resource-plan',str(plan_path),('audit-native' if argv[1]=='audit' else argv[1]),*argv[2:]]
            return result
        prepare.commands=routed
        return modules
    bridge.case=lambda:paths['root'];bridge.validate_plan=validate;bridge.recipe=resource_recipe;bridge.cache_pins=pins;bridge.bootstrap=bootstrap
    try:yield bridge
    finally:pass  # Detached module only; no original file/global namespace modified.

def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--resource-plan',type=Path)
    parser.add_argument('mode',choices=('cache-producer','cache-audit','deliver','runtime','native','audit-native'))
    args,rest=parser.parse_known_args(argv)
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--plan',type=Path)
    parsed,_=p.parse_known_args(rest);plan_path=args.resource_plan or parsed.plan
    require(plan_path is not None,'New resource plan is required')
    require(args.resource_plan is None or parsed.plan is None or args.resource_plan==parsed.plan,'CLI resource plan differs from actual --plan')
    plan=read(plan_path)
    if args.mode.startswith('cache-'):
        mode='producer' if args.mode=='cache-producer' else 'audit'
        with cache_module(plan,mode) as (module,proof):
            previous=sys.argv
            try:sys.argv=[str(Path(__file__).resolve()),*rest];module.main()
            finally:sys.argv=previous
    elif args.mode=='deliver':
        load_context(plan)
        # Original delivery only publishes real full-cache audit-bound metadata.
        p=argparse.ArgumentParser();p.add_argument('--plan',required=True,type=Path);p.add_argument('--audit-report',required=True,type=Path);p.add_argument('--producer-exit-file',required=True,type=Path);d=p.parse_args(rest)
        paths=metadata_paths(plan)
        require(d.plan==paths['home']/'plan.json' and d.audit_report==paths['home']/'full_audit_v1/report.json','Delivery metadata paths differ')
        require(d.producer_exit_file==paths['runtime']/(GROUP+'_01_encode_full_exit.json'),'Wrong actual producer exit')
        require(not (paths['home']/'delivery.json').exists(),'Delivery already exists; preserve it')
        with scoped_tiny_bridge(plan,plan_path) as bridge:
            with bridge._cache_scope():
                sys.path.insert(0,str(Path(plan['workspace'])/'foundation_visual_gciql_v1/cache'))
                spec=importlib.util.spec_from_file_location('_resource_delivery',Path(plan['workspace'])/'foundation_visual_gciql_v1/cache/deliver.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                module.actual_exit(d.producer_exit_file);module.publish(d.plan,d.audit_report,False,d.producer_exit_file)
    else:
        with scoped_tiny_bridge(plan,plan_path) as bridge:
            previous=sys.argv
            try:sys.argv=[str(Path(__file__).resolve()),'audit' if args.mode=='audit-native' else args.mode,*rest];bridge.main()
            finally:sys.argv=previous

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if getattr(exc,'exit_code',None)==75:raise SystemExit(75) from exc
        raise
