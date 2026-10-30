"""One closed P3 final-model + cache/Native evidence package; default plan only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
Q=ROOT/'cube3_pldm_gciql_v1'
GROUP='P3_cube3_pldm_rgb'
PACK_SHA='bd4a240d9a39d0d7140f873de62ce3f760651ae2223598a68882f680f30e69d4'
COPY_SHA='bcb574502929e6c128806c7f714788d0cfe02f991592b9ae1f8024f6bec1e574'
CONFIG_SHA='c7e5955bac93de0de16a9eb3e8f9c6e7725f89c412194302c34d2c90e108c225'
ACCEPT_SHA='f225b145663242f0991dddff8dcd81520cd0496d38d355476536cc3a548f6be9'
MODEL_SHA='0ed6b328afcecf8858af3d46061bed1026a9c75ab7472a855496c907ba495b67'
PHASES=('04_prepare_train','05_GCIQL_gate','06_GCIQL_reload','07_GCIQL_train','08_GCIQL_accept',
        '09_prepare_native','10_native_tiny','11_native_tiny_audit','12_native_full','13_native_full_audit')
OLD=('00_prepare','01_cache_pilot','02_cache_full','03_cache_audit')
COUNTERS=[1]+list(range(100000,1000001,100000))
ALLOWED_STEPS=set()
PACK_SOURCES=set()
META={'.json','.py','.sh','.yaml','.yml','.txt','.toml','.xml','.diff'}


def load_source(path,expected,edits,name):
    raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
    original=raw.decode();effective=original
    for before,after in edits:
        assert effective.count(before)==1,(name,before)
        effective=effective.replace(before,after)
    restored=effective
    for before,after in reversed(edits):
        assert restored.count(after)==1
        restored=restored.replace(after,before)
    assert restored==original
    module=types.ModuleType('_P3_closed_'+name);module.__file__=str(path)
    exec(compile(effective,str(path),'exec'),module.__dict__)
    module.parameter_proof=dict(source_sha256=expected,changes=len(edits),inverse_exact=True,
        effective_sha256=hashlib.sha256(effective.encode()).hexdigest())
    return module


def load_pack(path):
    module=load_source(path,PACK_SHA,[("assert path == run / 'final_val_probe.npz'",
        "assert path == run / 'final_val_probe.npz' or path in _P3_ALLOWED_STEPS")],'pack')
    module._P3_ALLOWED_STEPS=ALLOWED_STEPS
    PACK_SOURCES.add(Path(path))
    return module


def load_copy(path=None):
    path=path or Path(__file__).resolve().parents[2]/'cube4_gciql_103032Test_20261020_v1/closed_backup_preparation_v2/copy_verified.py'
    return load_source(path,COPY_SHA,[
        ("choices=('G11', 'G12')","choices=('P3_cube3_pldm_rgb',)"),
        ("/cube4_visual_gciql_v1/closed_training_backup_v2/{args.group}/completed_v1",
         "/cube3_pldm_gciql_v1/closed_final_backup_v1/{args.group}/completed_v1"),
        ("prefix=f'quad_{args.group}_closed_'","prefix=f'p3_{args.group}_closed_'")],'copy')


def verify_closed(cfg,status,accepted,closure,exits,outer,wrapper,native):
    assert cfg['group_id']==cfg['role']==GROUP and cfg['cube_count']==3 and cfg['executable'] is True
    assert (cfg['seed'],cfg['batch_size'],cfg['total_updates'])==(0,1024,1000000)
    assert cfg['checkpoint_steps']==COUNTERS and cfg['input_contract']['zero_padding'] is False
    assert (cfg['input_contract']['current_dim'],cfg['input_contract']['goal_dim'])==(531,521)
    assert cfg['encoder_binding']['family']=='pldm' and cfg['encoder_binding']['latent_dim']==512
    assert str(outer)==str(wrapper)=='0'
    assert list(exits)==list(PHASES) and all(x['actual_exit_code']==0 and x['residual_owned_members']==[] for x in exits.values())
    assert (closure['actual_exit_code'],closure['completed_phases'],closure['training_updates'],closure['SR_episodes'])==(0,10,1000000,250)
    assert closure['cube_count']==3 and closure['data_id']=='cube-triple-play-v0'
    assert status['exit_code']==0 and status['completed_updates']==status['adam_count']==1000000
    assert status['network_step']==1000001 and status['validation_count']==200
    assert accepted['actual_exit_code']==0 and accepted['role']==GROUP and accepted['full153_state_exact']
    assert accepted['state']['leaves']==153 and accepted['finite_actions'] and accepted['synthetic_zero_padding'] is False
    assert (accepted['Adam_count'],accepted['network_step'],accepted['CSV_validation_records'],accepted['actual_current_dim'],accepted['actual_goal_dim'])==(1000000,1000001,200,531,521)
    assert [x['counter'] for x in accepted['checkpoint_transactions']]==COUNTERS
    for mode,count in (('tiny',1),('full',250)):
        prefix='native_sr_v1/'+mode+'/'
        report,audit,binding=(native[prefix+n] for n in ('report.json','independent_audit.json','native_source_binding.json'))
        assert report['actual_exit_code']==audit['actual_exit_code']==binding['actual_exit_code']==0
        assert report['group_id']==audit['group_id']==GROUP and report['episodes']==audit['episodes']==count
        assert report['unique_seeds']==audit['unique_seeds']==count
        assert report['encoding_protocol']=='fresh_current_and_goal_each_policy_call_2N'
        assert report['offline_embedding_cache_used_for_policy'] is False and report['synthetic_zero_padding'] is False
        assert (report['current_dim'],report['goal_dim'])==(531,521)
        assert audit['encoder_family']==binding['encoder_family']=='pldm' and audit['latent_dim']==binding['latent_dim']==512
        assert audit['fresh_goal_then_current_encoding_required'] and audit['successes']==report['successes']
        if mode=='full':assert report['max_episode_steps']==1000


def select(q,group,u,out):
    assert group==GROUP
    ALLOWED_STEPS.clear()
    run=q/'runs'/GROUP;rt=q/'control_recovery_v3/runtime_v1';cache=q/'cache_v1/cache'
    config=q/f'functional_v1/configs/{GROUP}.v1.json';report=q/'acceptance/P3_full_acceptance.json'
    assert u.sha(config)==CONFIG_SHA and u.sha(report)==ACCEPT_SHA
    cfg,status,accepted=u.read(config),u.read(run/'status.json'),u.read(report)
    exits={name:u.read(rt/(name+'_exit.json')) for name in PHASES}
    native={f'native_sr_v1/{mode}/{name}':u.read(q/f'native_sr_v1/{mode}/{name}')
        for mode in ('tiny','full') for name in ('report.json','independent_audit.json','native_source_binding.json')}
    verify_closed(cfg,status,accepted,u.read(rt/'closure.json'),exits,
        (rt/'outer_exit_code').read_text().strip(),(q/'control_recovery_v3/wrapper_exit_code').read_text().strip(),native)
    assert (run/'trainer_exit_code').read_text().strip()=='0'
    resolved=u.read(run/'resolved.json');assert resolved['binding']['config']==cfg
    assert resolved['resumed_from'] is None and (resolved['initial_counter'],resolved['initial_network_step'],resolved['initial_adam_count'])==(0,1,0)
    assert u.read(run/'final_state_fingerprint.json')==accepted['state']
    paths={config,report,Path(__file__),Path(u.__file__)}|PACK_SOURCES
    paths.update(run/n for n in ('train.csv','resolved.json','status.json','trainer_exit_code','final_state_fingerprint.json','final_val_probe.npz'))
    native_binding=q/'functional_v1/NATIVE_BINDING_v1.json';nb=u.read(native_binding)
    assert nb['actual_exit_code']==0 and nb['final_counter']==1000000 and nb['group']['manifest_sha256']==accepted['final_manifest_sha256']
    pins=dict(resolved['binding']['code'])
    for mode,count in (('tiny',1),('full',250)):
        folder=q/'native_sr_v1'/mode
        assert (folder/'actual_exit_code').read_text().strip()=='0'
        assert native[f'native_sr_v1/{mode}/native_source_binding.json']['binding_sha256']==u.sha(native_binding)
        paths.update(folder/n for n in ('report.json','status.json','actual_exit_code','native_source_binding.json','independent_audit.json'))
        for task in range(1,6):
            for index in range(1 if mode=='tiny' and task==1 else 0 if mode=='tiny' else 50):
                ep=folder/'episodes'/f'task{task}'/f'episode_{index:03d}'
                item=u.read(ep/'result.json');assert item['actual_exit_code']==0 and item['group_id']==GROUP
                assert (item['task_id'],item['episode_index'],item['seed'])==(task,index,970000000+task*100000+index)
                assert (ep/'attempt_exit_code').read_text().strip()=='0'
                steps=ep/'steps.npz';ALLOWED_STEPS.add(steps)
                paths.update((ep/'result.json',ep/'attempt_exit_code',steps))
    assert len(ALLOWED_STEPS)==251
    ledger=[]
    for item in accepted['checkpoint_transactions']:
        counter=item['counter'];cp=run/'checkpoints'/f'update_{counter:09d}';mp=cp/'manifest.json'
        assert u.sha(mp)==item['manifest_sha256'];identity=u.read(mp);assert identity['counter']==counter
        policy,side=cp/f'params_{counter}.pkl',cp/'run_state.pkl'
        ledger.append(dict(counter=counter,manifest=str(mp),manifest_sha256=item['manifest_sha256'],policy=str(policy),policy_sha256=identity['agent_sha256'],policy_stat=u.stat(policy),sidecar=str(side),sidecar_sha256=identity['sidecar_sha256'],sidecar_stat=u.stat(side),included=counter==1000000))
        paths.add(mp)
        if counter==1000000:
            assert str(cp)==accepted['final_checkpoint'] and u.sha(mp)==accepted['final_manifest_sha256']
            assert u.sha(policy)==identity['agent_sha256'] and u.sha(side)==identity['sidecar_sha256']
            paths.update((policy,side))
    u.new('checkpoint_SHA_ledger.json',dict(actual_exit_code=0,group_id=GROUP,checkpoints=ledger,only_final_model_payload=True))
    manifest,seal,audit=(u.read(p) for p in (cache/'cache_manifest.json',cache/'cache_seal.json',q/'cache_v1/independent_audit/report.json'))
    assert manifest['actual_exit_code']==seal['actual_exit_code']==audit['actual_exit_code']==0
    assert manifest['checkpoint_sha256']==manifest['protocol']['checkpoint_sha256']==MODEL_SHA
    assert (audit['episodes_verified'],audit['raw_rows_verified'],audit['latent_dim'],audit['dtype'])==(3300,3303300,512,'float32')
    assert audit['manifest_sha256']==seal['manifest_sha256']==u.sha(cache/'cache_manifest.json')
    assert audit['producer_seal_sha256']==u.sha(cache/'cache_seal.json') and audit['plan_sha256']==u.sha(q/'cache_v1/producer_plan.json')
    assert audit['all_new_latent_SHA_shape_dtype_finite'] and audit['all_raw_ID_and_true_terminal_next_contracts']
    receipts=[]
    for split,count in (('train',3000),('val',300)):
        assert len(manifest['splits'][split])==count
        for index,item in enumerate(manifest['splits'][split]):
            assert item['episode']==index
            path=Path(item['receipt_path']);assert path.is_relative_to(cache) and path.name=='receipt.json'
            assert u.sha(path)==item['receipt_sha256'];paths.add(path)
            receipts.append(dict(split=split,episode=index,path=str(path),sha256=item['receipt_sha256']))
    u.new('cache_receipt_SHA_ledger.json',dict(actual_exit_code=0,encoder_family='pldm',modality='rgb',receipts=receipts,rows=3303300,vector_values_read=False))
    old=q/'control_v1/runtime_v1'
    assert (old/'outer_exit_code').read_text().strip()==(q/'control_v1/wrapper_exit_code').read_text().strip()=='1'
    for folder,names in ((old,OLD),(rt,PHASES)):
        for name in names:
            value=u.read(folder/(name+'_exit.json'));assert value['actual_exit_code']==0 and value['residual_owned_members']==[]
            paths.update(folder/(name+suffix) for suffix in ('_argv.json','_exit.json','.log'))
        paths.update(folder/n for n in ('status.json','actual_dispatch.json','resources.jsonl','outer_exit_code'))
    paths.update(old/('04_prepare_train'+suffix) for suffix in ('_argv.json','_exit.json','.log'))
    paths.update((old/'failure.json',rt/'closure.json',q/'control_v1/wrapper_exit_code',q/'control_recovery_v3/wrapper_exit_code'))
    for control in ('control_v1','control_recovery_v3'):
        launch=u.read(q/control/'launch_plan.json');paths.add(q/control/'launch_plan.json')
        pins.update(launch['source_pins'])
    for path,digest in pins.items():
        path=Path(path)
        if path.suffix in META:
            assert u.sha(path)==digest;paths.add(path)
    for relative in ('functional_v1/TRAIN_CONFIG_PREPARED_v1.json','functional_v1/NATIVE_BINDING_v1.json','functional_v1/NATIVE_PREPARED_v1.json',
        'control_recovery_v3/CACHE_ADOPTION_v3.json','control_recovery_v3/ROOT_RELEASE_v3.json',
        'cache_v1/PREPARED.json','cache_v1/producer_plan.json','cache_v1/independent_audit/report.json','cache_v1/independent_audit/actual_exit_code',
        'cache_v1/cache/cache_manifest.json','cache_v1/cache/cache_seal.json','cache_v1/cache/protocol.json','cache_v1/cache/preprocess_selection.json','cache_v1/cache/batch1_alignment.json','cache_v1/cache/pilot_receipt.json','cache_v1/cache/status.json',
        'cache_v1/source_arrays/source_arrays.json','cache_v1/source_arrays/data_acceptance.json','cache_v1/source_arrays/stat-freeze.json',
        'closed_backup_preparation_v1/copy_p3.py','closed_backup_preparation_v1/SOURCE_FREEZE_v1.json','closed_backup_preparation_v1/HANDOFF_v1.json'):
        paths.add(q/relative)
    for path in paths:
        if '/runs/' in path.as_posix():assert path.is_relative_to(run)
        if path.suffix not in ('.pkl','.npz'):assert path.stat().st_size<=32*1024*1024
    final_review=q/'independent_final_readback_v1/report.json'
    if final_review.is_file():
        assert u.sha(final_review)=='15fa442b0cf0a696da3d2f668026315bf406d1b3791f09b094300c30ed76a948'
        paths.add(final_review)
    static_review=Path(__file__).with_name('STATIC_REVIEW_v1.json')
    if static_review.is_file():paths.add(static_review)
    u.new('closure_readback.json',dict(actual_exit_code=0,group_id=GROUP,encoder_family='pldm',latent_dim=512,
        config_sha256=u.sha(config),CPU_acceptance_sha256=u.sha(report),final_manifest_sha256=accepted['final_manifest_sha256'],
        cache_manifest_sha256=u.sha(cache/'cache_manifest.json'),cache_audit_sha256=u.sha(q/'cache_v1/independent_audit/report.json'),
        cache_receipts=3300,new_closed_phases=10,Native_tiny_episodes=1,Native_full_episodes=250,
        Native_successes=native['native_sr_v1/full/report.json']['successes'],Native_steps_files=len(ALLOWED_STEPS),
        generic_pack_source_sha256=PACK_SHA,generic_copy_source_sha256=COPY_SHA,
        only_two_final_model_payload_files=True,no_vector_WM_raw_periodic_payload=True))
    paths.update(out/n for n in ('checkpoint_SHA_ledger.json','cache_receipt_SHA_ledger.json','closure_readback.json'))
    return run,paths


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--build-child',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--pack-source',type=Path,default=ROOT/'cube4_visual_gciql_v1/closed_backup_preparation_v2/closed_role.py')
    parser.add_argument('--reference',type=Path,default=ROOT/'cube23_distance_gciql_v1/state_priority_v1/closed_training_backup_v1/build_closed_remaining.py')
    args=parser.parse_args();args.q=Q;args.group=GROUP;args.output=Q/'closed_final_backup_v1'/GROUP/'completed_v1'
    if not args.execute and not args.build_child:
        print(json.dumps(dict(prepared_only=True,group=GROUP,output=str(args.output),model_payload_files=2,cache_vectors_included=False)));return 0
    if args.build_child:
        code=1
        try:
            base=load_pack(args.pack_source);base.select=select;base.build(args);code=0
        finally:
            with (args.output/'builder_actual_exit_code').open('x') as s:s.write(str(code)+'\n')
        return code
    args.output.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),'--build-child','--pack-source',str(args.pack_source),'--reference',str(args.reference)]
    with (args.output/'builder_actual_argv.json').open('x') as s:json.dump(dict(argv=argv,started=time.time(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),generic_pack_source_sha256=PACK_SHA),s)
    with (args.output/'builder_stdout.log').open('xb') as a,(args.output/'builder_stderr.log').open('xb') as b:
        result=subprocess.run(argv,stdout=a,stderr=b)
    with (args.output/'builder_outer_exit_code').open('x') as s:s.write(str(result.returncode)+'\n')
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
