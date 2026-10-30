"""Metadata-only recovery of the preserved failed PCD orchestration."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import types

OLD_SHA='cd76e395ed8ec4fa8020ba06ae43ec684b63f63fe37ad3acb6f48a0f13ea4c15'
OLD_PHASES=('00_wait_G11_acceptance','01_reference','02_fresh','03_GPU_preprocess')
NEW_PHASES=('04_profile','05_pilot','06_full','07_audit')
MODEL='95e7980f665a0ad888a0723499f5d0771c8cf797065b1f1b39c3eae39fd951d8'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable(path,records,expected=None,json_value=True):
    path=Path(path);a=path.stat();blob=path.read_bytes();b=path.stat()
    assert (a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(b.st_size,b.st_mtime_ns,b.st_ctime_ns)
    digest=hashlib.sha256(blob).hexdigest();assert expected is None or digest==expected,str(path)
    records[str(path)]={'sha256':digest,'bytes':len(blob),'mtime_ns':b.st_mtime_ns,'ctime_ns':b.st_ctime_ns}
    return json.loads(blob) if json_value else blob.decode().strip()


def load_binding(q,pcd,path,digest,records):
    binding=stable(path,records,digest)
    assert binding['schema_version']==2
    assert Path(binding['q']).resolve()==q and Path(binding['pcd_case']).resolve()==pcd
    return binding


def chain(q,pcd,binding,read):
    def zero(path):assert read(path,json_value=False)=='0',str(path)
    old=pcd/'runtime_v1';new=pcd/'runtime_v2'
    for path in (old/'outer_exit_code',q/'control_v1/PCD/wrapper_exit_code'):
        assert read(path,json_value=False)=='1',str(path)
    assert read(old/'status.json')['actual_exit_code']==1
    for folder,names in ((old,OLD_PHASES),(new,NEW_PHASES)):
        for name in names:
            item=read(folder/f'{name}_exit.json')
            assert item['actual_exit_code']==0 and item['residual_owned_members']==[],name
    zero(new/'outer_exit_code');zero(q/'control_v2/PCD/wrapper_exit_code')
    state=read(new/'status.json')
    assert state['actual_exit_code']==0 and state['stage']=='all_embedding_phases_completed'
    closed=read(new/'closure.json')
    assert (closed['actual_exit_code'],closed['completed_phases'],closed['cube_count'],closed['expected_features_per_modality'])==(0,4,4,5505500)
    assert closed['data_id']=='cube-quadruple-play-v0' and closed['training_updates']==closed['SR_episodes']==0
    def pinned(key,path=None):
        ref=binding[key];target=Path(ref['path']).resolve()
        if path is not None:assert target==path.resolve(),key
        return read(target,ref['sha256'])
    plan=pinned('remaining_launch_plan',q/'control_v2/PCD/launch_plan.json')
    assert [item['name'] for item in plan['phases']]==list(NEW_PHASES)
    gate=pinned('old_GPU_report',pcd/'3d/GPU_preprocess_report.json')
    assert gate['actual_exit_code']==0 and gate['accepted'] is False
    assert gate['prepared_bitexact'] and gate['latent_bitexact'] and gate['model_frozen_exact']
    assert gate['prepared_max_abs_diff']==gate['latent_max_abs_diff']==0
    assert gate['training_updates']==gate['render_frames']==0 and gate['checkpoint_sha256']==MODEL
    gain=(gate['baseline_median_seconds']-gate['candidate_median_seconds'])/gate['baseline_median_seconds']
    assert 0<gain<.05
    decision=pinned('engineering_decision')
    assert decision['actual_exit_code']==0 and decision['original_engineering_5percent_gate_passed'] is False
    assert decision['positive_gain_released'] is True and decision['unchanged_numeric_tolerances'] is True
    assert decision['gate_report_sha256']==binding['old_GPU_report']['sha256']
    assert abs(decision['positive_gain_fraction']-gain)<1e-15
    # The old complete-cache contracts are checked by the original preparer below.
    return dict(original_runtime_v1_actual_exit_code=1,original_GPU_gate_accepted=False,
        inherited_child_phases=4,new_runtime_completed_phases=4,logical_completed_phases=8,
        positive_gain_fraction=gain,original_engineering_5percent_gate_passed=False)


def checked_receipt(q,pcd,binding,binding_path,binding_sha,read):
    receipt=read(pcd/'recovery_receipt_v2.json')
    assert receipt['actual_exit_code']==0 and receipt['schema_version']==2
    assert receipt['publisher_sha256']==sha(__file__) and receipt['old_preparer_sha256']==OLD_SHA
    assert receipt['binding_sha256']==binding_sha and receipt['binding_path']==str(binding_path)
    for path,item in receipt['metadata_input_files'].items():read(path,item['sha256'],json_value=False)
    assert receipt['chain']==chain(q,pcd,binding,read)
    assert receipt['cube_count']==4 and receipt['data_id']=='cube-quadruple-play-v0'
    assert receipt['PCD_episodes']==5500 and receipt['PCD_rows']==5505500
    return receipt


def original(q,closure):
    path=q/'functional_v1/prepare_G12.py';raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==OLD_SHA
    source=raw.decode();start=source.index("    runtime=pcd/'runtime_v1'\n");end=source.index("    g11=q/'control_v1/G11'\n",start)
    source=source[:start]+'    recovery_closure(q,pcd,read)\n'+source[end:]
    edits=[("output=q/'functional_v1/configs/G12_cube4_pcd.v2.json'","output=q/'functional_v2/configs/G12_cube4_pcd.v3.json'"),
           ("receipt=q/'functional_v1/G12_CONFIG_PREPARED_v2.json'","receipt=q/'functional_v2/G12_CONFIG_PREPARED_v3.json'"),
           ('PCD_eight_phases_and_wrappers_actual0=True','PCD_recovery_chain_actual0=True,original_PCD_runtime_v1_actual_exit_code=1,original_GPU_gate_accepted=False')]
    for before,after in edits:assert source.count(before)==1;source=source.replace(before,after)
    module=types.ModuleType('_original_G12_recovery');module.__file__=str(path)
    exec(compile(source,str(path),'exec'),module.__dict__);module.recovery_closure=closure
    return module


def publish_recovery(q,pcd,binding_path,binding_sha):
    q,pcd,binding_path=Path(q).resolve(),Path(pcd).resolve(),Path(binding_path).resolve()
    target=pcd/'recovery_receipt_v2.json'
    if target.exists():raise FileExistsError('Preserve existing recovery receipt')
    records={};binding=load_binding(q,pcd,binding_path,binding_sha,records)
    read=lambda path,expected_sha=None,json_value=True:stable(path,records,expected_sha,json_value)
    facts=chain(q,pcd,binding,read)
    module=original(q,lambda _q,_p,reader:chain(_q,_p,binding,reader))
    captured={};module.publish=lambda path,value:captured.update({str(path):value})
    original_sha=module.sha
    module.sha=lambda path:hashlib.sha256((json.dumps(captured[str(path)],indent=2,allow_nan=False)+'\n').encode()).hexdigest() if str(path) in captured else original_sha(path)
    verified=module.prepare_config(q,pcd)  # All original contracts; publication intercepted in memory.
    assert verified['actual_exit_code']==0 and len(captured)==2
    for path,item in verified['metadata_input_files'].items():
        read(path,item['sha256'],json_value=False)
    receipt=dict(schema_version=2,actual_exit_code=0,completed=time.time(),cube_count=4,
        data_id='cube-quadruple-play-v0',PCD_episodes=5500,PCD_rows=5505500,
        chain=facts,old_preparer_sha256=OLD_SHA,publisher_sha256=sha(__file__),
        binding_path=str(binding_path),binding_sha256=binding_sha,metadata_input_files=records,
        old_runtime_and_failed_engineering_gate_preserved=True,all_original_G11_manifest_audit_guards_reused=True,
        config_not_published=True,payload_model_raw_cache_bytes_rehashed=False,NN_GPU_training_started=False)
    module=original(q,lambda *_:None);module.publish(target,receipt)
    return receipt


def prepare(q,pcd,binding_path,binding_sha):
    q,pcd,binding_path=Path(q).resolve(),Path(pcd).resolve(),Path(binding_path).resolve()
    records={};binding=load_binding(q,pcd,binding_path,binding_sha,records)
    def closure(_q,_p,reader):
        reader(binding_path,binding_sha)
        checked_receipt(_q,_p,binding,binding_path,binding_sha,reader)
    (q/'functional_v2/configs').mkdir(exist_ok=True)
    return original(q,closure).prepare_config(q,pcd)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('publish-recovery','prepare'))
    parser.add_argument('--q',type=Path,required=True);parser.add_argument('--pcd-case',type=Path,required=True)
    parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--binding-sha256',required=True)
    args=parser.parse_args();function=publish_recovery if args.action=='publish-recovery' else prepare
    print(json.dumps(function(args.q,args.pcd_case,args.binding,args.binding_sha256)))
