"""Publish only a new G12 config after accepted G11 and complete PCD metadata."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
PCD_CHECKPOINT_SHA='95e7980f665a0ad888a0723499f5d0771c8cf797065b1f1b39c3eae39fd951d8'
STATIC_PINS={'quad_visual.py':'a3f735677d83760c53b2d48ff032c03060ed40558cd7fa7dad5671f06b7696f8',
    'configs/G11_cube4_rgb.json':'57211622e3d426eda0bbbae32c8266d6c1a02a05e500d9e2b83d309fed8c988c',
    'configs/G12_cube4_pcd.TEMPLATE.json':'d79a152595b553a1f4658f18a7cf52ba9221eb3439a2c3d8f480638ade8a28ea'}
PCD_PHASES=('00_wait_G11_acceptance','01_reference','02_fresh','03_GPU_preprocess',
            '04_profile','05_pilot','06_full','07_audit')
G11_PHASES=('01_gate_G11','02_reload_G11','03_train_G11','04_accept_G11')
COUNTERS=[1]+list(range(100000,1000001,100000))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def publish(path,value):
    temporary=path.with_name(path.name+f'.pending_{os.getpid()}')
    with temporary.open('xb') as stream:
        stream.write((json.dumps(value,indent=2,allow_nan=False)+'\n').encode());stream.flush();os.fsync(stream.fileno())
    os.link(temporary,path)  # Atomic create, never overwrite an existing config/receipt.
    temporary.unlink()


def prepare_config(q,pcd):
    q,pcd=Path(q).resolve(),Path(pcd).resolve()
    assert q.name=='cube4_visual_gciql_v1' and pcd.name=='pcd_training_camera_v1'
    output=q/'functional_v1/configs/G12_cube4_pcd.v2.json'
    receipt=q/'functional_v1/G12_CONFIG_PREPARED_v2.json'
    if output.exists() or receipt.exists():raise FileExistsError('Never overwrite G12 config/preparation receipt')
    records={}
    def read(path,expected_sha=None,json_value=True):
        path=Path(path);before=path.stat();blob=path.read_bytes();after=path.stat()
        assert (before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns)
        digest=hashlib.sha256(blob).hexdigest();assert expected_sha is None or digest==expected_sha,str(path)
        records[str(path)]={'sha256':digest,'bytes':len(blob),'mtime_ns':after.st_mtime_ns,'ctime_ns':after.st_ctime_ns}
        return json.loads(blob) if json_value else blob.decode().strip()
    def zero(path):assert read(path,json_value=False)=='0',str(path)
    runtime=pcd/'runtime_v1'
    zero(runtime/'outer_exit_code');zero(q/'control_v1/PCD/wrapper_exit_code')
    state=read(runtime/'status.json');assert state['actual_exit_code']==0 and state['stage']=='all_embedding_phases_completed'
    closed=read(runtime/'closure.json')
    assert (closed['actual_exit_code'],closed['completed_phases'],closed['cube_count'],closed['expected_features_per_modality'])==(0,8,4,5505500)
    assert closed['data_id']=='cube-quadruple-play-v0' and closed['training_updates']==closed['SR_episodes']==0
    for name in PCD_PHASES:
        phase=read(runtime/f'{name}_exit.json');assert phase['actual_exit_code']==0 and phase['residual_owned_members']==[]
    g11=q/'control_v1/G11'
    zero(g11/'runtime/outer_exit_code');zero(g11/'wrapper_exit_code')
    state=read(g11/'runtime/status.json')
    assert state['actual_exit_code']==0 and state['completed_phases']==4 and state['trained_groups']==state['accepted_groups']==1
    for name in G11_PHASES:
        phase=read(g11/'runtime'/name/'exit.json');assert phase['actual_exit_code']==0 and phase['residual_owned_members']==[]
        zero(g11/'runtime'/name/'actual_exit_code')
    for relative,digest in STATIC_PINS.items():read(q/'functional_v1'/relative,digest,json_value=False)
    g11cfg=read(q/'functional_v1/configs/G11_cube4_rgb.json')
    template=read(q/'functional_v1/configs/G12_cube4_pcd.TEMPLATE.json')
    assert template['executable'] is False and template['group_id']=='G12' and template['modality']=='pcd'
    for key in ('agent_config','reward','input_contract','seed','batch_size','total_updates','checkpoint_steps','validation_interval','save_interval','save_step1'):
        assert template[key]==g11cfg[key],key
    accepted=read(q/'acceptance/G11_full_acceptance.json')
    assert accepted['actual_exit_code']==0 and accepted['full153_state_exact'] is True and accepted['state']['leaves']==153
    assert accepted['Adam_count']==1000000 and accepted['network_step']==1000001 and accepted['CSV_validation_records']==200
    assert accepted['actual_current_dim']==211 and accepted['actual_goal_dim']==204 and accepted['synthetic_zero_padding'] is False
    assert [item['counter'] for item in accepted['checkpoint_transactions']]==COUNTERS
    assert accepted['final_checkpoint']==g11cfg['output']+'/checkpoints/update_001000000'
    plan_path=pcd/'workspace/official_state_3d_cache_v1/producer_plan.json';plan=read(plan_path)
    cache=pcd/'3d/cache';manifest_path=cache/'cache_manifest.json';seal_path=cache/'cache_seal.json'
    assert Path(plan['cache_dir']).resolve()==cache and plan['checkpoint_sha256']==PCD_CHECKPOINT_SHA
    assert plan['split_episodes']=={'train':5000,'val':500} and plan['raw_rows_per_episode']==1001 and plan['encoder_batch']==32
    manifest=read(manifest_path);seal=read(seal_path)
    assert manifest['actual_exit_code']==seal['actual_exit_code']==0
    assert manifest['checkpoint_sha256']==manifest['protocol']['checkpoint_sha256']==PCD_CHECKPOINT_SHA
    assert object_sha(manifest['protocol'])==manifest['protocol_sha256']==seal['protocol_sha256']
    assert seal['manifest_sha256']==records[str(manifest_path)]['sha256'] and seal['full_raw_ID_coverage'] is True
    assert (seal['train_rows'],seal['val_rows'])==(5005000,500500)
    assert len(bytes.fromhex(manifest['source_encoder_state_sha256']))==32
    assert manifest['training_updates']==manifest['SR_episodes']==0
    zero(pcd/'3d/independent_audit/actual_exit_code');audit=read(pcd/'3d/independent_audit/report.json')
    assert audit['actual_exit_code']==0 and audit['plan_sha256']==records[str(plan_path)]['sha256']
    assert audit['manifest_sha256']==records[str(manifest_path)]['sha256'] and audit['producer_seal_sha256']==records[str(seal_path)]['sha256']
    assert audit['checkpoint_sha256']==PCD_CHECKPOINT_SHA and audit['protocol_sha256']==manifest['protocol_sha256']
    assert (audit['episodes_verified'],audit['raw_rows_verified'],audit['main_rows_verified'],audit['latent_dim'],audit['dtype'])==(5500,5505500,5500000,192,'float32')
    assert audit['all_new_latent_SHA_shape_dtype_finite'] and audit['all_canonical_ID_and_terminal_next_contracts'] and audit['tracked_stats_stable']
    arrays_ref=manifest['source_arrays_receipt'];arrays=read(arrays_ref['path'],arrays_ref['sha256'])
    assert arrays_ref['sha256']==plan['source_arrays_receipt_sha256']==audit['source_arrays_receipt_sha256']
    cfg=copy.deepcopy(template)
    cfg.update(executable=True,status='G11_and_complete_PCD_cache_accepted_pending_G12_real_gate',output=str(q/'runs/G12_cube4_pcd'),GPU_execution='serial G12 after accepted G11 and complete accepted PCD cache')
    cfg.pop('disabled_reason',None)
    cfg['feature_loader_arguments']={}
    for split,count in plan['split_episodes'].items():
        assert [entry['episode'] for entry in manifest['splits'][split]]==list(range(count))
        source=arrays['splits'][split];bound=cfg['data_binding']['splits'][split]
        assert source['source_npz']==bound['path'] and source['accepted_stat']==bound['accepted_stat'] and source['rows']==count*1001
        cfg['feature_loader_arguments'][split]=dict(cache_dir=str(cache),split=split,
            manifest_sha256=records[str(manifest_path)]['sha256'],checkpoint_sha256=PCD_CHECKPOINT_SHA,
            source_npz=source['source_npz'],expected_rows=count*1001)
    cfg['cache_binding']=dict(cube_count=4,data_id='cube-quadruple-play-v0',modality='pcd',cache_directory=str(cache),
        layout='raw1001_rows_per_episode_single_npy',latent_dim=192,dtype='float32',split_episodes=plan['split_episodes'],
        canonical_raw_rows_per_episode=1001,canonical_total_rows=5505500,
        manifest={'path':str(manifest_path),'sha256':records[str(manifest_path)]['sha256']},
        seal={'path':str(seal_path),'sha256':records[str(seal_path)]['sha256']},
        checkpoint_sha256=PCD_CHECKPOINT_SHA,PCD=plan['pcd'],camera=plan['camera'],
        full_current_host_audit={'path':str(pcd/'3d/independent_audit/report.json'),'sha256':records[str(pcd/'3d/independent_audit/report.json')]['sha256']},
        source_rebind_required=False,RGB_rebind_not_used=True)
    cfg['parent_frozen_config']=dict(path=str(q/'functional_v1/configs/G11_cube4_rgb.json'),sha256=STATIC_PINS['configs/G11_cube4_rgb.json'],parent_not_modified=True,minimum_modality_and_cacheasset_binding_adaptation=True)
    for name,item in records.items():
        current=Path(name).stat();assert (current.st_size,current.st_mtime_ns,current.st_ctime_ns)==(item['bytes'],item['mtime_ns'],item['ctime_ns'])
    publish(output,cfg)
    result=dict(actual_exit_code=0,completed=time.time(),config_path=str(output),config_sha256=sha(output),
        group_id='G12',current_dim=211,goal_dim=204,G11_full_CPUacceptance=True,
        PCD_eight_phases_and_wrappers_actual0=True,PCD_episodes=5500,PCD_rows=5505500,
        metadata_input_files=records,old_template_and_core_unchanged=True,
        payload_model_raw_cache_bytes_rehashed=False,NN_GPU_training_started=False)
    publish(receipt,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--q',type=Path,default=ROOT/'cube4_visual_gciql_v1')
    parser.add_argument('--pcd-case',type=Path,default=ROOT/'cube4_official_embeddings_v1/pcd_training_camera_v1')
    args=parser.parse_args();print(json.dumps(prepare_config(args.q,args.pcd_case)))
