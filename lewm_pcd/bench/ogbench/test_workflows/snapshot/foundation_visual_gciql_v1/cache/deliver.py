"""Make separate real pilot fixtures or publish audit-bound full-cache metadata."""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
from common import HERE,ROOT,sha,stat,save
from audit_bridge import check_camera

def actual_exit(path):
    value=Path(path).read_text().strip()
    if value=='0':return dict(actual_exit_code=0,format='plain_integer')
    record=json.loads(value)
    assert record['actual_exit_code']==0 and record['residual_owned_members']==[], 'Producer must have exited and drained workers'
    return record

def publish(plan_path,audit_path=None,pilot=False,producer_exit_file=None):
    plan_path=Path(plan_path);plan=json.loads(plan_path.read_text())
    group=plan_path.parent;source_cache=Path(plan['cache_dir']);n,d=plan['cube_count'],plan['latent_dim']
    data=json.loads((group/'data_binding.json').read_text())
    if pilot:
        assert producer_exit_file is not None,'Actual completed producer child exit file is required'
        exit_record=actual_exit(producer_exit_file)
        original=json.loads((source_cache/'pilot_receipt.json').read_text())
        assert original['actual_exit_code']==0 and original['not_full_cache'] is True
        proof_path=source_cache/'batch1_alignment.json';proof=json.loads(proof_path.read_text())
        assert proof['actual_exit_code']==0 and proof['passed'] is True
        assert proof['encoder_spec_sha256']==plan['encoder_spec_sha256']
        target=group/'pilot_fixture';cache=target/'cache';data=copy.deepcopy(data)
        manifest=dict(schema_version=1,purpose='pilot',splits={},checkpoint_sha256=plan['checkpoint_sha256'],
            actual_exit_code=0,source_pilot_receipt=dict(path=str(source_cache/'pilot_receipt.json'),sha256=sha(source_cache/'pilot_receipt.json')))
        for split in ('train','val'):
            source_npz=Path(data['splits'][split]['path']);target_npz=target/'data'/f'{split}.npz'
            target_npz.parent.mkdir(parents=True,exist_ok=True)
            if not target_npz.exists():
                with np.load(source_npz,allow_pickle=False) as raw:
                    fields={k:raw[k][:1001].copy() for k in ('observations','actions','qpos','qvel','terminals')}
                with target_npz.open('xb') as f:np.savez_compressed(f,**fields)
            with np.load(target_npz,allow_pickle=False) as raw:
                assert raw['observations'].shape==(1001,19+9*n) and raw['actions'].shape==(1001,5)
                assert np.array_equal(np.flatnonzero(raw['terminals']),[1000])
            provenance=data['splits'][split]
            data['splits'][split]=dict(path=str(target_npz),accepted_stat=stat(target_npz),episodes=1,raw_rows=1001,
                observation_shape=[1001,19+9*n],action_shape=[1001,5],accepted_raw_sha256=sha(target_npz),
                original_official_source=provenance,subset='first complete true1001 episode of each official split')
            source_receipt=source_cache/split/'episode_000000/receipt.json'
            record=json.loads(source_receipt.read_text());assert record['actual_exit_code']==0
            original_latent=Path(record['latent']['path']);path=cache/split/'episode_000000/latent.npy'
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():os.link(original_latent,path)
            assert sha(path)==record['latent']['sha256']
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            assert value.shape==(1001,d) and value.dtype==np.float32 and np.isfinite(value).all()
            assert record['raw_row_start']==0 and record['terminal_raw_id']==1000
            check_camera(record['camera'],plan['camera'])
            record=copy.deepcopy(record)
            record.update(purpose='pilot',original_producer_receipt=dict(path=str(source_receipt),sha256=sha(source_receipt)),
                source_npz=dict(path=str(target_npz),accepted_stat=stat(target_npz)))
            record['latent']['path']=str(path)
            receipt=path.parent/'receipt.json';save(receipt,record)
            manifest['splits'][split]=[dict(episode=0,receipt_path=str(receipt),receipt_sha256=sha(receipt))]
        save(cache/'cache_manifest.json',manifest)
        audit_path=target/'cache_audit.json'
        save(audit_path,dict(actual_exit_code=0,purpose='pilot',episodes_verified=2,raw_rows_verified=2002,
            latent_dim=d,dtype='float32',manifest_sha256=sha(cache/'cache_manifest.json'),
            source_pilot_receipt=manifest['source_pilot_receipt'],full_cache_claim=False,
            batch1_alignment=dict(path=str(proof_path),sha256=sha(proof_path)),
            actual_producer_exit=dict(path=str(producer_exit_file),sha256=sha(producer_exit_file),exit_code=0,
                                      guard_record=exit_record),
            all_new_latent_SHA_shape_dtype_finite=True,all_raw_ID_and_true_terminal_next_contracts=True))
        destination=target/'delivery.json'
    else:
        assert audit_path is not None
        audit_path=Path(audit_path);audit=json.loads(audit_path.read_text())
        assert audit['actual_exit_code']==0 and audit['episodes_verified']==sum(plan['split_episodes'].values())
        assert (audit_path.parent/'actual_exit_code').read_text().strip()=='0'
        assert audit['plan_sha256']==sha(plan_path)
        cache=source_cache
        assert audit['manifest_sha256']==sha(cache/'cache_manifest.json')
        destination=group/'delivery.json'
    delivery=dict(purpose='pilot' if pilot else 'formal',group_id=group.name,cube_count=n,latent_dim=d,
        encoder_binding=dict(family=plan['encoder_family'],latent_dim=d,checkpoint_sha256=plan['checkpoint_sha256'],
            spec_path=plan['encoder_spec'],spec_sha256=plan['encoder_spec_sha256'],
            adapter=dict(path=plan['encoder_adapter'],sha256=plan['encoder_adapter_sha256'])),
        data_binding=data,camera=plan['camera'],cache_audit=dict(path=str(audit_path),sha256=sha(audit_path)),
        feature_loader_arguments={split:dict(cache_dir=str(cache),split=split,
            manifest_sha256=sha(cache/'cache_manifest.json'),checkpoint_sha256=plan['checkpoint_sha256'],
            source_npz=item['path'],expected_rows=item['raw_rows']) for split,item in data['splits'].items()})
    save(destination,delivery)
    print(json.dumps(dict(delivery=str(destination),purpose=delivery['purpose'],actual_exit_code=0)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True,type=Path)
    p.add_argument('--audit-report',type=Path);p.add_argument('--pilot',action='store_true')
    p.add_argument('--producer-exit-file',type=Path);a=p.parse_args()
    publish(a.plan,a.audit_report,a.pilot,a.producer_exit_file)
