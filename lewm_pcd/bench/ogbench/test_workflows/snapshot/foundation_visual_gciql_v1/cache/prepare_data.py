"""CPU-only official bytes verification and current-host array binding, once per cube."""
import argparse
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from common import ROOT, HERE, NAMES, COUNTS, sha, stat, digest, save

OLD_ARRAYS = {2: ROOT/'cube2_pldm_gciql_v1/cache_v1/source_arrays/source_arrays.json',
              3: ROOT/'cube3_pldm_gciql_v1/cache_v1/source_arrays/source_arrays.json',
              4: ROOT/'cube4_visual_gciql_v1/migration_rebind_v1/source_arrays_view_v1/source_arrays.json'}

def prepare(n):
    out = HERE/'data'/f'cube{n}'
    out.mkdir(parents=True,exist_ok=True)
    receipt = out/'inventory.json'
    if receipt.exists():
        record=json.loads(receipt.read_text())
        for split,item in record['data_binding']['splits'].items():
            assert stat(item['path'])==item['accepted_stat']
        for item in record['arrays']['splits'].values():
            for key in ('qpos','qvel'): assert stat(item[key]['path'])==item[key]['stat']
        print(json.dumps(dict(cube=n,status='existing_verified_current_host_metadata_reused')));return record
    download=ROOT/'official_state_download_all_v1/status.json'
    prior=json.loads(download.read_text());assert prior['actual_exit_code']==0
    refs={x['file']:x for x in prior['results']}
    old=json.loads(OLD_ARRAYS[n].read_text()) if n in OLD_ARRAYS else None
    data=dict(cube_count=n,data_id=f'cube-{NAMES[n]}-play-v0',splits={})
    arrays=dict(splits={},original_NPZ_rehashed=True,semantic_array_origin='official_qpos_qvel_only')
    started=time.time()
    for split,count in COUNTS[n].items():
        path=ROOT/'data/official_state'/f'cube-{NAMES[n]}-play-v0{"-val" if split=="val" else ""}.npz'
        before=stat(path); actual=sha(path); ref=refs[path.name]
        assert actual==ref['sha256'] and ref['archive_CRC_all_members'] is True
        rows=count*1001; headers={}
        with zipfile.ZipFile(path) as archive:
            for name in ('observations','actions','terminals','qpos','qvel'):
                with archive.open(name+'.npy') as f:
                    version=np.lib.format.read_magic(f)
                    shape,fortran,dtype=np.lib.format._read_array_header(f,version)
                    assert not fortran
                    headers[name]=dict(shape=list(shape),dtype=str(dtype))
        assert headers['observations']==dict(shape=[rows,19+9*n],dtype='float32')
        assert headers['actions']==dict(shape=[rows,5],dtype='float32')
        assert headers['qpos']==dict(shape=[rows,14+7*n],dtype='float32')
        assert headers['qvel']==dict(shape=[rows,14+6*n],dtype='float32')
        with np.load(path,allow_pickle=False) as raw:
            terminals=raw['terminals']
            assert terminals.dtype==np.bool_ and terminals.shape==(rows,)
            assert np.array_equal(np.flatnonzero(terminals),np.arange(count)*1001+1000)
            item=dict(source_npz=str(path),accepted_stat=before,rows=rows)
            for key,width in (('qpos',14+7*n),('qvel',14+6*n)):
                if old:
                    source=old['splits'][split][key]; target=Path(source['path'])
                    b=stat(target); h=sha(target)
                    assert h==source['sha256'], 'Migrated source array byte identity differs'
                    value=np.load(target,mmap_mode='r',allow_pickle=False)
                    assert value.shape==(rows,width) and value.dtype==np.float32
                    assert np.isfinite(value).all() and stat(target)==b
                    # Existing extraction provenance is retained, not rewritten.
                    provenance=dict(path=str(OLD_ARRAYS[n]),sha256=sha(OLD_ARRAYS[n]),
                                    original_array_record=source,verified_same_bytes_this_run=True)
                else:
                    target=out/f'{split}_{key}.npy'
                    assert not target.exists(),'Uncommitted state array: inspect, do not overwrite'
                    value=raw[key]
                    assert value.shape==(rows,width) and value.dtype==np.float32 and np.isfinite(value).all()
                    with target.open('xb') as f:np.save(f,value,allow_pickle=False)
                    b=stat(target);h=sha(target)
                    provenance=dict(new_extraction=True,source_NPZ_sha256=actual)
                item[key]=dict(path=str(target),shape=[rows,width],dtype='float32',stat=b,sha256=h,provenance=provenance)
                del value
        assert stat(path)==before
        arrays['splits'][split]=item
        data['splits'][split]=dict(path=str(path),accepted_stat=before,accepted_raw_sha256=actual,
            observation_shape=[rows,19+9*n],action_shape=[rows,5],episodes=count,raw_rows=rows,
            headers=headers,terminal_indices_verified_this_run=True)
    save(out/'data_acceptance.json',dict(stage='complete',actual_exit_code=0,datasets=data['splits'],
        whole_NPZ_SHA_verified_current_host=True,prior_full_ZIP_CRC_reused_from=dict(path=str(download),sha256=sha(download))))
    acceptance=out/'data_acceptance.json'
    frozen=dict(actual_exit_code=0,acceptance_sha256=sha(acceptance),datasets={
        ('validation' if k=='val' else k):dict(path=v['path'],accepted_stat=v['accepted_stat']) for k,v in data['splits'].items()})
    save(out/'stat-freeze.json',frozen)
    contract=dict(data_acceptance_sha256=sha(acceptance),source_stat_freeze_sha256=sha(out/'stat-freeze.json'),
        split_episodes=COUNTS[n],raw_rows_per_episode=1001,qpos_width=14+7*n,qvel_width=14+6*n,
        extractor_source_sha256=sha(__file__))
    arrays.update(contract=contract,protocol_sha256=digest(contract),source_stat_freeze_sha256=contract['source_stat_freeze_sha256'])
    save(out/'source_arrays.json',arrays)
    result=dict(cube_count=n,data_binding=data,arrays=arrays,started=started,completed=time.time(),actual_exit_code=0,
        source_arrays_receipt=dict(path=str(out/'source_arrays.json'),sha256=sha(out/'source_arrays.json')))
    save(receipt,result)
    print(json.dumps(dict(cube=n,status='official_bytes_headers_terminals_and_arrays_verified',splits=COUNTS[n])))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cube',type=int,choices=range(1,5));a=p.parse_args()
    for n in ([a.cube] if a.cube else range(1,5)):prepare(n)
