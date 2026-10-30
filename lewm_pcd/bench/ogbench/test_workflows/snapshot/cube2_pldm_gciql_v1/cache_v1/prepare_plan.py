"""One metadata view and necessary migration byteproof; no model or rendering."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

from producer_bridge import ROOT,ORIGINAL,ORIGINAL_SHA,ADAPTER,ASSET
from encoder_adapter import ASSET_ID,MANIFEST_SHA,ADDENDUM_SHA,CHECKPOINT_SHA

OLD_PLAN_SHA='12454be9c3411d444840d323e75a7db8a3a63de79e0d231e02bfb35f29bffff9'
OLD_ARRAYS_SHA='28eedab709578fdf874d2261984c7cfc26cd2b7c41b4d191fc3fc2dfd83bb35e'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def object_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def stat(path):
    v=Path(path).stat();return dict(size=v.st_size,mtime_ns=v.st_mtime_ns,ctime_ns=v.st_ctime_ns,device=v.st_dev,inode=v.st_ino)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def prepare(case):
    case=Path(case).resolve();assert case==ROOT/'cube2_pldm_gciql_v1'
    out=case/'cache_v1';planpath=out/'producer_plan.json'
    if planpath.exists() or (out/'PREPARED.json').exists():raise FileExistsError('Prepared metadata is immutable')
    original=ROOT/'cube2_official_embeddings_v1/workspace/official_state_2d_cache_v1/producer_plan.json';assert sha(original)==OLD_PLAN_SHA
    old=json.loads(original.read_text());arrayspath=Path(old['shared_state_root'])/'source_arrays.json'
    assert sha(arrayspath)==OLD_ARRAYS_SHA
    arrays=json.loads(arrayspath.read_text());assert arrays['contract']['extractor_source_sha256']==ORIGINAL_SHA
    assert object_sha(arrays['contract'])==arrays['protocol_sha256']
    assert arrays['contract']['split_episodes']=={'train':1000,'val':100} and arrays['contract']['raw_rows_per_episode']==1001
    assert arrays['contract']['extractor_source_sha256']=='fa6ff3dcb2d8fb709de069ea3aebfa5970cf3686a4a8aec20def899bd9848cd8'
    assert sha(old['data_acceptance'])==old['data_acceptance_sha256'] and sha(old['source_stat_freeze'])==old['source_stat_freeze_sha256']
    accepted=json.loads(Path(old['data_acceptance']).read_text());freeze=json.loads(Path(old['source_stat_freeze']).read_text())
    assert accepted['actual_exit_code']==freeze['actual_exit_code']==0
    assert accepted['data_id']=='cube-double-play-v0' and accepted['cube_count']==2
    view=copy.deepcopy(arrays);proof=[]
    def rebind(path,expected_stat,expected_sha):
        before=stat(path);observed={k:before[k] for k in expected_stat}
        changed=observed!=expected_stat
        if changed:assert sha(path)==expected_sha,'Migrated bytes differ: '+str(path)
        assert stat(path)==before,'Source changed during migration check'
        proof.append(dict(path=str(path),original_stat=expected_stat,current_stat=observed,
            expected_sha256=expected_sha,current_bytes_SHA_verified=changed,stat_unchanged_original_byteproof_reused=not changed))
        return observed
    for split,count in [('train',1000),('val',100)]:
        item=view['splits'][split];source=freeze['datasets']['validation' if split=='val' else 'train']
        assert item['source_npz']==source['path'] and item['accepted_stat']==source['accepted_stat']
        assert item['rows']==count*1001
        item['accepted_stat']=rebind(item['source_npz'],item['accepted_stat'],source['raw_sha256'])
        source['accepted_stat']=item['accepted_stat']
        for key,width in [('qpos',28),('qvel',26)]:
            data=item[key];assert data['shape']==[count*1001,width] and data['dtype']=='float32'
            data['stat']=rebind(data['path'],data['stat'],data['sha256'])
    meta=out/'source_arrays'
    accepted.update(original_acceptance={'path':old['data_acceptance'],'sha256':old['data_acceptance_sha256']},
        current_migration_byteproof=proof,arrays_reextracted=False,old_files_modified=False)
    acceptance=meta/'data_acceptance.json';write(acceptance,accepted)
    freeze.update(acceptance_path=str(acceptance),acceptance_sha256=sha(acceptance),
        original_stat_freeze={'path':old['source_stat_freeze'],'sha256':old['source_stat_freeze_sha256']})
    frozen=meta/'stat-freeze.json';write(frozen,freeze)
    view['contract'].update(data_acceptance_sha256=sha(acceptance),source_stat_freeze_sha256=sha(frozen))
    view.update(protocol_sha256=object_sha(view['contract']),source_stat_freeze_sha256=sha(frozen),
        original_receipt={'path':str(arrayspath),'sha256':OLD_ARRAYS_SHA,'contract':arrays['contract'],'protocol_sha256':arrays['protocol_sha256']},
        arrays_reextracted=False,current_migration_byteproof=proof,original_extractor_not_relabelled=True,
        original_NPZ_rehashed=any(item['current_bytes_SHA_verified'] for item in proof if item['path'].endswith('.npz')))
    write(meta/'source_arrays.json',view)
    asset=ROOT/ASSET;assert sha(asset/'MANIFEST.json')==MANIFEST_SHA and sha(asset/'RUNTIME_ADDENDUM.json')==ADDENDUM_SHA
    manifest=json.loads((asset/'MANIFEST.json').read_text());addon=json.loads((asset/'RUNTIME_ADDENDUM.json').read_text())
    assert manifest['asset_id']==ASSET_ID and manifest['embedding_dim']==512 and manifest['global_step']==378900
    assert manifest['members']['weights/weights_epoch_10.pt']['sha256']==CHECKPOINT_SHA
    pins={key:value for key,value in old['source_pins'].items() if key.startswith('source/ogbench/')}
    old_render_key='official_state_2d_cache_v1/render_adapter.py'
    pins['cube2_official_embeddings_v1/workspace/official_state_2d_cache_v1/render_adapter.py']=old['source_pins'][old_render_key]
    for name,row in manifest['members'].items():
        if (name.startswith('source/') and Path(name).suffix in ('.py','.json','.yaml','.yml','.toml')) or name=='encoder.py':pins[ASSET+'/'+name]=row['sha256']
    for name,h in addon['files'].items():
        if name.endswith('.py'):pins[ASSET+'/'+name]=h
    for name in ['MANIFEST.json','RUNTIME_ADDENDUM.json']:pins[ASSET+'/'+name]=sha(asset/name)
    for name in ['encoder_adapter.py','producer_bridge.py','audit_bridge.py','prepare_plan.py']:
        pins['cube2_pldm_gciql_v1/cache_v1/'+name]=sha(out/name)
    for name,h in pins.items():assert sha(ROOT/name)==h,'Source pin changed: '+name
    plan=copy.deepcopy(old);plan.update(workspace=str(ROOT),cache_dir=str(out/'cache'),shared_state_root=str(meta),
        checkpoint=str(asset/'weights/weights_epoch_10.pt'),checkpoint_sha256=CHECKPOINT_SHA,
        data_acceptance=str(acceptance),data_acceptance_sha256=sha(acceptance),source_stat_freeze=str(frozen),source_stat_freeze_sha256=sha(frozen),
        source_pins=pins,latent_dim=512,encoder_family='pldm',encoder_asset_id=ASSET_ID,
        encoder_asset_manifest_sha256=MANIFEST_SHA,encoder_runtime_addendum_sha256=ADDENDUM_SHA,
        encoder_preprocess=dict(input_dtype='uint8',input_shape=[3,224,224],output_dtype='float32',output_shape=[3,65,65],divisor=255.,antialias=True,ImageNet_normalization=False,history_duplication=False,Native_B1_alignment_required=True),
        source_arrays_original_extractor_sha256=ORIGINAL_SHA,source_arrays_original_receipt_sha256=OLD_ARRAYS_SHA,
        source_arrays_view_sha256=sha(meta/'source_arrays.json'),training_updates=0,SR_episodes=0)
    write(planpath,plan)
    result=dict(actual_exit_code=0,completed=time.time(),plan_path=str(planpath),plan_sha256=sha(planpath),
        source_arrays_view=str(meta/'source_arrays.json'),source_arrays_view_sha256=sha(meta/'source_arrays.json'),
        original_arrays_receipt_sha256=OLD_ARRAYS_SHA,original_extractor_sha256=ORIGINAL_SHA,
        migration_files=len(proof),current_bytes_SHA_verified_files=sum(item['current_bytes_SHA_verified'] for item in proof),
        arrays_reextracted=False,old_files_modified=False,model_or_cache_vectors_read=False,NN_GPU_started=False,
        encoder_family='pldm',latent_dim=512,episodes=1100,canonical_rows=1101100)
    write(out/'PREPARED.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(prepare(args.case)))
