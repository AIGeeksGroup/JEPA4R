"""Only true raw1000 terminal frames; original cache/model/preprocessor stay immutable."""
import argparse,ast,datetime,hashlib,importlib.util,json,os,sys
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()
def stat(path):
    s=Path(path).stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
def read(path):return json.loads(Path(path).read_text())
def save(path,data):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));os.replace(tmp,p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def terminal_raw_ids(count):return np.arange(count,dtype=np.int64)*1001+1000
def padding_for64(valid):
    if not 32<=valid<=64:raise ValueError('Fixed64 source split requires32..64 real frames per final batch')
    return 64-valid
def validate_identity(fields,episode):
    expected=dict(source_row=episode*1001+1000,source_episode=episode,source_step=1000,source_terminal=True,valid_transition=False)
    if any(fields[k]!=v for k,v in expected.items()):raise ValueError('Real source row1000 terminal identity differs')
def validate_payload(ids,z,count):
    if ids.dtype!=np.dtype('int64') or ids.shape!=(count,) or not np.array_equal(ids,terminal_raw_ids(count)):raise ValueError('Terminal raw IDs differ; no compact/cross-episode substitution')
    if z.dtype!=np.dtype('float32') or z.shape!=(count,192) or not np.isfinite(z).all():raise ValueError('Real finite FP32 terminal latent192 required')
def classify(manifest):
    tag=manifest.get('source_modality')
    if tag=='pcd' and 'source_pcd_index' in manifest and 'source_rgb_index' not in manifest:return 'pcd'
    if tag in (None,'rgb') and 'source_rgb_index' in manifest and 'source_pcd_index' not in manifest:return 'rgb'
    raise ValueError('Unknown/ambiguous historical modality; original manifest must not be edited')
def parse_args(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--execute',action='store_true');return p.parse_args(argv)
def alias(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def register_official_namespace(config,expected_init_sha):
    import importlib
    w=Path(config['workspace']);official=w/config['official'];repo=w/config['repo']/'lewm_pcd'
    sys.path[:0]=[str(official/'impls'),str(official),str(repo),str(repo/'third_party/stable-worldmodel'),str(w/Path(config['helper']).parent)]
    package=importlib.import_module('utils');expected=(official/'impls/utils/__init__.py').resolve()
    if Path(package.__file__).resolve()!=expected or sha(expected)!=expected_init_sha:raise ValueError('Original official utils package namespace differs')
    return dict(utils_source=str(expected),sha256=expected_init_sha)
def source_input(entry,cached,episode,kind,step=1000,reader=None):
    import hdf5plugin,h5py
    p=Path(entry['path']);before=stat(p);prefix='pcd' if kind=='pcd' else 'rgb'
    if before!=cached['source_'+prefix+'_stat_before'] or str(p)!=cached['source_'+prefix+'_path']:raise ValueError('Original frame owner/stat changed')
    with h5py.File(p,'r') as f:
        fields={k:f[k][step].item() for k in ['source_row','source_episode','source_step','source_terminal','valid_transition']}
        if step==1000:validate_identity(fields,episode)
        elif fields['source_row']!=episode*1001+step or fields['source_episode']!=episode or fields['source_step']!=step:raise ValueError('Boundary reference source identity differs')
        keys=['pixels'] if kind=='rgb' else ['pcd_xyz','pcd_rgb']
        rows={k:np.array(f[k][step:step+1],copy=True) for k in keys} if reader is None else None
    if reader is not None:
        role='next_observations' if step==1000 else 'value_goals'
        rows=reader.read(np.array([[episode*1001+step]],dtype=np.int64),role)
    if kind=='rgb':
        if rows['pixels'].shape!=(1,224,224,3) or rows['pixels'].dtype!=np.uint8:raise ValueError('Original opaque224RGB schema')
    elif rows['pcd_xyz'].shape!=(1,1200,3) or rows['pcd_xyz'].dtype!=np.float32 or rows['pcd_rgb'].shape!=(1,1200,3) or rows['pcd_rgb'].dtype!=np.uint8:raise ValueError('Original PCD1200 XYZ/uint8 RGB schema')
    if not all(np.isfinite(x).all() for x in rows.values()) or stat(p)!=before:raise ValueError('Real source input nonfinite/mutated')
    receipt=dict(source_path=str(p),source_file_sha256_reference=entry['sha256'],source_file_SHA_scope='accepted old index; no HDF file rescan',source_stat_before=before,source_stat_after=stat(p),source_identity=fields,input_fields={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in rows.items()})
    return rows,receipt
def prepare(plan):
    kind=plan['modality'];m=read(plan['cache_manifest']);s=read(plan['cache_seal']);prefix=kind
    if kind!=classify(m) or sha(plan['cache_manifest'])!=plan['cache_manifest_sha256'] or sha(plan['cache_seal'])!=plan['cache_seal_sha256'] or s['manifest_sha256']!=plan['cache_manifest_sha256']:raise ValueError('Approved canonical cache identity differs')
    if m['status']!='passed' or m['exit_code']!=0 or s['status']!='passed' or s['exit_code']!=0:raise ValueError('Previously completed cache required')
    idx_path=m['source_'+prefix+'_index'];index=read(idx_path)
    if sha(idx_path)!=m['source_'+prefix+'_index_sha256']:raise ValueError('Original source index changed')
    for split,n in [('train',1000),('val',100)]:
        if len(index[split])!=len(m['splits'][split]) or len(index[split])!=n or [x['episode'] for x in index[split]]!=list(range(n)):raise ValueError('Official split/source episode count differs')
    for p,h in plan['source_pins'].items():
        if sha(p)!=h:raise ValueError('Frozen dependency changed '+p)
    if plan['source_checkpoint_sha256']!=m['source_checkpoint_sha256'] or plan['batch']!=64:raise ValueError('Original WM/B64 required')
    return m,index
def execute(plan,m,index,plan_sha):
    import torch
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    status=dict(stage='starting',started=now(),updated=now(),completed_frames=0,exit_code=None)
    save(out/'status.json',status)
    try:
        w=Path(plan['workspace']);cfg=read(plan['bridge_config']);repo=w/cfg['repo']/'lewm_pcd'
        register_official_namespace(cfg,plan['official_utils_init_sha256'])
        bridge=alias('terminal_original_frozen_bridge',plan['bridge_source'])
        torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=m['precision']['cudnn_allow_tf32']
        if not torch.cuda.is_available():raise ValueError('Explicit root-approved CUDA stage required')
        kind=plan['modality'];checkpoint=w/cfg['routes'][kind]['checkpoint']
        if str(checkpoint)!=m['source_checkpoint'] or sha(checkpoint)!=m['source_checkpoint_sha256']:raise ValueError('Actual original fullWM SHA')
        frozen=bridge.FrozenLatents(cfg,kind,reader=None,device='cuda')
        trainer=Path(plan['original_trainer_source']);tree=ast.parse(trainer.read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='module_hash');scope=dict(hashlib=hashlib,np=np,torch=torch)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(trainer),'exec'),scope);module_hash=scope['module_hash']
        if module_hash(frozen.encoder)!=m['encoder_hashes']['trainer_numpy_dtype']:raise ValueError('Same actual encoder+projector tensors required')
        snapshot=frozen.snapshot();splits={};all_rows=[];boundary=[];readers={}
        for split,count in [('train',1000),('val',100)]:
            # Eligibility metadata from the accepted1001-row index; each actual HDF terminal is checked by original RawRows.
            valids=np.ones(count*1001,dtype=np.float32);valids[terminal_raw_ids(count)]=0
            readers[split]=bridge.RawRows(cfg,kind,{'valids':valids},split)
        def infer(images):
            valid=len(images);pad=padding_for64(valid);inputs=torch.stack(images)
            if pad:inputs=torch.cat((inputs,inputs[:pad]),0)
            with torch.inference_mode():z=frozen.encoder(inputs.to('cuda')).logits.float().cpu().numpy()[:valid].copy()
            if z.shape!=(valid,192) or not np.isfinite(z).all():raise ValueError('Real terminal forward nonfinite/wrong shape')
            return z
        # Four source-specific frame999 references; no main-cache/HDF full re-encoding.
        for split in ['train','val']:
            ep=0;rows,_=source_input(index[split][ep],m['splits'][split][ep],ep,kind,999,readers[split]);x=bridge.preprocess_rows(frozen.preprocessor,rows,frozen.input_key)[0]
            actual=infer([x]*64)[0];expected=np.load(m['splits'][split][ep]['path'],mmap_mode='r',allow_pickle=False)[999].copy()
            if not np.allclose(actual,expected,atol=plan['boundary_atol'],rtol=plan['boundary_rtol']):raise ValueError('Same B64 source boundary/cache parity failed')
            boundary.append(dict(split=split,episode=ep,raw_id=999,max_abs=float(np.max(np.abs(actual-expected))),bitexact=bool(np.array_equal(actual,expected)),atol=plan['boundary_atol'],rtol=plan['boundary_rtol']))
        for split,count in [('train',1000),('val',100)]:
            values=[];images=[]
            for ep in range(count):
                rows,record=source_input(index[split][ep],m['splits'][split][ep],ep,kind,reader=readers[split])
                x=bridge.preprocess_rows(frozen.preprocessor,rows,frozen.input_key)[0]
                expected_shape=(3,224,224) if kind=='rgb' else (1200,6)
                if tuple(x.shape)!=expected_shape or x.dtype!=torch.float32 or not torch.isfinite(x).all():raise ValueError('Original source singleton preprocessor contract')
                images.append(x);record.update(split=split,episode=ep,raw_id=ep*1001+1000);all_rows.append(record)
                if len(images)==64 or ep==count-1:
                    values.append(infer(images));images=[];status.update(stage='encoding_'+split,updated=now(),completed_frames=len(all_rows));save(out/'status.json',status)
            ids=terminal_raw_ids(count);z=np.concatenate(values);validate_payload(ids,z,count);path=out/(split+'_terminal_next.npz')
            with path.open('xb') as f:np.savez(f,raw_ids=ids,latents=z)
            splits[split]=dict(path=str(path.resolve()),sha256=sha(path),shape=list(z.shape),dtype='float32',raw_ids_shape=list(ids.shape),raw_ids_dtype='int64',raw_ids_sha256=hashlib.sha256(ids.astype('<i8',copy=False).tobytes()).hexdigest())
            frozen.check_frozen(snapshot)
        if module_hash(frozen.encoder)!=m['encoder_hashes']['trainer_numpy_dtype']:raise ValueError('Encoder state changed')
        ledger=out/'terminal_source_rows.jsonl'
        with ledger.open('x') as f:
            for row in all_rows:f.write(json.dumps(row,allow_nan=False)+'\n')
        manifest=dict(status='passed',exit_code=0,modality=kind,source_modality=kind,cache_manifest_sha256=plan['cache_manifest_sha256'],cache_seal_sha256=plan['cache_seal_sha256'],source_checkpoint_sha256=m['source_checkpoint_sha256'],preprocess_source_sha256=m['preprocess_source_sha256'],encoder_hashes=m['encoder_hashes'],precision=m['precision'],raw_id_mapping='episode*1001+1000',splits=splits,producer_source_sha256=sha(__file__),plan_sha256=plan_sha,source_pins=plan['source_pins'],source_checkpoint=m['source_checkpoint'],source_row_ledger=str(ledger.resolve()),source_row_ledger_sha256=sha(ledger),actual_real_terminal_frames=1100,boundary_cache_reuse=boundary,terminal_producer_parameters=dict(batch=64,train_last_valid=40,train_padding=24,val_last_valid=36,val_padding=28,frozen_wrapper='original FrozenLatents.encoder = original LeWMFrozenEncoder',input_kind='pixels' if kind=='rgb' else 'pcd',no_new_action_head=True,no_BC_sampler=True,no_maincache_reencode=True,terminal_action_not_read=True),actual_outer_exit_scope='producer report written before process exit; require actual wrapper exit and independent CPU audit0')
        save(out/'terminal_manifest.json',manifest);status.update(stage='producer_complete_pending_actual_outer_and_CPU_audit',updated=now(),exit_code=0);save(out/'status.json',status)
    except BaseException:
        import traceback
        status.update(stage='failed',updated=now(),exit_code=1,error=traceback.format_exc());save(out/'status.json',status);raise
def main(argv=None):
    a=parse_args(argv);plan=read(a.plan);m,index=prepare(plan)
    if not a.execute:
        print(json.dumps(dict(status='CPU_prepared_no_HDF_model_encoder_GPU',modality=plan['modality'],terminal_frames=1100,split_shapes={'train':[1000,192],'val':[100,192]},precision=m['precision'],output=plan['output'],execute=False)));return 0
    execute(plan,m,index,sha(a.plan));return 0
if __name__=='__main__':raise SystemExit(main())
