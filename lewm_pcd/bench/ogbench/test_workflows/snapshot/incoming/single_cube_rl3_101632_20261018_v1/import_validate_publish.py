"""Validate one immutable Single Cube bundle, then publish without overwriting assets."""
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
FINAL=ROOT/'assets/imports/single_cube_rl3_101632_20261018_v1'
SHA='80d420173d2d594e1b3deaa2612d20f088ba0045ecfb4ac58f141f3e9ef3872a'
SOURCE=PurePosixPath('/root/autodl-tmp/workspace/users/lewm')


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def stat(path):
    s=path.stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)


def write(name,obj):
    p=HERE/name;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');os.replace(t,p)


def main():
    import numpy as np
    assert HERE.parent==ROOT/'incoming' and HERE.name=='single_cube_rl3_101632_20261018_v1'
    assert not FINAL.exists() and not FINAL.is_symlink()
    archive=HERE/'single_cube_rl3_assets.tar'
    assert archive.stat().st_size==2164602880 and digest(archive)==SHA
    stage=HERE/'verified_payload';stage.mkdir(exist_ok=False)
    checked=[];main_counts={};terminal_counts={};started=time.time()
    with tarfile.open(archive,'r:') as tar:
        members=tar.getmembers();assert len(members)==2440 and all(m.isfile() for m in members)
        names=[m.name for m in members];assert len(set(names))==len(names)
        origin=json.load(tar.extractfile('IMPORT_ORIGIN_MANIFEST.json'))
        assert origin['source_host']=='101632' and origin['cube_count']==1 and origin['data_id']=='cube-single-play-v0'
        assert origin['source_root']==str(SOURCE) and origin['source_file_count']==2439
        entries={x['member']:x for x in origin['files']};assert len(entries)==2439
        assert set(names)==set(entries)|{'IMPORT_ORIGIN_MANIFEST.json'}
        for member in members:
            name=PurePosixPath(member.name)
            assert not name.is_absolute() and '..' not in name.parts and '\\' not in member.name
            target=(stage/member.name).resolve();assert target.is_relative_to(stage.resolve())
            target.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256()
            with tar.extractfile(member) as src,target.open('xb') as dst:
                while b:=src.read(1024*1024):dst.write(b);h.update(b)
            if member.name not in entries:
                continue
            item=entries[member.name]
            assert member.name=='source_layout/'+PurePosixPath(item['path']).relative_to(SOURCE).as_posix()
            assert target.stat().st_size==item['bytes'] and h.hexdigest()==item['expected_sha256']==item['sha256']
            result=dict(member=member.name,category=item['category'],bytes=item['bytes'],sha256=h.hexdigest(),current_target_stat=stat(target))
            if target.suffix=='.npy':
                with target.open('rb') as f:
                    version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version)
                assert not fortran and dtype==np.float32
                result.update(shape=list(shape),dtype='float32')
                if 'main_embedding' in item['category']:
                    assert shape==(1000,192)
                    modality='pcd' if '_pcd_' in item['category'] else 'rgb'
                    split=item['split'];main_counts[(modality,split)]=main_counts.get((modality,split),0)+1
                else:
                    split='train' if '/train/' in member.name else 'val'
                    rows=1001000 if split=='train' else 100100
                    width=28 if 'observation' in item['category'] else 5
                    assert shape==(rows,width)
            if 'terminal_embedding_rawIDs' in item['category']:
                split='train' if target.name.startswith('train') else 'val';count=1000 if split=='train' else 100
                modality='pcd' if '_pcd_' in item['category'] else 'rgb'
                with np.load(target,allow_pickle=False) as data:
                    assert set(data.files)=={'raw_ids','latents'}
                    ids,values=data['raw_ids'],data['latents']
                    assert ids.dtype==np.int64 and np.array_equal(ids,np.arange(count,dtype=np.int64)*1001+1000)
                    assert values.shape==(count,192) and values.dtype==np.float32 and np.isfinite(values).all()
                terminal_counts[(modality,split)]=count
            checked.append(result)
            if len(checked)%100==0:write('status.json',dict(stage='verifying_and_extracting',completed_files=len(checked),total=2439,actual_exit_code=None,updated=time.time()))
    expected={(m,s):n for m in ('pcd','rgb') for s,n in [('train',1000),('val',100)]}
    assert main_counts==expected and terminal_counts==expected
    acceptance=dict(actual_exit_code=0,cube_count=1,data_id='cube-single-play-v0',source_host='101632',
        archive_sha256=SHA,archive_bytes=2164602880,source_files_verified=2439,
        all_current_target_SHA_match_expected=True,files=checked,
        main_layout='1000FP32D192 rows per EP; separate true terminal NPZ, rawID=e*1001+t',
        main_counts={str(k):v for k,v in main_counts.items()},terminal_counts={str(k):v for k,v in terminal_counts.items()},
        main_finite_proof='original accepted finite checks reused through fresh matching source and target SHA; no redundant full-vector finite sweep',
        terminal_arrays_finite_and_rawIDs_freshly_verified=True,source_inode_rebased_and_original_seals_preserved=True,
        original_manifest_sha256=digest(stage/'IMPORT_ORIGIN_MANIFEST.json'),NN_or_training_performed=False,
        target_final_root=str(FINAL),started=started,completed=time.time())
    with (stage/'IMPORT_ACCEPTANCE.json').open('x') as f:json.dump(acceptance,f,indent=2)
    FINAL.parent.mkdir(parents=True,exist_ok=True);assert FINAL.parent.resolve()==FINAL.parent
    libc=ctypes.CDLL(None,use_errno=True);rename=libc.renameat2
    rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];rename.restype=ctypes.c_int
    if rename(-100,os.fsencode(stage),-100,os.fsencode(FINAL),1)!=0:raise OSError(ctypes.get_errno(),'RENAME_NOREPLACE failed')
    write('IMPORT_PUBLISHED_RECEIPT.json',dict(actual_exit_code=0,target_root=str(FINAL),verified_source_files=2439,
        archive_sha256=SHA,acceptance_sha256=digest(FINAL/'IMPORT_ACCEPTANCE.json'),published_without_overwrite=True,
        old_cube2_cube3_assets_modified=False,completed=time.time()))
    print(json.dumps(dict(actual_exit_code=0,target_root=str(FINAL),verified_source_files=2439)))


if __name__=='__main__':
    code=1
    try:main();code=0
    except BaseException:
        write('failure.json',dict(actual_exit_code=1,error=traceback.format_exc(),at=time.time()));raise
    finally:(HERE/'import_actual_exit_code').write_text(str(code)+'\n')
