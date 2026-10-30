"""Shared bounded-memory SHA/tar/CRC/copy primitives for all eight closed roles."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2)
def stat(path):
    value=Path(path).stat()
    return dict(bytes=value.st_size,mtime_ns=value.st_mtime_ns,ctime_ns=value.st_ctime_ns,inode=value.st_ino,device=value.st_dev)

APPROVED_MAPPINGS=(
    ('foundation_visual_gciql_v1/runs','/autodl-fs/data/lewm/foundation_visual_gciql_31435_20261024_v1_new/runs'),
    ('foundation_visual_gciql_v1/native_sr','/autodl-fs/data/lewm/foundation_visual_gciql_31435_20261024_v1_new/native_sr'),
    ('foundation_visual_gciql_v1/cache/w8_groups/vjepa21_vitl1024_cube3',
     '/root/foundation_visual_gciql_31435_20261024_v1_outputs/cache_groups/w8_vjepa21_vitl1024_cube3'),
)

def strict_scope(path,root,mappings=None):
    """Keep a logical ROOT-relative name and verify its exact approved target."""
    root=Path(os.path.abspath(root));logical=Path(os.path.abspath(path))
    assert logical.is_relative_to(root),str(path)
    resolved=logical.resolve()
    for relative,physical in APPROVED_MAPPINGS if mappings is None else mappings:
        prefix=root/relative;physical=Path(physical)
        if logical.is_relative_to(prefix):
            assert prefix.resolve()==physical,str(prefix)
            assert resolved==physical/logical.relative_to(prefix),str(path)
            return logical
    assert resolved.is_relative_to(root.resolve()),str(path)
    return logical

def verify_archive(archive,expected_sha=None,expected_bytes=None):
    archive=Path(archive)
    if expected_sha is not None:assert sha(archive)==expected_sha
    if expected_bytes is not None:assert archive.stat().st_size==expected_bytes
    with gzip.open(archive,'rb') as stream:
        while stream.read(1<<20):pass
    with tarfile.open(archive,'r:gz') as bundle:
        inventory=read_stream(bundle.extractfile('member_inventory.json'))
        expected=inventory['members'];members=bundle.getmembers()
        assert len(members)==len(expected)+1 and len({m.name for m in members})==len(members)
        for member in members:
            assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
            with bundle.extractfile(member) as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if member.name!='member_inventory.json':
                assert digest==expected[member.name]['sha256'] and member.size==expected[member.name]['stat']['bytes']
    return dict(member_count=len(members),full_gzip_CRC=True,every_member_SHA_verified=True)

def read_stream(stream):
    with stream:return json.load(stream)

def pack(root,out,paths,metadata,expected_hashes=None):
    root,out=Path(os.path.abspath(root)),Path(out);before={}
    expected_hashes={str(Path(path).resolve()):digest for path,digest in (expected_hashes or {}).items()}
    for path in sorted(set(map(Path,paths))):
        path=strict_scope(path,root)
        assert path.is_file()
        snapshot=stat(path);digest=sha(path);assert stat(path)==snapshot
        if expected_hashes and str(path.resolve()) in expected_hashes:assert digest==expected_hashes[str(path.resolve())],str(path)
        before[path.relative_to(root).as_posix()]=dict(stat=snapshot,sha256=digest)
    save(out/'member_inventory.json',dict(members=before,metadata=metadata,created=time.time()))
    archive=out/'closed_final.tar.gz'
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz',compresslevel=1) as bundle:
            for name in before:bundle.add(root/name,arcname=name,recursive=False)
            bundle.add(out/'member_inventory.json',arcname='member_inventory.json',recursive=False)
    for name,record in before.items():assert stat(root/name)==record['stat'] and sha(root/name)==record['sha256']
    checks=verify_archive(archive)
    receipt=dict(actual_exit_code=0,archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),
        source_before_after_stat_SHA_exact=True,NN_forward=False,completed=time.time(),**checks)
    save(out/'receipt.json',receipt)
    return receipt

def verified_copy(scratch,destination):
    scratch,destination=Path(scratch),Path(destination);receipt=read(scratch/'receipt.json')
    assert receipt['actual_exit_code']==0
    for name in ('inner_exit_code','outer_exit_code'):assert (scratch/name).read_text().strip()=='0'
    checks=verify_archive(scratch/'closed_final.tar.gz',receipt['sha256'],receipt['bytes'])
    destination.mkdir(parents=True,exist_ok=False)
    for name in ('closed_final.tar.gz','receipt.json','inner_exit_code','outer_exit_code','builder.stdout.log','builder.stderr.log'):
        source=scratch/name;target=destination/name
        with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
        assert sha(source)==sha(target) and source.stat().st_size==target.stat().st_size
    checks=verify_archive(destination/'closed_final.tar.gz',receipt['sha256'],receipt['bytes'])
    result=dict(actual_exit_code=0,archive=str((destination/'closed_final.tar.gz').resolve()),
        bytes=receipt['bytes'],sha256=receipt['sha256'],remote_inner_exit_code=0,remote_outer_exit_code=0,
        scratch_and_project_SHA_size_exact=True,completed=time.time(),**checks)
    save(destination/'local_copy_receipt.json',result)
    return result
