"""Official eight-file state Play download: reuse, bounded Range retry, streaming verification."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import struct
import sys
import threading
import time
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'data/official_state'
MIRROR_PLAN=json.loads((ROOT/'mirror_plan_v1.json').read_text())
BASE=MIRROR_PLAN['base_url']
OFFICIAL_BASE='https://rail.eecs.berkeley.edu/datasets/ogbench/'
FILES=[('single',1,1000,256728298,25687020),('double',2,1000,297435656,29726524),
       ('triple',3,3000,1003544255,100613267),('quadruple',4,5000,1861419607,185890898)]
CHUNK=1024*1024
BUFFER=65536
STREAMS=8
LOCK=threading.Lock()
STOP=threading.Event()
RESULTS=[]
RUN_ID=str(time.time_ns())
OWNS_LOCK=False


def write(name,record):
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(record,indent=2)+'\n');os.replace(tmp,path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def memory():
    with open('/proc/self/status') as f:
        rss=int(next(l.split()[1] for l in f if l.startswith('VmRSS:')))*1024
    return dict(own_RSS_bytes=rss,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                CPU_affinity=list(os.sched_getaffinity(0)),GPU_calls=0)


def status(stage,**extra):
    write('status.json',dict(stage=stage,updated=time.time(),completed_files=len(RESULTS),
        expected_files=8,accepted_bytes=sum(x['bytes'] for x in RESULTS),
        expected_bytes=3761045525,results=RESULTS,memory=memory(),
        actual_exit_code=None,training_started=False,**extra))


def npy_header(stream):
    assert stream.read(6)==b'\x93NUMPY'
    major,minor=stream.read(2)
    n=struct.unpack('<H' if major==1 else '<I',stream.read(2 if major==1 else 4))[0]
    assert n<1024*1024
    return ast.literal_eval(stream.read(n).decode('utf8' if major==3 else 'latin1'))


def validate(path,name,cubes,episodes,size,reused):
    before=path.stat();assert before.st_size==size,(name,before.st_size,size)
    prior=ROOT/'receipts'/ (name+'.json')
    if reused and prior.exists():
        old=json.loads(prior.read_text())
        if (old.get('actual_exit_code')==0 and old.get('archive_CRC_all_members') and
                old.get('bytes')==size and old.get('mtime_ns')==before.st_mtime_ns and
                old.get('inode')==before.st_ino and old.get('episode_count')==episodes and old.get('sha256')==MIRROR_PLAN['files'][name]['sha256']):
            return dict(old,reused_existing=True,prior_verified_receipt_reused=True)
    status('validating_existing' if reused else 'validating_download',file=name)
    headers={}
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None,'ZIP CRC failure'
        for member in archive.namelist():
            with archive.open(member) as stream:headers[member]=npy_header(stream)
        rows=episodes*1001
        assert tuple(headers['observations.npy']['shape'])==(rows,19+9*cubes)
        assert tuple(headers['actions.npy']['shape'])==(rows,5)
        assert tuple(headers['terminals.npy']['shape'])==(rows,)
        assert headers['terminals.npy']['descr'] in ['|b1','|?']
        with archive.open('terminals.npy') as stream:
            npy_header(stream)
            expected=b'\0'*1000+b'\1'
            for _ in range(episodes):assert stream.read(1001)==expected,'Unexpected episode boundary'
            assert not stream.read(1)
    digest=sha(path);after=path.stat()
    assert digest==MIRROR_PLAN['files'][name]['sha256'],('Pinned mirror SHA mismatch',name,digest)
    assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
    record=dict(file=name,path=str(path),url=BASE+name,bytes=size,sha256=digest,
        official_url=OFFICIAL_BASE+name,mirror_revision=MIRROR_PLAN['revision'],
        pinned_SHA_exact=True,mixed_official_and_mirror_chunks_allowed=True,
        archive_CRC_all_members=True,episode_count=episodes,rows=rows,rows_per_episode=1001,
        valid_transitions_per_episode=1000,headers=headers,reused_existing=reused,
        source_before_after_stat_exact=True,mtime_ns=after.st_mtime_ns,inode=after.st_ino,
        actual_exit_code=0,completed=time.time(),arrays_not_loaded_to_RAM=True)
    write('receipts/'+name+'.json',record)
    print(json.dumps(dict(event='accepted',file=name,bytes=size,reused_existing=reused,
                          episodes=episodes,sha256=digest)),flush=True)
    return record


def download(name,size):
    url=BASE+name;target=DATA/name
    ledger=ROOT/'chunks'/name;ledger.mkdir(parents=True,exist_ok=True)
    headers=None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,method='HEAD',headers={'User-Agent':'OGBench-dataset-verification'}),timeout=20) as response:
                headers=dict(response.headers);resolved_url=response.url
            assert int(headers.get('Content-Length',headers.get('content-length')))==size
            break
        except Exception as exc:
            with (ROOT/'retry.log').open('a') as log:log.write(json.dumps(dict(file=name,phase='HEAD',attempt=attempt,error=repr(exc),at=time.time()))+'\n')
            if attempt==5:raise
            time.sleep(min(2**attempt,20))
    etag=headers.get('ETag',headers.get('etag'));assert etag
    write('headers_mirror_v3/'+name+'.json',dict(url=url,size=size,etag=etag,headers=headers,revision=MIRROR_PLAN['revision']))
    origin=ROOT/'headers'/(name+'.json')
    official=json.loads(origin.read_text()) if origin.exists() else {}
    official_etag=official.get('etag') if official.get('url')==OFFICIAL_BASE+name and official.get('size')==size else None
    accepted={}
    for start in range(0,size,CHUNK):
        p=ledger/(str(start)+'.part');r=ledger/(str(start)+'.json')
        if p.exists() and r.exists():
            rec=json.loads(r.read_text())
            assert rec['etag'] in [etag,official_etag] and rec['bytes']==min(CHUNK,size-start)
            assert p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256']
            accepted[start]=rec
    write('mirror_chunk_adoption_v3/'+name+'.json',dict(at=time.time(),official_etag=official_etag,
        expected_file_sha256=MIRROR_PLAN['files'][name]['sha256'],
        adopted_chunks=[dict(start=k,**v) if 'start' not in v else v for k,v in sorted(accepted.items())],
        adopted_bytes=sum(x['bytes'] for x in accepted.values()),old_chunk_receipts_unchanged=True))
    def one(start):
        if start in accepted:return accepted[start]
        end=min(size,start+CHUNK)-1
        final=ledger/(str(start)+'.part')
        for attempt in range(12):
            if STOP.is_set():raise RuntimeError('Owned download cancelled after a failure')
            tmp=ledger/(str(start)+'.working_'+str(os.getpid())+'_'+str(attempt))
            try:
                request=urllib.request.Request(resolved_url,headers={'User-Agent':'OGBench-dataset-verification','Range':f'bytes={start}-{end}','If-Match':etag})
                begin=time.monotonic();count=0;h=hashlib.sha256()
                with urllib.request.urlopen(request,timeout=20) as response,tmp.open('xb') as out:
                    assert response.status==206
                    assert response.headers.get('Content-Range')==f'bytes {start}-{end}/{size}'
                    while count<end-start+1:
                        if STOP.is_set() or time.monotonic()-begin>150:raise TimeoutError('Bounded range request deadline')
                        block=response.read1(min(BUFFER,end-start+1-count))
                        if not block:break
                        out.write(block);h.update(block);count+=len(block)
                    assert count==end-start+1,('short_response',count,end-start+1)
                    out.flush();os.fsync(out.fileno())
                os.replace(tmp,final)
                rec=dict(start=start,end=end,bytes=count,sha256=h.hexdigest(),etag=etag,url=url,mirror_revision=MIRROR_PLAN['revision'])
                with LOCK:write('chunks/'+name+'/'+str(start)+'.json',rec)
                return rec
            except Exception as exc:
                # Only this attempt's uncommitted, incomplete scratch is removed.
                if tmp.exists():tmp.unlink()
                with LOCK:
                    with (ROOT/'retry.log').open('a') as log:log.write(json.dumps(dict(file=name,start=start,attempt=attempt,error=repr(exc),at=time.time()))+'\n')
                if attempt==11:raise
                time.sleep(min(2**attempt,30))
    status('downloading',file=name,file_completed_bytes=sum(x['bytes'] for x in accepted.values()),file_expected_bytes=size,streams=STREAMS)
    with ThreadPoolExecutor(STREAMS) as pool:
        jobs=[pool.submit(one,x) for x in range(0,size,CHUNK)]
        try:
            for future in as_completed(jobs):
                record=future.result();accepted[record['start']]=record
                status('downloading',file=name,file_completed_bytes=sum(x['bytes'] for x in accepted.values()),file_expected_bytes=size,streams=STREAMS)
        except BaseException:
            STOP.set()
            for future in jobs:future.cancel()
            raise
    assembled=DATA/(name+'.assembling_'+str(os.getpid()))
    with assembled.open('xb') as out:
        for start in sorted(accepted):
            with (ledger/(str(start)+'.part')).open('rb') as src:shutil.copyfileobj(src,out,BUFFER)
        out.flush();os.fsync(out.fileno())
    assert assembled.stat().st_size==size
    return assembled


def main():
    global OWNS_LOCK
    os.nice(10);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
    os.environ['CUDA_VISIBLE_DEVICES']=''
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OWNS_LOCK=True
    assert shutil.disk_usage(DATA).free>12*1024**3
    write('actual_argv.json',dict(argv=sys.argv,pid=os.getpid(),started=time.time(),
        CPU_cores=1,address_space_limit_bytes=512*1024**2,read_buffer_per_stream=BUFFER,
        streams=STREAMS,source='pinned HF mirror; reuse verified official chunks; full expected SHA before publication',GPU_calls=0))
    write('runs/'+RUN_ID+'/argv.json',dict(argv=sys.argv,pid=os.getpid(),started=time.time()))
    entries=[]
    for kind,cubes,episodes,train_size,val_size in FILES:
        for suffix,ep,size in [('',episodes,train_size),('-val',episodes//10,val_size)]:
            entries.append((f'cube-{kind}-play-v0{suffix}.npz',cubes,ep,size))
    # Validate/reuse existing complete files before any network transfer.
    for name,cubes,episodes,size in entries:
        if (DATA/name).exists():RESULTS.append(validate(DATA/name,name,cubes,episodes,size,True))
    for name,cubes,episodes,size in entries:
        if any(x['file']==name for x in RESULTS):continue
        path=download(name,size)
        record=validate(path,name,cubes,episodes,size,False)
        # Publish only after complete CRC/schema/SHA validation; refuse overwrite.
        target=DATA/name
        os.link(path,target)
        path.unlink()
        record['path']=str(target)
        record['validated_before_atomic_publish']=True
        write('receipts/'+name+'.json',record)
        RESULTS.append(record)
    write('status.json',dict(stage='all8_complete_and_verified',updated=time.time(),actual_exit_code=0,
        completed_files=8,accepted_bytes=sum(x['bytes'] for x in RESULTS),results=RESULTS,
        memory=memory(),GPU_calls=0,training_started=False))


if __name__=='__main__':
    code=1
    try:main();code=0
    except BaseException as exc:
        if OWNS_LOCK:
            failure=dict(stage='failed',actual_exit_code=1,error=repr(exc),at=time.time(),results=RESULTS)
            write('failure.json',failure);write('status.json',failure)
        raise
    finally:
        if OWNS_LOCK:
            receipt=dict(actual_exit_code=code,finished=time.time(),run_id=RUN_ID)
            write('outer_exit.json',receipt);write('runs/'+RUN_ID+'/outer_exit.json',receipt)
