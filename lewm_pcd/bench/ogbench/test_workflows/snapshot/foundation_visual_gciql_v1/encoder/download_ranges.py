"""Four resumable ranges in one file; no duplicate full checkpoint."""
import concurrent.futures,json,os,pathlib,time,urllib.request,datetime,traceback
ROOT=pathlib.Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
R=ROOT/'foundation_visual_gciql_v1/encoder'
P=ROOT/'assets/baselines/foundation/vjepa21_vitl1024_official_v1/vjepa2_1_vitl_dist_vitG_384.pt.part'
N=5151198524
URL='https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt'
ETAG='"75aac8d267d11f285683714d390dc0ed-615"'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
state={'started_at':now(),'stage':'four_ranges_one_file','exit_code':None,'total_bytes':N,'URL':URL,'ranges':{}}
def worker(i):
 start=N*i//4;end=N*(i+1)//4-1
 receipt=R/f'range_{i}.json'
 saved=json.loads(receipt.read_text())['saved_bytes'] if receipt.exists() else (min(prefix,end+1) if i==0 else 0)
 for attempt in range(1,7):
  try:
   if start+saved>end:break
   req=urllib.request.Request(URL,headers={'Range':f'bytes={start+saved}-{end}','If-Match':ETAG})
   with urllib.request.urlopen(req,timeout=90) as resp:
    assert resp.status==206 and resp.headers.get('Content-Range')==f'bytes {start+saved}-{end}/{N}' and resp.headers.get('ETag')==ETAG
    with (R/f'range_{i}.responses.jsonl').open('a') as log:log.write(json.dumps({'at':now(),'attempt':attempt,'headers':dict(resp.headers)})+'\n')
    fd=os.open(str(P),os.O_WRONLY)
    try:
     while True:
      data=resp.read(1<<20)
      if not data:break
      assert start+saved+len(data)<=end+1
      written=os.pwrite(fd,data,start+saved);assert written==len(data);saved+=written
      record={'saved_bytes':saved,'range':[start,end],'updated_at':now(),'exit_code':None}
      atomic(receipt,record);state['ranges'][str(i)]=record
    finally:os.close(fd)
   assert saved==end-start+1
   break
  except BaseException as e:
   with (R/f'range_{i}.errors.jsonl').open('a') as log:log.write(json.dumps({'at':now(),'attempt':attempt,'saved':saved,'error':repr(e)})+'\n')
   if attempt==6:raise
   time.sleep(5)
 record={'saved_bytes':saved,'range':[start,end],'updated_at':now(),'exit_code':0};atomic(receipt,record);state['ranges'][str(i)]=record
try:
 prefix=P.stat().st_size if P.exists() and not (R/'ranges_initialized.json').exists() else 0
 if not (R/'ranges_initialized.json').exists():
  with P.open('ab') as f:f.truncate(N)
  atomic(R/'ranges_initialized.json',{'prefix_reused':prefix,'bytes':N,'at':now()})
 beginning=time.monotonic();initial=prefix
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  fs=[pool.submit(worker,i) for i in range(4)]
  while not all(f.done() for f in fs):
   total=sum(x['saved_bytes'] for x in state['ranges'].values());speed=max(0,total-initial)/max(1,time.monotonic()-beginning)
   state.update(updated_at=now(),downloaded_bytes=total,MB_per_s=speed/1e6,ETA_seconds=(N-total)/speed if speed else None);atomic(R/'ranges_status.json',state);time.sleep(5)
  [f.result() for f in fs]
 state.update(stage='ranges_complete',exit_code=0,updated_at=now());atomic(R/'ranges_status.json',state);(R/'ranges.actual_exit_code').write_text('0\n')
except BaseException as e:
 state.update(stage='failed',error=repr(e),traceback=traceback.format_exc(),exit_code=1);atomic(R/'ranges_status.json',state);(R/'ranges.actual_exit_code').write_text('1\n');raise
