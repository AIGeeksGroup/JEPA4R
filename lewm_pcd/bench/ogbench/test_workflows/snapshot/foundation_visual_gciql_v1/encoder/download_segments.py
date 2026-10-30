"""Resume completed prefixes with 16 bounded HTTP ranges in one owned file."""
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
state={'started_at':now(),'stage':'sixteen_segments_one_file','exit_code':None,'total_bytes':N,'URL':URL,'segments':{}}
def worker(item):
 i,start,end=item; receipt=R/f'segment_{i}.json';saved=json.loads(receipt.read_text())['saved_bytes'] if receipt.exists() else 0
 for attempt in range(1,7):
  try:
   if start+saved>end:break
   request_start=start+saved
   req=urllib.request.Request(URL,headers={'Range':f'bytes={request_start}-{end}','If-Match':ETAG})
   with urllib.request.urlopen(req,timeout=90) as resp:
    assert resp.status==206 and resp.headers.get('Content-Range')==f'bytes {request_start}-{end}/{N}' and resp.headers.get('ETag')==ETAG
    with (R/f'segment_{i}.responses.jsonl').open('a') as log:log.write(json.dumps({'at':now(),'attempt':attempt,'headers':dict(resp.headers)})+'\n')
    fd=os.open(str(P),os.O_WRONLY)
    try:
     while True:
      data=resp.read(1<<20)
      if not data:break
      assert start+saved+len(data)<=end+1
      written=os.pwrite(fd,data,start+saved);assert written==len(data);saved+=written
      record={'saved_bytes':saved,'range':[start,end],'updated_at':now(),'exit_code':None};atomic(receipt,record);state['segments'][str(i)]=record
    finally:os.close(fd)
   assert saved==end-start+1
   break
  except BaseException as e:
   with (R/f'segment_{i}.errors.jsonl').open('a') as log:log.write(json.dumps({'at':now(),'attempt':attempt,'saved':saved,'error':repr(e)})+'\n')
   if attempt==6:raise
   time.sleep(5)
 record={'saved_bytes':saved,'range':[start,end],'updated_at':now(),'exit_code':0};atomic(receipt,record);state['segments'][str(i)]=record
try:
 init=R/'segments_initialized.json'
 if not init.exists():
  segments=[];reused=0
  for i in range(4):
   previous=json.loads((R/f'range_{i}.json').read_text());begin,end=previous['range'];saved=previous['saved_bytes'];reused+=saved
   begin+=saved;remaining=end-begin+1
   for j in range(4):segments.append([i*4+j,begin+remaining*j//4,begin+remaining*(j+1)//4-1])
  atomic(init,{'segments':segments,'reused_bytes':reused,'at':now()})
 plan=json.loads(init.read_text());beginning=time.monotonic();initial=sum(json.loads((R/f'segment_{i}.json').read_text())['saved_bytes'] if (R/f'segment_{i}.json').exists() else 0 for i,a,b in plan['segments'])
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
  fs=[pool.submit(worker,item) for item in plan['segments']]
  while not all(f.done() for f in fs):
   segment_total=sum(x['saved_bytes'] for x in state['segments'].values());total=plan['reused_bytes']+segment_total;speed=max(0,segment_total-initial)/max(1,time.monotonic()-beginning)
   state.update(updated_at=now(),reused_bytes=plan['reused_bytes'],downloaded_bytes=total,MB_per_s=speed/1e6,ETA_seconds=(N-total)/speed if speed else None);atomic(R/'segments_status.json',state);time.sleep(5)
  [f.result() for f in fs]
 state.update(stage='segments_complete',exit_code=0,updated_at=now(),downloaded_bytes=N);atomic(R/'segments_status.json',state);(R/'segments.actual_exit_code').write_text('0\n')
except BaseException as e:
 state.update(stage='failed',error=repr(e),traceback=traceback.format_exc(),exit_code=1);atomic(R/'segments_status.json',state);(R/'segments.actual_exit_code').write_text('1\n');raise
