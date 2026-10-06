"""Record resource use for this collection only; never signals workers."""
import argparse,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();started=time.time()
with (a.root/'resource_samples.jsonl').open('x',buffering=1) as out:
 while True:
  row={'time':time.time(),'memory_current':int(Path('/sys/fs/cgroup/memory.current').read_text()),'memory_max':int(Path('/sys/fs/cgroup/memory.max').read_text()),'free_disk':os.statvfs(a.root).f_bavail*os.statvfs(a.root).f_frsize,'loadavg':os.getloadavg()}
  try:row['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
  except Exception as e:row['gpu_error']=str(e)
  q=a.root/'scheduler_state.json'
  if q.exists():
   state=json.loads(q.read_text());row.update(phase=state['phase'],committed=state['committed_seed_count'],active=len(state['active']),failed=len(state['failed']))
  out.write(json.dumps(row)+'\n')
  if (a.root/'launcher.exit_code').exists():break
  if time.time()-started>86400:raise RuntimeError('resource monitor safety limit reached')
  time.sleep(5)
