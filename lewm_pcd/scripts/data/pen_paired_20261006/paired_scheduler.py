"""Bounded DAG scheduler; one RGB->PCD->paired-commit transaction per active job."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from paired_parent import sha,save

def main(args):
 config=json.loads(args.config.read_text());root=Path(config['root']).resolve();root.mkdir(parents=True,exist_ok=True)
 import fcntl
 scheduler_lock=(root/'scheduler.lock').open('a+')
 try:fcntl.flock(scheduler_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('another scheduler owns this run root')
 seal=sha(args.config);statepath=root/'scheduler_state.json'
 if statepath.exists():assert json.loads(statepath.read_text())['config_sha256']==seal,'resume configuration changed'
 started_at=json.loads(statepath.read_text()).get('started_at',time.time()) if statepath.exists() else time.time()
 assert all(len(j['commands'])==3 for j in config['jobs']),'only RGB/PCD/writer stages are authorized'
 assert config['concurrency'] in (1,2,4,8,16),'concurrency requires resource calibration'
 assert all(j['id'] and all(c.isalnum() or c in '_-' for c in j['id']) for j in config['jobs']),'unsafe job ID'
 jobs={j['id']:j for j in config['jobs']};assert len(jobs)==len(config['jobs'])
 planned=[ep for j in jobs.values() for ep in j['episodes']]
 assert len({e['seed'] for e in planned})==len(planned) and len({e['episode_id'] for e in planned})==len(planned),'duplicate planned seed/id'
 for j in jobs.values():
  lo,hi=config['seed_ranges_half_open'][j['split']]
  assert all(lo<=e['seed']<hi and 0<=e['seed']<2**32 for e in j['episodes'])
 completed={};failed={};active={};registry={'seed':set(),'native_initial_sha256':set(),'full_initial_sha256':set(),'id':set(),'episode_id':set()}
 def register(result):
  temporary={key:set(values) for key,values in registry.items()}
  for ep in result['episodes']:
   for key,values in temporary.items():
    assert ep[key] not in values,'duplicate committed '+key
    values.add(ep[key])
  for ep in result['episodes']:
   for key,values in registry.items():values.add(ep[key])
 def verify_source():
  for row in config['source_manifest']:assert sha(row['path'])==row['sha256'],'sealed source changed'
 def verify_commit(job,attempt):
  for stage in range(len(job['commands'])):assert (attempt/('stage_%d.exit_code'%stage)).read_text().strip()=='0','stage exit code missing/failed'
  paired_state=json.loads((attempt/'paired/state.json').read_text());assert paired_state['phase']=='committed' and paired_state['exit_code']==0
  assert (attempt/'paired/exit_code').read_text().strip()=='0'
  assert json.loads((attempt/'job_config.json').read_text())==job,'attempt job configuration differs'
  commit=attempt/'paired/commit.json';r=json.loads(commit.read_text());assert r['manifest_sha256']==sha(attempt/'paired_manifest.json'),'attempt input manifest changed'
  assert r['qc']=='PASS' and r['exit_code']==0
  assert [(v['episode_id'],v['seed']) for v in r['episodes']]==[(v['episode_id'],v['seed']) for v in job['episodes']]
  assert sha(r['policy_rng_end']['path'])==r['policy_rng_end']['sha256'],'policy RNG source changed'
  for row in r['files']:assert sha(commit.parent/row['name'])==row['sha256']
  return r
 for name,job in jobs.items():
  directory=root/name;directory.mkdir(exist_ok=True)
  status=directory/'state.json'
  if status.exists():
   old=json.loads(status.read_text())
   if old['phase']=='running' and old.get('pid'):
    try:os.kill(int(old['pid']),0)
    except ProcessLookupError:pass
    else:raise RuntimeError('prior worker PID is still alive; do not launch duplicate job '+name)
   if old['phase']=='committed' or (old.get('attempt') and (Path(old['attempt'])/'paired/commit.json').exists()):
    register(verify_commit(job,Path(old['attempt'])));old['phase']='committed';completed[name]=old;save(status,old)
 def persist(phase='running',exit_code=None):
  save(statepath,dict(phase=phase,started_at=started_at,config_sha256=seal,updated_at=time.time(),completed=list(completed),committed_seed_count=sum(len(jobs[k]['episodes']) for k in completed),expected_seed_count=config.get('expected_committed_episodes',sum(len(j['episodes']) for j in jobs.values())),seed_ranges=config.get('seed_ranges_half_open',{}),failed=failed,active={k:dict(pid=v['process'].pid,stage=v['stage'],attempt=str(v['attempt'])) for k,v in active.items()},exit_code=exit_code))
 def launch(name,job,attempt,stage):
  values={'attempt':str(attempt),'episodes':str(attempt/'episodes.json'),'manifest':str(attempt/'paired_manifest.json')}
  argv=[str(v).format_map(values) for v in job['commands'][stage]]
  log=(attempt/('stage_%d.log'%stage)).open('ab')
  try:
   env=os.environ.copy();env.update(config.get('environment',{}))
   p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  finally:log.close()
  active[name]=dict(process=p,stage=stage,attempt=attempt,job=job)
  save(root/name/'state.json',dict(phase='running',stage=stage,pid=p.pid,started_at=json.loads((attempt/'start.json').read_text())['started_at'],attempt=str(attempt),config_sha256=seal,updated_at=time.time()))
 def start(name,job):
  directory=root/name;index=1
  while (directory/('attempt_%03d'%index)).exists():index+=1
  attempt=directory/('attempt_%03d'%index);attempt.mkdir();save(attempt/'start.json',dict(started_at=time.time()))
  episodes={'episodes':job['episodes'],'policy_seed':job['policy_seed']}
  if job.get('policy_parent'):
   parent=Path(completed[job['policy_parent']]['attempt'])/'rgb/policy_rng_end.json';episodes['policy_rng_state']=json.loads(parent.read_text())
   save(attempt/'policy_parent.json',dict(path=str(parent),sha256=sha(parent)))
  save(attempt/'episodes.json',episodes);save(attempt/'job_config.json',job);launch(name,job,attempt,0)
 try:
  verify_source();persist()
  while len(completed)+len(failed)<len(jobs):
   for name,item in list(active.items()):
    result=item['process'].poll()
    if result is None:continue
    (item['attempt']/('stage_%d.exit_code'%item['stage'])).write_text(str(result)+'\n')
    del active[name]
    if result:
     failed[name]=dict(stage=item['stage'],exit_code=result,attempt=str(item['attempt']));save(root/name/'state.json',dict(phase='failed',**failed[name]));continue
    try:
     verify_source();stage=item['stage']+1
     if stage==2:
      episodes=[dict(episode_id=e['episode_id'],rgb=str(item['attempt']/'rgb'/('ep_%d_rgb.npz'%e['episode_id'])),pcd=str(item['attempt']/'pcd'/('ep_%d_pcd.npz'%e['episode_id'])),metadata=str(item['attempt']/'pcd'/('ep_%d_pcd.json'%e['episode_id']))) for e in item['job']['episodes']]
      save(item['attempt']/'paired_manifest.json',dict(episodes=episodes,image_size=config.get('image_size',256),config_sha256=config['worker_config_sha256'],expected_source_hashes=config['expected_source_hashes'],policy_rng_end=str(item['attempt']/'rgb/policy_rng_end.json')))
     if stage<len(item['job']['commands']):launch(name,item['job'],item['attempt'],stage)
     else:
      register(verify_commit(item['job'],item['attempt']));record=dict(phase='committed',exit_code=0,completed=len(item['job']['episodes']),config_sha256=seal,attempt=str(item['attempt']),updated_at=time.time());completed[name]=record;save(root/name/'state.json',record)
    except BaseException as error:failed[name]=dict(error=repr(error),attempt=str(item['attempt']));save(root/name/'state.json',dict(phase='failed',**failed[name]))
   for name,job in jobs.items():
    if len(active)>=config['concurrency']:break
    if name in completed or name in failed or name in active:continue
    deps=job.get('depends_on',[])
    if any(d in failed for d in deps):failed[name]={'error':'dependency failed; no rollout started'};continue
    if all(d in completed for d in deps):start(name,job)
   persist()
   if not active and not all(name in completed or name in failed for name in jobs):raise RuntimeError('dependency cycle or missing dependency')
   time.sleep(2)
  seen_seeds=set();seen_initial=set();seen_native=set();counts={0:0,1:0,2:0}
  for name,record in completed.items():
   for ep in verify_commit(jobs[name],Path(record['attempt']))['episodes']:
    assert ep['seed'] not in seen_seeds and ep['full_initial_sha256'] not in seen_initial and ep['native_initial_sha256'] not in seen_native,'global duplicate seed/initial state'
    seen_seeds.add(ep['seed']);seen_initial.add(ep['full_initial_sha256']);seen_native.add(ep['native_initial_sha256']);counts[ep['sample_type']]+=1
  code=1 if failed else 0;persist('failed' if failed else 'complete',code);save(root/'final_qc.json',dict(episodes=len(seen_seeds),sample_type_counts=counts,failed=failed,exit_code=code));(root/'exit_code').write_text(str(code)+'\n');return code
 except BaseException as error:
  persist('scheduler_failed',1);(root/'exit_code').write_text('1\n');save(root/'scheduler_error.json',dict(error=repr(error),active_pids=[v['process'].pid for v in active.values()]));raise
 finally:
  scheduler_lock.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);raise SystemExit(main(p.parse_args()))
