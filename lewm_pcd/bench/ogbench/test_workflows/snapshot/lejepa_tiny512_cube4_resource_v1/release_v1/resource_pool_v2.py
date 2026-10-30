"""Own-pool resource tiers over original rendering; no encoder/learning math."""
import hashlib,json,os,time,uuid
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

POLICY_SHA='6d0ede59a7977f026d3cdae583e8c4014d7cf5feb45ad479ab39a965f50511b6'
TIERS=(16,12,8)
class ResourcePaused(RuntimeError):
    exit_code=75
class IdentityViolation(ValueError):pass

def require(value,message):
    if not value:raise ValueError(message)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def raw_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def new_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def state_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('x',encoding='utf-8') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)  # Only this pool's private mutable status.
def policy_defaults():
    return dict(initial_workers=16,fallback_workers=[12,8],prefetch_per_worker=2,allowed_affinity=list(range(16)),nice=10,sample_interval_seconds=15,consecutive_windows=2,warmup_commits=2,policy_source_sha256=POLICY_SHA)
def validate_policy(p):
    require(p==policy_defaults(),'Wrong authorized16->12->8/PF2/affinity/nice/policy')
    require(all(type(p[k]) is int for k in ['initial_workers','prefetch_per_worker','nice','sample_interval_seconds','consecutive_windows','warmup_commits']),'Policy integers required')
def window_for(p,workers):
    validate_policy(p);require(workers in TIERS,'Unknown resource tier');return workers*p['prefetch_per_worker']
def apply_affinity(allowed):
    wanted=set(allowed);require(wanted==set(range(16)) and wanted<=set(os.sched_getaffinity(0)),'Original serial16 affinity unavailable')
    os.sched_setaffinity(0,wanted);require(set(os.sched_getaffinity(0))==wanted,'Own affinity differs')
    os.setpriority(os.PRIO_PROCESS,0,10);require(os.getpriority(os.PRIO_PROCESS,0)==10,'Own nice differs')
def worker_init(original,args,allowed):
    apply_affinity(allowed)
    if original is not None:original(*args)

class TriggerState:
    def __init__(self):self.quota=0;self.scheduler=0
    def observe(self,m,workers):
        if m is None:self.quota=self.scheduler=0;return None
        quota=m['quota_utilization']>=.90 and m['throttled_period_fraction']>=.20
        scheduler=m['eligible_busy']>=.90 and m['cgroup_psi_some']>=.05
        self.quota=self.quota+1 if quota else 0;self.scheduler=self.scheduler+1 if scheduler else 0
        label='QUOTA_SATURATION' if self.quota>=2 else 'SHARED_SCHEDULER_PRESSURE' if self.scheduler>=2 else None
        if label is None:return None
        target=TIERS[TIERS.index(workers)+1] if workers!=8 else 8
        external=m.get('host_psi_some',0)>=.05 or m.get('other_visible_cores',0)>=.035325+.50
        return dict(action='pause' if workers==8 else 'lower',target=target,evidence=dict(trigger=label,quota_streak=self.quota,scheduler_streak=self.scheduler,metrics=m,cause_label='externally_corroborated_pressure' if external else 'own_affinity_or_unattributed_pressure',causal_harm_to_neighbors_proved=False))

def cpulist(value):
    result=[]
    for item in value.strip().split(','):
        if not item:continue
        a,*b=item.split('-');result.extend(range(int(a),int(b[0])+1) if b else [int(a)])
    return result

def psi(path):
    for line in Path(path).read_text().splitlines():
        if line.startswith('some '):return int(dict(x.split('=') for x in line.split()[1:])['total'])
    raise ValueError('CPU PSI some missing')
def pairs(path):return {k:int(v) for k,v in (line.split() for line in Path(path).read_text().splitlines())}
def proc_stat(pid):
    raw=(Path('/proc')/str(pid)/'stat').read_text();parts=raw[raw.rindex(')')+2:].split();status={}
    for line in (Path('/proc')/str(pid)/'status').read_text().splitlines():
        if ':' in line:k,v=line.split(':',1);status[k]=v.strip()
    return dict(pid=pid,parent=int(parts[1]),pgid=int(parts[2]),ticks=int(parts[11])+int(parts[12]),nice=int(parts[16]),threads=int(parts[17]),start=int(parts[19]),RSS=int(parts[21])*os.sysconf('SC_PAGE_SIZE'),affinity=cpulist(status['Cpus_allowed_list']))
def sample_linux():
    parent=os.getpid();pgid=os.getpgrp();allproc={}
    for q in Path('/proc').iterdir():
        if not q.name.isdigit():continue
        try:allproc[int(q.name)]=proc_stat(int(q.name))
        except (OSError,ValueError,KeyError,IndexError):pass
    require(parent in allproc,'Own parent identity unavailable');owned={parent};changed=True
    while changed:
        before=len(owned);owned.update(pid for pid,v in allproc.items() if v['parent'] in owned and v['pgid']==pgid);changed=len(owned)!=before
    cpus={}
    for line in Path('/proc/stat').read_text().splitlines():
        key,*v=line.split()
        if key.startswith('cpu') and key[3:].isdigit() and int(key[3:]) in range(16):cpus[int(key[3:])]=list(map(int,v[:8]))
    cg=Path('/sys/fs/cgroup');quota,period=(cg/'cpu.max').read_text().split();quota_capacity=16 if quota=='max' else min(16,int(quota)/int(period))
    return dict(at=time.time(),monotonic=time.monotonic(),parent_pid=parent,pgid=pgid,parent_start=allproc[parent]['start'],owned={str(pid):allproc[pid] for pid in sorted(owned)},other={str(pid):v for pid,v in allproc.items() if pid not in owned},cpus=cpus,cpu_stat=pairs(cg/'cpu.stat'),quota_capacity=quota_capacity,cpu_max=(cg/'cpu.max').read_text().strip(),cgroup_psi_some_total=psi(cg/'cpu.pressure'),host_psi_some_total=psi('/proc/pressure/cpu'),memory_events=pairs(cg/'memory.events'),HZ=os.sysconf('SC_CLK_TCK'))

def derive_window(a,b,allowed):
    dt=b['monotonic']-a['monotonic'];require(dt>0,'Nonpositive sample interval')
    if (a['parent_pid'],a['parent_start'],a['pgid'],a['cpu_max'])!=(b['parent_pid'],b['parent_start'],b['pgid'],b['cpu_max']):raise IdentityViolation('Own identity/quota changed')
    require(set(a['owned'])==set(b['owned']),'Own membership changed')
    for pid,v in b['owned'].items():
        if v['start']!=a['owned'][pid]['start'] or v['affinity']!=allowed or v['nice']!=10:raise IdentityViolation('Own starttime/affinity/nice changed')
    delta=lambda key:b['cpu_stat'][key]-a['cpu_stat'][key]
    require(all(delta(k)>=0 for k in ['usage_usec','nr_periods','nr_throttled','throttled_usec']),'CPU counter reset')
    total=busy=0
    for cpu in allowed:
        aa=a['cpus'][str(cpu)] if str(cpu) in a['cpus'] else a['cpus'][cpu];bb=b['cpus'][str(cpu)] if str(cpu) in b['cpus'] else b['cpus'][cpu]
        d=[y-x for x,y in zip(aa,bb)];require(len(d)==8 and min(d)>=0,'Eligible CPU counter reset');total+=sum(d);busy+=sum(d[i] for i in [0,1,2,5,6,7])
    require(total>0 and b['quota_capacity']>0,'CPU capacity unknown')
    stable_other=[pid for pid,v in b['other'].items() if pid in a['other'] and v['start']==a['other'][pid]['start']]
    other=sum(max(0,b['other'][pid]['ticks']-a['other'][pid]['ticks']) for pid in stable_other)/b['HZ']/dt
    cg=b['cgroup_psi_some_total']-a['cgroup_psi_some_total'];host=b['host_psi_some_total']-a['host_psi_some_total'];require(cg>=0 and host>=0,'PSI counter reset')
    return dict(elapsed_seconds=dt,quota_utilization=delta('usage_usec')/(dt*1e6*b['quota_capacity']),throttled_period_fraction=delta('nr_throttled')/max(delta('nr_periods'),1),throttled_usec=delta('throttled_usec'),eligible_busy=busy/total,cgroup_psi_some=cg/(dt*1e6),host_psi_some=host/(dt*1e6),other_visible_cores=other,own_RSS_sum_bytes=sum(v['RSS'] for v in b['owned'].values()),RSS_is_not_PSS=True)

class ResourceMonitor:
    def __init__(self,policy,state_dir,sample_fn=sample_linux,clock=time.monotonic):
        validate_policy(policy);self.policy=policy;self.home=Path(state_dir);self.sample_fn=sample_fn;self.clock=clock;self.counter=0;self.reset_generation()
    def reset_generation(self):self.last=None;self.last_check=None;self.commits=0;self.warmed=False;self.trigger=TriggerState()
    def after_commit(self,workers,worker_pids):
        self.commits+=1;now=self.clock()
        if self.last_check is not None and now-self.last_check<15:return None
        self.last_check=now;self.counter+=1
        try:
            value=self.sample_fn();pids={str(p) for p in worker_pids}
            require(len(pids)==workers and pids<=set(value['owned']),'Not all workers identified yet')
            if value['quota_capacity']!=16 or any(v['affinity']!=self.policy['allowed_affinity'] or v['nice']!=10 for v in value['owned'].values()):raise IdentityViolation('Own resource identity outside original serial16/nice10')
            if self.last is None or self.commits<2:self.last=value;self._log(value,None,'warming');return None
            before=self.last;self.last=value;metrics=derive_window(before,value,self.policy['allowed_affinity'])
            if not self.warmed:self.warmed=True;self._log(value,metrics,'first_stable15s_warmup_excluded');return None
            decision=self.trigger.observe(metrics,workers);self._log(value,metrics,'trigger' if decision else 'healthy',decision);return decision
        except IdentityViolation as exc:
            decision=dict(action='pause',target=workers,hard_identity=True,evidence=dict(error=repr(exc),status='source_or_affinity_identity_violation_wait_for_Root'));self._log(None,None,'hard_identity_pause',decision,error=repr(exc));return decision
        except (OSError,ValueError,KeyError,IndexError) as exc:
            self.trigger.observe(None,workers);self.last=None;self.warmed=False;self._log(None,None,'unknown_window_reset',error=repr(exc));return None
    def _log(self,sample,metrics,status,decision=None,error=None):new_json(self.home/'samples'/f'{self.counter:08d}.json',dict(actual_exit_code=0,sampling_status=status,sample=sample,metrics=metrics,decision=decision,error=error,NN_performed=False))

class TrackedFuture:
    def __init__(self,pool,job,future,binding):self.pool=pool;self.job=job;self.future=future;self.binding=binding;self.result_called=False;self.worker_pid=None
    def result(self,timeout=None):
        value=self.future.result(timeout=timeout);require(isinstance(value,dict) and type(value.get('worker_pid')) is int,'Original renderer result missing workerPID');self.worker_pid=value['worker_pid'];self.result_called=True;return value

class StagePool:
    def __init__(self,max_workers,mp_context,initializer,initargs,policy,state_dir,monitor=None,executor_factory=ProcessPoolExecutor,expected_new_episodes=None):
        validate_policy(policy);require(max_workers==16,'Entry must be authorized initial16');require(expected_new_episodes is None or (type(expected_new_episodes) is int and expected_new_episodes>=0),'Expected newEP count invalid');self.expected_new_episodes=expected_new_episodes;self.finished=False;self.policy=policy;self.workers=16;self.generation=0;self.home=Path(state_dir);self.home.mkdir(parents=True,exist_ok=True)
        self.monitor=monitor if monitor is not None else ResourceMonitor(policy,self.home);self.factory=executor_factory;self.context=mp_context;self.initializer=initializer;self.initargs=initargs;self.jobs={};self.committed=set();self.pending=None;self.executor=None;self.stage_binding=None;self.counter=0;self.probe_count=0
    @property
    def window(self):return window_for(self.policy,self.workers)
    def __enter__(self):
        require(self.expected_new_episodes is None or self.expected_new_episodes>0,'Zero pending: no resource pool/stage needed; return to metadata control');self._start();return self
    def _start(self):
        require(self.executor is None and not self.jobs,'Concurrent generations prohibited');self.executor=self.factory(max_workers=self.workers,mp_context=self.context,initializer=worker_init,initargs=(self.initializer,self.initargs,self.policy['allowed_affinity']))
        path=self.home/'stages'/f'generation_{self.generation:03d}.json';value=dict(actual_exit_code=0,generation=self.generation,actual_workers=self.workers,prefetch_per_worker=2,window=self.window,allowed_affinity=self.policy['allowed_affinity'],nice=10,policy_source_sha256=POLICY_SHA,resource_module_sha256=raw_sha(__file__),started=time.time())
        new_json(path,value);self.stage_binding=dict(generation=self.generation,actual_workers=self.workers,prefetch_per_worker=2,window=self.window,stage_record_path=str(path),stage_record_sha256=raw_sha(path),policy_source_sha256=POLICY_SHA);self.monitor.reset_generation();self._state('running')
    def probe(self,fn,job):
        require(self.executor is not None and self.generation==0 and not self.jobs and not self.committed and self.probe_count<2,'Resource gate probes allowed only twice before first new EP');self.probe_count+=1;return self.executor.submit(fn,job).result()
    def submit(self,fn,job):
        job=tuple(job);require(not self.finished and (self.expected_new_episodes is None or len(self.committed)+len(self.jobs)<self.expected_new_episodes),'No further declared newEP work');require(job not in self.jobs and job not in self.committed,'Duplicate logicalEP');require(self.pending is None and len(self.jobs)<self.window,'Refill while draining or abovePF2 window')
        future=TrackedFuture(self,job,self.executor.submit(fn,job),dict(self.stage_binding));self.jobs[job]=future;return future
    def receipt_binding(self,job):
        future=self.jobs[tuple(job)];require(future.result_called,'Receipt before actual renderer result');return dict(future.binding)
    def after_commit(self,job,receipt_path,queued_count):
        job=tuple(job);require(job in self.jobs and queued_count==len(self.jobs)-1,'Producer deque/owned job registry differs');future=self.jobs[job];require(future.result_called,'Commit before render completion')
        receipt_sha=None
        if receipt_path is not None:
            receipt=json.loads(Path(receipt_path).read_bytes());require(receipt['actual_exit_code']==0 and receipt['resource_stage']==future.binding,'Committed receipt resource provenance mismatch');receipt_sha=raw_sha(receipt_path)
        self.jobs.pop(job);self.committed.add(job);self.counter+=1;new_json(self.home/'commits'/f'{self.counter:08d}.json',dict(actual_exit_code=0,job=list(job),receipt_path=str(receipt_path) if receipt_path else None,receipt_sha256=receipt_sha,resource_stage=future.binding,renderer_worker_pid=future.worker_pid,queued_EP_count=queued_count))
        pids=list(getattr(self.executor,'_processes',{}) or {}) or [future.worker_pid]
        if self.pending is None:
            decision=self.monitor.after_commit(self.workers,pids)
            if decision:
                target=decision['target'];action=decision['action'];require((self.workers in (16,12) and action=='lower' and target==TIERS[TIERS.index(self.workers)+1]) or (self.workers==8 and action=='pause' and target==8) or (decision.get('hard_identity') is True and action=='pause' and target==self.workers),'Unauthorized skip/upgrade/pause tier');self.pending=decision;self._state('draining_no_refill')
        if self.expected_new_episodes is not None and len(self.committed)==self.expected_new_episodes and not self.jobs:
            if self.pending is not None:new_json(self.home/'completion_resource_decision.json',dict(actual_exit_code=0,decision=self.pending,late_decision_no_further_work=True,next_pool_started=False,committed_new_EP=len(self.committed)))
            self.pending=None;self.finished=True;self._state('all_expected_new_EP_committed');return
        if self.pending is not None and not self.jobs:
            decision=self.pending;self._stop('drained');self.pending=None
            if decision['action']=='pause':self._state('paused_resource_contention',exit_code=75);raise ResourcePaused('Own resource/identity pause after committedEP window drained; tier='+str(self.workers)+'; wait for Root')
            self.workers=decision['target'];self.generation+=1;self._start()
    def refill_slots(self,queued_count):
        require(queued_count==len(self.jobs),'Producer deque count differs');return 0 if self.finished or self.pending is not None else self.window-queued_count
    def _state(self,status,exit_code=None):state_json(self.home/'status.json',dict(stage=status,updated=time.time(),generation=self.generation,actual_workers=self.workers,prefetch_per_worker=2,window=self.window,committed_episodes=len(self.committed),pending_decision=self.pending,queued_EP_count=len(self.jobs),actual_exit_code=exit_code,automatic_upgrade=False,automatic_restart=False))
    def _stop(self,reason):
        if self.executor is None:return
        owned=list((getattr(self.executor,'_processes',{}) or {}).values());self.executor.shutdown(wait=True,cancel_futures=False);new_json(self.home/'stages'/f'generation_{self.generation:03d}_closed.json',dict(actual_exit_code=0,reason=reason,owned_worker_exitcodes={str(p.pid):p.exitcode for p in owned},no_next_generation_overlap=True,queued_commits_remaining=len(self.jobs),completed=time.time()));self.executor=None
    def __exit__(self,kind,value,tb):
        self._stop('context_exit');code=75 if isinstance(value,ResourcePaused) else 1 if value is not None else 0
        new_json(self.home/'pool_exit.json',dict(actual_exit_code=code,all_submissions_committed=not self.jobs and (self.expected_new_episodes is None or len(self.committed)==self.expected_new_episodes),expected_new_episodes=self.expected_new_episodes,committed_episodes=len(self.committed),generation_count=self.generation+1,completed=time.time()));return False
