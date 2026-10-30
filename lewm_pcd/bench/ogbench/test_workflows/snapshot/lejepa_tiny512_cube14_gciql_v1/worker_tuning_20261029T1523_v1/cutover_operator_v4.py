"""Root-only, explicit owned C4 cutover. Default inspect performs no mutation."""
import argparse,base64,hashlib,json,os,re,signal,time
from pathlib import Path,PurePosixPath
ROOT=PurePosixPath('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE=ROOT/'lejepa_tiny512_cube14_gciql_v1';CONTROL=CASE/'control/full';RUNTIME=CONTROL/'runtime'
G='lejepa_tiny512_cube4';GROUP=CASE/'cache/groups'/G;CACHE=CASE/'outputs/cache'/G/'cache'
BRIDGE=CASE/'release_v1/tiny_bridge.py';OLD_PLAN=CONTROL/'launch_plan.json'
OLD_PLAN_SHA='ba4ca64b31fb60b8f02b4637d0f56c73d1615bfc8641c0e94bd4a9c6baa9d74b'
BRIDGE_SHA='98d684a1065b4c44f026ada58ac9664d026998fd74cdbe3872bd09913bb51c62'
LEASE=ROOT/'cube4_visual_gciql_v1/control_v1/gpu_lease.py'
LEASE_SHA='d3a1757ec4b22746da404f15cea2f99e7e830c1edf0f07e1ac20a96955a13727'
PY=ROOT/'env_runtime/bin/python'
STORE=PurePosixPath('/root/autodl-tmp/lejepa_tiny512_cube14_20261029_v1_outputs')
HOLD=STORE/'resource_c4_cutover_hold_v1';OPERATOR_ROOT=CASE/'worker_tuning_20261029T1523_v1'
SUFFIXES=('01_encode_full','02_cache_audit','03_deliver','04_hooks','05_gate','06_reload','07_native_tiny','08_tiny_audit','09_train','10_accept','11_native_full','12_native_audit')
PHASE=G+'_01_encode_full';FORBIDDEN={'.pkl','.npy','.npz','.pt','.pth','.ckpt','.h5','.hdf5'}
SOURCE_LINKS={
 str(ROOT/'assets/baselines/cube4/cube4_lejepa_tiny512_epoch10_20261026_v1/source/jepa_baselines.py'):dict(target='/autodl-fs/data/lewm/lejepa_cube4_tiny512_epoch10_20261026_v1/support_source/reference_checkout/lewm_pcd/bench/cortexbench/jepa_baselines.py',sha256='8c46fe85b054a2521842a3fc753f719736ab0ccd4d584f9999ee9306704d06bc',bytes=8653),
 str(ROOT/'assets/baselines/cube4/cube4_lejepa_tiny512_epoch10_20261026_v1/weights/config.json'):dict(target='/autodl-fs/data/lewm/lejepa_cube4_tiny512_epoch10_20261026_v1/original_training_closure/parallel24_home_v1/checkpoints/lejepa_quadruple600_cortexbench_seed0_w24pf2_v1/config.json',sha256='68b252a64def5326c6f11d985acd58fdfc83f50081163ff8c96c179834f55235',bytes=424)}
MESH_ROOT=ROOT/'source/ogbench/ogbench/manipspace/descriptions/universal_robots_ur5e/assets'
LARGE_SOURCES={str(MESH_ROOT/name):dict(bytes=size,sha256=sha) for name,size,sha in (
 ('shoulder_0.obj',2947794,'477387a7e7312782a6040484a3c2c2fcee33f40109f419076e31b6f6e1692c68'),
 ('shoulder_2.obj',2440344,'291714518416bb8f0bafbf123867db011ecbf5146a05245ca5893ee8c57d4e21'),
 ('upperarm_2.obj',3677923,'86c05350d60eb0e7e28a81b544267478b7db997baba41cef19088ebbeffecfa1'),
 ('upperarm_3.obj',5346165,'b257dab441e6fab64ea03b4b87029d8a96554ad1a513cce944d56acba0ad948a'),
 ('wrist1_1.obj',2470967,'ee6d5c562e6becef94109b0543329080d285d1aef521a01ed98e18a5bf1e8684'),
 ('wrist2_1.obj',2156900,'a721d06dc61d3b3a8ec58c035c9e3358ed30d6991a72e077d8abf3c3c203b633'),
 ('wrist2_2.obj',2141392,'6df4335c8bd5e881fda2c224f770115ffe9ddd0cd321ebe3c0a4a310796f0987'))}
def require(value,message):
    if not value:raise ValueError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return digest(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())

class Machine:
    def path(self,p):return Path(p)
    def read(self,p):
        require(PurePosixPath(p).suffix not in FORBIDDEN,'Payload read forbidden: '+str(p))
        require(self.stat(p)['size']<=2<<20,'Metadata exceeds2MiB: '+str(p));return self.path(p).read_bytes()
    def stat(self,p):
        v=self.path(p).stat();return dict(size=v.st_size,mtime_ns=v.st_mtime_ns,ctime_ns=v.st_ctime_ns,inode=v.st_ino,device=v.st_dev)
    def exists(self,p):return self.path(p).exists()
    def symlink(self,p):return self.path(p).is_symlink()
    def lstat(self,p):
        v=self.path(p).lstat();return dict(size=v.st_size,mtime_ns=v.st_mtime_ns,ctime_ns=v.st_ctime_ns,inode=v.st_ino,device=v.st_dev)
    def link_text(self,p):return os.readlink(self.path(p))
    def resolved_name(self,p):return str(self.path(p).resolve())
    def hash_source(self,p):
        require(str(p) in LARGE_SOURCES and PurePosixPath(p).suffix=='.obj' and not self.symlink(p),'Only exact frozen regular mesh sources may stream')
        require(self.stat(p)['size']==LARGE_SOURCES[str(p)]['bytes']<=6<<20,'Wrong frozen mesh size')
        value=hashlib.sha256()
        with self.path(p).open('rb') as stream:
            for block in iter(lambda:stream.read(1<<20),b''):value.update(block)
        return value.hexdigest()
    def resolve(self,p):return self.path(p).resolve()
    def children(self,p):return sorted(x.name for x in self.path(p).iterdir()) if self.exists(p) else []
    def mkdir(self,p):self.path(p).mkdir(parents=True,exist_ok=False)
    def new(self,p,value):
        with self.path(p).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    def rename(self,source,dest):
        require(not self.exists(dest),'Never overwrite hold destination');self.path(dest).parent.mkdir(parents=True,exist_ok=True);self.path(source).rename(self.path(dest))
    def now(self):return time.monotonic()
    def wall(self):return time.time()
    def sleep(self,s):time.sleep(s)
    def uid(self):return os.getuid()
    def proc(self,pid):
        try:
            root=Path('/proc')/str(pid);raw=(root/'stat').read_text();tail=raw.rsplit(')',1)[1].split()
            return dict(pid=pid,starttime=int(tail[19]),PGID=int(tail[2]),session=int(tail[3]),ppid=int(tail[1]),uid=root.stat().st_uid,argv=[x.decode() for x in (root/'cmdline').read_bytes().split(b'\0') if x],cwd=str((root/'cwd').resolve()))
        except FileNotFoundError:return None
    def members(self,pgid):
        result=[]
        for path in Path('/proc').iterdir():
            if path.name.isdigit():
                try:value=self.proc(int(path.name))
                except (PermissionError,ProcessLookupError):continue
                if value and value['PGID']==pgid:result.append(value)
        return sorted(result,key=lambda x:x['pid'])
    def signal_group(self,pgid):os.killpg(pgid,signal.SIGTERM)

def capture(fs,path):
    require(not fs.symlink(path),'Metadata symlink refused: '+str(path))
    before=fs.stat(path);raw=fs.read(path);after=fs.stat(path)
    require(before==after and len(raw)==before['size'],'Metadata changed during read: '+str(path))
    return dict(sha256=digest(raw),bytes=len(raw),source_before=before,source_after=after,raw_b64=base64.b64encode(raw).decode())
def raw(proof):return base64.b64decode(proof['raw_b64'],validate=True)
def read(fs,path):return json.loads(raw(capture(fs,path)))
def capture_source_pin(fs,path,expected_sha):
    """Only the two diagnosed frozen source links may follow their exact target."""
    if str(path) in LARGE_SOURCES:
        known=LARGE_SOURCES[str(path)];before=fs.stat(path)
        require(not fs.symlink(path) and expected_sha==known['sha256'] and before['size']==known['bytes']<=6<<20,'Frozen mesh identity/size differs')
        actual=fs.hash_source(path);after=fs.stat(path)
        require(before==after and actual==expected_sha,'Mesh source SHA/stat changed')
        evidence=dict(path=str(path),sha256=actual,bytes=before['size'],source_before=before,source_after=after,stream_hash_only=True,raw_body_captured=False)
        return dict(sha256=actual,bytes=before['size'],source_before=before,source_after=after,source_mesh=evidence)
    if not fs.symlink(path):
        proof=capture(fs,path);require(proof['sha256']==expected_sha,'Old source pin differs: '+str(path));return proof
    require(str(path) in SOURCE_LINKS,'Unknown source symlink refused: '+str(path))
    known=SOURCE_LINKS[str(path)];require(expected_sha==known['sha256'],'Source link is not the frozen pin SHA')
    require(PurePosixPath(path).suffix not in FORBIDDEN and PurePosixPath(known['target']).suffix not in FORBIDDEN,'Payload symlink forbidden')
    link_before=fs.lstat(path);text_before=fs.link_text(path);resolved_before=fs.resolved_name(path)
    require(text_before==resolved_before==known['target'],'Source link target is not the exact diagnosed target')
    before=fs.stat(path);content=fs.read(path);after=fs.stat(path)
    link_after=fs.lstat(path);text_after=fs.link_text(path);resolved_after=fs.resolved_name(path)
    require(link_before==link_after and text_before==text_after and resolved_before==resolved_after and before==after,'Source link/target changed while reading')
    require(len(content)==before['size']==known['bytes'] and digest(content)==expected_sha,'Source symlink target SHA/size differs')
    proof=dict(sha256=expected_sha,bytes=len(content),source_before=before,source_after=after,raw_b64=base64.b64encode(content).decode())
    proof['source_symlink']=dict(path=str(path),expected_sha256=expected_sha,bytes=len(content),link_before=link_before,link_after=link_after,readlink_before=text_before,readlink_after=text_after,resolved_before=resolved_before,resolved_after=resolved_after,target_before=before,target_after=after)
    return proof
def absent(fs):
    for path in (CACHE/'cache_seal.json',CACHE/'cache_manifest.json',RUNTIME/'closure.json',RUNTIME/'failure.json',RUNTIME/(PHASE+'_exit.json'),RUNTIME/'outer_exit_code',CONTROL/'wrapper_exit_code',CONTROL/'worker_failure.json',CONTROL/'worker_outcome.json'):
        require(not fs.exists(path),'Old phase no longer exclusively running: '+str(path))
def group_identity(fs,expected,phase):
    require(type(expected['pid']) is int and expected['pid']==expected['pgid'] and expected['pid']>1,'Expected phase leader/PGID required')
    value=fs.proc(expected['pid']);require(value is not None,'Expected phase leader vanished')
    require(value['starttime']==expected['starttime'] and value['PGID']==value['session']==expected['pgid'],'PID starttime/PGID/session changed')
    prefix=[str(PY),'-B',str(LEASE),'--']
    suffix=[str(PY),'-B',str(BRIDGE),'cache-producer','--plan',str(GROUP/'plan.json'),'--execute']
    require(phase['name']==PHASE and phase['argv']==prefix+suffix,'Frozen phase is not the exact known lease-wrapped C4 producer')
    require(capture_source_pin(fs,LEASE,LEASE_SHA)['sha256']==LEASE_SHA,'Known execv lease source differs')
    require(value['argv']==phase['argv'][4:]==suffix and fs.resolve(value['cwd'])==fs.resolve(phase['cwd']),'Actual /proc argv/cwd differs from exact known execv suffix')
    value=dict(value,argv_admission='exact_known_gpu_lease_execv_suffix',lease_source_sha256=LEASE_SHA,recorded_wrapper_prefix=prefix)
    members=fs.members(expected['pgid']);require(members and any(x['pid']==expected['pid'] for x in members),'Owned group missing leader')
    ids={x['pid'] for x in members}
    require(all(x['uid']==fs.uid() and x['PGID']==x['session']==expected['pgid'] and (x['pid']==expected['pid'] or x['ppid'] in ids) for x in members),'Foreign/reparented process in target group')
    return value,members

def inspect(fs,expected):
    absent(fs);proof=capture(fs,OLD_PLAN);require(proof['sha256']==OLD_PLAN_SHA,'Frozen old global plan SHA differs')
    old=json.loads(raw(proof));require(len(old['phases'])==48 and old['phases'][36]['name']==PHASE,'Wrong old48 phase list')
    require(digest(fs.read(BRIDGE))==BRIDGE_SHA,'Frozen Tiny bridge SHA differs')
    source_symlinks={};source_meshes={}
    for path,sha in old['source_pins'].items():
        pin=capture_source_pin(fs,path,sha)
        if 'source_symlink' in pin:source_symlinks[path]=pin['source_symlink']
        if 'source_mesh' in pin:source_meshes[path]=pin['source_mesh']
    exits={}
    for phase in old['phases'][:36]:
        value=read(fs,RUNTIME/(phase['name']+'_exit.json'));require(type(value['actual_exit_code']) is int and value['actual_exit_code']==0 and value['residual_owned_members']==[],'Old36 not0/drained');exits[phase['name']]=value
    status=read(fs,RUNTIME/'status.json');require(status['stage']==PHASE and status['actual_exit_code'] is None,'Old controller not at C4 encode')
    phase=old['phases'][36];argv=read(fs,RUNTIME/(PHASE+'_argv.json'))
    require(argv['pid']==expected['pid'] and argv['argv']==phase['argv'] and argv['cwd']==phase['cwd'],'Recorded current phase identity differs')
    proc,members=group_identity(fs,expected,phase)
    plan=read(fs,GROUP/'plan.json');require(plan['cube_count']==4 and plan['latent_dim']==512 and plan['encoder_family']=='lejepa_tiny512','Wrong C4 Tiny plan')
    require((plan['renderer_workers'],plan['prefetch'],plan['encoder_batch'],plan['torch_cpu_threads'])==(2,2,32,1) and plan['inference_precision']=='fp32' and plan['storage_dtype']=='float32' and plan['cache_dir']==str(CACHE),'Legacy resource/science/cache identity differs')
    protocol_proof=capture(fs,CACHE/'protocol.json');protocol=json.loads(raw(protocol_proof));require(protocol['producer_source_sha256']==BRIDGE_SHA and protocol['renderer_workers']==protocol['prefetch']==2,'Wrong original W2 protocol')
    progress=read(fs,CACHE/'status.json');require(progress['phase']=='encoding' and progress['failed_episodes']==0 and progress['protocol_sha256']==canonical(protocol) and progress['completed_rows']==1001*progress['completed_episodes'],'Producer progress differs')
    return dict(old36_count=36,old=old,plan=plan,old_phase_exits=exits,protocol=protocol,protocol_raw_sha=protocol_proof['sha256'],argv_record=argv,progress=progress,proc=proc,members=members,source_symlinks=source_symlinks,source_meshes=source_meshes)

def inventory(fs,context):
    files={};entries=[];uncommitted=[];protocol=canonical(context['protocol'])
    for split,count in (('train',5000),('val',500)):
        for name in fs.children(CACHE/split):
            require(re.fullmatch(r'episode_\d{6}',name) is not None,'Unknown episode directory')
            ep=int(name[-6:]);require(ep<count,'Episode outside fixed split');directory=CACHE/split/name
            require(not fs.symlink(directory) and fs.resolve(CACHE) in fs.resolve(directory).parents,'Episode path escaped fixed cache')
            receipt_path=directory/'receipt.json';latent=directory/'latent.npy';committed=fs.exists(receipt_path)
            if committed:
                proof=capture(fs,receipt_path);receipt=json.loads(raw(proof));require(receipt['status']=='episode_committed' and receipt['actual_exit_code']==0 and receipt['split']==split and receipt['episode']==ep,'Invalid committed receipt')
                require(receipt['protocol_sha256']==protocol and receipt['producer_source_sha256']==BRIDGE_SHA and receipt['latent']['path']==str(latent),'Committed source identity differs')
                require(not fs.symlink(latent) and fs.exists(latent),'Committed latent missing/symlink')
                stat=fs.stat(latent);require(stat['size']==receipt['latent']['bytes'],'Latent size differs; payload remains unread')
                files[str(receipt_path)]=proof;entries.append(dict(split=split,episode=ep,receipt_path=str(receipt_path),receipt_sha256=proof['sha256'],receipt_stat=proof['source_before'],latent=dict(path=str(latent),sha256=receipt['latent']['sha256'],stat=stat)))
            for filename in fs.children(directory):
                path=directory/filename
                if committed and filename in ('receipt.json','latent.npy'):continue
                allowed=(filename=='latent.npy' and not committed) or (filename.startswith('latent.partial_') and filename.endswith('.npy')) or re.fullmatch(r'\.?receipt(?:\.json)?(?:[._].*)?\.tmp',filename)
                require(allowed and not fs.symlink(path) and fs.resolve(CACHE) in fs.resolve(path).parents,'Unknown/unowned episode file: '+str(path))
                uncommitted.append(dict(path=str(path),stat=fs.stat(path),reason='no committed transaction for this file'))
    return files,entries,uncommitted

def execute(fs,expected,evidence_dir,isolate=False,drain_timeout=90):
    evidence_dir=PurePosixPath(evidence_dir)
    require(fs.resolve(OPERATOR_ROOT) in fs.resolve(evidence_dir).parents,'Evidence directory outside own operator root')
    require(not fs.exists(evidence_dir),'Evidence exists; no duplicate execution')
    context=inspect(fs,expected);require(not isolate or not fs.exists(HOLD),'Hold directory already exists')
    fs.mkdir(evidence_dir);events=[]
    def record(name,value):fs.new(evidence_dir/name,value)
    record('intent.json',dict(operation='explicit_user_owned_C4_cutover',expected=expected,pre_proc=context['proc'],pre_members=context['members'],source_symlinks=context['source_symlinks'],source_meshes=context['source_meshes'],baseline_completed=context['progress']['completed_episodes'],at_unix=fs.wall(),operator_source_sha256=digest(Path(__file__).read_bytes()),automatic_SIGKILL=False))
    try:
        start=fs.now();baseline=context['progress']['completed_episodes']
        while True:
            require(fs.now()-start<15,'No newly committed EP within15seconds; no signal sent')
            value=read(fs,CACHE/'status.json')
            if value['completed_episodes']>baseline:
                require(value['phase']=='encoding' and value['failed_episodes']==0 and value['completed_rows']==value['completed_episodes']*1001,'Invalid commit boundary status')
                receipt=capture(fs,CACHE/value['split']/f"episode_{value['episode']:06d}"/'receipt.json');r=json.loads(raw(receipt))
                require(r['actual_exit_code']==0 and r['protocol_sha256']==canonical(context['protocol']) and r['producer_source_sha256']==BRIDGE_SHA,'New commit not closed under original protocol');break
            fs.sleep(.25)
        before=inspect(fs,expected);require(before['members']==context['members'] and before['old_phase_exits']==context['old_phase_exits'] and before['protocol_raw_sha']==context['protocol_raw_sha'] and before['argv_record']==context['argv_record'] and before['source_symlinks']==context['source_symlinks'] and before['source_meshes']==context['source_meshes'],'Owned worker/source/old36 identity changed before signal')
        record('pre_signal.json',dict(at_unix=fs.wall(),proc=before['proc'],members=before['members'],commit_status=value,receipt=receipt))
        record('signal_intent.json',dict(at_unix=fs.wall(),signal='SIGTERM',PGID=expected['pgid']))
        fs.signal_group(expected['pgid']);record('signal_sent.json',dict(at_unix=fs.wall(),signal='SIGTERM',PGID=expected['pgid']))
        started=fs.now()
        while True:
            require(fs.now()-started<=drain_timeout,'SIGTERM did not produce closed/drained old failure; no SIGKILL attempted')
            require(not fs.exists(RUNTIME/'closure.json') and not fs.exists(CACHE/'cache_seal.json'),'Old pipeline completed/raced; refuse interrupted adoption')
            paths=[RUNTIME/(PHASE+'_exit.json'),RUNTIME/'outer_exit_code',CONTROL/'wrapper_exit_code',RUNTIME/'failure.json',RUNTIME/'status.json',CONTROL/'worker_outcome.json']
            if not fs.members(expected['pgid']) and all(fs.exists(p) for p in paths):
                stop=read(fs,paths[0]);require(type(stop['actual_exit_code']) is int and stop['actual_exit_code']!=0 and stop['residual_owned_members']==[],'Old producer did not actually fail/drain')
                require(int(fs.read(paths[1]))!=0 and int(fs.read(paths[2]))!=0,'Old outer/wrapper must retain true nonzero')
                require(all(type(read(fs,p)['actual_exit_code']) is int and read(fs,p)['actual_exit_code']!=0 for p in paths[3:]),'Old failure/outcome must be real nonzero');break
            fs.sleep(.25)
        fixed=[OLD_PLAN,BRIDGE,GROUP/'plan.json',GROUP/'data_binding.json',GROUP/'camera.json',GROUP/'encoder_spec.json',CACHE/'protocol.json',RUNTIME/(PHASE+'_argv.json'),*paths]
        fixed += [RUNTIME/(p['name']+'_exit.json') for p in context['old']['phases'][:36]]
        fixed += [PurePosixPath(context['plan']['shared_state_root'])/'source_arrays.json',PurePosixPath(context['plan']['encoder_spec']),PurePosixPath(context['plan']['data_acceptance']),PurePosixPath(context['plan']['source_stat_freeze'])]
        originals={str(p):capture(fs,p) for p in fixed};receipt_files,entries,unknown=inventory(fs,context)
        require(originals[str(OLD_PLAN)]['sha256']==OLD_PLAN_SHA and originals[str(BRIDGE)]['sha256']==BRIDGE_SHA and originals[str(CACHE/'protocol.json')]['sha256']==context['protocol_raw_sha'],'Original source changed after stop')
        require(all(json.loads(raw(originals[str(RUNTIME/(name+'_exit.json'))]))==value for name,value in context['old_phase_exits'].items()),'Old36 records changed after stop')
        require({('train',0),('train',1)}<={(x['split'],x['episode']) for x in entries} and len(entries)<5500,'Bounded reference EP0/1 required and cache must remain incomplete')
        if unknown:
            require(isolate,'Uncommitted files preserved; explicit isolation required before ledger acceptance')
            fs.mkdir(HOLD)
            for index,item in enumerate(unknown):
                source=PurePosixPath(item['path']);require(fs.resolve(CACHE) in fs.resolve(source).parents and fs.stat(source)==item['stat'] and not fs.symlink(source),'Uncommitted source changed/escaped')
                require(not fs.exists(source.parent/'receipt.json') or source.name not in ('receipt.json','latent.npy'),'Refuse committed file movement')
                destination=HOLD/source.relative_to(CACHE);require(fs.resolve(HOLD) in fs.resolve(destination).parents and not fs.exists(destination),'Hold destination escaped/exists')
                record(f'isolation_{index:06d}_intent.json',dict(source=str(source),destination=str(destination),source_before=item['stat'],at=fs.now()))
                fs.rename(source,destination);after=fs.stat(destination)
                require(all(after[k]==item['stat'][k] for k in ('size','mtime_ns','inode','device')),'Rename changed content identity')
                event=dict(source=str(source),destination=str(destination),source_before=item['stat'],destination_after=after,actual_action='rename',at=fs.now());events.append(event);record(f'isolation_{index:06d}_actual.json',event)
            receipt_files,new_entries,unknown=inventory(fs,context);require(new_entries==entries and not unknown,'Committed adoption changed during isolation')
        originals.update(receipt_files)
        require(not fs.members(expected['pgid']) and not fs.exists(RUNTIME/'closure.json'),'Owned group/closure changed after capture')
        ledger=dict(schema_version=1,group_id=G,cube_count=4,old_global_plan_sha256=OLD_PLAN_SHA,old_phase_exits=context['old_phase_exits'],post_stop=dict(actual_exit_code=stop['actual_exit_code'],residual_owned_members=[],owned_PGID=expected['pgid'],old_runtime_root=str(RUNTIME)),old_global_closure_absent=True,original_protocol_sha256=canonical(context['protocol']),original_producer_source_sha256=BRIDGE_SHA,original_files=originals,entries=entries,uncommitted_files=unknown,isolated_records=events)
        record('full_input_ledger.json',ledger);record('operator_outcome.json',dict(actual_exit_code=0,cutover_stopped_only=True,new_NN_started=False,committed_episodes=len(entries),isolated_files=len(events),at=fs.now()))
        return ledger
    except BaseException as exc:
        record('operator_failure.json',dict(actual_exit_code=1,error=repr(exc),at=fs.now(),remaining_owned_members=fs.members(expected['pgid']),automatic_SIGKILL=False,isolated_records=events));raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-pid',required=True,type=int);p.add_argument('--expected-pgid',required=True,type=int);p.add_argument('--expected-starttime',required=True,type=int)
    p.add_argument('--execute-user-cutover',action='store_true');p.add_argument('--isolate-owned-uncommitted',action='store_true');p.add_argument('--evidence-dir',type=PurePosixPath);a=p.parse_args()
    fs=Machine();expected=dict(pid=a.expected_pid,pgid=a.expected_pgid,starttime=a.expected_starttime)
    if a.execute_user_cutover:
        require(a.evidence_dir is not None,'Explicit fresh evidence directory required');ledger=execute(fs,expected,a.evidence_dir,a.isolate_owned_uncommitted)
        print(json.dumps(dict(actual_exit_code=0,committed_episodes=len(ledger['entries']),new_NN_started=False,evidence_dir=str(a.evidence_dir))))
    else:
        require(not a.isolate_owned_uncommitted and a.evidence_dir is None,'Inspect creates no evidence/hold paths')
        value=inspect(fs,expected);print(json.dumps(dict(actual_exit_code=0,inspect_only=True,old36_count=value['old36_count'],progress=value['progress'],proc=value['proc'],members=value['members'],source_symlinks=value['source_symlinks'],source_meshes=value['source_meshes'],signals_sent=False)))
if __name__=='__main__':main()
