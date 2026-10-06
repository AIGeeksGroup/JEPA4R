"""Gated sequential original RGB then DexArt WM training; no auto resume/relaunch."""
import argparse,csv,fcntl,hashlib,json,os,subprocess,sys,time,importlib.util,inspect
from pathlib import Path
ACTIVE_PLAN_SHA='3264fad372810eca69ae10b785d0b1d3361be47730da856a9260abad3bdc6609'
PROTOCOL={'epochs':10,'batch':128,'precision':'bf16','history':3,'frameskip':5,'num_preds':1,'action':24,'input':120,'pcd_dim':6,'img':224}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def check_report(entry):
 p=Path(entry['path']);assert sha(p)==entry['sha256'],'prerequisite report changed: '+str(p)
 x=json.loads(p.read_text())
 assert entry['assertions'],'unreviewed prerequisite cannot be an empty check'
 for assertion in entry['assertions']:
  value=x
  if assertion['path'] not in ('','$'):
   for key in assertion['path'].split('.'):value=value[int(key)] if isinstance(value,list) else value[key]
  assert value==assertion['equals'],'prerequisite failed '+assertion['path']
 return x

def gpu_idle(index,minimum):
 p=subprocess.run(['nvidia-smi','-i',str(index),'--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 row=next(csv.reader(p.stdout.splitlines()));uuid=row[0].strip();free=int(row[1].strip());util=int(row[2].strip())
 apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
 peers=[v for v in csv.reader(apps.splitlines()) if len(v)>=2 and v[1].strip()==uuid]
 assert not peers,'GPU has an existing compute process; no preemption'
 assert util==0,'GPU is currently utilized; wait instead of competing'
 assert free>=minimum,'GPU free memory below actual batch128 gate requirement'
 return {'uuid':uuid,'free_mib':free,'utilization_percent':util,'compute_process_count':0}

def run(config_path):
 cfg=json.loads(config_path.read_text());full=Path(cfg['full_root']).resolve();home=Path(cfg['training_root']).resolve()
 assert home.is_relative_to(full/'training'),'fresh training home must stay under full_root/training'
 assert not home.exists(),'existing training run is never relaunched or overwritten'
 assert cfg.get('gpu_index',0)==0 and int(cfg['num_workers'])>=1
 assert sha(cfg['active_plan'])==ACTIVE_PLAN_SHA and cfg['active_plan_sha256']==ACTIVE_PLAN_SHA
 required={'scheduler','launcher','publisher','publication_manifest','full_qc','loader_rgb','loader_pcd'}
 assert set(cfg['prerequisites'])==required
 evidence={name:check_report(entry) for name,entry in cfg['prerequisites'].items()}
 scheduler=evidence['scheduler'];assert scheduler['phase']=='complete' and scheduler['exit_code']==0
 assert scheduler['config_sha256']==ACTIVE_PLAN_SHA and scheduler['committed_seed_count']==scheduler['expected_seed_count']==10150
 assert not scheduler['failed'] and not scheduler['active']
 assert cfg['source_manifest'] and cfg['payload_manifest']
 core=Path(cfg['code_root']).resolve()/'lewm_pcd'
 mandatory=['train.py','utils.py','module.py','jepa.py','config/train/lewm.yaml','config/train/data/adroit_pen.yaml','config/train/data/adroit_pen_pcd.yaml','config/train/data/dmc.yaml','config/train/data/ogb_pcd.yaml','config/train/model/lewm.yaml','config/train/model/lewm_pcd_dexart.yaml','encoders/pointnet_dexart.py']
 sealed_paths={Path(row['path']).resolve() for row in cfg['source_manifest']}
 assert all((core/name).resolve() in sealed_paths for name in mandatory),'missing frozen training core/config hashes'
 assert Path(cfg['verifier']).resolve() in sealed_paths and Path(cfg['recorded_train']).resolve() in sealed_paths and Path(__file__).resolve() in sealed_paths,'runner/verifier must be sealed too'
 for row in cfg['source_manifest']:assert sha(row['path'])==row['sha256'],'sealed train/source dependency changed'
 for key in ('train_rgb','train_pcd','test_rgb','test_pcd'):
  entry=cfg['datasets'][key];assert sha(entry['path'])==entry['sha256']
 for row in cfg['payload_manifest']:assert Path(row['path']).is_file() and sha(row['path'])==row['sha256'],'VDS payload missing or changed'
 for route in ('rgb','pcd'):
  gate=check_report(cfg['gpu_gates'][route]);assert gate['gate']=='PASS' and gate['active_plan_sha256']==ACTIVE_PLAN_SHA
  assert gate['train_rgb_sha']==cfg['datasets']['train_rgb']['sha256'] and gate['train_pcd_sha']==cfg['datasets']['train_pcd']['sha256']
  assert gate['protocol']==PROTOCOL,'GPU gate protocol differs'
 home.parent.mkdir(parents=True,exist_ok=True);lock=(home.parent/'gpu0-training.lock').open('a+')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 gpu=gpu_idle(cfg.get('gpu_index',0),cfg['minimum_free_gpu_mib']);home.mkdir()
 logs=home/'logs';logs.mkdir();state={'phase':'preflight_passed','started_at':time.time(),'updated_at':time.time(),'completed_stages':0,'failed_stages':0,'errors':[],'exit_code':None,'config_sha256':sha(config_path),'rng_policy':'unchanged original train/SPT; cfg.seed explicitly split/loader only','protocol':PROTOCOL,'gpu':gpu}
 save(home/'runner_config.json',cfg);save(home/'state.json',state)
 env=os.environ.copy();env.update({'STABLEWM_HOME':str(home),'SPT_CACHE_DIR':str(home/'checkpoints/_spt_cache'),'PYTHONPATH':str(Path(cfg['code_root'])/'lewm_pcd'),'CUDA_VISIBLE_DEVICES':str(cfg.get('gpu_index',0)),'HYDRA_FULL_ERROR':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
 code=Path(cfg['code_root'])/'lewm_pcd';python=cfg['python'];results=[]
 def call(argv,path):
  with path.open('xb') as stream:p=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,cwd=code,env=env,start_new_session=True)
  # Parent closes the log immediately; child retains its inherited stdout.
  state.update(pid=p.pid,updated_at=time.time());save(home/'state.json',state)
  while p.poll() is None:state['updated_at']=time.time();save(home/'state.json',state);time.sleep(20)
  result=p.returncode;path.with_suffix('.exit_code').write_text(str(result)+'\n');assert result==0,'stage command failed '+str(path);return result
 try:
  for route in ('rgb','pcd'):
   assert sha(config_path)==state['config_sha256'],'runner input config changed'
   for row in cfg['source_manifest']:assert sha(row['path'])==row['sha256']
   state.update(phase=route+'_preflight',updated_at=time.time());save(home/'state.json',state)
   save(logs/(route+'_gpu_before.json'),gpu_idle(cfg.get('gpu_index',0),cfg['minimum_free_gpu_mib']))
   name=cfg['run_names'][route];assert name and all(c.isalnum() or c in '_-' for c in name)
   target=home/'checkpoints'/name;assert not target.exists(),'original Manager auto-resume is forbidden for this new run'
   dataset=cfg['datasets']['train_'+route]['path'];assert dataset!=cfg['datasets']['test_'+route]['path']
   overrides=['data='+('adroit_pen' if route=='rgb' else 'adroit_pen_pcd'),'model='+('lewm' if route=='rgb' else 'lewm_pcd_dexart'),'data.dataset.name='+dataset,'subdir='+name,'output_model_name='+name,'trainer.devices=1','num_workers='+str(cfg['num_workers']),'loader.prefetch_factor='+str(cfg['prefetch_factor'])]
   call([python,'train.py','--cfg','job','--resolve','model.action_encoder.input_dim=120']+overrides,logs/(route+'_resolved.log'))
   state.update(phase=route+'_training',updated_at=time.time());save(home/'state.json',state)
   call([python,cfg['recorded_train'],'--code-root',cfg['code_root']]+overrides,logs/(route+'_train.log'))
   effective=json.loads((target/'effective_spt_config.json').read_text())
   assert Path(effective['cache_dir']).resolve()==(home/'checkpoints/_spt_cache').resolve(),'SPT singleton output escaped fresh temp home'
   assert all(effective['cleanup'][key] is False for key in ('checkpoints','logs','callback_artifacts')),'current-run evidence cleanup still enabled'
   weights=target/'weights_epoch_10.pt';assert weights.is_file(),'missing explicit epoch10 weights'
   state.update(phase=route+'_strict_reload',updated_at=time.time());save(home/'state.json',state)
   report_dir=logs/(route+'_verify_full');assert not report_dir.exists(),'fresh verifier output directory required'
   report=report_dir/'report.json'
   argv=[python,cfg['verifier'],'--mode','verify-full','--code-root',cfg['code_root'],'--dataset',dataset,'--route',route,'--report',str(report_dir),'--checkpoint-root',str(target),'--weights','weights_epoch_10.pt','--train-rgb',cfg['datasets']['train_rgb']['path'],'--train-pcd',cfg['datasets']['train_pcd']['path'],'--active-plan',cfg['active_plan']]
   call(argv,logs/(route+'_verify_full.log'));assert (report_dir/'exit_code').read_text().strip()=='0'
   v=json.loads(report.read_text());assert v['gate']=='PASS' and v['finite'] and v['strict_reload']=='PASS'
   assert v['epochs_completed']==10 and v['optimizer_steps']>0 and v['checkpoint_sha256']==sha(weights)
   assert v['active_plan_sha256']==ACTIVE_PLAN_SHA and v['protocol']==PROTOCOL
   assert v['source_hashes']==json.loads(Path(cfg['gpu_gates'][route]['path']).read_text())['source_hashes'],'reload sources differ from qualified GPU gate'
   assert v['train_rgb_sha']==cfg['datasets']['train_rgb']['sha256'] and v['train_pcd_sha']==cfg['datasets']['train_pcd']['sha256']
   for row in cfg['source_manifest']:assert sha(row['path'])==row['sha256']
   results.append(dict(route=route,checkpoint=str(weights),sha256=sha(weights),verification_report=str(report),verification_sha256=sha(report)))
   state.update(completed_stages=len(results),phase=route+'_verified',updated_at=time.time());save(home/'state.json',state)
  state.update(phase='complete',exit_code=0,updated_at=time.time());save(home/'final_report.json',{'stages':results,'qc':'PASS','exit_code':0});return 0
 except BaseException as error:
  state.update(phase='failed',failed_stages=1,errors=[repr(error)],exit_code=1,updated_at=time.time());raise
 finally:save(home/'state.json',state);(home/'exit_code').write_text(str(state['exit_code'])+'\n');lock.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);raise SystemExit(run(p.parse_args().config))
