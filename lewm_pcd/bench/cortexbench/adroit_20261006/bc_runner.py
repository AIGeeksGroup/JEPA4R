"""Six original BC jobs, serial GPU; prepare-only until root explicitly launches."""
import argparse,fcntl,hashlib,json,math,os,pickle,subprocess,time,traceback
from pathlib import Path

EVAL_INDICES=list(range(5,100,5))+[99]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def sources(c):
 assert subprocess.check_output(['git','-C',c['code_root'],'rev-parse','HEAD'],text=True).strip()==c['head']
 for item in c['source_files']:assert sha(Path(c['code_root'])/item['path'])==item['sha256'],item['path']
 assert c['baseline_count']==len(c['source_files'])==len({i['path'] for i in c['source_files']})==221 and len(c['bc_leaf_files'])==5
 for item in c['bc_leaf_files']:assert sha(Path(c['code_root'])/item['path'])==item['sha256'],item['path']
def gpu(c):
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'GPU compute process already active'
 free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=c['minimum_free_mib'],free
 return {'free_mib':free,'compute_processes':0}
def preflight(c):
 assert c['head']=='f1cc56dade6937de8ceec6834d0e796ff8bb407f'
 assert c['python']=='/nix/lewm/pen105-bc/bin/python'
 sources(c)
 for item in c['sealed_inputs']:assert sha(item['path'])==item['sha256'],item['path']
 for path in c['gate_exit_files']:assert Path(path).read_text().strip()=='0',path
 for route in ('rgb','pcd'):
  g=read(c['native_gates'][route]);assert g['gate']=='PASS_NATIVE_ROUTE_ENCODER' and g['finite'] and g['native_steps']==3
 smoke=read(c['smoke_reload_gate']);assert smoke['qc']=='PASS_REAL_BC_SMOKE_AND_POLICY_RELOAD'
 assert {r['route'] for r in smoke['results']}=={'rgb','pcd'}
 for r in smoke['results']:assert r['policy_reload']=='PASS' and r['finite_model'] and r['input']==600 and r['action']==24
 q=read(c['pcd100_gate']);assert q['qc']=='PASS_OFFICIAL100_PCD_REPLAY_LABELS_PRESERVED' and q['demos']==100 and q['frames']==10000 and q['native_real_replay_state_reward_all100checked'] and q['all_original_labels_initstates_proprio_values_preserved']
def job_qc(directory,seed,route,c):
 from omegaconf import OmegaConf
 cfg=OmegaConf.to_container(OmegaConf.load(directory/'.hydra/config.yaml'),resolve=True)
 required={'num_demos':100,'eval_num_traj':25,'epochs':100,'eval_frequency':5,'save_frequency':10,'env':'pen-v0','seed':seed,'algorithm':'BC','camera':'vil_camera'}
 for k,v in required.items():assert cfg[k]==v,(k,cfg[k],v)
 assert cfg['embedding']==c['embedding'][route] and Path(cfg['data_dir']).resolve()==Path(c['data_dir'][route]).resolve()
 kw=cfg['env_kwargs'];assert kw['suite']=='adroit' and kw['history_window']==3 and kw['add_proprio'] and kw['image_width']==kw['image_height']==256
 bc=cfg['bc_kwargs'];assert bc['loss_type']=='MSE' and bc['batch_size']==256 and bc['lr']==.001 and bc['dropout']==0 and bc['nonlinearity']=='relu' and str(bc['hidden_sizes']).replace(' ','')=='(256,256,256)'
 rows=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines() if line.strip()]
 losses=[r for r in rows if 'epoch_loss' in r];assert len(losses)==100 and [r['_step'] for r in losses]==list(range(1,101))
 assert all(math.isfinite(r['epoch_loss']) for r in losses),'nonfinite loss'
 evaluations=[r for r in rows if 'eval/epoch' in r];assert [r['eval/epoch'] for r in evaluations]==EVAL_INDICES
 assert all(math.isfinite(r['eval/success']) and 0<=r['eval/success']<=100 for r in evaluations),'invalid native SR'
 policy=directory/cfg['job_name']/'iterations/best_policy.pickle';policy_info={'policy_saved':policy.is_file(),'policy_reload':'UNVERIFIED_NO_ORIGINAL_SAVED_POLICY','policy_dependency':'UNVERIFIED'}
 if policy.is_file():
  import torch
  with policy.open('rb') as f:actor=pickle.load(f)
  assert actor.n==600 and actor.m==24 and all(v.device.type=='cpu' and torch.isfinite(v).all() for v in actor.model.state_dict().values()),'policy CPU reload invalid'
  policy_info.update(best_policy_sha256=sha(policy),policy_reload='PASS_CPU_FINITE_PARAMETERS_AND_BUFFERS',policy_dependency='PASS')
 return dict(qc='PASS',epochs=100,losses=100,evaluation_indices=EVAL_INDICES,native_success=[r['eval/success'] for r in evaluations],policy_selection='unchanged original highest reward; no replacement save',metrics_sha256=sha(directory/'metrics.jsonl'),resolved_config_sha256=sha(directory/'.hydra/config.yaml'),**policy_info)
def main(a):
 c=read(a.config);assert sha(__file__)==c['runner_sha256'];env=dict(os.environ,**read(c['runtime_env']),STABLEWM_HOME=c['base']);os.environ.update(env)
 preflight(c)
 if a.dry_run:print(json.dumps({'qc':'PASS_CPU_PREFLIGHT','scope':'no BC/GPU compute launched','config_sha256':sha(a.config)}));return
 output=Path(c['output_root']);assert not output.exists(),'fresh formal output required; no auto-resume'
 lock=Path(c['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu(c)
 output.mkdir();controller=output/'controller';controller.mkdir();(output/'runs').mkdir()
 state=dict(phase='starting',pid=os.getpid(),started_at=time.time(),updated_at=time.time(),completed=0,failed=0,errors=[],exit_code=None,config_sha256=sha(a.config),jobs=[])
 def save():
  state['updated_at']=time.time();tmp=controller/'state.tmp';tmp.write_text(json.dumps(state,indent=2));os.replace(tmp,controller/'state.json')
 save()
 try:
  here=Path(c['code_root'])/'lewm_pcd/bench/cortexbench'
  for seed in (100,200,300):
   for route in ('rgb','pcd'):
    sources(c);availability=gpu(c);name=f'{route}_s{seed}';directory=output/'runs'/name;assert not directory.exists()
    entry='run_bc.py' if route=='rgb' else 'run_adroit_bc3d.py'
    cmd=[c['python'],str(here/entry),'--config-name','Adroit_BC_config','env=pen-v0','embedding='+c['embedding'][route],f'seed={seed}','data_dir='+c['data_dir'][route],'hydra.run.dir='+str(directory),'hydra.job.chdir=True','wandb.project=local','wandb.entity=local']
    item=dict(name=name,seed=seed,route=route,command=cmd,gpu=availability,started_at=time.time(),phase='running',exit_code=None);state['jobs'].append(item);state['phase']=name;save()
    (controller/(name+'.command.json')).write_text(json.dumps(cmd,indent=2))
    with (controller/(name+'.stdout.log')).open('xb') as log:child=subprocess.Popen(cmd,cwd=here,env=env,stdout=log,stderr=subprocess.STDOUT)
    item['pid']=child.pid;save()
    while child.poll() is None:save();time.sleep(10)
    item['exit_code']=child.returncode;(controller/(name+'.exit_code')).write_text(str(child.returncode)+'\n');assert child.returncode==0,(name,child.returncode)
    sources(c);item['qc']=job_qc(directory,seed,route,c);item.update(phase='complete',finished_at=time.time());state['completed']+=1;save()
  state.update(phase='complete',exit_code=0);save();(controller/'final_report.json').write_text(json.dumps(state,indent=2))
 except BaseException as error:
  state.update(phase='failed_stop_dependencies',failed=1,errors=[repr(error)],exit_code=1);save();(controller/'error.log').write_text(traceback.format_exc());raise
 finally:
  (controller/'exit_code').write_text(str(state['exit_code'] if state['exit_code'] is not None else 1)+'\n');lock.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--dry-run',action='store_true');main(p.parse_args())
