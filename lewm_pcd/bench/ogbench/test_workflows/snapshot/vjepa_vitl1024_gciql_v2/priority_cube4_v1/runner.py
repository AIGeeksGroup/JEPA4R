"""Cube4's four original formal phases after its five diagnostic gates close."""
import argparse,copy,hashlib,json,math,sys
from pathlib import Path
R=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1');C=R/'vjepa_vitl1024_gciql_v2';H=C/'priority_cube4_v1';T=H/'control/full';G='vjepa21_vitl1024_cube4'
PILOT_SHA='ed3a8057b81cab2e1397d83e3e280a5d1a9dde9110d59b22c6dbce16c09766b1';FORMAL_SHA='5ad453ab2cdd60b9097c1abb6e3b0bfe5969ecdd49c0a3b66b7f32a62ce2daad';MANIFEST_SHA='963bcdedfb4be6eb8bfdc09f6790f9e334be684b5dc9050f3235cc96c3c92470';FULL_SHA='258f0e54f240ee9a1710a36394e52fd112e81e44083956f97ca159d0c8ea1eb9'
GATE_SUFFIXES=('01_verify_hooks','02_gate','04_reload','05_native_tiny','06_tiny_audit')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stat(p):
 s=Path(p).stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino,device=s.st_dev)
def phases(old):
 out=[]
 for suffix in ('01_train','03_accept','04_native_full','05_native_audit'):
  found=[x for x in old['phases'] if x['name']==G+'_'+suffix];assert len(found)==1
  x=copy.deepcopy(found[0]);assert str(C/'configs/formal'/f'{G}.json') in x['argv'] and 'backup_gate.py' not in json.dumps(x);x['name']='priority4_'+x['name'];out.append(x)
 return out
def config_contract(cfg,purpose):
 assert cfg['group_id']==cfg['role']==G and cfg['cube_count']==4 and cfg['purpose']==purpose
 assert cfg['fresh'] and cfg['resume_from'] is None and (cfg['seed'],cfg['batch_size'],cfg['total_updates'],cfg['native_max_episode_steps'])==(0,1024,1000000,1000)
 assert cfg['input_contract']==dict(current_dim=1043,goal_dim=1036,latent_dim=1024,zero_padding=False)
 assert cfg['feature_lookup_workers']==4 and cfg['runtime_encoder_forwards']==cfg['WM_dynamics_calls']==0
 assert cfg['output']==str(C/('pilot/runs' if purpose=='pilot' else 'runs')/G)
 e=cfg['encoder_binding'];assert e['family']=='vjepa21_vitl1024' and e['latent_dim']==1024
 assert e['checkpoint_sha256']=='7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6'
 assert e['spec_sha256']=='319355585e08d686337a6d84a43fab75f64cf4c5e4dd33f041f0201d6f25d951' and e['adapter']['sha256']=='8ac34a03554671ccdbe5cdd63d6a7f8881850a25e31afeb7fbc3321f18766463'
 assert cfg['data_binding']['data_id']=='cube-quadruple-play-v0' and cfg['data_binding']['cube_count']==4
 for split,rows in [('train',5005000),('val',500500)]:
  f=cfg['feature_loader_arguments'][split];d=cfg['data_binding']['splits'][split]
  assert f['cache_dir']==str(R/'foundation_visual_gciql_v1/cache/fp16_bulk_w8_groups'/G/'cache') and f['expected_rows']==rows and f['split']==split
  assert f['manifest_sha256']==MANIFEST_SHA and f['checkpoint_sha256']==e['checkpoint_sha256'] and f['source_npz']==d['path']
  assert Path(f['source_npz']).name==('cube-quadruple-play-v0.npz' if split=='train' else 'cube-quadruple-play-v0-val.npz')
  assert d['raw_rows']==rows and d['observation_shape']==[rows,55] and d['action_shape']==[rows,5]
def model_gates_contract(closure,exits,outer,wrapper,status,gate,reload,tiny,audit,binding,pilot):
 assert closure['actual_exit_code']==0 and closure['completed_phases']==5 and closure['training_updates']==closure['SR_episodes']==0
 assert outer==wrapper==0 and status['actual_exit_code']==0
 assert set(exits)=={'gate4_cont1_'+G+'_'+s for s in GATE_SUFFIXES}
 for x in exits.values():assert x['actual_exit_code']==0 and x['residual_owned_members']==[]
 assert gate['actual_exit_code']==0 and gate['role']==G and (gate['actual_backend'],gate['updates'],gate['batch_size'],gate['network_step'],gate['adam_count'])==('gpu',2,1024,3,2)
 assert gate['saved_state']['leaves']==153 and len(gate['update_metrics'])==2
 assert all(math.isfinite(v) for x in gate['update_metrics'] for v in x.values()) and all(x['grad/norm']>0 for x in gate['update_metrics'])
 assert all(math.isfinite(v) for v in gate['validation'].values()) and gate['encoder_calls']==gate['WM_dynamics_calls']==0
 kernels=gate['first_layer_kernels']
 for name in ('modules_actor/actor_net/Dense_0/kernel','modules_value/value_net/Dense_0/kernel'):assert kernels[name]==[2079,512]
 for name in ('modules_critic/value_net/Dense_0/kernel','modules_target_critic/value_net/Dense_0/kernel'):assert kernels[name]==[2,2084,512]
 assert gate['actual_source_binding']['config']==pilot
 assert reload['actual_exit_code']==0 and reload['role']==G and reload['full153_state_exact'] and reload['next_batch_and_rawIDs_exact'] and reload['finite_actions'] and reload['state']==gate['saved_state']
 assert reload['applied_host_RNG']==['numpy','python','torchCPU'] and reload['backend']=='cpu' and reload['no_updates']
 assert tiny['actual_exit_code']==audit['actual_exit_code']==binding['actual_exit_code']==0 and tiny['group_id']==audit['group_id']==binding['group_id']==G
 assert tiny['episodes']==tiny['unique_seeds']==audit['episodes']==audit['unique_seeds']==1 and tiny['online_encoder'] and not tiny['offline_embedding_cache_used_for_policy']
 assert tiny['encoding_protocol']=='fresh_current_and_goal_each_policy_call_2N' and (tiny['current_dim'],tiny['goal_dim'],tiny['max_episode_steps'])==(1043,1036,4)
 assert tiny['temperature']==0 and tiny['gaussian_noise'] is None and tiny['extra_policy_warmup']==0 and not tiny['synthetic_zero_padding']
 assert audit['actual_step_arrays_checked'] and audit['diagnostic_excluded_from_formal_SR'] and audit['config_sha256']==binding['config_sha256']==PILOT_SHA
 assert binding['mode']=='tiny' and binding['gate_smoke'] is True and binding['encoder_family']=='vjepa21_vitl1024' and binding['latent_dim']==1024 and binding['camera']==pilot['camera']
def require_model_gates():
 t=C/'priority_cube4_model_gates_v1/control/gates';r=t/'runtime';assert not(r/'failure.json').exists()
 names=['gate4_cont1_'+G+'_'+s for s in GATE_SUFFIXES]
 gate=read(C/'pilot/gates'/G/'report.json');reload=read(C/'pilot/gates'/f'{G}_reload.json');tiny=read(C/'pilot/native_sr'/G/'report.json');audit=read(C/'pilot/native_sr'/G/'independent_audit.json');bp=C/'pilot/native_sr'/G/'native_source_binding.json'
 model_gates_contract(read(r/'closure.json'),{n:read(r/(n+'_exit.json')) for n in names},int((r/'outer_exit_code').read_text()),int((t/'wrapper_exit_code').read_text()),read(r/'status.json'),gate,reload,tiny,audit,read(bp),read(C/'configs/pilot'/f'{G}.json'))
 assert audit['source_binding_sha256']==sha(bp)
 return t,names
def fresh_outputs(home,acceptance,native):
 assert not home.exists() and not home.is_symlink(),'Existing entire formal home must be inspected, not overwritten'
 for p in (acceptance,native):assert not p.exists() and not p.is_symlink(),'Existing formal result must be inspected, not repeated'
def new_runtime(directory):
 for name in ('runtime','wrapper_exit_code'):assert not(directory/name).exists(),'Existing queue must not be started again'
def validate(plan):
 oldp=C/'control/full/launch_plan.json';old=read(oldp);rel=read(H/'GROUP_RELEASE.json')
 assert rel['actual_exit_code']==0 and rel['scope']=='Cube4 fresh1M and formalNative250 only' and sha(oldp)==rel['original_full_plan_sha256']==FULL_SHA
 for path,h in rel['references'].items():assert sha(path)==h,path
 for key in ('checkpoint_payload_stat_no_rehash','source_npz_stat_no_rehash'):
  for path,want in rel[key].items():assert stat(path)==want,path
 assert plan['phases']==phases(old) and plan['cube_counts']==[4] and plan['families']==['vjepa21_vitl1024']
 assert (plan['queue'],plan['mode'],plan['training_updates'],plan['SR_episodes'])==('vj_priority_cube4','full',1000000,250)
 for path,h in plan['source_pins'].items():assert sha(path)==h,path
 for purpose,h in [('pilot',PILOT_SHA),('formal',FORMAL_SHA)]:
  cp=C/'configs'/purpose/f'{G}.json';assert sha(cp)==h;config_contract(read(cp),purpose)
 require_model_gates()
 if not(T/'runtime').exists():fresh_outputs(C/'runs'/G,C/'acceptance'/f'{G}.json',C/'native_sr'/G)
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args();validate(read(T/'launch_plan.json'))
 if not a.execute:print(json.dumps({'prepared_only':True,'NN_started':False,'algorithm':'GCIQL','group':G}));return 0
 new_runtime(T);sys.path.insert(0,str(R/'foundation_visual_gciql_v1/control'));import serial
 serial.validate_plan=validate;sys.argv=[str(__file__),'--plan-directory',str(T),'--execute'];code=1
 try:code=serial.main();return code
 finally:
  with (T/'wrapper_exit_code').open('x') as f:f.write(str(code)+'\n')
if __name__=='__main__':raise SystemExit(main())
