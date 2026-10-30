"""Cube4's five original diagnostic model gates, without a copy hard gate."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
R=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1');C=R/'vjepa_vitl1024_gciql_v2';H=C/'priority_cube4_model_gates_v1';T=H/'control/gates';G='vjepa21_vitl1024_cube4';G3='vjepa21_vitl1024_cube3'
CONT=C/'continuation_v1/control/smoke/launch_plan.json';CORE=C/'control/full/launch_plan.json'
CONT_SHA='e44214e1ef74ba4ed519cea4c32b93a0d40d7b95be8d30174fd0a1c039d9f03a';CORE_SHA='258f0e54f240ee9a1710a36394e52fd112e81e44083956f97ca159d0c8ea1eb9'
PILOT_SHA='ed3a8057b81cab2e1397d83e3e280a5d1a9dde9110d59b22c6dbce16c09766b1';FORMAL_SHA='5ad453ab2cdd60b9097c1abb6e3b0bfe5969ecdd49c0a3b66b7f32a62ce2daad';C3_FORMAL_SHA='062a0dfcd77998cba5025700cfc8bd076c596cfd4b7987b98babf7244960f06e'
MANIFEST_SHA='963bcdedfb4be6eb8bfdc09f6790f9e334be684b5dc9050f3235cc96c3c92470'
SUFFIXES=('01_verify_hooks','02_gate','04_reload','05_native_tiny','06_tiny_audit')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stat_contract(p,want):
 st=Path(p).stat();actual=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns,inode=st.st_ino,device=st.st_dev)
 assert actual==want,str(p)
def phases(old):
 out=[]
 for suffix in SUFFIXES:
  name='cont1_'+G+'_'+suffix;found=[x for x in old['phases'] if x['name']==name];assert len(found)==1,name
  x=copy.deepcopy(found[0]);assert str(C/'configs/pilot'/f'{G}.json') in x['argv'] and 'backup_gate.py' not in json.dumps(x)
  x['name']='gate4_'+name;out.append(x)
 assert [x['environment']['CUDA_VISIBLE_DEVICES'] for x in out]==['','0','','0','']
 assert [x['environment']['JAX_PLATFORMS'] for x in out]==['cpu','cuda','cpu','cpu','cpu']
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
def C3_contract(closure,exits,outer,wrapper,status,cpu,report,audit,binding):
 assert closure['actual_exit_code']==0 and closure['completed_phases']==4 and closure['training_updates']==1000000 and closure['SR_episodes']==250
 assert outer==wrapper==0 and status['actual_exit_code']==0
 expected={'priority3_'+G3+'_'+s for s in ('01_train','03_accept','04_native_full','05_native_audit')};assert set(exits)==expected
 for ex in exits.values():assert ex['actual_exit_code']==0 and ex['residual_owned_members']==[]
 assert cpu['actual_exit_code']==0 and cpu['role']==G3 and cpu['full153_state_exact'] and cpu['Adam_count']==1000000 and cpu['network_step']==1000001
 assert report['actual_exit_code']==audit['actual_exit_code']==binding['actual_exit_code']==0 and report['group_id']==audit['group_id']==binding['group_id']==G3 and report['mode']=='full'
 assert report['episodes']==audit['episodes']==report['unique_seeds']==audit['unique_seeds']==250 and audit['actual_step_arrays_checked']
 assert audit['config_sha256']==binding['config_sha256']==C3_FORMAL_SHA
 assert binding['mode']=='full' and binding['gate_smoke'] is False and binding['encoder_family']=='vjepa21_vitl1024' and binding['latent_dim']==1024
def require_C3_closed():
 t=C/'priority_cube3_v1/control/full';r=t/'runtime';names=['priority3_'+G3+'_'+s for s in ('01_train','03_accept','04_native_full','05_native_audit')]
 assert not (r/'failure.json').exists(),'C3 failure evidence must be inspected'
 C3_contract(read(r/'closure.json'),{n:read(r/(n+'_exit.json')) for n in names},int((r/'outer_exit_code').read_text()),int((t/'wrapper_exit_code').read_text()),read(r/'status.json'),read(C/'acceptance'/f'{G3}.json'),read(C/'native_sr'/G3/'report.json'),read(C/'native_sr'/G3/'independent_audit.json'),read(C/'native_sr'/G3/'native_source_binding.json'))
 assert read(C/'native_sr'/G3/'independent_audit.json')['source_binding_sha256']==sha(C/'native_sr'/G3/'native_source_binding.json')
def fresh_outputs():
 paths=[C/'pilot/gates'/G,C/'pilot/gates'/f'{G}_reload.json',C/'pilot/native_sr'/G,C/'pilot/runs'/G,C/'runs'/G,C/'acceptance'/f'{G}.json',C/'native_sr'/G]
 paths += [C/'continuation_v1/control/smoke/runtime'/('cont1_'+G+'_'+s+'_exit.json') for s in SUFFIXES]
 for p in paths:assert not p.exists() and not p.is_symlink(),'Existing Cube4 artifact must be inspected: '+str(p)
def new_runtime(directory):
 # Dispatch logs/receipt can already exist when its child runner reaches here.
 for name in ('runtime','wrapper_exit_code'):
  assert not (directory/name).exists(),'Existing Cube4 dispatch/runtime must not be repeated'
def validate(plan,require_c3=True):
 rel=read(H/'RELEASE.json');assert rel['actual_exit_code']==0 and rel['prepared_only'] and rel['NN_started'] is False and rel['backup_advisory']
 assert sha(CORE)==rel['original_full_plan_sha256']==CORE_SHA and sha(CONT)==rel['original_continuation_plan_sha256']==CONT_SHA
 for path,h in rel['references'].items():assert sha(path)==h,path
 assert plan['phases']==phases(read(CONT)) and plan['cube_counts']==[4] and plan['families']==['vjepa21_vitl1024']
 assert (plan['queue'],plan['mode'],plan['training_updates'],plan['SR_episodes'],plan['diagnostic_optimizer_updates'],plan['diagnostic_tiny_episodes'])==('vj_priority_cube4_model_gates','model_gates',0,0,2,1)
 for path,h in plan['source_pins'].items():assert sha(path)==h,path
 for purpose,want in [('pilot',PILOT_SHA),('formal',FORMAL_SHA)]:
  cp=C/'configs'/purpose/f'{G}.json';assert sha(cp)==want;config_contract(read(cp),purpose)
 if require_c3:require_C3_closed()
 if not (T/'runtime').exists():fresh_outputs()
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--metadata-only',action='store_true');a=p.parse_args();assert not(a.execute and a.metadata_only)
 validate(read(T/'launch_plan.json'),require_c3=not a.metadata_only)
 if not a.execute:print(json.dumps({'prepared_only':True,'NN_started':False,'C3_live_closure_required':not a.metadata_only,'scope':'Cube4 diagnostic five model gates only'}));return 0
 new_runtime(T);sys.path.insert(0,str(R/'foundation_visual_gciql_v1/control'));import serial
 serial.validate_plan=validate;sys.argv=[str(__file__),'--plan-directory',str(T),'--execute'];code=1
 try:code=serial.main();return code
 finally:
  with (T/'wrapper_exit_code').open('x') as f:f.write(str(code)+'\n')
if __name__=='__main__':raise SystemExit(main())
