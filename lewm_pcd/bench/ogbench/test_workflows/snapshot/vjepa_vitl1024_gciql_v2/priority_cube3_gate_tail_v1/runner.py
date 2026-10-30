"""Only the remaining Cube3 CPU reload and diagnostic Native/audit phases."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
R=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1');C=R/'vjepa_vitl1024_gciql_v2';H=C/'priority_cube3_gate_tail_v1';T=H/'control/tail';G='vjepa21_vitl1024_cube3';G2='vjepa21_vitl1024_cube2';CONT=C/'continuation_v1/control/smoke/launch_plan.json'
PILOT_SHA='516ffea7f7276eb9d5d36be7e7dd3ff36789b16f991e337b107d1505beeef01d';FORMAL_SHA='062a0dfcd77998cba5025700cfc8bd076c596cfd4b7987b98babf7244960f06e';CONT_SHA='e44214e1ef74ba4ed519cea4c32b93a0d40d7b95be8d30174fd0a1c039d9f03a'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def phases(old):
 out=[]
 for suffix in ('04_reload','05_native_tiny','06_tiny_audit'):
  name='cont1_'+G+'_'+suffix;found=[x for x in old['phases'] if x['name']==name];assert len(found)==1
  x=copy.deepcopy(found[0]);assert G in json.dumps(x) and 'vjepa21_vitl1024_cube4' not in json.dumps(x);x['name']='tail3_'+x['name'];out.append(x)
 return out
def config_contract(cfg,purpose):
 assert cfg['group_id']==cfg['role']==G and cfg['cube_count']==3 and cfg['purpose']==purpose
 assert cfg['input_contract']==dict(current_dim=1043,goal_dim=1033,latent_dim=1024,zero_padding=False) and cfg['feature_lookup_workers']==4
 e=cfg['encoder_binding'];assert e['family']=='vjepa21_vitl1024' and e['latent_dim']==1024 and e['checkpoint_sha256']=='7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6' and e['spec_sha256']=='319355585e08d686337a6d84a43fab75f64cf4c5e4dd33f041f0201d6f25d951'
 assert cfg['data_binding']['data_id']=='cube-triple-play-v0' and cfg['data_binding']['cube_count']==3
 for split,rows in [('train',3003000),('val',300300)]:
  f=cfg['feature_loader_arguments'][split];assert f['cache_dir']==str(R/'foundation_visual_gciql_v1/cache/fp16_bulk_w8_groups'/G/'cache') and f['expected_rows']==rows and f['split']==split
  assert f['manifest_sha256']=='74ece0c101f3500761d4f9cfabd2f855b3ed18e34e65192eb6f94b2d422ec6ee' and f['checkpoint_sha256']==e['checkpoint_sha256'] and f['source_npz']==cfg['data_binding']['splits'][split]['path']
  assert Path(f['source_npz']).name==('cube-triple-play-v0.npz' if split=='train' else 'cube-triple-play-v0-val.npz')
def checkpoint_contract(manifest,counter):
 assert manifest['counter']==counter and len(manifest['agent_sha256'])==len(manifest['sidecar_sha256'])==64
 for key in ('agent_sha256','sidecar_sha256'):assert all(ch in '0123456789abcdef' for ch in manifest[key])
def C2_contract(closure,exits,outer,wrapper,status,cpu,report,audit,binding):
 assert isinstance(closure,dict) and closure['actual_exit_code']==0 and closure['completed_phases']==4 and closure['training_updates']==1000000 and closure['SR_episodes']==250
 assert outer==wrapper==0 and status['actual_exit_code']==0
 expected={'priority2_'+G2+'_'+s for s in ('01_train','03_accept','04_native_full','05_native_audit')};assert set(exits)==expected
 for ex in exits.values():assert ex['actual_exit_code']==0 and ex['residual_owned_members']==[]
 assert cpu['actual_exit_code']==0 and cpu['role']==G2 and cpu['full153_state_exact'] and cpu['Adam_count']==1000000 and cpu['network_step']==1000001
 assert report['actual_exit_code']==audit['actual_exit_code']==binding['actual_exit_code']==0 and report['group_id']==audit['group_id']==binding['group_id']==G2 and report['mode']=='full'
 assert report['episodes']==audit['episodes']==report['unique_seeds']==audit['unique_seeds']==250 and audit['actual_step_arrays_checked']
 assert audit['config_sha256']==binding['config_sha256']=='ed40b0962ba45657bb386d7ed61bee087b53461267992bc5c1491628f9e6ffc9'
 assert binding['mode']=='full' and binding['gate_smoke'] is False and binding['encoder_family']=='vjepa21_vitl1024' and binding['latent_dim']==1024
def require_C2_closed():
 t=C/'priority_cube2_v1/control/full';r=t/'runtime';names=['priority2_'+G2+'_'+s for s in ('01_train','03_accept','04_native_full','05_native_audit')]
 assert not (r/'failure.json').exists(),'Current C2 queue has failure evidence'
 C2_contract(read(r/'closure.json'),{n:read(r/(n+'_exit.json')) for n in names},int((r/'outer_exit_code').read_text()),int((t/'wrapper_exit_code').read_text()),read(r/'status.json'),read(C/'acceptance'/f'{G2}.json'),read(C/'native_sr'/G2/'report.json'),read(C/'native_sr'/G2/'independent_audit.json'),read(C/'native_sr'/G2/'native_source_binding.json'))
def fresh_outputs(reload_report,native):
 assert not reload_report.exists() and not native.exists(),'Existing Cube3 reload/Native output must be inspected/adopted, not overwritten'
def new_runtime(directory):
 assert not (directory/'runtime').exists() and not (directory/'wrapper_exit_code').exists(),'Existing tail queue must not be started again'
def validate(plan,require_c2=True):
 rel=read(H/'RELEASE.json');core=C/'control/full/launch_plan.json'
 assert rel['actual_exit_code']==0 and rel['prepared_only'] and rel['C2_closed_claimed_at_publication'] is False and rel['dispatch_requires_live_C2_closure'] is True
 assert sha(core)==rel['original_full_plan_sha256'] and sha(CONT)==rel['original_continuation_plan_sha256']==CONT_SHA
 for path,h in rel['references'].items():assert sha(path)==h,path
 for path,want in rel['checkpoint_payload_stat_no_rehash'].items():
  st=Path(path).stat();assert dict(size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns,inode=st.st_ino,device=st.st_dev)==want,path
 assert plan['phases']==phases(read(CONT)) and plan['cube_counts']==[3] and plan['families']==['vjepa21_vitl1024']
 assert (plan['queue'],plan['mode'],plan['training_updates'],plan['SR_episodes'],plan['diagnostic_optimizer_updates'],plan['diagnostic_tiny_episodes'])==('vj_priority_cube3_gate_tail','gate_tail',0,0,0,1)
 for path,h in plan['source_pins'].items():assert sha(path)==h,path
 for purpose,want in [('pilot',PILOT_SHA),('formal',FORMAL_SHA)]:
  cp=C/'configs'/purpose/f'{G}.json';assert sha(cp)==want;config_contract(read(cp),purpose)
 if require_c2:require_C2_closed()
 if not (T/'runtime').exists():fresh_outputs(C/'pilot/gates'/f'{G}_reload.json',C/'pilot/native_sr'/G)
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--metadata-only',action='store_true');a=p.parse_args();assert not (a.execute and a.metadata_only)
 validate(read(T/'launch_plan.json'),require_c2=not a.metadata_only)
 if not a.execute:print(json.dumps({'prepared_only':True,'NN_started':False,'C2_live_closure_required':not a.metadata_only,'scope':'Cube3 remaining diagnostic gates only'}));return 0
 new_runtime(T);sys.path.insert(0,str(R/'foundation_visual_gciql_v1/control'));import serial
 serial.validate_plan=validate;sys.argv=[str(__file__),'--plan-directory',str(T),'--execute'];code=1
 try:code=serial.main();return code
 finally:
  with (T/'wrapper_exit_code').open('x') as f:f.write(str(code)+'\n')
if __name__=='__main__':raise SystemExit(main())
