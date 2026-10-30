"""Only Cube2's frozen train/accept/Native/audit phases; original learning code unchanged."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
R=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1');C=R/'vjepa_vitl1024_gciql_v2';H=C/'priority_cube2_v1';T=H/'control/full';G='vjepa21_vitl1024_cube2'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def phases(old):
 out=[copy.deepcopy(old['phases'][i]) for i in (5,7,8,9)]
 for x in out:x['name']='priority2_'+x['name']
 return out
def config_contract(cfg):
 assert cfg['group_id']==cfg['role']==G and cfg['cube_count']==2 and cfg['purpose']=='formal'
 assert cfg['resume_from'] is None and cfg['fresh'] and (cfg['seed'],cfg['batch_size'],cfg['total_updates'],cfg['native_max_episode_steps'])==(0,1024,1000000,500)
 assert cfg['input_contract']==dict(current_dim=1043,goal_dim=1030,latent_dim=1024,zero_padding=False)
 e=cfg['encoder_binding'];assert e['family']=='vjepa21_vitl1024' and e['latent_dim']==1024 and e['checkpoint_sha256']=='7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6'
 assert e['spec_sha256']=='319355585e08d686337a6d84a43fab75f64cf4c5e4dd33f041f0201d6f25d951' and e['adapter']['sha256']=='8ac34a03554671ccdbe5cdd63d6a7f8881850a25e31afeb7fbc3321f18766463'
 assert cfg['data_binding']['data_id']=='cube-double-play-v0' and cfg['data_binding']['cube_count']==2
 for split,rows in [('train',1001000),('val',100100)]:
  f=cfg['feature_loader_arguments'][split]
  assert f['cache_dir']==str(R/'foundation_visual_gciql_v1/cache/fp16_bulk_w8_groups'/G/'cache') and f['split']==split and f['expected_rows']==rows
  assert f['manifest_sha256']=='ac32c5a0bf18f6722a47bd7b1ee62ed0381645286f4fa2b9efa65dce79870cd1' and f['checkpoint_sha256']==e['checkpoint_sha256']
  assert f['source_npz']==cfg['data_binding']['splits'][split]['path'] and Path(f['source_npz']).name==('cube-double-play-v0.npz' if split=='train' else 'cube-double-play-v0-val.npz')
 assert cfg['feature_lookup_workers']==4 and cfg['runtime_encoder_forwards']==cfg['WM_dynamics_calls']==0
 assert cfg['output']==str(C/'runs'/G)
def fresh_outputs(home,acceptance,native):
 assert all(not (home/n).exists() for n in ('status.json','resolved.json','train.csv')),'Existing formal trainer must be inspected, not repeated'
 assert not list((home/'checkpoints').glob('update_*')),'Existing checkpoints must not be overwritten'
 assert not acceptance.exists() and not native.exists(),'Existing downstream results must be inspected, not repeated'
def new_runtime(directory):
 assert not (directory/'runtime').exists() and not (directory/'wrapper_exit_code').exists(),'Existing queue must not be started again'
def validate(plan):
 oldp=C/'control/full/launch_plan.json';old=read(oldp);rel=read(H/'GROUP_RELEASE.json')
 assert rel['actual_exit_code']==0 and rel['scope']=='Cube2 fresh1M and formalNative250 only' and sha(oldp)==rel['original_full_plan_sha256']
 for path,h in rel['references'].items():assert sha(path)==h,path
 assert plan['phases']==phases(old) and plan['cube_counts']==[2] and plan['families']==['vjepa21_vitl1024']
 assert (plan['queue'],plan['mode'],plan['training_updates'],plan['SR_episodes'])==('vj_priority_cube2','full',1000000,250)
 for path,h in plan['source_pins'].items():assert sha(path)==h,path
 config_contract(read(C/'configs/formal'/f'{G}.json'))
 if not (T/'runtime').exists():fresh_outputs(C/'runs'/G,C/'acceptance'/f'{G}.json',C/'native_sr'/G)
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args();validate(read(T/'launch_plan.json'))
 if not a.execute:print(json.dumps({'prepared_only':True,'NN_started':False,'algorithm':'GCIQL','group':G}));return 0
 new_runtime(T);sys.path.insert(0,str(R/'foundation_visual_gciql_v1/control'));import serial
 serial.validate_plan=validate;sys.argv=[str(__file__),'--plan-directory',str(T),'--execute'];code=1
 try:code=serial.main();return code
 finally:
  with (T/'wrapper_exit_code').open('x') as f:f.write(str(code)+'\n')
if __name__=='__main__':raise SystemExit(main())
