"""One approved Cube1 formal training phase; original learning code unchanged."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
R=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1');C=R/'vjepa_vitl1024_gciql_v2';H=C/'priority_cube1_v1';G='vjepa21_vitl1024_cube1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate(plan):
 old=read(C/'control/full/launch_plan.json');release=read(H/'GROUP_RELEASE.json')
 assert release['actual_exit_code']==0 and release['scope']=='Cube1 fresh formal train only'
 assert sha(C/'control/full/launch_plan.json')==release['original_full_plan_sha256']
 for p,h in release['references'].items():assert sha(p)==h,p
 expected=copy.deepcopy(old['phases'][0]);expected['name']='priority1_'+expected['name']
 assert plan['phases']==[expected] and plan['cube_counts']==[1] and plan['families']==['vjepa21_vitl1024']
 assert (plan['training_updates'],plan['SR_episodes'])==(1000000,0)
 cfg=read(C/'configs/formal'/f'{G}.json')
 assert cfg['purpose']=='formal' and cfg['resume_from'] is None and cfg['seed']==0 and cfg['batch_size']==1024 and cfg['total_updates']==1000000
 assert cfg['input_contract']==dict(current_dim=1043,goal_dim=1027,latent_dim=1024,zero_padding=False)
 if not (H/'control/train/runtime').exists():assert not (Path(cfg['output'])/'status.json').exists(),'Existing formal trainer must not be duplicated'
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args();T=H/'control/train';validate(read(T/'launch_plan.json'))
 if not a.execute:print(json.dumps({'prepared_only':True,'NN_started':False,'algorithm':'GCIQL','group':G}));return 0
 assert not (T/'runtime').exists() and not (T/'wrapper_exit_code').exists()
 sys.path.insert(0,str(R/'foundation_visual_gciql_v1/control'));import serial
 serial.validate_plan=validate
 sys.argv=[str(__file__),'--plan-directory',str(T),'--execute'];code=1
 try:code=serial.main();return code
 finally:
  with (T/'wrapper_exit_code').open('x') as f:f.write(str(code)+'\n')
if __name__=='__main__':raise SystemExit(main())
