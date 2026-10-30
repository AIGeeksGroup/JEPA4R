"""Relay only Cube1's original accept/Native/audit after its training closes."""
import argparse,copy,csv,hashlib,io,json,math,sys
from pathlib import Path
R=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
C=R/'vjepa_vitl1024_gciql_v2';H=C/'priority_cube1_post_train_v1';G='vjepa21_vitl1024_cube1'
T=H/'control/post_train';A=C/'priority_cube1_v1/control/train';TRAIN_NAME='priority1_'+G+'_01_train'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def phases(old):
 out=copy.deepcopy(old['phases'][2:5])
 for x in out:x['name']='post1_'+x['name']
 return out
def train_closed():
 runtime=A/'runtime'
 for p in (runtime/'outer_exit_code',A/'wrapper_exit_code'):assert int(p.read_text())==0,str(p)
 ex=read(runtime/(TRAIN_NAME+'_exit.json'));assert ex['actual_exit_code']==0 and ex['residual_owned_members']==[]
 cl=read(runtime/'closure.json');assert cl['actual_exit_code']==0 and cl['completed_phases']==1 and cl['training_updates']==1000000 and cl['SR_episodes']==0
 assert not (runtime/'failure.json').exists(),'Current train has failure evidence'
 s=read(C/'runs'/G/'status.json');assert s['completed_updates']==1000000 and s['network_step']==1000001 and s['exit_code']==0 and s['failed_stages']==0
 rows=list(csv.DictReader(io.StringIO((C/'runs'/G/'train.csv').read_text())))
 assert [int(x['step']) for x in rows]==list(range(5000,1000001,5000))
 for row in rows:
  assert all(math.isfinite(float(v)) for v in row.values())
  assert float(row['training/grad/norm'])>0
 cp=C/'runs'/G/'checkpoints';steps=[1]+list(range(100000,1000001,100000))
 assert sorted(x.name for x in cp.glob('update_*'))==[f'update_{x:09d}' for x in steps]
 for n in steps:
  p=cp/f'update_{n:09d}';m=read(p/'manifest.json');assert m['counter']==n
  assert len(m['agent_sha256'])==len(m['sidecar_sha256'])==64
  assert (p/f'params_{n}.pkl').is_file() and (p/'run_state.pkl').is_file()
def validate(plan,require_train=True):
 old_path=C/'control/full/launch_plan.json';old=read(old_path);rel=read(H/'RELEASE.json')
 assert rel['actual_exit_code']==0 and rel['scope']=='Cube1 post_train accept Native250 audit only'
 assert sha(old_path)==rel['original_full_plan_sha256']
 prior=C/'priority_cube1_v1/GROUP_RELEASE.json';assert sha(prior)==rel['training_release_sha256']
 prior_rel=read(prior);assert prior_rel['actual_exit_code']==0
 for p,h in prior_rel['references'].items():assert sha(p)==h,p
 assert plan['phases']==phases(old) and plan['cube_counts']==[1] and plan['families']==['vjepa21_vitl1024']
 assert (plan['queue'],plan['mode'],plan['training_updates'],plan['SR_episodes'])==('vj_priority_cube1_post_train','post_train',0,250)
 for p,h in plan['source_pins'].items():assert sha(p)==h,p
 cfg=read(C/'configs/formal'/f'{G}.json');assert cfg['purpose']=='formal' and cfg['native_max_episode_steps']==200
 if require_train:train_closed()
 if not (T/'runtime').exists():
  assert not (C/'acceptance'/f'{G}.json').exists(),'Existing acceptance must be inspected/adopted, not overwritten'
  assert not (C/'native_sr'/G).exists(),'Existing Native output must be inspected/adopted, not repeated'
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--metadata-only',action='store_true');a=p.parse_args()
 assert not (a.execute and a.metadata_only)
 validate(read(T/'launch_plan.json'),require_train=not a.metadata_only)
 if not a.execute:print(json.dumps({'prepared_only':True,'training_closed_required':not a.metadata_only,'NN_started':False}));return 0
 assert not (T/'runtime').exists() and not (T/'wrapper_exit_code').exists()
 sys.path.insert(0,str(R/'foundation_visual_gciql_v1/control'));import serial
 serial.validate_plan=validate
 sys.argv=[str(__file__),'--plan-directory',str(T),'--execute'];code=1
 try:code=serial.main();return code
 finally:
  with (T/'wrapper_exit_code').open('x') as f:f.write(str(code)+'\n')
if __name__=='__main__':raise SystemExit(main())
