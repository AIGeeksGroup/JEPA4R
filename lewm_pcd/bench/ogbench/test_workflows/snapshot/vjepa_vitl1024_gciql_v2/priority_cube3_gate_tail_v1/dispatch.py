"""Single diagnostic tail dispatch only after C2's complete live closure."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,subprocess,time
R=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1');C=R/'vjepa_vitl1024_gciql_v2';H=C/'priority_cube3_gate_tail_v1';T=H/'control/tail';PY=R/'env_runtime/bin/python'
p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
assert not (T/'runtime').exists() and not (T/'dispatch_receipt.json').exists() and not (T/'wrapper_exit_code').exists()
plan=json.loads((T/'launch_plan.json').read_text())
for path,h in plan['source_pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
probe=subprocess.run([str(PY),'-B',str(H/'runner.py')],cwd=R,capture_output=True,text=True);assert probe.returncode==0,probe.stderr
spec=importlib.util.spec_from_file_location('_space_only',C/'prepare.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);space=m.space_admission()
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True);assert gpu.returncode==0 and not gpu.stdout.strip(),'GPU occupied; no preemption'
if not a.execute:print(json.dumps({'prepared_only':True,'NN_started':False,'space':space}));raise SystemExit(0)
argv=[str(PY),'-B',str(H/'runner.py'),'--execute']
with (T/'dispatch.stdout.log').open('xb') as out,(T/'dispatch.stderr.log').open('xb') as err:child=subprocess.Popen(argv,cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
r={'actual_dispatch_exit_code':0,'started':time.time(),'pid_locator_only':child.pid,'argv':argv,'plan_sha256':hashlib.sha256((T/'launch_plan.json').read_bytes()).hexdigest(),'space_admission':space,'queue_completed':False,'backup_advisory':True,'scope':'Cube3 reload diagnostic1EP audit only; no optimizer/formal training/formal SR'}
with (T/'dispatch_receipt.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r,separators=(',',':')))
