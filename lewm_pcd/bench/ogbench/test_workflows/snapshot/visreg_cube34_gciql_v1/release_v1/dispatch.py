"""Root-only one-time dispatch after reviewed metadata publication."""
import argparse
import subprocess
import time
import common as m
from chain_runner import validate_chain

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args();validate_chain()
    t=m.CASE/'control/chain'
    for name in ('runtime','dispatch_receipt.json','dispatch.stdout.log','dispatch.stderr.log'):
        assert not (t/name).exists(),'Existing dispatch must be inspected, not repeated: '+name
    for stage in m.STAGES:m.fresh_outputs(stage)
    probe=subprocess.run([str(m.PY),'-B',str(m.H/'chain_runner.py')],cwd=m.ROOT,capture_output=True,text=True)
    assert probe.returncode==0,probe.stderr
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
    assert gpu.returncode==0 and not gpu.stdout.strip(),'GPU occupied; no preemption'
    space=m.space_admission()
    if not a.execute:
        print(m.json.dumps(dict(actual_exit_code=0,prepared_only=True,NN_started=False,space=space,GPU_compute_apps_empty=True)));return 0
    argv=[str(m.PY),'-B',str(m.H/'chain_runner.py'),'--execute']
    t.mkdir(parents=True,exist_ok=True)
    with (t/'dispatch.stdout.log').open('x') as out,(t/'dispatch.stderr.log').open('x') as err:
        child=subprocess.Popen(argv,cwd=m.ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    r=dict(actual_dispatch_exit_code=0,started=time.time(),pid_locator_only=child.pid,argv=argv,
           chain_sha256=m.sha(m.CASE/'CHAIN.json'),space=space,backup_advisory=True,queue_completed=False,
           scope='VISReg Cube3 gates/full, then Cube4 gates/full only',formal_updates=2000000,formal_SR_episodes=500)
    m.write_new(t/'dispatch_receipt.json',r);print(m.json.dumps(r));return 0

if __name__=='__main__':raise SystemExit(main())
