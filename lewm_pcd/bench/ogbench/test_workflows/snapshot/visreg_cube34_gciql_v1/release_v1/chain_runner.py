"""Sequence four separate stage runtimes; no encoder/training implementation."""
import argparse
import subprocess
import time
import traceback
import common as m

def validate_chain():
    chain=m.read(m.CASE/'CHAIN.json');prepared=m.read(m.CASE/'PREPARED.json')
    assert prepared['actual_exit_code']==0 and prepared['NN_started'] is False and chain['stages']==list(m.STAGES)
    assert prepared['chain_sha256']==m.sha(m.CASE/'CHAIN.json') and chain['release_sha256']==m.sha(m.CASE/'RELEASE.json')
    assert chain['backup_advisory'] and chain['automatic_restart'] is False
    m.pins_contract(chain['stage_plans']);m.pins_contract(prepared['source_pins'])
    return chain

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args();chain=validate_chain()
    runtime=m.CASE/'control/chain/runtime'
    assert not runtime.exists(),'Existing chain must be inspected; no automatic restart'
    for stage in m.STAGES:m.fresh_outputs(stage)
    m.validate_plan(m.read(m.stage_dir(m.STAGES[0])/'launch_plan.json'),m.STAGES[0])
    space=m.space_admission()
    if not a.execute:
        print(m.json.dumps(dict(actual_exit_code=0,prepared_only=True,NN_started=False,space=space,stages=list(m.STAGES))));return 0
    runtime.mkdir(parents=True);code=1
    m.write_new(runtime/'actual_dispatch.json',dict(at=time.time(),chain_sha256=m.sha(m.CASE/'CHAIN.json'),stages=list(m.STAGES)))
    try:
        accepted=[]
        for stage in m.STAGES:
            m.atomic(runtime/'status.json',dict(stage=stage,started=time.time(),updated=time.time(),completed_stages=len(accepted),failed_stages=0,actual_exit_code=None))
            argv=[str(m.PY),'-B',str(m.H/'stage_runner.py'),'--stage',stage,'--execute']
            with (runtime/(stage+'.stdout.log')).open('x') as out,(runtime/(stage+'.stderr.log')).open('x') as err:
                result=subprocess.run(argv,cwd=m.ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            m.write_new(runtime/(stage+'_exit.json'),dict(actual_exit_code=result.returncode,argv=argv,completed=time.time()))
            assert result.returncode==0,stage+' failed; no automatic restart'
            accepted.append(m.stage_closed(stage))
            m.atomic(runtime/'status.json',dict(stage=stage+'_accepted',updated=time.time(),completed_stages=len(accepted),failed_stages=0,actual_exit_code=None))
        m.write_new(runtime/'closure.json',dict(actual_exit_code=0,completed=time.time(),stages=accepted,formal_updates=2000000,formal_SR_episodes=500,backup_advisory=True))
        m.atomic(runtime/'status.json',dict(stage='complete',updated=time.time(),completed_stages=4,failed_stages=0,actual_exit_code=0));code=0;return 0
    except BaseException:
        m.write_new(runtime/'failure.json',dict(actual_exit_code=1,error=traceback.format_exc(),at=time.time(),automatic_restart=False))
        m.atomic(runtime/'status.json',dict(stage='failed',updated=time.time(),completed_stages=len(accepted),failed_stages=1,actual_exit_code=1));raise
    finally:
        for name in ('outer_exit_code','wrapper_exit_code'):
            with (runtime/name).open('x') as f:f.write(str(code)+'\n')

if __name__=='__main__':raise SystemExit(main())
