"""Persistent metadata waiter and sequential handoff; no ML imports or retries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from barrier import read,sha,require_smoke,validate_plan,_require_adoption
from build_queues import CASE,CONTROL,PY,ROOT,FAMILIES

RUN=CONTROL/'bootstrap_runtime'

def save(name,value):
    path=RUN/name;tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)

def phase(name,argv):
    save('status.json',dict(stage=name,updated=time.time(),actual_exit_code=None))
    save(name+'_argv.json',dict(argv=argv,cwd=str(ROOT),started=time.time()))
    with (RUN/(name+'.log')).open('x') as log:
        child=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,
            PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
            LD_LIBRARY_PATH=str(ROOT/'runtime/lib')+':/usr/lib/x86_64-linux-gnu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        code=child.wait()
    save(name+'_exit.json',dict(actual_exit_code=code,completed=time.time()))
    assert code==0,f'{name} failed with actual exit {code}; no automatic restart'

def require_assets(release,random):
    kind='random_assets' if random else 'official_assets';evidence=release[kind]
    assert set(evidence)==set(release['active_families'])
    for family,record in evidence.items():
        specpath=Path(record['spec']);assert sha(specpath)==record['spec_sha256']
        spec=read(specpath)
        assert (spec['family'],spec['output_dim'])==(family,FAMILIES[family])
        assert spec.get('diagnostic_only',False) is random
        if random:assert spec['initialization']=='random' and spec['official_pretrained'] is False
        for path in record['actual_exit_files']:assert Path(path).read_text().strip()=='0',path
        for report in record['reports']:
            assert sha(report['path'])==report['sha256']
            content=read(report['path']);assert content[report['exit_field']]==0
        assert Path(spec['checkpoint']).stat().st_size==record['checkpoint_bytes']
    save(kind+'_readback.json',dict(actual_exit_code=0,evidence=evidence,at=time.time(),NN_forward_performed=False))

def main():
    global RUN
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--runtime-directory',type=Path,default=RUN)
    parser.add_argument('--random-recovery-directory',type=Path);args=parser.parse_args()
    RUN=args.runtime_directory
    RUN.mkdir(exist_ok=False)
    code=1
    try:
        release=read(args.release);registry=read(CONTROL/'registry.json')
        active=release['active_families'];assert active==registry['active_families']
        assert all(registry['families'][family]['enabled'] for family in active)
        for path,digest in release['source_pins'].items():assert sha(path)==digest,path
        save('dispatch.json',dict(pid=os.getpid(),argv=sys.argv,release_sha256=sha(args.release),started=time.time()))
        require_assets(release,True)
        if args.random_recovery_directory:
            directory=args.random_recovery_directory
            recovery=read(directory/'launch_plan.json');validate_plan(recovery);_require_adoption(recovery)
            phase('00c_remaining_random_smokes',[PY,'-B',str(CONTROL/'serial.py'),'--plan-directory',str(directory),'--execute'])
            random_barrier=require_smoke(directory,read(directory/'launch_plan.json'))
            save('random_diagnostic_acceptance.json',dict(**random_barrier,at=time.time(),formal_training=False,formal_SR=False))
        else:
            adopted={}
            for family in active:
                random_spec=CASE/'encoder/specs'/f'random_diagnostic_{family}.json'
                paths=[CASE/'cache/groups'/f'random_init_v1_{family}_cube{n}'/'plan.json' for n in range(1,5)]
                if not any(path.exists() for path in paths):
                    phase('00a_materialize_random_'+family,[PY,'-B',str(CASE/'cache/prepare_plans.py'),'--family',family,
                        '--spec',str(random_spec),'--group-prefix','random_init_v1_','--pilot-only'])
                assert all(path.exists() for path in paths),'Partial random plans; no automatic overwrite'
                for n,path in enumerate(paths,1):
                    plan=read(path)
                    assert plan['diagnostic_only'] is True and plan['pilot_only'] is True
                    assert plan['cube_count']==n and plan['latent_dim']==FAMILIES[family]
                    assert plan['encoder_spec_sha256']==sha(random_spec)
                    for relative,digest in plan['source_pins'].items():assert sha(ROOT/relative)==digest,relative
                    adopted[str(path)]=sha(path)
            save('00a_existing_random_plans_readback.json',dict(actual_exit_code=0,plans=adopted,at=time.time(),producer_started=False))
            phase('00b_random_queues',[PY,'-B',str(CONTROL/'build_queues.py'),'--release',str(args.release),'--diagnostic-random'])
            phase('00c_random_smokes',[PY,'-B',str(CONTROL/'serial.py'),'--plan-directory',str(CONTROL/'random_smoke'),'--execute'])
            random_barrier=require_smoke(CONTROL/'random_smoke',read(CONTROL/'random_smoke/launch_plan.json'))
            save('random_diagnostic_acceptance.json',dict(**random_barrier,at=time.time(),formal_training=False,formal_SR=False))
        require_assets(release,False)
        for family in active:
            phase('01_materialize_'+family,[PY,'-B',str(CASE/'cache/prepare_plans.py'),'--family',family])
        phase('02_build_queues',[PY,'-B',str(CONTROL/'build_queues.py'),'--release',str(args.release)])
        phase('03_all_active_smokes',[PY,'-B',str(CONTROL/'serial.py'),'--plan-directory',str(CONTROL/'smoke'),'--execute'])
        barrier=require_smoke(CONTROL/'smoke',read(CONTROL/'formal/launch_plan.json'))
        save('formal_release_barrier.json',dict(**barrier,at=time.time()))
        phase('04_formal_serial',[PY,'-B',str(CONTROL/'serial.py'),'--plan-directory',str(CONTROL/'formal'),'--execute'])
        closure=read(CONTROL/'formal/runtime/closure.json')
        assert closure['actual_exit_code']==0 and closure['completed_phases']==32*len(active)
        assert (CONTROL/'formal/runtime/outer_exit_code').read_text().strip()=='0'
        save('status.json',dict(stage='all_active_groups_train_and_final_SR_completed_backup_pending',actual_exit_code=0,
             completed=time.time(),families=active,training_updates=4000000*len(active),SR_episodes=1000*len(active)))
        code=0
    except BaseException:
        save('failure.json',dict(actual_exit_code=1,error=traceback.format_exc(),at=time.time(),automatic_restart=False))
        save('status.json',dict(stage='failed',actual_exit_code=1,updated=time.time()))
        raise
    finally:
        with (RUN/'outer_exit_code').open('x') as out:out.write(str(code)+'\n')

if __name__=='__main__':main()
