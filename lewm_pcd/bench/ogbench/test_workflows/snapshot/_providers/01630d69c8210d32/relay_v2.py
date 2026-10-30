"""Thin serial original launch + frozen CPU acceptance; Native argv supplied only by its owner."""
import argparse,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY='/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python'
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
load=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))
RUN=None

def validate_entry(item):
    argv=item['argv']; cfg=item['config']
    assert cfg['goal_mode']=='latent_object_xyz' and cfg['batch_size']==1024 and cfg['total_updates']==1000000
    assert cfg['save_interval']==100000 and cfg['validation_interval']==5000 and cfg['seed']==0 and cfg['workers'] in (2,4) and cfg['evaluation_episodes']==0
    assert cfg['output']==item['output'] and '--resume' not in argv and '--execute' in argv
    for flag,key in [('--config','config_path'),('--output','output'),('--launcher-output','launcher_output'),('--gate-report','gate_report'),('--gate-sha256','gate_sha256')]:
        assert argv[argv.index(flag)+1]==item[key]
    assert argv[1]==item['core_root']+'/launch.py'
    assert len(item['gate_sha256'])==64

def validate_gate(report,item):
    assert report['status']=='passed' and report['exit_code']==0
    assert report['formal_config_sha256']==item['config_sha256'] and report['actual_rl_batch']==1024 and report['train_updates']==2
    assert report['fresh_next_batch_full_state_exact'] and report['source_binding']['config']['goal_mode']=='latent_object_xyz'

def native_ready_pointer(plan):
    direct=plan.get('native_handoff')
    if direct is not None:
        pointer=direct
    else:
        slot=plan.get('native_ready_file')
        if not slot or not Path(slot).is_file(): return None
        pointer=load(slot)
        if not pointer.get('ready'): return None
    assert isinstance(pointer.get('path'),str) and Path(pointer['path']).is_absolute()
    assert isinstance(pointer.get('sha256'),str) and len(pointer['sha256'])==64
    return pointer


def status(stage,**fields):
    data=dict(stage=stage,updated_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields)
    temporary=RUN/'status.tmp';temporary.write_text(json.dumps(data,indent=2),encoding='utf-8');os.replace(temporary,RUN/'status.json')

def run(stage,argv,env=None):
    directory=RUN/stage;directory.mkdir(exist_ok=False)
    (directory/'actual_argv.json').write_text(json.dumps(argv,indent=2),encoding='utf-8')
    with (directory/'run.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
        status(stage,child_pid=child.pid,argv=argv,log=str(directory/'run.log'),exit_code=None)
        code=child.wait()
    (directory/'actual_exit_code').write_text(str(code)+'\n',encoding='utf-8')
    assert code==0,f'{stage} actual exit {code}; stopped, evidence retained'

def accept(handoff,mode,env):
    for name,expected in handoff['source_sha256'].items():assert sha(name)==expected
    reports=[]
    for stage in handoff['modes'][mode]['stages']:
        run(mode+'_'+stage['name'],stage['argv'],env)
        proof=load(stage['report']);assert proof['status']==stage['status'] and proof['exit_code']==0
        reports.append(dict(path=stage['report'],sha256=sha(stage['report']),status=proof['status']))
    assert len(reports)==2,'Full-state and fresh action evidence required'
    (RUN/(mode+'_acceptance_receipt.json')).write_text(json.dumps(dict(status='accepted_fresh1M_fullstate_and_action',exit_code=0,reports=reports),indent=2),encoding='utf-8')

def empty_GPU():
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip(),'GPU occupied; no process is stopped'

def main(args):
    global RUN
    assert sha(args.plan)==args.plan_sha256
    plan=load(args.plan);assert set(plan['modes'])=={'M0','M6','M5'} and plan['ready']
    for mode,item in plan['modes'].items():
        validate_entry(item);assert sha(item['config_path'])==item['config_sha256'];assert load(item['config_path'])==item['config']
        assert sha(item['gate_report'])==item['gate_sha256'];validate_gate(load(item['gate_report']),item)
        freeze=Path(plan['core_freeze_path'])
        assert sha(freeze)==plan['core_freeze_sha256']
        for file in load(freeze)['files']:assert sha(file['path'])==file['sha256']
    assert sha(plan['afterhook_handoff'])==plan['afterhook_handoff_sha256'];handoff=load(plan['afterhook_handoff'])
    for path,h in handoff['source_sha256'].items():assert sha(path)==h
    for path,h in plan['source_sha256'].items():assert sha(path)==h
    if not args.execute:print(json.dumps(dict(status='ready_plan_only',exit_code=0)));return
    RUN=args.output;RUN.mkdir(parents=True,exist_ok=False);code=1
    try:
        cfg=plan['modes']['M0']['config'];profile=Path(cfg['runtime_profile']);assert sha(profile)==cfg['runtime_profile_sha256']
        raw=subprocess.check_output(['bash','-ec','source "$1"; env -0','profile',str(profile)])
        env=dict(part.split('=',1) for part in raw.decode().split('\0') if part)
        source='/root/autodl-tmp/workspace/users/lewm/downstream-smoke-20261005-2133/state/official-source'
        env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONUNBUFFERED='1',PYTHONPATH=source+'/impls:'+source+':'+env.get('PYTHONPATH',''))
        for block in plan['training_blocks']:
            for mode in block['groups']:
                item=plan['modes'][mode]
                assert not Path(item['output']).exists() and not Path(item['launcher_output']).exists()
            empty_GPU();run(block['stage'],block['argv'])
            empty_GPU()
            for mode in block['groups']:
                item=plan['modes'][mode]
                exits=[Path(item['output'])/'trainer_exit_code',Path(item['launcher_output'])/'child_exit_code',Path(item['launcher_output'])/'driver_exit_code',Path(item['outer_exit'])]
                assert all(int(p.read_text())==0 for p in exits)
                current=load(Path(item['output'])/'status.json')
                assert current['completed_updates']==current['adam_count']==1000000 and current['network_step']==1000001 and current['validation_count']==200 and current['failed_stages']==0
                accept(handoff,mode,env)
        native=native_ready_pointer(plan)
        while native is None:
            status('all_three1M_training_CPU_accepted_waiting_native_ready',exit_code=None,training_complete=True,native_complete=False,waiting_for=plan.get('native_ready_file'))
            time.sleep(30)
            native=native_ready_pointer(plan)
        assert sha(native['path'])==native['sha256'];nh=load(native['path'])
        for path,h in nh['source_sha256'].items():assert sha(path)==h
        assert nh['ready'] and set(nh['modes'])=={'M0','M6','M5'}
        assert nh['selected_updates']==1000000 and nh['goal_dims']==195
        for mode,item in plan['modes'].items():assert nh['modes'][mode]['current_dims']==item['current_dims']
        for mode in ['M0','M6','M5']:
            for stage in nh['modes'][mode]['stages']:
                if not stage.get('cpu',False):empty_GPU()
                run(mode+'_native_'+stage['name'],stage['argv'],env if stage.get('cpu',False) else None)
                for path in stage['required_zero_exits']:assert int(Path(path).read_text())==0
                proof=load(stage['report']);assert proof['status']==stage['status'] and proof['exit_code']==0
            final=load(nh['modes'][mode]['final_report']);assert final['exit_code']==0 and final['episodes']==250 and str(final['status']).startswith('passed')
        status('all_three1M_training_CPU_and_native_owned_stages_complete',exit_code=0,training_complete=True,native_complete=True);code=0
    except BaseException:status('failed',exit_code=1,error=traceback.format_exc());raise
    finally:(RUN/'actual_exit_code').write_text(str(code)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');main(p.parse_args())
