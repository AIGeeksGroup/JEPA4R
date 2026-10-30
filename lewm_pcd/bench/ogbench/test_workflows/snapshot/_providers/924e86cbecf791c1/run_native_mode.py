"""Thin serial orchestration of frozen binding, tiny, original250 driver and audits."""
import argparse,hashlib,json,os,shlex,subprocess,time,traceback
from pathlib import Path
N=Path(__file__).resolve().parents[1];Q=N.parent
B=Path('/root/autodl-tmp/workspace/users/lewm/ogbench-official-single-play-20261006')
P='/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python'
DRIVER=B/'official_lewm_gcbc_20261006/native_sr_v1/run_full_sequence.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)

def main(a):
    runtime=N/f'aux/native_runtime_v1/{a.mode}';runtime.mkdir(parents=True,exist_ok=False)
    status=dict(stage='validating_true_final1M',exit_code=None,steps=[],updated_at=time.time());code=1
    def persist():status['updated_at']=time.time();write(runtime/'status.json',status)
    def run(name,argv,cpu=False):
        status.update(stage=name,argv=argv);persist()
        environment=dict(os.environ)
        if cpu:environment.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (runtime/(name+'.log')).open('x') as log:
            result=subprocess.run(argv,cwd=N,env=environment,stdout=log,stderr=subprocess.STDOUT)
        path=runtime/(name+'.exit_code');path.write_text(str(result.returncode)+'\n')
        status['steps'].append(dict(name=name,actual_exit_code=result.returncode,argv=argv));persist()
        assert result.returncode==0,f'{name} failed; original evidence retained'
        return path
    persist()
    try:
        freeze=N/'aux/freeze_v1/manifest.json'
        assert sha(freeze)=='d8f4adfa41aaeb002198dc1f134503e27d2033f0250c847baee2ac9cb91283dc'
        assert all(sha(N/name)==value for name,value in load(freeze)['files'].items())
        assert sha(DRIVER)=='bbbb96caf3ea7ac013a2841cede2c0a83c622ba2e7cb65b86fbd0574233552f1'
        assert sha(a.auditor)==a.auditor_sha256 and sha(a.audit_shell)==a.audit_shell_sha256
        plan_path=Q/'relay_v1/formal_plan_v2.json'
        assert sha(plan_path)=='bd76211d4db28d96f811aece34ff3c66723d76467f56351a209bae9cae2f8927'
        training=load(plan_path)['modes'][a.mode]
        assert sha(training['config_path'])==training['config_sha256']
        audit=Q/f'afterhook_v1/{a.mode}_final/report.json';probe=Q/f'afterhook_v1/{a.mode}_action/report.json'
        assert load(audit)['counter']==1000000 and load(audit)['status']=='passed_CPU_final_acceptance' and load(audit)['exit_code']==0
        assert load(probe)['restored_network_step']==1000001 and load(probe)['status']=='passed_CPU_fresh_cached_action_probe' and load(probe)['exit_code']==0
        cfg=N/f'configs/{a.mode}_frozen.json';out=B/f'rl3_{a.mode.lower()}_gciql_1m_b1024_native_20261013_v1'
        assert not out.exists() and not cfg.exists()
        binding_argv=[P,str(N/'prepare_binding.py'),'--training-config',training['config_path'],'--train-root',training['output'],
            '--selected-updates','1000000','--final-audit',str(audit),'--final-probe',str(probe),'--launcher',training['launcher_output'],
            '--outer-exit',training['outer_exit'],
            '--audit-exit',str(Q/f'relay_v1/runtime/{a.mode}_CPU_fullstate/actual_exit_code'),
            '--probe-exit',str(Q/f'relay_v1/runtime/{a.mode}_CPU_cached_action/actual_exit_code'),
            '--native-output',str(out),'--output',str(cfg)]
        run('bind_true_final1M',binding_argv,cpu=True)
        actual_cfg=load(cfg);assert actual_cfg['selected_updates']==1000000 and actual_cfg['goal_dims']==195 and actual_cfg['current_dims']==training['current_dims']
        assert actual_cfg['training_code_root']==str(Q/'core_v1')
        write(runtime/'binding_receipt.json',dict(status='passed_true_final1M_binding',exit_code=0,config_sha256=sha(cfg),argv=binding_argv))
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip(),'GPU occupied; no process is stopped'
        wrappers=N/'aux/stage_launches';wrappers.mkdir(exist_ok=True);before=set(wrappers.iterdir())
        tiny_outer=run('tiny',['bash',str(N/'run_stage.sh'),str(cfg),'gate'])
        matches=[]
        for folder in set(wrappers.iterdir())-before:
            if (folder/'argv.txt').is_file():
                argv=shlex.split((folder/'argv.txt').read_text())
                if '--output' in argv and argv[argv.index('--output')+1]==str(out) and argv[argv.index('--stage')+1]=='gate':matches.append(folder)
        assert len(matches)==1;stage_exit=matches[0]/'exit_code';assert stage_exit.read_text().strip()=='0'
        tiny_audit=N/f'aux/final_audit_v1/{a.mode}_tiny_result_v1'
        run('tiny_CPU',[P,str(a.auditor),'--mode','tiny','--config',str(cfg),'--expected-config-sha256',sha(cfg),
            '--output',str(tiny_audit),'--required-zero-exit',str(tiny_outer),'--required-zero-exit',str(stage_exit)],cpu=True)
        tiny_report=tiny_audit/'report.json'
        assert load(tiny_report)['status']=='passed_tiny_metadata_source_contract_only' and load(tiny_report)['exit_code']==0
        assert (tiny_audit/'inner_exit_code').read_text().strip()=='0'
        run('seal_full_plan',[P,str(N/'aux/prepare_full_plan.py'),'--kind',a.mode,'--tiny-review',str(tiny_report),
            '--tiny-review-sha',sha(tiny_report),'--tiny-outer-exit',str(tiny_outer),'--tiny-stage-exit',str(stage_exit)],cpu=True)
        full_plan=N/f'aux/sequence_plans/{a.mode}.json'
        full_outer=run('full250',[P,str(DRIVER),'--plan',str(full_plan)])
        driver=Path(load(full_plan)['driver_output']);assert (driver/'exit_code').read_text().strip()=='0'
        full_audit=N/f'aux/final_audit_v1/{a.mode}_full_result_v1'
        run('full_CPU',['bash',str(a.audit_shell),str(cfg),sha(cfg),str(driver),str(full_plan),sha(full_plan),str(full_audit),str(full_outer),str(tiny_outer),str(stage_exit)],cpu=True)
        final=load(full_audit/'report.json')
        assert final['status']=='passed_full250_metadata_and_source_with_explicit_limits' and final['exit_code']==0 and final['episodes']==250
        assert (full_audit/'inner_exit_code').read_text().strip()=='0' and Path(str(full_audit)+'_launch/outer_exit_code').read_text().strip()=='0'
        result=dict(status='passed_native_true1M_tiny_full250_and_CPU_audit',exit_code=0,mode=a.mode,episodes=250,
            selected_updates=1000000,current_dims=actual_cfg['current_dims'],goal_dims=195,config_sha256=sha(cfg),plan_sha256=sha(full_plan),
            final_report=str(full_audit/'report.json'),final_report_sha256=sha(full_audit/'report.json'),steps=status['steps'])
        write(runtime/'report.json',result);code=0;status.update(stage='complete',exit_code=0)
    except BaseException:status.update(stage='failed',exit_code=1,error=traceback.format_exc());raise
    finally:persist();(runtime/'actual_exit_code').write_text(str(code)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['M0','M6','M5'],required=True)
    p.add_argument('--auditor',type=Path,required=True);p.add_argument('--auditor-sha256',required=True)
    p.add_argument('--audit-shell',type=Path,required=True);p.add_argument('--audit-shell-sha256',required=True)
    main(p.parse_args())
