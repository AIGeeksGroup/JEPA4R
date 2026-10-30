"""Metadata-only continuation of the frozen VJ v2 queues; no NN/math copy."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE=ROOT/'vjepa_vitl1024_gciql_v2'
CONTROL=CASE/'continuation_v1'
RUN_SHA='d5a04ebfe48cb45584578db685cb9796a449f29df8ab8de00f5a35db376a199e'
FAMILY='vjepa21_vitl1024'
SMOKE_LABELS=('verify_hooks','gate','backup_gate','reload','native_tiny','tiny_audit')
FULL_LABELS=('train','backup_gate','accept','native_full','native_audit')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def reference(item):
    path=Path(item['path']);assert path.is_absolute(),'Evidence must use a real remote absolute path'
    assert sha(path)==item['sha256'],str(path)
    return path


def references(value):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:yield value
        for child in value.values():yield from references(child)
    elif isinstance(value,list):
        for child in value:yield from references(child)


def run_api():
    path=CASE/'run.py';assert sha(path)==RUN_SHA,'Original v2 guard source changed'
    spec=importlib.util.spec_from_file_location('_frozen_v2_launch_guard',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # These remain the original CASE, never the continuation log namespace.
    module.ROOT,module.CASE=ROOT,CASE
    return module


def source_plans():
    adoption=read(CONTROL/'ADOPTED_V2_PHASES.json')
    resolution=read(CONTROL/'TRANSPORT_RESOLUTION.json')
    assert adoption['actual_exit_code']==resolution['actual_exit_code']==0
    run_api().check_backup_failure()
    plans={}
    for mode,labels in [('smoke',SMOKE_LABELS),('full',FULL_LABELS)]:
        item=adoption['old_'+mode+'_plan'];path=reference(item)
        assert path==CASE/'control'/mode/'launch_plan.json'
        plan=read(path)
        assert plan['mode']==mode and plan['cube_counts']==[1,2,3,4] and plan['families']==[FAMILY]
        assert [p['name'] for p in plan['phases']]==[f'{FAMILY}_cube{n}_{i:02d}_{label}' for n in (1,2,3,4) for i,label in enumerate(labels,1)]
        plans[mode]=plan
    runtime=CASE/'control/smoke/runtime'
    assert (runtime/'outer_exit_code').read_text().strip()==(runtime.parent/'wrapper_exit_code').read_text().strip()=='1'
    assert read(runtime/'failure.json')['actual_exit_code']==1
    assert read(runtime/'status.json')['actual_exit_code']==1
    failed=read(runtime/(plans['smoke']['phases'][2]['name']+'_exit.json'))
    assert failed['actual_exit_code']==1 and failed['residual_owned_members']==[]
    assert not (CASE/'control/full/runtime').exists(),'Original formal queue must not have started'
    assert [p['name'] for p in adoption['adopted_phases']]==[p['name'] for p in plans['smoke']['phases'][:2]]
    for proof,step in zip(adoption['adopted_phases'],plans['smoke']['phases'][:2]):
        exit_record=read(reference(proof['exit_file']));argv=read(reference(proof['argv_file']));reference(proof['log_file'])
        assert exit_record['actual_exit_code']==0 and exit_record['residual_owned_members']==[]
        assert argv['argv']==step['argv'] and argv['cwd']==step['cwd']
    gate=adoption['gate'];config_path=reference(gate['config']);config=read(config_path)
    assert config_path==CASE/'configs/pilot/vjepa21_vitl1024_cube1.json'
    assert config['group_id']==FAMILY+'_cube1' and config['purpose']=='pilot'
    report_path=reference(gate['report']);assert report_path==Path(config['durability']['pilot_output'])/'report.json'
    report=read(report_path)
    assert (report['actual_exit_code'],report['actual_backend'],report['updates'],report['batch_size'],report['network_step'],report['adam_count'])==(0,'gpu',2,1024,3,2)
    assert report['encoder_calls']==report['WM_dynamics_calls']==0 and report['saved_state']['leaves']==153
    assert {shape[-2] for shape in report['first_layer_kernels'].values()}=={2070,2075}
    assert [p['counter'] for p in gate['checkpoints']]==[1,2]
    for cp in gate['checkpoints']:
        manifest_path=reference(cp['manifest']);ack_path=reference(cp['ack'])
        expected=Path(config['durability']['pilot_output'])/'checkpoints'/f"update_{cp['counter']:09d}"/'manifest.json'
        assert manifest_path==expected and read(manifest_path)['counter']==cp['counter']
        ack=read(ack_path)
        assert ack_path==Path(config['durability']['ack_root'])/'pilot'/config['group_id']/f"update_{cp['counter']:09d}.json"
        assert ack['actual_exit_code']==0 and ack['counter']==cp['counter'] and ack['stage']=='pilot' and ack['group_id']==config['group_id']
        assert ack['config_sha256']==gate['config']['sha256'] and ack['manifest_sha256']==cp['manifest']['sha256']
        assert ack['local_receipt_sha256']==cp['local_receipt_sha256'] and ack['scp_exit_codes']==[0,0,0,0]
    assert resolution['classification']=='transport_timeout'
    assert read(reference(resolution['failure_record']))['actual_exit_code']==1
    assert read(reference(resolution['timeout_command']))['actual_exit_code'] is None
    reference(resolution['retry_record'])
    assert resolution['retry_record']['actual_exit_code']==0 and reference(resolution['retry_record']['exit_file']).read_text().strip()=='0'
    assert read(reference(resolution['manual_backup_record']))['actual_exit_code']==resolution['manual_backup_record']['actual_exit_code']==0
    assert resolution['old_control_failure_preserved'] and resolution['experiment_sources_configs_unchanged']
    assert resolution['archived_sticky']
    failure_root=CASE/'control/backup_failures'
    for item in resolution['archived_sticky']:
        original,archive=Path(item['original_path']),Path(item['archive_path'])
        assert original.is_relative_to(failure_root) and not original.exists()
        assert not archive.is_relative_to(failure_root) and sha(archive)==item['sha256']
    receiver=read(CONTROL/'backup_config.json')
    expected_receiver=read(CASE/'control/backup_config.json')
    expected_receiver.update(case_root=str(CONTROL),remote_config=str(CONTROL/'backup_config.json'))
    assert receiver==expected_receiver,'Recovery receiver may change only its log namespace and own path'
    expected_configs=[str(CASE/'configs'/purpose/f'{FAMILY}_cube{n}.json') for purpose in ('pilot','formal') for n in (1,2,3,4)]
    assert receiver['case_root']==str(CONTROL) and receiver['config_paths']==expected_configs
    assert receiver['remote_config']==str(CONTROL/'backup_config.json') and receiver['remote_script']==str(CASE/'backup_watch.py')
    return adoption,resolution,plans


def derived_plan(mode):
    assert mode in ('smoke','full')
    adoption,resolution,plans=source_plans();original=plans[mode]
    plan=copy.deepcopy(original);steps=original['phases'][2:] if mode=='smoke' else original['phases']
    plan['phases']=copy.deepcopy(steps)
    for step in plan['phases']:
        old_name=step['name'];step['name']='cont1_'+old_name
        if old_name.endswith('_backup_gate'):
            index=step['argv'].index('--report')+1
            step['argv'][index]=str(CONTROL/'control'/mode/'durability_reports'/Path(step['argv'][index]).name)
    pins=dict(original['source_pins'])
    for p in (Path(__file__).resolve(),CONTROL/'ADOPTED_V2_PHASES.json',CONTROL/'TRANSPORT_RESOLUTION.json',CONTROL/'backup_config.json'):
        pins[str(p)]=sha(p)
    for value in (adoption,resolution):
        for item in references(value):pins[str(reference(item))]=item['sha256']
    for item in resolution['archived_sticky']:pins[item['archive_path']]=item['sha256']
    for path in (CASE/'control/smoke/runtime/failure.json',CASE/'control/smoke/runtime/status.json',CASE/'control/smoke/runtime/outer_exit_code',CASE/'control/smoke/wrapper_exit_code',CASE/'control/smoke/runtime'/('vjepa21_vitl1024_cube1_03_backup_gate_exit.json')):
        pins[str(path)]=sha(path)
    plan['source_pins']=pins
    plan['continuation']=dict(original_plan_sha256=adoption['old_'+mode+'_plan']['sha256'],adopted_phase_count=2 if mode=='smoke' else 0,executed_phase_count=len(plan['phases']),effective_smoke_phase_count=24,adopted_diagnostic_updates=2 if mode=='smoke' else 0,remaining_diagnostic_updates=6 if mode=='smoke' else 0)
    return plan


def require_effective_smoke():
    adoption,_,_=source_plans();plan=derived_plan('smoke');runtime=CONTROL/'control/smoke/runtime'
    assert read(runtime.parent/'launch_plan.json')==plan
    assert (runtime/'outer_exit_code').read_text().strip()==(runtime.parent/'wrapper_exit_code').read_text().strip()=='0'
    closure=read(runtime/'closure.json')
    assert closure['actual_exit_code']==0 and closure['completed_phases']==22
    assert closure['cube_counts']==[1,2,3,4] and closure['families']==[FAMILY]
    assert closure['training_updates']==closure['SR_episodes']==0
    for step in plan['phases']:
        record=read(runtime/(step['name']+'_exit.json'))
        assert record['actual_exit_code']==0 and record['residual_owned_members']==[]
    for n in (1,2,3,4):
        group=f'{FAMILY}_cube{n}';cfg_path=CASE/'configs/pilot'/(group+'.json');cfg=read(cfg_path);home=Path(cfg['execution_root'])
        gate=read(home/'gates'/group/'report.json');reload=read(home/'gates'/(group+'_reload.json'))
        assert (gate['actual_exit_code'],gate['actual_backend'],gate['updates'],gate['batch_size'],gate['network_step'],gate['adam_count'])==(0,'gpu',2,1024,3,2)
        assert gate['encoder_calls']==gate['WM_dynamics_calls']==0
        assert reload['actual_exit_code']==0 and reload['full153_state_exact'] and reload['finite_actions']
        backup=read(runtime.parent/'durability_reports'/(group+'.json'))
        assert backup['actual_exit_code']==0 and backup['stage']=='gate' and backup['checkpoint_count']==2
        assert backup['backup_stage']=='pilot' and backup['group_id']==group and backup['config_sha256']==sha(cfg_path)
        assert [cp['counter'] for cp in backup['checkpoints']]==[1,2]
        for cp in backup['checkpoints']:
            ack=Path(cp['ack_path']);assert sha(ack)==cp['ack_sha256']
            assert ack==Path(cfg['durability']['ack_root'])/'pilot'/group/f"update_{cp['counter']:09d}.json"
            item=read(ack);assert item['actual_exit_code']==0 and item['config_sha256']==sha(cfg_path)
            assert (item['group_id'],item['stage'],item['counter'])==(group,'pilot',cp['counter'])
            assert item['manifest_sha256']==cp['manifest_sha256'] and item['local_receipt_sha256']==cp['local_receipt_sha256']
            manifest=Path(cfg['durability']['pilot_output'])/'checkpoints'/f"update_{cp['counter']:09d}"/'manifest.json'
            assert sha(manifest)==cp['manifest_sha256']
        tiny=read(home/'native_sr'/group/'report.json');audit=read(home/'native_sr'/group/'independent_audit.json')
        assert tiny['actual_exit_code']==audit['actual_exit_code']==0 and tiny['episodes']==audit['episodes']==1
        assert (tiny['current_dim'],tiny['goal_dim'],tiny['max_episode_steps'])==(1043,1024+3*n,4)
        assert tiny['encoding_protocol']=='fresh_current_and_goal_each_policy_call_2N'
        assert audit['config_sha256']==sha(cfg_path) and audit['actual_step_arrays_checked'] and audit['diagnostic_excluded_from_formal_SR']
    return dict(actual_exit_code=0,adopted_phases=len(adoption['adopted_phases']),executed_phases=22,effective_completed_phases=24,old_outer_exit_code=1,old_failure_preserved=True)


def validate(plan):
    run_api().check_backup_failure()
    assert plan==derived_plan(plan['mode']),'Continuation must be an exact derivation of frozen original plans'
    if plan['mode']=='full':require_effective_smoke()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--queue',choices=('smoke','full'));parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.prepare:
        assert not args.queue and not args.execute
        for mode in ('smoke','full'):save(CONTROL/'control'/mode/'launch_plan.json',derived_plan(mode))
        print(json.dumps(dict(actual_exit_code=0,prepared_only=True,remaining_smoke_phases=22,formal_phases=20,NN_started=False)));return 0
    assert args.queue and not args.prepare
    directory=CONTROL/'control'/args.queue;validate(read(directory/'launch_plan.json'))
    if not args.execute:
        print(json.dumps(dict(prepared_only=True,NN_started=False,queue=args.queue)));return 0
    assert not (directory/'runtime').exists() and not (directory/'wrapper_exit_code').exists(),'Continuation runtime cannot be restarted'
    sys.path.insert(0,str(ROOT/'foundation_visual_gciql_v1/control'));import serial
    serial.validate_plan=validate;run_api().install_launch_guard(serial)
    sys.argv=[str(__file__),'--plan-directory',str(directory),'--execute']
    code=1
    try:
        code=serial.main();return code
    finally:
        with (directory/'wrapper_exit_code').open('x') as stream:stream.write(str(code)+'\n')


if __name__=='__main__':raise SystemExit(main())
