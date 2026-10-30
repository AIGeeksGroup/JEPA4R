"""VJ-only contract wrapper over the unchanged original serial supervisor."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE=ROOT/'vjepa_vitl1024_gciql_v1'


def read(path): return json.loads(Path(path).read_text())


def require_smoke():
    from prepare import COUNTS,FAMILY,sha
    runtime=CASE/'control/smoke/runtime'
    assert (runtime/'outer_exit_code').read_text().strip()==(runtime.parent/'wrapper_exit_code').read_text().strip()=='0'
    closure=read(runtime/'closure.json')
    assert closure['actual_exit_code']==0 and closure['completed_phases']==24
    assert closure['cube_counts']==[1,2,3,4] and closure['families']==[FAMILY]
    labels=('verify_hooks','gate','backup_gate','reload','native_tiny','tiny_audit')
    for n in COUNTS:
        group=f'{FAMILY}_cube{n}'
        for i,label in enumerate(labels,1):
            phase=read(runtime/f'{group}_{i:02d}_{label}_exit.json')
            assert phase['actual_exit_code']==0 and phase['residual_owned_members']==[]
        config_path=CASE/'configs/pilot'/(group+'.json');cfg=read(config_path);home=Path(cfg['execution_root'])
        gate=read(home/'gates'/group/'report.json');reload=read(home/'gates'/(group+'_reload.json'))
        assert (gate['actual_exit_code'],gate['actual_backend'],gate['updates'],gate['batch_size'],gate['network_step'],gate['adam_count'])==(0,'gpu',2,1024,3,2)
        assert gate['encoder_calls']==gate['WM_dynamics_calls']==0
        assert reload['actual_exit_code']==0 and reload['full153_state_exact'] and reload['finite_actions']
        backup=read(runtime.parent/'durability_reports'/(group+'.json'))
        assert backup['actual_exit_code']==0 and backup['stage']=='gate' and backup['backup_stage']=='pilot'
        assert backup['group_id']==group and backup['config_sha256']==sha(config_path) and backup['checkpoint_count']==2
        assert [p['counter'] for p in backup['checkpoints']]==[1,2]
        tiny=read(home/'native_sr'/group/'report.json');audit=read(home/'native_sr'/group/'independent_audit.json')
        assert tiny['actual_exit_code']==audit['actual_exit_code']==0 and tiny['episodes']==audit['episodes']==1
        assert (tiny['current_dim'],tiny['goal_dim'],tiny['max_episode_steps'])==(1043,1024+3*n,4)
        assert tiny['encoding_protocol']=='fresh_current_and_goal_each_policy_call_2N'
        assert audit['config_sha256']==sha(config_path) and audit['actual_step_arrays_checked'] and audit['diagnostic_excluded_from_formal_SR']


def validate(plan):
    from prepare import COUNTS,FAMILY,cache_admission,build_queue
    assert plan['queue']=='vj_downstream' and plan['mode'] in ('smoke','full')
    assert plan['families']==[FAMILY] and plan['cube_counts']==[1,2,3,4]
    configs={}
    for n in COUNTS:
        for purpose in ('pilot','formal'):
            path=CASE/'configs'/purpose/f'{FAMILY}_cube{n}.json';cfg=read(path)
            assert cfg['group_id']==f'{FAMILY}_cube{n}' and cfg['encoder_binding']['family']==FAMILY
            assert cfg['input_contract']==dict(current_dim=1043,goal_dim=1024+3*n,latent_dim=1024,zero_padding=False)
            configs[n,purpose]=(path,cfg)
    assert plan==build_queue(configs,plan['mode'],plan['source_pins']),'Exact VJ-only phase/config contract differs'
    cache_admission()
    if plan['mode']=='full':require_smoke()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--queue',choices=('smoke','full'),required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    from prepare import space_admission,save
    directory=CASE/'control'/args.queue
    validate(read(directory/'launch_plan.json'))
    if args.execute:
        assert not (directory/'runtime').exists() and not (directory/'wrapper_exit_code').exists(),'Existing experiment runtime or exit must not be restarted'
        save(directory/'SPACE_ADMISSION.json',space_admission())
    sys.path.insert(0,str(ROOT/'foundation_visual_gciql_v1/control'))
    import serial
    serial.validate_plan=validate
    sys.argv=[str(__file__),'--plan-directory',str(directory)]+(['--execute'] if args.execute else [])
    return serial.main()


if __name__=='__main__':raise SystemExit(main())
