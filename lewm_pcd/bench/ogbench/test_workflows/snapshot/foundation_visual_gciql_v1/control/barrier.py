"""Small metadata-only acceptance barrier for the existing serial supervisor."""
import hashlib
import json
from pathlib import Path

FAMILIES={'visreg_vitb768':768,'vjepa21_vitl1024':1024}

def read(path):return json.loads(Path(path).read_text())

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def validate_plan(plan):
    if 'adoption' in plan:
        old=_recovery_source(plan)
        validate_plan(old)
        for key in old:
            if key not in ('phases','source_pins'):
                assert plan[key]==old[key],key
        combined=dict(plan);combined.pop('adoption');combined.pop('recovery_runtime')
        combined['phases']=old['phases'][:6]+plan['phases']
        validate_plan(combined)
        assert len(plan['phases'])==58
        for before,after in zip(old['phases'][6:],plan['phases']):
            expected=dict(before);argv=list(before['argv'])
            if before['name'].endswith('_02_deliver_pilot'):
                index=argv.index('--producer-exit-file')+1
                assert Path(argv[index]).parent==Path(plan['adoption']['runtime'])
                argv[index]=str(Path(plan['recovery_runtime'])/Path(argv[index]).name)
            expected['argv']=argv
            assert after==expected,before['name']
        return
    families=plan['families'];assert families and len(set(families))==len(families)
    assert all(family in FAMILIES for family in families)
    assert plan['cube_counts']==[1,2,3,4]
    prefix=plan.get('group_prefix','');assert prefix in ('','random_init_v1_')
    assert plan.get('diagnostic_random',False)==bool(prefix)
    assert not prefix or plan['queue']=='smoke'
    groups=[dict(group_id=f'{prefix}{f}_cube{n}',family=f,cube_count=n,latent_dim=FAMILIES[f]) for f in families for n in range(1,5)]
    assert plan['groups']==groups
    formal=plan['queue']=='formal';assert plan['queue'] in ('smoke','formal')
    assert (plan['training_updates'],plan['SR_episodes'])==((4000000*len(families),1000*len(families)) if formal else (0,0))
    assert plan['expected_features']==(11011000 if formal else 8008)*len(families)
    names=[phase['name'] for phase in plan['phases']]
    assert len(names)==len(set(names))==len(groups)*8
    for group in groups:
        assert sum(name.startswith(group['group_id']+'_') for name in names)==8


def _recovery_source(plan):
    adoption=plan['adoption']
    assert plan['queue']=='smoke' and plan['diagnostic_random'] is True
    assert adoption['phase_count']==6
    assert sha(adoption['plan_path'])==adoption['plan_sha256']
    old=read(adoption['plan_path']);assert 'adoption' not in old
    assert len(old['phases'])==64
    assert Path(adoption['runtime'])!=Path(plan['recovery_runtime'])
    group=old['groups'][0]['group_id']
    assert [p['name'] for p in old['phases'][:6]]==[group+'_'+suffix for suffix in
        ('01_encode_pilot','02_deliver_pilot','03_config','04_hooks','05_gate','06_reload')]
    return old


def _require_adoption(plan):
    old=_recovery_source(plan);adoption=plan['adoption'];runtime=Path(adoption['runtime'])
    evidence=adoption['evidence'];assert evidence
    required=[Path(adoption['plan_path']),runtime/'outer_exit_code',runtime/'actual_dispatch.json']
    assert (runtime/'outer_exit_code').read_text().strip()=='1'
    assert read(runtime/'actual_dispatch.json')['launch_plan_sha256']==adoption['plan_sha256']
    for phase in old['phases'][:6]:
        exit_path=runtime/(phase['name']+'_exit.json');argv_path=runtime/(phase['name']+'_argv.json')
        required.extend((exit_path,argv_path))
        record=read(exit_path)
        assert record['actual_exit_code']==0 and record['residual_owned_members']==[]
        assert read(argv_path)['argv']==phase['argv']
    assert all(str(path) in evidence for path in required)
    for path,digest in evidence.items():assert sha(path)==digest,path
    backups=adoption['changed_source_backups']
    assert set(backups)<=set(old['source_pins'])
    for path,digest in old['source_pins'].items():
        assert sha(backups.get(path,path))==digest,path
    return 6

def require_smoke(directory,formal):
    directory=Path(directory);plan=read(directory/'launch_plan.json');validate_plan(plan)
    adopted=_require_adoption(plan) if 'adoption' in plan else 0
    assert plan['queue']=='smoke' and plan['families']==formal['families'] and plan['groups']==formal['groups']
    runtime=directory/'runtime'
    if adopted:assert runtime==Path(plan['recovery_runtime'])
    assert (runtime/'outer_exit_code').read_text().strip()=='0'
    closure=read(runtime/'closure.json')
    assert closure['actual_exit_code']==0 and closure['completed_phases']==len(plan['phases'])
    assert closure['families']==plan['families']
    assert read(runtime/'actual_dispatch.json')['launch_plan_sha256']==sha(directory/'launch_plan.json')
    for phase in plan['phases']:
        record=read(runtime/(phase['name']+'_exit.json'))
        assert record['actual_exit_code']==0 and record['residual_owned_members']==[]
        assert read(runtime/(phase['name']+'_argv.json'))['argv']==phase['argv']
        if phase['name'].endswith('_08_native_audit'):
            args=phase['argv'];output=Path(args[args.index('--output')+1]);config=Path(args[args.index('--config')+1])
            audit=read(output/'independent_audit.json')
            assert audit['actual_exit_code']==0 and audit['mode']=='tiny' and audit['diagnostic_excluded_from_formal_SR']
            assert audit['config_sha256']==sha(config)
    assert plan['source_pins']==formal['source_pins']
    for path,digest in plan['source_pins'].items():assert sha(path)==digest,path
    result=dict(actual_exit_code=0,smoke_phases=len(plan['phases']),groups=len(plan['groups']))
    if adopted:result.update(adopted_phases=adopted,executed_phases=len(plan['phases']),effective_completed_phases=adopted+len(plan['phases']))
    return result
