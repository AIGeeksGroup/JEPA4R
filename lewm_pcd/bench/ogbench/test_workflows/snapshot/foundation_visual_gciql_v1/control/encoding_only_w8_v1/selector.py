"""Exact eighteen-phase encoding-only projection of the frozen w8 queue."""
import copy
import hashlib
from pathlib import Path,PurePosixPath

ROOT=PurePosixPath('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE=ROOT/'foundation_visual_gciql_v1';CONTROL=CASE/'control'
DIRECTORY=CONTROL/'encoding_only_w8_v1';BASE=CONTROL/'formal_w8/launch_plan.json'
BASE_SHA='0cfaba2fefefcca1483cea98c0b3bd04bcfbf6cd6c1536636f3477958473f5e1'
GROUPS=['visreg_vitb768_cube3','visreg_vitb768_cube4']+[f'vjepa21_vitl1024_cube{n}' for n in range(1,5)]
SUFFIXES=('01_encode_full','02_cache_audit','03_deliver')

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def make_plan(base,pins):
    phases=[]
    for group in GROUPS:
        for suffix in SUFFIXES:
            matches=[p for p in base['phases'] if p['name']==group+'_'+suffix];assert len(matches)==1
            phase=copy.deepcopy(matches[0]);args=phase['argv']
            if suffix=='03_deliver':
                i=args.index('--producer-exit-file')+1
                assert PurePosixPath(args[i]).parent==CONTROL/'formal_w8/runtime'
                args[i]=str(DIRECTORY/'runtime'/PurePosixPath(args[i]).name)
            phases.append(phase)
    result=copy.deepcopy(base)
    result.update(queue='encoding_only',groups=[g for g in base['groups'] if g['group_id'] in GROUPS],
        phases=phases,training_updates=0,SR_episodes=0,expected_features=19819800,source_pins=pins,
        encoding_only=True,base_plan_path=str(BASE),base_plan_sha256=BASE_SHA)
    assert [g['group_id'] for g in result['groups']]==GROUPS
    return result

def validate_selection(plan,base,pins):
    assert plan==make_plan(base,pins),'Only exact selected base phases and delivery exit routing are allowed'
    assert len(plan['phases'])==18 and plan['training_updates']==plan['SR_episodes']==0
    assert plan['diagnostic_optimizer_updates']==plan['diagnostic_SR_episodes']==0
    allowed={str(CASE/'cache'/name) for name in ('resource_w8.py','resource_audit_w8.py','deliver.py')}
    lease=str(ROOT/'cube4_visual_gciql_v1/control_v1/gpu_lease.py')
    for phase in plan['phases']:
        args=phase['argv'];scripts={arg for arg in args if arg.endswith('.py')}
        assert scripts and scripts<=allowed|{lease}
        assert '--config' not in args and not any(Path(arg).name in ('runtime.py','native_sr.py','prepare.py') for arg in args)
        assert phase['name'].rsplit('_',2)[-2:]!=['05','train']
        i=args.index('--plan')+1;group=next(g for g in GROUPS if phase['name'].startswith(g+'_'))
        assert args[i]==str(CASE/'cache/w8_groups'/group/'plan.json')
    return True
