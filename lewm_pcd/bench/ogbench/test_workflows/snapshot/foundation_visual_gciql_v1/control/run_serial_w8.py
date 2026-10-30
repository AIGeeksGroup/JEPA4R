"""Adopt accepted encoder smokes; select only the eight-renderer resource entry."""
import copy,json,sys,time
from pathlib import Path
import serial as original
from barrier import read,sha,require_smoke as original_barrier,validate_plan
from build_queues import CASE,CONTROL,ROOT

BASE=CONTROL/'formal/launch_plan.json'
NEW=CONTROL/'formal_w8'
RESOURCE_PRODUCER=CASE/'cache/resource_w8.py'
RESOURCE_AUDITOR=CASE/'cache/resource_audit_w8.py'

def expected_phases(base):
    phases=copy.deepcopy(base['phases'])
    for phase in phases:
        args=phase['argv']
        for group in base['groups']:
            name=group['group_id']
            old=CASE/'cache/groups'/name;new=CASE/'cache/w8_groups'/name
            args[:]=[str(new)+value[len(str(old)):] if value.startswith(str(old)+'/') else value for value in args]
        if phase['name'].endswith('_01_encode_full'):
            args[args.index(str(CASE/'cache/producer_bridge.py'))]=str(RESOURCE_PRODUCER)
            for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):phase['environment'][key]='1'
        elif phase['name'].endswith('_02_cache_audit'):
            args[args.index(str(CASE/'cache/audit_bridge.py'))]=str(RESOURCE_AUDITOR)
        if '--producer-exit-file' in args:
            index=args.index('--producer-exit-file')+1
            assert Path(args[index]).parent==CONTROL/'formal/runtime'
            args[index]=str(NEW/'runtime'/Path(args[index]).name)
    return phases

def require_resource_smokes(directory,plan):
    base=read(BASE);accepted=original_barrier(CONTROL/'smoke',base)
    validate_plan(plan)
    for key in base:
        if key not in ('phases','source_pins'):assert plan[key]==base[key],key
    assert plan['phases']==expected_phases(base)
    assert all(plan['source_pins'][key]==value for key,value in base['source_pins'].items())
    for group in base['groups']:
        name=group['group_id'];old=read(CASE/'cache/groups'/name/'plan.json')
        path=CASE/'cache/w8_groups'/name/'plan.json';new=read(path)
        for key,value in old.items():
            if key not in ('renderer_workers','cache_dir','source_pins'):assert new[key]==value,(name,key)
        assert new['renderer_workers']==new['prefetch']==8 and new['encoder_batch']==32
        assert new['cache_dir']==str(path.parent/'cache')
        assert all(new['source_pins'][key]==value for key,value in old['source_pins'].items())
        for key,value in new['source_pins'].items():assert sha(ROOT/key)==value,key
    for key,value in plan['source_pins'].items():assert sha(key)==value,key
    result=dict(actual_exit_code=0,official_smoke=accepted,resource_workers=8,prefetch=8,encoder_batch=32,
        base_plan_sha256=sha(BASE),new_plan_sha256=sha(NEW/'launch_plan.json'),scientific_configuration_unchanged=True,
        previous_w16_trial_bitwise_failure_preserved=True,bitwise_to_earlier_render_runs_claimed=False,at=time.time())
    path=NEW/'resource_admission.json'
    if not path.exists():
        with path.open('x') as out:json.dump(result,out,indent=2)
    return result

if __name__=='__main__':
    original.require_smoke=require_resource_smokes
    raise SystemExit(original.main())
