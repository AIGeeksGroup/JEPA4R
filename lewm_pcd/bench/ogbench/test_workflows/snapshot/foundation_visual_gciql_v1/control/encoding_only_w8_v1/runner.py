"""Reuse the frozen original supervisor/guard; admit only encoding/cache delivery."""
import argparse
import json
from pathlib import Path
import sys
import time
from selector import ROOT,CASE,CONTROL,DIRECTORY,BASE,BASE_SHA,GROUPS,sha,validate_selection
ROOT,CASE,CONTROL,DIRECTORY,BASE=map(Path,(ROOT,CASE,CONTROL,DIRECTORY,BASE))

def read(path):return json.loads(Path(path).read_text())

def frozen_inputs(plan):
    assert sha(BASE)==BASE_SHA
    base=read(BASE);pins=base['source_pins'].copy();pins[str(BASE)]=BASE_SHA
    for name in ('selector.py','runner.py'):pins[str(DIRECTORY/name)]=sha(DIRECTORY/name)
    validate_selection(plan,base,pins)
    return base

def require_adopted_metadata(base):
    # Adopt completed diagnostic evidence; never dispatch their original phases.
    accepted=read(CONTROL/'bootstrap_runtime_v3/random_diagnostic_acceptance.json')
    assert accepted['actual_exit_code']==0 and accepted['effective_completed_phases']==64
    assert accepted['adopted_phases']==6 and accepted['executed_phases']==58
    random_runtime=CONTROL/'random_smoke_recovery_v1/runtime'
    assert (random_runtime/'outer_exit_code').read_text().strip()=='0'
    assert read(random_runtime/'closure.json')['completed_phases']==58
    smoke=CONTROL/'smoke/runtime';closure=read(smoke/'closure.json')
    assert (smoke/'outer_exit_code').read_text().strip()=='0'
    assert closure['actual_exit_code']==0 and closure['completed_phases']==64
    admission=read(CONTROL/'formal_w8/resource_admission.json')
    assert admission['actual_exit_code']==0 and admission['new_plan_sha256']==BASE_SHA
    assert admission['official_smoke']['actual_exit_code']==0 and admission['official_smoke']['smoke_phases']==64
    for group in ('visreg_vitb768_cube1','visreg_vitb768_cube2'):
        home=CASE/'cache/w8_groups'/group;planpath=home/'plan.json';audit=read(home/'full_audit_v1/report.json')
        assert audit['actual_exit_code']==0 and audit['episodes_verified']==1100 and audit['raw_rows_verified']==1101100
        assert audit['plan_sha256']==sha(planpath) and audit['manifest_sha256']==sha(home/'cache/cache_manifest.json')
        assert (home/'full_audit_v1/actual_exit_code').read_text().strip()=='0'
        assert read(home/'delivery.json')['cache_audit']['sha256']==sha(home/'full_audit_v1/report.json')
        for suffix in ('01_encode_full','02_cache_audit','03_deliver'):
            record=read(CONTROL/'formal_w8/runtime'/f'{group}_{suffix}_exit.json')
            assert record['actual_exit_code']==0 and record['residual_owned_members']==[]
    return dict(actual_exit_code=0,random_effective_phases=64,official_smoke_phases=64,already_accepted_cache_groups=2,
        no_diagnostic_or_RL_or_SR_dispatch=True,at=time.time())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    plan=read(DIRECTORY/'launch_plan.json');base=frozen_inputs(plan)
    if not args.execute:
        print(json.dumps(dict(prepared_only=True,actual_exit_code=0,phases=18,groups=GROUPS,training_updates=0,SR_episodes=0)));return
    adoption=require_adopted_metadata(base)
    assert not (DIRECTORY/'runtime').exists(),'No overwrite, resume or automatic retry'
    sys.path.insert(0,str(CONTROL));import serial as original
    def validate(actual):frozen_inputs(actual)
    original.validate_plan=validate
    original_loader=original.load_supervisor
    def load(directory):
        module=original_loader(directory);save=module.save
        def metadata_save(name,value):
            if name=='closure.json':value=dict(value,encoding_only=True,group_ids=GROUPS,accepted_cache_groups_before_run=2)
            save(name,value)
        module.save=metadata_save
        return module
    original.load_supervisor=load
    # queue=encoding_only bypasses the formal RL barrier branch; our own exact
    # selection and metadata adoption above are the only replacement admission.
    with (DIRECTORY/'adopted_metadata.json').open('x') as stream:json.dump(adoption,stream,indent=2)
    sys.argv=[str(Path(__file__)), '--plan-directory',str(DIRECTORY),'--execute']
    return original.main()

if __name__=='__main__':raise SystemExit(main())
