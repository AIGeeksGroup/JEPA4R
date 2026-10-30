"""Reuse the original serial supervisor; admit fresh canonical1 encoding only."""
import argparse
import json
from pathlib import Path
import sys
import time
import selector as s

def read(path):return json.loads(Path(path).read_text())
def inputs(plan):
    assert s.sha(s.BASE)==s.BASE_SHA
    base=read(s.BASE);pins=base['source_pins'].copy();pins[str(s.BASE)]=s.BASE_SHA
    for name in ('selector.py','runner.py'):pins[str(s.HERE/name)]=s.sha(s.HERE/name)
    inventory=read(s.HERE/'source_inventory.json')
    for group in s.GROUPS:
        home=s.cache_home(group);path=home/'plan.json';actual=read(path)
        assert actual==s.cache_plan(inventory['groups'][group]['plan.json'],group)
        for name in ('data_binding.json','camera.json'):assert read(home/name)==inventory['groups'][group][name]
        pins[str(path)]=s.sha(path)
        for relative,digest in actual['source_pins'].items():pins[str(s.ROOT/relative)]=digest
    pins[str(s.HERE/'source_inventory.json')]=s.sha(s.HERE/'source_inventory.json')
    s.validate(plan,base,pins)
    assert Path(s.cache_home(s.GROUPS[2])).is_symlink()
    assert Path(s.cache_home(s.GROUPS[2])).resolve()==Path(s.ROOT_DISK_TARGET)

def admission(plan):
    control=Path(s.CASE/'control')
    for n in range(1,5):
        group=f'visreg_vitb768_cube{n}';home=Path(s.CASE/'cache/w8_groups'/group)
        audit=read(home/'full_audit_v1/report.json');assert audit['actual_exit_code']==0
        assert audit['episodes_verified']=={1:1100,2:1100,3:3300,4:5500}[n]
        assert audit['manifest_sha256']==s.sha(home/'cache/cache_manifest.json')
        assert read(home/'delivery.json')['cache_audit']['sha256']==s.sha(home/'full_audit_v1/report.json')
        runtime=control/('formal_w8' if n<3 else 'encoding_only_w8_v1')/'runtime'
        for suffix in ('01_encode_full','02_cache_audit','03_deliver'):
            record=read(runtime/(group+'_'+suffix+'_exit.json'))
            assert record['actual_exit_code']==0 and record['residual_owned_members']==[]
    official=read(control/'smoke/runtime/closure.json');assert official['actual_exit_code']==0 and official['completed_phases']==64
    random=read(control/'bootstrap_runtime_v3/random_diagnostic_acceptance.json')
    assert random['actual_exit_code']==0 and random['effective_completed_phases']==64
    release=read(s.HERE/'ROOT_RELEASE.json')
    assert release['actual_exit_code']==0 and release['authorized_launch'] is True
    assert release['launch_plan_sha256']==s.sha(s.HERE/'launch_plan.json')
    assert release['inference_precision']=='fp16' and release['offline_encoder_batch']==32 and release['online_reference_batch']==1
    assert release['batch_reference_policy']=='same_batch_official_fp16_reference_v1'
    assert release['training_updates']==release['SR_episodes']==0
    proof_path=Path(release['precision_reference_proof_path'])
    assert proof_path.resolve().is_relative_to(Path(s.AUX).resolve())
    assert release['precision_reference_proof_sha256']==s.sha(proof_path)
    sys.path.insert(0,str(s.CASE/'cache'))
    from fp16_reference_producer_v1 import validate_reference_proof
    proof=read(proof_path)
    validate_reference_proof(proof,dict(encoder_spec_sha256=s.SPEC_SHA),proof['frozen_encoder_state_sha256'])
    return dict(actual_exit_code=0,VIS_cache_groups_adopted=4,old_smoke_metadata_only=True,
        new_precision_reference_proof=str(proof_path),offline_encoder_batch=32,online_reference_batch=1,
        legacy_cross_batch_diagnostic=proof['legacy_diagnostic'],training_updates=0,SR_episodes=0,at=time.time())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    plan=read(s.HERE/'launch_plan.json');inputs(plan)
    if not args.execute:print(json.dumps(dict(prepared_only=True,phases=12,offline_encoder_batch=32,online_reference_batch=1,NN_forward=False)));return
    adopted=admission(plan);directory=Path(s.HERE)
    assert not (directory/'runtime').exists()
    with (directory/'adopted_metadata.json').open('x') as stream:json.dump(adopted,stream,indent=2)
    sys.path.insert(0,str(s.CASE/'control'));import serial as original
    original.validate_plan=inputs
    old_loader=original.load_supervisor
    def loader(path):
        module=old_loader(path);old_save=module.save
        def save(name,value):
            if name=='closure.json':value=dict(value,group_ids=s.GROUPS,inference_precision='fp16',offline_encoder_batch=32,online_reference_batch=1,storage_dtype='float32')
            old_save(name,value)
        module.save=save;return module
    original.load_supervisor=loader
    sys.argv=[str(Path(__file__)),'--plan-directory',str(directory),'--execute']
    return original.main()

if __name__=='__main__':raise SystemExit(main())
