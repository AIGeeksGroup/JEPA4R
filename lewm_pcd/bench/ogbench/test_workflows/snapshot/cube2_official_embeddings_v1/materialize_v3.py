"""Finish the failed pre-GPU materialization without editing frozen v2 or sources."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

V2_SHA = '28175b49b1548e512cc1b27fc720b988f8625f34b4dd1dd785779fe8a99216a5'
REF = 'official_state_3d_cache_v1/test_contracts.py'
REF_SHA = '2351c4be997958f6b241d63cce421ce1d0fb43d623d3a81c213e9ff9b436e919'
LINKS = ('source','external','reference','env_runtime','checkpoint_encoder.py','checkpoints')


def v2_module():
    path = Path(__file__).with_name('materialize.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == V2_SHA, 'Frozen v2 changed'
    spec = importlib.util.spec_from_file_location('_frozen_Double_materializer_v2',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_effective(module,original,shadow):
    generated,_ = module.build_sources(original)
    for name,expected in generated.items():
        actual = shadow/name
        assert actual.is_file() and not actual.is_symlink(), name
        assert actual.read_bytes() == expected, 'Changed effective source: '+name
    for name in LINKS:
        assert (shadow/name).is_symlink(), name
        assert (shadow/name).resolve() == (original/name).resolve(), 'Wrong shared reference: '+name
    return {name:module.sha(shadow/name) for name in generated}


def verify_all_pin_paths(original,shadow,module):
    checked=[]
    for mode,directory in [('2d',module.P2),('3d',module.P3)]:
        plan=json.loads((original/directory/'producer_plan.json').read_text())
        for relative in plan['source_pins']:
            relative=relative.replace(f'checkpoints/{mode}_epoch10/',f'checkpoints/cube2/{mode}_epoch10/')
            item=Path(relative)
            assert not item.is_absolute() and '..' not in item.parts,relative
            path=shadow/item
            assert path.is_file(), 'Unresolved source pin: '+relative
            checked.append(dict(mode=mode,relative=relative,resolved=str(path.resolve()),sha256=module.sha(path)))
    return checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root',type=Path,required=True)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--phase',choices=('finish-materialize',),required=True)
    args=parser.parse_args()
    original,case=args.original_root.resolve(),args.case.resolve()
    assert case != original and case.is_relative_to(original)
    shadow=case/'workspace'
    module=v2_module()
    effective=verify_effective(module,original,shadow)
    reference,source=shadow/REF,original/REF
    assert module.sha(source)==REF_SHA
    if reference.exists() or reference.is_symlink():
        assert reference.is_symlink() and reference.resolve()==source.resolve()
    else:
        reference.symlink_to(source)
    assert module.sha(reference)==REF_SHA
    pins=verify_all_pin_paths(original,shadow,module)
    all_plans=module.plans(original,case,shadow)
    target2,template3=shadow/module.P2/'producer_plan.json',case/'3d/producer_plan.template.json'
    assert not target2.exists() and not template3.exists(), 'Never overwrite a previous plan'
    module.write(target2,all_plans['2d'])
    module.write(template3,all_plans['3d'])
    module.write(case/'materialization_completion_v3.json',dict(actual_exit_code=0,
        original_v2_sha256=V2_SHA,effective_files=effective,all_source_pin_paths=pins,
        reference_only_test_not_executed=dict(path=str(reference),sha256=REF_SHA),
        plans=dict(twoD=str(target2),twoD_sha256=module.sha(target2),
                   threeD_template=str(template3),threeD_template_sha256=module.sha(template3)),
        source_v2_or_effective_modified=False,GPU_NN_or_old_tests_executed=False))
    print(json.dumps(dict(actual_exit_code=0,phase=args.phase,source_pin_entries_resolved=len(pins),
                         GPU_NN_or_old_tests_executed=False)))


if __name__=='__main__':
    main()
