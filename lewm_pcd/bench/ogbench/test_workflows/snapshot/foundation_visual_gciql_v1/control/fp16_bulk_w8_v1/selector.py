"""Exact twelve-phase V-JEPA canonical1 projection; immutable original NN/guard."""
import copy
import hashlib
import json
from pathlib import Path,PurePosixPath

ROOT=PurePosixPath('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE=ROOT/'foundation_visual_gciql_v1';AUX=CASE/'vj_precision_switch_v1';HERE=CASE/'control/fp16_bulk_w8_v1'
BASE=CASE/'control/encoding_only_w8_v1/launch_plan.json'
BASE_SHA='af2049822ec1e033dfaaed723f2ce298d6dd8884805634f71dbd0ea3bf39a596'
GROUPS=[f'vjepa21_vitl1024_cube{n}' for n in range(1,5)]
BRIDGE=AUX/'precision_bridge.py'
BRIDGE_SHA='8ac34a03554671ccdbe5cdd63d6a7f8881850a25e31afeb7fbc3321f18766463'
SPEC=AUX/'specs/vjepa21_vitl1024_fp16_reference_v1.json'
SPEC_SHA='319355585e08d686337a6d84a43fab75f64cf4c5e4dd33f041f0201d6f25d951'
ROOT_DISK_TARGET=PurePosixPath('/root/foundation_visual_gciql_31435_20261024_v1_outputs/cache_groups/fp16bulk_vjepa21_vitl1024_cube3')

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def cache_home(group):return CASE/'cache/fp16_bulk_w8_groups'/group
def cache_plan(old,group):
    value=copy.deepcopy(old)
    value.update(cache_dir=str(cache_home(group)/'cache'),encoder_spec=str(SPEC),encoder_spec_sha256=SPEC_SHA,
        encoder_adapter=str(BRIDGE),encoder_adapter_sha256=BRIDGE_SHA,inference_precision='fp16',batch_reference_policy='same_batch_official_fp16_reference_v1')
    value['source_pins'][str(BRIDGE.relative_to(ROOT))]=BRIDGE_SHA
    value['source_pins'][str((AUX/'precision_bridge.py').relative_to(ROOT))]='8ac34a03554671ccdbe5cdd63d6a7f8881850a25e31afeb7fbc3321f18766463'
    value['source_pins'][str(SPEC.relative_to(ROOT))]=SPEC_SHA
    assert value['renderer_workers']==value['prefetch']==8 and value['encoder_batch']==32
    assert value['output_dtype']=='float32'
    value['source_pins']['foundation_visual_gciql_v1/cache/fp16_reference_producer_v1.py']='20f936b9b81754f006c24723fd335e9a67990188e681e160707723e58773540c'
    value['source_pins']['foundation_visual_gciql_v1/cache/fp16_reference_audit_v1.py']='c1b8f1c77636effadbf67ce159c36118e46d2bbc996276de880899830aff1316'
    return value

def make_plan(base,pins):
    phases=[]
    for group in GROUPS:
        old=CASE/'cache/w8_groups'/group;new=cache_home(group)
        for suffix in ('01_encode_full','02_cache_audit','03_deliver'):
            found=[p for p in base['phases'] if p['name']==group+'_'+suffix];assert len(found)==1
            phase=copy.deepcopy(found[0]);args=phase['argv']
            args[:]=[str(new)+arg[len(str(old)):] if arg.startswith(str(old)+'/') else arg for arg in args]
            args[:]=[arg.replace(str(CASE/'cache/resource_w8.py'),str(CASE/'cache/fp16_reference_producer_v1.py')).replace(str(CASE/'cache/resource_audit_w8.py'),str(CASE/'cache/fp16_reference_audit_v1.py')) for arg in args]
            if suffix=='03_deliver':
                i=args.index('--producer-exit-file')+1;assert PurePosixPath(args[i]).parent==BASE.parent/'runtime'
                args[i]=str(HERE/'runtime'/PurePosixPath(args[i]).name)
            phases.append(phase)
    result=copy.deepcopy(base)
    result.update(groups=[g for g in base['groups'] if g['group_id'] in GROUPS],families=['vjepa21_vitl1024'],phases=phases,
        expected_features=11011000,source_pins=pins,base_plan_path=str(BASE),base_plan_sha256=BASE_SHA,
        inference_precision='fp16',offline_encoder_batch=32,online_reference_batch=1,batch_reference_policy='same_batch_official_fp16_reference_v1',storage_dtype='float32',
        training_updates=0,SR_episodes=0)
    return result

def validate(plan,base,pins):
    assert plan==make_plan(base,pins),'Only exact VJ phase subset and new path bindings are permitted'
    assert len(plan['phases'])==12 and plan['training_updates']==plan['SR_episodes']==0
    assert [g['group_id'] for g in plan['groups']]==GROUPS
    for phase in plan['phases']:
        args=phase['argv'];assert '--config' not in args
        assert not any(PurePosixPath(arg).name in ('runtime.py','native_sr.py','prepare.py') for arg in args)
    return True
