"""Publish only a new metadata/configuration release; no NN package imports."""
import argparse
from pathlib import Path
import shutil
import time
import common as m

def preflight(here,weight_path):
    freeze=m.read(here/'FREEZE.json')
    for name,digest in freeze['files'].items():
        assert Path(name).name==name and m.sha(here/name)==digest,name
    assert m.sha(here/'expected_cache_bindings_v1.json')==m.EXPECTED_SHA
    expected=m.read(here/'expected_cache_bindings_v1.json')
    cache=m.cache_contract(expected);weight=m.read(weight_path);m.weight_contract(weight,expected)
    full=m.read(m.FORMAL);release=m.read(m.F/'control/ROOT_RELEASE_w8_v1.json')
    assert len(full['source_pins'])==255 and full['source_pins']==release['source_pins']
    m.pins_contract(full['source_pins'])
    assert full['source_pins'][str(m.F/'functional/native_sr.py')]=='ae16abe24a609719630357f69c2c472d4df2f58a63f044edd9fbe028f4f10e41'
    for p in (m.CASE,m.STORE):assert not p.exists() and not p.is_symlink(),'Existing new root must be inspected: '+str(p)
    for n in (3,4):
        g=f'visreg_vitb768_cube{n}'
        for p in (m.F/'runs'/g,m.F/'acceptance'/f'{g}.json',m.F/'native_sr'/g):
            assert not p.exists() and not p.is_symlink(),'Existing formal result must be inspected: '+str(p)
    prep=m.module('_original_prepare_for_new_IO',m.F/'functional/prepare.py')
    configs={}
    for n in (3,4):
        configs[n]={}
        for purpose in ('pilot','formal'):
            cfg=m.remap_config(m.original_config(n,purpose,prep),m.cfgpath(n,purpose),m.home(purpose),prep.commands)
            m.config_contract(cfg,n,purpose,prep);configs[n][purpose]=cfg
        # Purpose selects the bounded gate; every scientific/data field is identical.
        keep=lambda c:{k:v for k,v in c.items() if k not in m.IO_KEYS|{'purpose'}}
        assert keep(configs[n]['pilot'])==keep(configs[n]['formal'])
        for fullstage in (False,True):m.selected_phases(n,fullstage,configs[n])
    return freeze,full,configs,cache,m.space_admission()

def main():
    p=argparse.ArgumentParser();p.add_argument('--weight-identity',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    here=Path(__file__).resolve().parent
    freeze,full,configs,cache,space=preflight(here,a.weight_identity)
    if not a.execute:
        print(m.json.dumps(dict(actual_exit_code=0,prepared_only=True,NN_started=False,space=space,cache=cache)));return 0
    # Only fresh task-owned roots. A failed preparation remains inspectable; no cleanup/retry in place.
    m.CASE.mkdir();m.STORE.mkdir();m.H.mkdir()
    for name in [*freeze['files'],'FREEZE.json']:shutil.copyfile(here/name,m.H/name)
    (m.CASE/'outputs').symlink_to(m.STORE,target_is_directory=True)
    identity=m.CASE/'references/weight_identity.json';identity.parent.mkdir();shutil.copyfile(a.weight_identity,identity)
    pins=full['source_pins'].copy()
    for name,digest in freeze['files'].items():pins[str(m.H/name)]=digest
    pins[str(m.H/'FREEZE.json')]=m.sha(m.H/'FREEZE.json')
    refs=[m.FORMAL,m.SMOKE,m.F/'control/ROOT_RELEASE_w8_v1.json',m.TEMPLATE,identity]
    for n in (3,4):
        for purpose,cfg in configs[n].items():
            cp=m.cfgpath(n,purpose);m.write_new(cp,cfg);pins[str(cp)]=m.sha(cp)
        refs.append(m.delivery_path(n))
    for path in refs:pins[str(path)]=m.sha(path)
    rel=dict(schema_version=1,actual_exit_code=0,prepared_only=True,NN_started=False,created=time.time(),
             families=['visreg_vitb768'],cube_counts=[3,4],stages=list(m.STAGES),
             original_NN_sources_changed=False,original_configs_changed=0,configuration_IO_fields=sorted(m.IO_KEYS),
             full_data_pilot=True,backup_advisory=True,no_copy_phase=True,
             expected_cache_bindings_sha256=m.EXPECTED_SHA,source_pins=pins,cache_adoption=cache,space_at_publish=space,
             output_mapping=dict(logical=str(m.CASE/'outputs'),physical=str(m.STORE)),
             initial_artifact_budget_bytes=3100000000,reserve_bytes=512*(1<<20),checkpoint_count_including_pilots=26)
    m.write_new(m.CASE/'RELEASE.json',rel);plans={}
    for stage in m.STAGES:
        n=int(stage[4]);fullstage=stage.endswith('_full')
        plan=dict(schema_version=1,queue='visreg34_'+stage,mode='full' if fullstage else 'model_gates',
                  families=['visreg_vitb768'],cube_counts=[n],expected_features=0,
                  training_updates=1000000 if fullstage else 0,SR_episodes=250 if fullstage else 0,
                  diagnostic_optimizer_updates=0 if fullstage else 2,diagnostic_tiny_episodes=0 if fullstage else 1,
                  process_guard=str(m.ROOT/'prepare_runtime.py'),own_RSS_limit_bytes=60*(1<<30),
                  source_pins=pins,release_sha256=m.sha(m.CASE/'RELEASE.json'),phases=m.selected_phases(n,fullstage,configs[n]))
        path=m.stage_dir(stage)/'launch_plan.json';m.write_new(path,plan);plans[str(path)]=m.sha(path)
        m.validate_plan(plan,stage,predecessor=False)
    chain=dict(schema_version=1,stages=list(m.STAGES),stage_plans=plans,release_sha256=m.sha(m.CASE/'RELEASE.json'),
               NN_started=False,backup_advisory=True,automatic_restart=False,formal_updates=2000000,formal_SR_episodes=500)
    m.write_new(m.CASE/'CHAIN.json',chain)
    m.write_new(m.CASE/'PREPARED.json',dict(actual_exit_code=0,NN_started=False,source_pins=pins,
                release_sha256=m.sha(m.CASE/'RELEASE.json'),chain_sha256=m.sha(m.CASE/'CHAIN.json'),stage_plans=plans,
                config_count=4,cache=cache,space=space,created=time.time()))
    print(m.json.dumps(dict(actual_exit_code=0,prepared_only=True,NN_started=False,case=str(m.CASE),stages=list(m.STAGES),space=space)))
    return 0

if __name__=='__main__':raise SystemExit(main())
