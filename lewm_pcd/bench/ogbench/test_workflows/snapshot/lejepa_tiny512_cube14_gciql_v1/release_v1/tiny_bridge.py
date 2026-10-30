"""Tiny-specific metadata and counted512 boundaries over frozen existing code."""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys

ROOT=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE_NAME='lejepa_tiny512_cube14_gciql_v1'
FAMILY='lejepa_tiny512'
NAMES={1:'single',2:'double',3:'triple',4:'quadruple'}
COUNTS={1:dict(train=1000,val=100),2:dict(train=1000,val=100),3:dict(train=3000,val=300),4:dict(train=5000,val=500)}
FUNCTIONAL_BRIDGE=ROOT/'pldm_cube14_gciql_v1/functional.py'
CACHE_BRIDGE=ROOT/'pldm_cube14_embeddings_v1/cache_bridge.py'
FUNCTIONAL_BRIDGE_SHA='d777c27d82a357ae0ca429a6bea7721c51e9860f9ca688c2c22dc3ed4dedbbfa'
CACHE_BRIDGE_SHA='f898715a33c48148d5deacb35a434ffd74db38c1d561947739c87c1fd6b53409'
SOURCE_SHA={'inputs':'ff66c68f50a6c73bd7928e4281a30adf4686538a2829d31d6981a4fd86b5f2ec','prepare':'17141ea3dffd485455f9a54dbcb52a3b569c583b2f1326d7074d2d96b4a78464','runtime':'03986a4c4dd03598b9ae06e57955ce164ad3a73fcc655dcc73acf12df0632326','audit_native':'c2840c183d9279623be16b5f1e882a4b121a856039fd74e8f4584fca82f8501a','native_sr':'ae16abe24a609719630357f69c2c472d4df2f58a63f044edd9fbe028f4f10e41'}
WIDTH_EDITS={'inputs':('dim in (768, 1024)','dim in (512, 768, 1024)'), 'prepare':('d in (768,1024)','d in (512,768,1024)'), 'runtime':('d in (768,1024)','d in (512,768,1024)')}
CPU_EDITS={'producer':[("assert policy == 'shared_gpu'","assert policy == 'shared_cpu'"),("if policy == 'shared_gpu':\n        activate('cuda', None)","if policy == 'shared_cpu':\n        activate('cpu', None)"),("assert selected_device == 'cuda'  # shared encoder API; no CPU/GPU speed selection","assert selected_device == 'cpu'  # shared encoder API; no CPU/GPU speed selection")], 'audit':[("assert policy == 'shared_gpu', 'Shared encoder GPU preprocessing required'","assert policy == 'shared_cpu', 'Shared Tiny CPU preprocessing required'"),("selected_device = 'cuda'","selected_device = 'cpu'"),("if policy not in ('cpu', 'shared_gpu'):","if policy not in ('cpu', 'shared_cpu'):")]}
_MISSING=object()
_FUNCTIONAL_API=None


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def _load(path,digest,name):
    assert sha(path)==digest,str(path)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


ENCODER=_load(Path(__file__).with_name('tiny_encoder.py'),'d32c467fc02fd3680d3c12d737e692dd423a9f1b5d7a0f9f20c5e21584da3b86','_tiny_metadata_encoder')


def case():return ROOT/CASE_NAME


def group(n):
    assert type(n) is int and n in NAMES
    return f'lejepa_tiny512_cube{n}'


def cfgpath(n,purpose):return case()/'configs'/purpose/(group(n)+'.json')


def functional_api():
    global _FUNCTIONAL_API
    if _FUNCTIONAL_API is None:_FUNCTIONAL_API=_load(FUNCTIONAL_BRIDGE,FUNCTIONAL_BRIDGE_SHA,'_tiny_reused_functional')
    return _FUNCTIONAL_API


def functional_source(name,raw):
    assert name in SOURCE_SHA and hashlib.sha256(raw).hexdigest()==SOURCE_SHA[name]
    source=raw.decode()
    if name in WIDTH_EDITS:
        before,after=WIDTH_EDITS[name];assert source.count(before)==1
        source=source.replace(before,after,1)
    return source


def recipe():
    return dict(path=str(Path(__file__).resolve()),sha256=sha(__file__),reused_functional_SHA256=FUNCTIONAL_BRIDGE_SHA,
        sources={name:dict(original_SHA256=SOURCE_SHA[name],effective_SHA256=hashlib.sha256(functional_source(name,(ROOT/'foundation_visual_gciql_v1/functional'/f'{name}.py').read_bytes()).encode()).hexdigest(),counted_edits=[list(WIDTH_EDITS[name])] if name in WIDTH_EDITS else []) for name in SOURCE_SHA},original_native_audit_byteexact=True)


def check_config(cfg):
    n=cfg['cube_count'];enc=cfg['encoder_binding'];spec=ENCODER.validate_spec(read(enc['spec_path']))
    assert spec['cube_count']==n and sha(enc['spec_path'])==enc['spec_sha256']
    assert cfg['group_id']==cfg['role']==group(n) and cfg['modality']=='rgb'
    assert cfg['purpose'] in ('pilot','formal') and cfg['fresh'] and cfg['resume_from'] is None
    assert cfg['input_contract']==dict(current_dim=531,goal_dim=512+3*n,latent_dim=512,zero_padding=False)
    assert cfg['native_max_episode_steps']=={1:200,2:500,3:1000,4:1000}[n]
    assert cfg['encoder_binding']['family']==FAMILY and cfg['encoder_binding']['latent_dim']==512
    assert cfg['encoder_binding']['checkpoint_sha256']==spec['checkpoint_sha256']
    assert cfg['data_binding']['data_id']==f'cube-{NAMES[n]}-play-v0'
    assert cfg['camera']==spec['downstream_camera']


def _cache_paths(n):
    home=case()/'cache/groups'/group(n)
    runtime=case()/'control/full/runtime'
    exits=[runtime/(group(n)+suffix+'_exit.json') for suffix in ('_01_encode_full','_02_cache_audit','_03_deliver')]
    return home,exits


def cache_pins(cfg,require_seal=True):
    check_config(cfg);n=cfg['cube_count'];home,exits=_cache_paths(n)
    planpath=home/'plan.json';deliverypath=home/'delivery.json';auditpath=home/'full_audit_v1/report.json'
    plan,delivery,audit=map(read,(planpath,deliverypath,auditpath));validate_plan(plan)
    assert cfg['cache_plan_binding']==dict(path=str(planpath),sha256=sha(planpath))
    assert cfg['delivery_binding']==dict(path=str(deliverypath),sha256=sha(deliverypath))
    assert delivery['purpose']=='formal' and delivery['cube_count']==n and delivery['latent_dim']==512
    for key in ('encoder_binding','data_binding','feature_loader_arguments','camera','cache_audit'):assert cfg[key]==delivery[key],key
    enc=cfg['encoder_binding'];assert enc['spec_path']==plan['encoder_spec'] and enc['spec_sha256']==sha(plan['encoder_spec'])
    assert enc['adapter']==dict(path=plan['encoder_adapter'],sha256=sha(plan['encoder_adapter']))
    spec=ENCODER.validate_spec(read(plan['encoder_spec']));assert spec['cube_count']==n
    assert plan['checkpoint']==spec['checkpoint'] and plan['checkpoint_sha256']==spec['checkpoint_sha256']==enc['checkpoint_sha256']
    assert plan['camera']==spec['downstream_camera']
    assert cfg['cache_audit']==dict(path=str(auditpath),sha256=sha(auditpath))
    assert int((auditpath.parent/'actual_exit_code').read_text())==audit['actual_exit_code']==0
    assert audit['episodes_verified']==sum(COUNTS[n].values()) and audit['raw_rows_verified']==sum(COUNTS[n].values())*1001
    assert audit['main_rows_verified']==sum(COUNTS[n].values())*1000 and audit['latent_dim']==512 and audit['dtype']=='float32'
    assert all(audit[k] for k in ('all_new_latent_SHA_shape_dtype_finite','all_raw_ID_and_true_terminal_next_contracts','tracked_stats_stable'))
    manifest=Path(plan['cache_dir'])/'cache_manifest.json';seal=Path(plan['cache_dir'])/'cache_seal.json'
    assert audit['manifest_sha256']==sha(manifest) and audit['producer_seal_sha256']==sha(seal) and audit['plan_sha256']==sha(planpath)
    control=[]
    for path in exits:
        result=read(path);assert result['actual_exit_code']==0 and result['residual_owned_members']==[]
        control.append(dict(path=str(path),sha256=sha(path)))
    assert cfg['cache_control_binding']==control
    pins={str(q):sha(q) for q in [Path(__file__),Path(__file__).with_name('tiny_encoder.py'),planpath,deliverypath,auditpath,manifest,seal,*exits]}
    for rel,digest in plan['source_pins'].items():
        path=PurePosixPath(rel);assert not path.is_absolute() and '..' not in path.parts
        actual=ROOT.joinpath(*path.parts);assert sha(actual)==digest,str(actual);pins[str(actual)]=digest
    pins[spec['asset_manifest_path']]=spec['asset_manifest_sha256'];assert sha(spec['asset_manifest_path'])==spec['asset_manifest_sha256']
    for q,key in [(plan['encoder_spec'],'encoder_spec_sha256'),(plan['encoder_adapter'],'encoder_adapter_sha256')]:assert sha(q)==plan[key];pins[q]=plan[key]
    if require_seal:
        expected=case()/'configs/seals'/(group(n)+'.json');assert cfg['config_seal_path']==str(expected)
        record=read(expected);assert record['actual_exit_code']==0 and record['group_id']==group(n)
        assert record['functional_overlay']==recipe()
        assert record['delivery_binding']==cfg['delivery_binding'] and record['cache_plan_binding']==cfg['cache_plan_binding'] and record['cache_control_binding']==cfg['cache_control_binding']
        assert record['parent_template']==cfg['parent_template'] and sha(cfg['parent_template']['path'])==cfg['parent_template']['sha256']
        for purpose in ('pilot','formal'):
            path=cfgpath(n,purpose);assert record['configs'][purpose]==dict(path=str(path),sha256=sha(path));pins[str(path)]=sha(path)
        assert read(cfgpath(n,cfg['purpose']))==cfg
        pins[str(expected)]=sha(expected)
    return pins


def prepare_configs(pilot_path,delivery_path,formal_path,cache_control):
    """Pure metadata at original hooks phase: closed-cache proof -> exclusive cfg pair+seal."""
    pilot_path,delivery_path,formal_path,cache_control=map(Path,(pilot_path,delivery_path,formal_path,cache_control))
    matches=[n for n in NAMES if pilot_path==cfgpath(n,'pilot')]
    assert len(matches)==1,'Hooks config must be the exact new pilot path'
    n=matches[0];g=group(n);home,exits=_cache_paths(n)
    assert delivery_path==home/'delivery.json' and formal_path==cfgpath(n,'formal')
    assert cache_control==case()/'control/full/runtime'
    delivery=read(delivery_path);assert delivery['purpose']=='formal' and delivery['group_id']==g
    control=[]
    for path in exits:
        result=read(path);assert type(result['actual_exit_code']) is int and result['actual_exit_code']==0 and result['residual_owned_members']==[]
        control.append(dict(path=str(path),sha256=sha(path)))
    modules=bootstrap();template=ROOT/'cube23_distance_gciql_v1/five_group_extension_v1/configs/G08_cube3_rgb.json'
    sealpath=case()/'configs/seals'/(g+'.json');expected={}
    for purpose,path in [('pilot',pilot_path),('formal',formal_path)]:
        view=copy.deepcopy(delivery);view['purpose']=purpose
        cfg=modules['prepare'].make_config(read(template),view)
        original=copy.deepcopy(cfg);execution=case()/'outputs'/purpose
        cfg.update(execution_root=str(execution),output=str(execution/'runs'/g))
        assert {k:v for k,v in cfg.items() if k not in ('execution_root','output')}=={k:v for k,v in original.items() if k not in ('execution_root','output')}
        cfg.update(delivery_binding=dict(path=str(delivery_path),sha256=sha(delivery_path)),
            cache_plan_binding=dict(path=str(home/'plan.json'),sha256=sha(home/'plan.json')),
            cache_control_binding=control,parent_template=dict(path=str(template),sha256=sha(template)),
            config_seal_path=str(sealpath))
        cfg['commands']=modules['prepare'].commands(path,cfg)
        modules['runtime'].check_config(cfg);cache_pins(cfg,require_seal=False);expected[purpose]=cfg
    targets=[pilot_path,formal_path,sealpath]
    if any(path.exists() for path in targets):
        assert all(path.is_file() for path in targets),'Partial cfg/seal exists; preserve and inspect'
        assert read(pilot_path)==expected['pilot'] and read(formal_path)==expected['formal'],'Existing cfg bytes/fields differ'
        cache_pins(expected['pilot']);cache_pins(expected['formal'])
        return dict(actual_exit_code=0,created=False,seal_path=str(sealpath),seal_sha256=sha(sealpath),NN_performed=False)
    for purpose,path in [('pilot',pilot_path),('formal',formal_path)]:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:json.dump(expected[purpose],stream,indent=2,allow_nan=False);stream.write('\n')
    sealed=dict(actual_exit_code=0,group_id=g,configs={purpose:dict(path=str(path),sha256=sha(path)) for purpose,path in [('pilot',pilot_path),('formal',formal_path)]},
        delivery_binding=expected['formal']['delivery_binding'],cache_plan_binding=expected['formal']['cache_plan_binding'],cache_control_binding=control,
        parent_template=expected['formal']['parent_template'],functional_overlay=recipe(),NN_performed=False)
    sealpath.parent.mkdir(parents=True,exist_ok=True)
    with sealpath.open('x') as stream:json.dump(sealed,stream,indent=2,allow_nan=False);stream.write('\n')
    cache_pins(expected['pilot']);cache_pins(expected['formal'])
    return dict(actual_exit_code=0,created=True,seal_path=str(sealpath),seal_sha256=sha(sealpath),NN_performed=False)


def bootstrap():
    lib=functional_api();factory=lib.effective_source;oldroot=lib.ROOT
    names=tuple('_original14_'+x for x in lib.MODULES);saved={x:sys.modules.get(x,_MISSING) for x in names}
    try:
        lib.ROOT=ROOT
        lib.effective_source=lambda name,pldm=False:functional_source(name,factory(name,pldm=False).encode())
        modules=lib.bootstrap(pldm=False)
    finally:
        lib.effective_source=factory;lib.ROOT=oldroot
        for name,value in saved.items():
            if value is _MISSING:sys.modules.pop(name,None)
            else:sys.modules[name]=value
    prepare,runtime,audit=(modules[x] for x in ('prepare','runtime','audit_native'));prepare.CASE=case()
    oldmake,oldcommands,oldcheck=prepare.make_config,prepare.commands,runtime.check_config
    def make_config(template,delivery):
        cfg=oldmake(template,delivery);check_config(cfg);return cfg
    def commands(path,cfg):
        result=oldcommands(path,cfg)
        for key,argv in result.items():result[key]=[str(Path(__file__).resolve()),'runtime' if Path(argv[0]).name=='runtime.py' else 'native',*argv[1:]]
        result['native_audit']=[str(Path(__file__).resolve()),'audit',*[v for v in result['native'][2:] if v!='--gate-smoke']]
        return result
    def checked(cfg):oldcheck(cfg);check_config(cfg)
    prepare.make_config,prepare.commands,runtime.check_config=make_config,commands,checked
    oldinstall=runtime.install
    def install(cfg):
        cache_pins(cfg);installed,boundary=oldinstall(cfg);oldprepare=installed.prepare
        def traced(config,shared_datasets=None):
            pins=cache_pins(config);samplers,resources,binding=oldprepare(config,shared_datasets)
            binding['code'].update(pins);binding['runtime']['functional_overlay']=recipe()
            return samplers,resources,binding
        installed.prepare=lib.bound(traced,modules);boundary.prepare,boundary.get_batch=installed.prepare,installed.get_batch
        return installed,boundary
    runtime.install=lib.bound(install,modules)
    oldbinding=audit.source_binding
    def source_binding(cfg,path):
        pins=cache_pins(cfg);result=oldbinding(cfg,path);result['source_pins'].update(pins);result['functional_overlay']=recipe();return result
    audit.source_binding=source_binding
    return modules


def validate_plan(plan):
    n=plan['cube_count'];spec=ENCODER.validate_spec(read(plan['encoder_spec']))
    assert spec['cube_count']==n and sha(plan['encoder_spec'])==plan['encoder_spec_sha256']
    assert plan['encoder_family']==FAMILY and plan['latent_dim']==512
    assert plan['renderer_workers']==plan['prefetch']==2 and plan['encoder_batch']==32 and plan['torch_cpu_threads']==1
    assert plan['pixel_preprocess_device']=='shared_cpu' and plan['encoder_preprocessing_device']=='cpu' and plan['host_tensor_dtype']=='uint8'
    assert plan['inference_precision']=='fp32' and plan['storage_dtype']=='float32'
    assert plan['training_updates']==plan['SR_episodes']==0 and plan['raw_rows_per_episode']==1001
    assert plan['split_episodes']==COUNTS[n]
    assert plan['checkpoint']==spec['checkpoint'] and plan['checkpoint_sha256']==spec['checkpoint_sha256']
    assert Path(plan['encoder_adapter'])==case()/'release_v1/tiny_encoder.py' and plan['encoder_adapter_sha256']==sha(Path(__file__).with_name('tiny_encoder.py'))
    assert plan['encoder_preprocess']==spec['preprocessing']


@contextmanager
def _cache_scope():
    names=('common','producer_bridge','audit_bridge','render_adapter');before={n:sys.modules.get(n,_MISSING) for n in names};path=list(sys.path)
    try:
        for n in names:sys.modules.pop(n,None)
        yield
    finally:
        for n,value in before.items():
            if value is _MISSING:sys.modules.pop(n,None)
            else:sys.modules[n]=value
        sys.path[:]=path


def _cache_api(mode):
    assert mode in ('producer','audit')
    lib=_load(CACHE_BRIDGE,CACHE_BRIDGE_SHA,'_tiny_reused_cache');lib.ROOT=ROOT
    return lib.foundation_api(mode,expanded=True)


def _cpu_source(plan,mode,original):
    validate_plan(plan);source=original(plan)
    for before,after in CPU_EDITS[mode]:
        assert source.count(before)==1,(mode,before);source=source.replace(before,after,1)
    return source


def cache_effective_source(plan,mode='producer'):
    with _cache_scope():
        api=_cache_api(mode);return _cpu_source(plan,mode,api.effective_source if mode=='producer' else api.source)


def _cache_main(mode,args):
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--plan',type=Path,required=True)
    parsed,remaining=parser.parse_known_args(args);plan=read(parsed.plan);validate_plan(plan)
    spec=ENCODER.validate_spec(read(plan['encoder_spec']));assert spec['cube_count']==plan['cube_count']
    for key in ('checkpoint','checkpoint_sha256'):assert spec[key]==plan[key],key
    assert sha(plan['encoder_spec'])==plan['encoder_spec_sha256'] and sha(plan['encoder_adapter'])==plan['encoder_adapter_sha256']
    with _cache_scope():
        api=_cache_api(mode);original=api.effective_source if mode=='producer' else api.source
        if mode=='producer':api.effective_source=lambda value:_cpu_source(value,mode,original)
        else:api.source=lambda value:_cpu_source(value,mode,original)
        module=api.load(plan);module.__file__=str(Path(__file__).resolve())
        if mode=='producer':
            module.HERE=ROOT/'foundation_visual_gciql_v1/cache'
            module.CONTRACT_KEYS+=('prefetch','inference_precision','storage_dtype','encoder_preprocess')
        else:
            module.PRODUCER=CASE_NAME+'/release_v1/tiny_bridge.py';module.PROTOCOL_KEYS+=('prefetch','inference_precision','storage_dtype','encoder_preprocess')
        previous=list(sys.argv)
        try:sys.argv=[str(Path(__file__).resolve()),'--plan',str(parsed.plan),*remaining];return module.main()
        finally:sys.argv[:]=previous


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('mode',choices=('cache-producer','cache-audit','runtime','native','audit'))
    mode,args=parser.parse_known_args(argv)
    if mode.mode.startswith('cache-'):return _cache_main('producer' if mode.mode=='cache-producer' else 'audit',args)
    if mode.mode=='runtime' and args and args[0]=='verify-hooks':
        hooks=argparse.ArgumentParser();hooks.add_argument('action',choices=('verify-hooks',));hooks.add_argument('--config',type=Path,required=True)
        hooks.add_argument('--delivery',type=Path,required=True);hooks.add_argument('--formal-config',type=Path,required=True);hooks.add_argument('--cache-control',type=Path,required=True)
        h=hooks.parse_args(args);print(json.dumps(prepare_configs(h.config,h.delivery,h.formal_config,h.cache_control)))
        args=['verify-hooks','--config',str(h.config)]
    modules=bootstrap();target={'runtime':'runtime','native':'native_sr','audit':'audit_native'}[mode.mode];previous=list(sys.argv)
    try:sys.argv=[str(Path(__file__).resolve()),*args];return modules[target].main()
    finally:sys.argv[:]=previous


if __name__=='__main__':raise SystemExit(main())
