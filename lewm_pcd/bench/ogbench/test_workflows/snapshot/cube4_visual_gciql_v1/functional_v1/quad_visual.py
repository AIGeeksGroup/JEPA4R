"""Quad visual input guard and migration metadata around original GCIQL code."""
import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
import types

ROOT=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
LEGACY=Path(os.environ.get('GCIQL_LEGACY_CASE',str(ROOT/'cube23_distance_gciql_v1')))
DISTANCE_SHA='b9906dc2f4e7a38826f5ee05031713618a6ce408ee5555302801505452758f90'
EXTENSION_SHA='a9735b9b4f3fec3b01f189e910e12e985cbba3a8ba13e1fafeac878f1222c9c1'


def distance_module():
    name='_quad_visual_original_distance_guard4'
    if name not in sys.modules:
        path=LEGACY/'distance_inputs.py';source=path.read_bytes()
        assert hashlib.sha256(source).hexdigest()==DISTANCE_SHA
        before='self.cubes not in (2,3)';after='self.cubes not in (2,3,4)'
        text=source.decode();assert text.count(before)==1
        effective=text.replace(before,after);assert effective.replace(after,before)==text
        module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
        exec(compile(effective,str(path),'exec'),module.__dict__)
        module.quad_guard_proof=dict(original_source_sha256=DISTANCE_SHA,
            effective_source_sha256=hashlib.sha256(effective.encode()).hexdigest(),
            replacement_count=1,exact_replacement=[before,after],original_file_unchanged=True)
    return sys.modules[name]


def guard_proof():return dict(distance_module().quad_guard_proof)


class VisualInputs:
    """No future samples: delegate the original four already-sampled RAM roles."""
    pending=False

    def __init__(self,*args,**kwargs):self.inner=distance_module().DistanceInputs(*args,**kwargs)

    def sample(self,batch_size,idxs=None,evaluation=False,*,prefetch_next=True):
        return self.inner.sample(batch_size,idxs=idxs,evaluation=evaluation)

    def barrier(self):pass

    def close(self):self.inner.close()

    def __getattr__(self,name):return getattr(self.inner,name)


def load_cached_features(*,source_rebind=None,**arguments):
    """Retain all original byte/finite/ID guards; translate only proven inode stats."""
    module=distance_module();original_read=module.read_json
    if source_rebind is None:return module.load_cached_features(**arguments)
    proof=original_read(source_rebind['path'],source_rebind['sha256'])
    assert proof['actual_exit_code']==0 and proof['cube_count']==4
    assert proof['data_id']=='cube-quadruple-play-v0' and proof['modality'] in ('rgb','pcd')
    assert Path(proof['cache_dir']).resolve()==Path(arguments['cache_dir']).resolve()
    assert proof['cache_manifest_sha256']==arguments['manifest_sha256']
    assert proof['checkpoint_sha256']==arguments['checkpoint_sha256'] and proof['raw_rows_per_episode']==1001
    split=arguments['split'];item=proof['splits'][split]
    assert arguments['expected_rows']==proof['split_episodes'][split]*1001
    source=Path(arguments['source_npz']).resolve()
    assert Path(item['source_npz']).resolve()==source
    current=module.file_stat(source)
    assert item['current_accepted_stat']==current
    assert item['sourcebeforeafterstat']['before']==item['sourcebeforeafterstat']['after']==current
    assert item['original_accepted_stat']['size']==current['size']
    assert item['current_source_hash_actual_exit_code']==0
    assert item['original_raw_sha256']==item['current_raw_sha256']
    assert len(bytes.fromhex(item['current_raw_sha256']))==32
    prior=proof['cache_full_SHA_verification']
    assert prior['scope']=='prior_producer_host'
    prior_audit=original_read(prior['path'],prior['sha256'])
    assert prior_audit['actual_exit_code']==0 and prior_audit['manifest_sha256']==arguments['manifest_sha256']
    assert prior_audit['episodes_verified']==sum(proof['split_episodes'].values())
    assert prior_audit['raw_rows_verified']==sum(proof['split_episodes'].values())*1001
    assert prior_audit['latent_dim']==192 and prior_audit['dtype']=='float32'
    projected=0
    def read_with_proven_stat(path,expected_sha):
        nonlocal projected
        record=original_read(path,expected_sha)  # Original immutable bytes SHA/stat first.
        if 'source_npz' in record and 'latent' in record:
            assert record['split']==split and Path(record['source_npz']['path']).resolve()==source
            assert record['source_npz']['accepted_stat']==item['original_accepted_stat']
            record=copy.deepcopy(record)
            record['source_npz']['accepted_stat']=current  # Only an in-memory metadata projection.
            projected+=1
        return record
    module.read_json=read_with_proven_stat
    try:features,binding=module.load_cached_features(**arguments)
    finally:module.read_json=original_read
    assert projected==arguments['expected_rows']//1001
    binding['source_rebind']=dict(path=str(Path(source_rebind['path']).resolve()),sha256=source_rebind['sha256'],
        modality=proof['modality'],source_npz=str(source),original_accepted_stat=item['original_accepted_stat'],
        current_accepted_stat=current,raw_sha256=item['current_raw_sha256'],
        source_whole_bytes_SHA_proof_reused=True,old_receipt_files_unchanged=True,
        in_memory_projection_only='source_npz.accepted_stat after immutable receipt SHA verification',
        prior_cache_full_byte_audit_scope='prior_producer_host')
    binding['vector_verification_this_loader']=dict(split=split,rows=arguments['expected_rows'],
        current_bytes_SHA_and_finite_verified=True,all_original_per_NPY_guards_preserved=True,
        receipt_feature_stream_sha256=binding['loaded_receipt_feature_stream_sha256'])
    return features,binding


def install():
    path=LEGACY/'five_group_extension_v1/extension_runtime.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==EXTENSION_SHA
    sys.path[:0]=[str(LEGACY/'five_group_extension_v1'),str(LEGACY/'state_priority_v1'),str(LEGACY),str(ROOT)]
    spec=importlib.util.spec_from_file_location('_quad_preserved_extension_runtime',path)
    extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension)
    extension.GROUPS.update(G11=(4,'rgb',211,204),G12=(4,'pcd',211,204))
    runtime=extension.install('baseline','async100');original_prepare=runtime.prepare
    def make_input(dataset,cfg,*,cubes,mode,features,workers):
        assert cubes==4 and mode in ('rgb','pcd') and workers==4 and features is not None
        return VisualInputs(dataset,cfg,cubes=cubes,mode=mode,features=features,workers=workers)
    def prepare(config,shared_datasets=None):
        assert config.get('executable',True) is True,'Incomplete asset template is not executable'
        assert config['group_id'] in ('G11','G12') and config['data_binding']['data_id']=='cube-quadruple-play-v0'
        assert config['data_binding']['splits']['train']['episodes']==5000
        assert config['data_binding']['splits']['val']['episodes']==500
        samplers,resources,binding=original_prepare(config,shared_datasets)
        for source in binding['source_bindings'].values():
            rebind=source['feature'].get('source_rebind')
            if rebind is not None:assert rebind['modality']==config['modality']
        binding['code'][str(Path(__file__).resolve())]=runtime.utility.sha(Path(__file__).resolve())
        binding['runtime']['cube4_visual_guard_extension']=guard_proof()
        binding['runtime']['no_runtime_encoder']=True
        return samplers,resources,binding
    runtime.DistanceInputs=make_input
    runtime.load_cached_features=load_cached_features
    runtime.prepare=prepare
    return runtime


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('gate','reload','train','accept'))
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    assert config.get('executable',True) is True,'Asset template cannot run'
    runtime=install()
    if args.action=='train':return runtime.run(args.config)
    assert args.output is not None
    if args.report:args.report.parent.mkdir(parents=True,exist_ok=True)
    boundary=importlib.import_module('gciql_boundary')
    try:
        if args.action=='gate':boundary.gate(config,args.output)
        elif args.action=='reload':boundary.reload_gate(config,args.output,args.report)
        else:boundary.accept(config,args.output,args.report)
    except BaseException:
        runtime.utility.atomic_json(args.report or args.output/'failure.json',dict(actual_exit_code=1,
            action=args.action,role=config['role'],error=traceback.format_exc()))
        raise
    return 0


if __name__=='__main__':raise SystemExit(main())
