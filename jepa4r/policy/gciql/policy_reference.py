"""Device-labelled CPU endpoint reference, independently replayed from raw NPZ."""
import hashlib,json,importlib.metadata
from pathlib import Path

def config_digest(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def write_cpu_reference(runtime,gate,output,trainer,boundary,runtime_recipe):
    import flax,jax,numpy as np
    from jepa4r.policy.gciql.gates.arrays import compare_array_maps,array_identity
    from .gate_checks import _tool
    state_replay=_tool('state_replay')
    state_from_arrays,flatten_state=state_replay.state_from_arrays,state_replay.flatten_state
    from .workflow_downstream import read,pinned,sha,save_new
    if jax.default_backend()!='cpu':raise ValueError('CPU endpoint reference requires an actual CPU process')
    gate,output=Path(gate),Path(output)
    report=read(gate/'report.json')
    if report.get('actual_exit_code')!=0 or report.get('updates')not in (1,2):
        raise ValueError('reference requires an actual completed small-budget producer')
    if report['actual_source_binding']['config']!=runtime or read(pinned(runtime_recipe))!=runtime:
        raise ValueError('reference recipe differs from the real training endpoint')
    raw_path=gate/'full_state.npz';batch_path=gate/'next_batch.npz'
    if sha(raw_path)!=report['full_state_sha256']:raise ValueError('raw endpoint export SHA differs')
    with np.load(raw_path,allow_pickle=False)as data:raw={k:data[k].copy()for k in data.files}
    with np.load(batch_path,allow_pickle=False)as data:batch={k:data[k].copy()for k in data.files}
    template=trainer.create_agent(trainer.model.GCIQLAgent,runtime['seed'],batch,trainer.agent_config(runtime))
    state,mapping=state_from_arrays(flax.serialization.to_state_dict(template),raw)
    shadow=flax.serialization.from_state_dict(template,state)
    shadow=jax.device_put(shadow,jax.devices('cpu')[0])
    fingerprint=trainer.fingerprint(shadow)
    leaf_comparison=compare_array_maps(flatten_state(flax.serialization.to_state_dict(shadow)),raw,exact=True)
    if fingerprint!=report['saved_state']or fingerprint['leaves']!=153 or not leaf_comparison['passed']:
        raise ValueError('complete153 raw endpoint state mapping differs')
    original_versions=report['actual_source_binding']['versions']
    versions={name:importlib.metadata.version(name)for name in original_versions}
    if versions!=original_versions:raise ValueError('CPU endpoint reference library stack drifted')
    original_math=set(report['actual_source_binding']['code'].values())
    math_paths=[Path(trainer.model.__file__),Path(trainer.IMPL)/'utils/networks.py',
                Path(trainer.IMPL)/'utils/encoders.py',Path(trainer.IMPL)/'utils/flax_utils.py']
    math_sources={str(path):sha(path)for path in math_paths}
    if any(h not in original_math for h in math_sources.values()):
        raise ValueError('CPU reference factory/forward differs from the actual producer source')
    precision=str(jax.config.jax_default_matmul_precision)
    if precision!=report['actual_source_binding']['runtime']['JAX_default_matmul_precision']:
        raise ValueError('CPU endpoint reference changed the original matmul configuration')
    actions,proof=boundary.deterministic_action_reference(shadow,batch)
    action_path=output/'cpu_endpoint_actions.npy'
    with action_path.open('xb')as stream:np.save(stream,actions,allow_pickle=False)
    bind=lambda p:dict(path=str(p),sha256=sha(p))
    reference = dict(
        schema='gciql_cpu_endpoint_reference_v1',
        scope='new_CPU_reference_from_real_terminal_NPZ_not_GPU_original_output',
        actual_exit_code=0,
        backend='cpu',
        storage='terminal_full_state_npz',
        precision='fp32',
        updates=0,
        source_export=dict(
            producer_report=bind(gate / 'report.json'),
            full_state=bind(raw_path),
            next_batch=bind(batch_path),
            next_ids=bind(gate / 'next_ids.npz'),
            runtime_recipe=runtime_recipe,
        ),
        config_digest=config_digest(runtime),
        endpoint_fingerprint=fingerprint,
        complete153_mapping=leaf_comparison,
        leaf_mapping=mapping,
        math_sources=math_sources,
        versions=versions,
        jax_default_matmul_precision=precision,
        actions=dict(
            **bind(action_path),
            shape=list(actions.shape),
            dtype=str(actions.dtype),
            array_sha256=array_identity(actions),
        ),
        action_forward=proof,
        producer_source=bind(Path(__file__)),
        mapping_source=bind(Path(state_replay.__file__)),
        producer_source_sha256=sha(__file__),
    )
    target=output/'CPU_REFERENCE.json';save_new(target,reference)
    return dict(actual_exit_code=0,updates=0,cpu_reference=bind(target),endpoint_fingerprint=fingerprint,
                reference_semantics=reference['scope'])

def validate_cpu_reference(record,runtime,gate,expected):
    import numpy as np
    from .workflow_downstream import read,pinned,sha
    gate=Path(gate);reference=read(pinned(record))
    if (reference.get('schema')!='gciql_cpu_endpoint_reference_v1'or reference.get('actual_exit_code')!=0 or
        reference.get('backend')!='cpu'or reference.get('storage')!='terminal_full_state_npz'or
        reference.get('scope')!='new_CPU_reference_from_real_terminal_NPZ_not_GPU_original_output'or
        reference.get('updates')!=0 or reference.get('precision')!='fp32' or
        reference.get('config_digest')!=config_digest(runtime)):
        raise ValueError('guide reference is not a matching independent CPU endpoint reference')
    exports=reference['source_export']
    for name,path in [('producer_report',gate/'report.json'),('full_state',gate/'full_state.npz'),
                      ('next_batch',gate/'next_batch.npz'),('next_ids',gate/'next_ids.npz')]:
        if pinned(exports[name]).resolve()!=path.resolve()or exports[name]['sha256']!=sha(path):
            raise ValueError('CPU reference points to another real endpoint export: '+name)
    if (exports['full_state']['sha256']!=expected['full_state_sha256']or
            read(pinned(exports['runtime_recipe']))!=runtime or
            reference['endpoint_fingerprint']!=expected['saved_state']or
            reference['endpoint_fingerprint']['leaves']!=153 or
            not reference.get('complete153_mapping',{}).get('passed')or len(reference.get('leaf_mapping',[]))!=153):
        raise ValueError('CPU reference full-state/source/config mapping differs')
    original=expected['actual_source_binding']
    if reference['versions'] != original['versions'] or any(
        value not in set(original['code'].values()) for value in reference['math_sources'].values()
    ):
        raise ValueError('CPU reference source/library identities differ from its producer')
    for path,digest in reference['math_sources'].items():pinned(dict(path=path,sha256=digest))
    pinned(reference['producer_source']);pinned(reference['mapping_source'])
    if reference['producer_source']['sha256']!=reference['producer_source_sha256']:
        raise ValueError('CPU reference producer source identity differs')
    if reference['jax_default_matmul_precision']!=original['runtime']['JAX_default_matmul_precision']:
        raise ValueError('CPU reference changed matmul precision')
    actions=np.load(pinned(reference['actions']),allow_pickle=False)
    from jepa4r.policy.gciql.gates.arrays import array_identity
    with np.load(pinned(exports['full_state']),allow_pickle=False)as raw:
        rows=reference['leaf_mapping']
        if len(raw.files)!=153 or {row['path']for row in rows}!=set(raw.files):
            raise ValueError('CPU reference raw leaf-path mapping is incomplete')
        for row in rows:
            value=raw[row['path']]
            if (row['shape']!=list(value.shape)or row['endpoint_dtype']!=str(value.dtype)or
                    row['sha256']!=array_identity(value)):
                raise ValueError('CPU reference raw leaf dtype/shape/identity differs')
    proof=reference.get('action_forward',{})
    if (proof.get('key')!=0 or proof.get('temperature')!=0 or
            proof.get('full_agent_state_unchanged')is not True or proof.get('host_rng_unchanged')is not True):
        raise ValueError('CPU reference forward event/RNG contract differs')
    if (actions.shape!=(1024,5)or actions.dtype!=np.float32 or not np.isfinite(actions).all()or
            array_identity(actions)!=reference['actions']['array_sha256']):
        raise ValueError('complete CPU reference action array differs')
    return actions,reference
