"""Fresh CPU factory restore and real cached val4 action probe; no encoder/update."""
import argparse,csv,hashlib,json,os,pickle,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['JAX_PLATFORMS']='cpu'
W=Path('/root/autodl-tmp/workspace/users/lewm');Q=W/'coordination/rl3_m0_m5_m6_visual_proprio_smoke_109_20261013_v1/formal_start_v1'
C=Q/'core_v1';OUT=Q/'afterhook_v1/pcd_action_probe'
sys.path.insert(0,str(C))
import numpy as np
import jax,flax,torch
import train
from reader import CachedLatents
from verify_segment import validate_input_kernels,current_dims
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):
    leaves,tree=jax.tree_util.tree_flatten(flax.serialization.to_state_dict(x))
    h=hashlib.sha256(str(tree).encode())
    for v in leaves:h.update(train.digest_array(np.asarray(v)).encode())
    return h.hexdigest(),len(leaves)

def parse_args(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Q/'plans/zero192_config.json')
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--counter',type=int,default=1000000)
    return parser.parse_args(argv)


_owned_pool=None

def probe(args):
    global _owned_pool
    OUT=args.output
    OUT.mkdir(exist_ok=False)
    cfg=json.loads(args.config.read_text());R=args.run or Path(cfg['output']);counter=args.counter
    assert counter==cfg['total_updates']==1000000 and cfg['batch_size']==1024 and cfg['goal_mode']=='latent_object_xyz'
    assert cfg['save_interval']==100000 and cfg['validation_interval']==5000
    freeze_path=Q/'adapter_v1/freeze_v1/manifest.json';freeze=json.loads(freeze_path.read_text())
    assert sha(freeze_path)=='03f3c3551a46c80bc1f82f4472781369d806d1cdb3f773c7b8da6c53b9dbe75e'
    pins={Path(x['path']).name:x['sha256'] for x in freeze['files']}
    assert {k:sha(C/k) for k in pins}==pins
    bridge,samplers,pool,frozen,agent_cfg,probe_binding=train.prepare(cfg,'cpu')
    _owned_pool=pool
    assert (pool is frozen if cfg['workers'] else pool is None) and isinstance(frozen,CachedLatents) and not hasattr(frozen,'world_model')
    assert jax.default_backend()=='cpu' and not torch.cuda.is_initialized()
    snapshot=R/f'checkpoints/update_{counter:09d}';manifest=json.loads((snapshot/'manifest.json').read_text())
    assert sha(snapshot/f'params_{counter}.pkl')==manifest['agent_sha256']
    assert sha(snapshot/'run_state.pkl')==manifest['sidecar_sha256']
    with (snapshot/'run_state.pkl').open('rb') as f:side=pickle.load(f)
    assert side['binding']['config']==cfg
    assert side['binding']['cache_sources']==probe_binding['cache_sources']
    assert side['binding']['frozen_full_state']==train.frozen_hash(frozen)
    from utils.flax_utils import restore_agent
    template=train.create_agent(cfg,agent_cfg)
    agent=restore_agent(template,str(snapshot),counter)
    with (snapshot/f'params_{counter}.pkl').open('rb') as f:saved=pickle.load(f)['agent']
    aa,ta=jax.tree_util.tree_flatten(saved);bb,tb=jax.tree_util.tree_flatten(flax.serialization.to_state_dict(agent))
    assert ta==tb and len(aa)==153
    for x,y in zip(aa,bb):
        assert np.asarray(x).dtype==np.asarray(y).dtype and np.isfinite(np.asarray(y)).all()
        np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
    actual_input_kernels=validate_input_kernels(agent.network.params,cfg)
    before,leaves=digest(agent)
    assert int(agent.network.step)==counter+1 and int(agent.network.opt_state[0].count)==counter
    np.random.set_state(side['rng']['numpy']);rng_before=np.random.get_state()
    raw=samplers['val'].sample(4);frozen.reader.split='val';batch=frozen.batch(raw)
    assert batch['observations'].shape==batch['next_observations'].shape==(4,current_dims(cfg))
    assert batch['value_goals'].shape==batch['actor_goals'].shape==(4,195) and batch['actions'].shape==(4,5)
    assert cfg['visual_mode'] in ('cache','zero192')
    for role in ('observations','next_observations','value_goals','actor_goals'):
        latent=frozen.lookups['val'].lookup(role,raw[role],paired_current_ids=raw['observations'] if role=='next_observations' else None)
        np.testing.assert_array_equal(batch[role][:,:192],np.zeros_like(latent) if cfg['visual_mode']=='zero192' else latent)
    for role in ('observations','next_observations'):
        np.testing.assert_array_equal(batch[role][:,192:],frozen.compacts['val']['observations'][raw[role].reshape(-1),:current_dims(cfg)-192])
    assert all(x.dtype==np.float32 and np.isfinite(x).all() for x in batch.values())
    expected=(np.linalg.norm(frozen.compacts['val']['observations'][raw['observations'].reshape(-1),19:22]-batch['value_goals'][:,-3:],axis=-1)<=np.float32(.4))
    np.testing.assert_array_equal(batch['rewards'],expected.astype(np.float32)-1)
    np.testing.assert_array_equal(batch['masks'],1-expected.astype(np.float32))
    actions=np.asarray(agent.sample_actions(batch['observations'],batch['actor_goals'],seed=jax.random.PRNGKey(17),temperature=0.0))
    assert actions.shape==(4,5) and actions.dtype==np.float32 and np.isfinite(actions).all() and np.abs(actions).max()<=1
    after,_=digest(agent);assert before==after
    frozen.guard();assert train.frozen_hash(frozen)==side['binding']['frozen_full_state']
    np.random.set_state(rng_before)
    with (R/'train.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    segment_start=json.loads((R/'resolved.json').read_text())['initial_counter']
    assert len(rows)==counter//5000-segment_start//5000
    zero_fields=['timing/io_seconds','timing/preprocessing_transfer_seconds','timing/encoder_copy_seconds']
    assert all(float(row[k])==0. for row in rows for k in zero_fields)
    measured=[float(row['timing/cache_lookup_seconds']) for row in rows]
    assert all(np.isfinite(v) and v>0 for v in measured)
    report=dict(status='passed_CPU_fresh_cached_action_probe',exit_code=0,checkpoint_sha256=manifest['agent_sha256'],
        sidecar_sha256=manifest['sidecar_sha256'],full153_saved_state_exact=True,all_four_roles_visual192_exact_zero=cfg['visual_mode']=='zero192',all_four_roles_match_source_cached_latents_or_zero=True,restored_network_step=counter+1,adam_count=counter,
        original_actor_source=str(C/'gciql.py'),original_actor_source_sha256=sha(C/'gciql.py'),actual_actor_class=type(agent).__name__,
        actual_input_kernel_shapes=actual_input_kernels,current_next_shape=[4,current_dims(cfg)],goal_shape=[4,195],action_shape=[4,5],precision='float32 arrays/unchanged official JAX precision defaults',jax_default_matmul_precision_actual=jax.config.jax_default_matmul_precision,jax_backend='cpu',torch_cuda_initialized=False,
        raw_IDs={k:v.tolist() for k,v in raw.items() if k in bridge.OBS_FIELDS},batch_sha256={k:train.digest_array(v) for k,v in batch.items()},
        action_sha256=train.digest_array(actions),actions=actions.tolist(),agent_before_sha256=before,agent_after_sha256=after,
        source_binding_before_after=side['binding']['frozen_full_state'],source_binding_semantics='cached source fingerprint, not WM tensor hash',
        optimizer_updates_performed=0,encoder_forwards_performed=0,
        encoder_zero_evidence={'source':'prepare_cached constructs CachedLatents only; encode_batch returns frozen.batch immediately for that type; CachedLatents.batch contains only cache/NumPy lookup and concatenation; no encoder orWM object exists.',
          'runtime':'Restored runtime object is CachedLatents, optional owned RAM thread pool and no world_model attribute; production resolved.runtime.runtime_encoder_forwards=0.',
          'timing_rows':len(rows),'production_encoder_time_fields_all_zero':True,'cache_lookup_seconds_min':min(measured),'cache_lookup_seconds_max':max(measured),
          'limitation':'Encoder time fields are deliberately assigned zero in cached adapter, not an independent profiler counter. Zero calls follow the audited executed branch, not timing alone.'},
        limitations=['Probe draws4validation samples in its own process then restores its NumPy RNG; no training artifact/RNG/checkpoint is modified.',
            'CPU finite cached action verifies restored policy interface, not online encoder or native success rate.',
            'No runtimeWM exists in this cached path; source-binding equality is not presented as re-extracted WM-state equality.'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','checkpoint_sha256','full153_saved_state_exact','action_shape','optimizer_updates_performed','encoder_forwards_performed','action_sha256']},indent=2))

def main(args):
    try:return probe(args)
    finally:
        if _owned_pool is not None:_owned_pool.close()


if __name__=='__main__':main(parse_args())
