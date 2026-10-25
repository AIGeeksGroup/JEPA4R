"""Frozen PCD LeWM with untouched official GCIQL updates and exact boundary resume."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from reader import CachedLatents, CanonicalLatentLookup, ParallelRows, ReadPool, encode_batch, load_bridge, state_dim


def helpers(config):
    bridge = load_bridge(config)
    if bridge.sha256(config['driver_helpers']) != config['driver_helpers_sha256']:
        raise ValueError('Driver helpers changed')
    return bridge.load_alias('pcd_gciql_driver_helpers', config['driver_helpers'])


def pin_run_imports():
    directory = Path(__file__).resolve().parent
    sys.path[:] = [str(directory)] + [entry for entry in sys.path
                                     if Path(entry or os.getcwd()).resolve() != directory]


def digest_array(value):
    value = np.asarray(value)
    h = hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def frozen_hash(frozen):
    if isinstance(frozen, CachedLatents):
        return frozen.binding_hash  # Source-cache identity, not a runtime WM tensor hash.
    h = hashlib.sha256()
    for name, value in sorted(frozen.world_model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(digest_array(array).encode())
    return h.hexdigest()


def frozen_guard(frozen):
    if isinstance(frozen, CachedLatents):
        frozen.guard()
        return
    if frozen.world_model.training or any(p.requires_grad or p.grad is not None
                                        for p in frozen.world_model.parameters()):
        raise ValueError('Frozen WM gradient/mode violation')


def close_resources(logger, pool):
    errors = []
    for resource in (logger, pool):
        if resource is not None:
            try:
                resource.close()
            except BaseException:
                errors.append(traceback.format_exc())
    return getattr(pool, 'exit_codes', {}), errors


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def host_rng():
    import torch
    return dict(numpy=np.random.get_state(), python=random.getstate(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_host_rng(state):
    import torch
    np.random.set_state(state['numpy']); random.setstate(state['python'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def sampler_binding(samplers):
    return {split: dict(size=s.size, config=s.config,
            arrays={key: digest_array(getattr(s, key)) for key in ('terminal_locs', 'initial_locs')},
            valid_idxs=digest_array(s.dataset.valid_idxs)) for split, s in samplers.items()}


def prepare(config, device='cuda'):
    if config.get('input_profile', 'online') not in ('online', 'cached'):
        raise ValueError('Unknown explicit input profile')
    if config.get('input_profile', 'online') == 'cached':
        return prepare_cached(config, device)
    bridge = load_bridge(config)
    source_path = Path(config['source_config'])
    if bridge.sha256(source_path) != config['source_config_sha256']:
        raise ValueError('Frozen source config changed')
    source = json.loads(source_path.read_text())
    locks = bridge.setup(source)
    pin_run_imports()  # spawn must import this run's train/reader, retaining official utils below it.
    logger_path = Path(source['workspace']) / source['official'] / 'impls/utils/log_utils.py'
    if bridge.sha256(logger_path) != config['log_utils_sha256']:
        raise ValueError('Original logger changed')
    import jax
    import torch
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    if device == 'cuda':
        assert torch.cuda.is_available() and jax.default_backend() == 'gpu'
    compacts = {split: bridge.load_compact(source, split) for split in ('train', 'val')}
    assert len(compacts['train']['observations']) == 1001000
    assert len(compacts['val']['observations']) == 100100
    cfg = bridge.agent_config(config['batch_size'])
    samplers = {split: bridge.make_sampler(compact, cfg, ids=True) for split, compact in compacts.items()}
    pool = ReadPool(config, source, compacts)
    try:
        frozen = bridge.FrozenLatents(source, 'pcd', ParallelRows(pool), device)
        floating_dtypes = sorted({str(x.dtype) for x in frozen.world_model.state_dict().values() if torch.is_floating_point(x)})
        assert floating_dtypes == ['torch.float32'], 'Full frozen WM must retain FP32 tensors'
        binding = dict(config=config, agent_config=cfg, source_locks=locks,
            sampler=sampler_binding(samplers), frozen_full_state=frozen_hash(frozen),
            code={p.name: bridge.sha256(p) for p in Path(__file__).parent.iterdir()
                  if p.name in ('train.py', 'reader.py', 'launch.py', 'test_formal.py', 'config.json')},
            versions={key: importlib.metadata.version(key) for key in
                      ('torch', 'torchvision', 'stable-pretraining', 'jax', 'jaxlib', 'flax', 'optax', 'numpy', 'h5py', 'hdf5plugin')},
            runtime=dict(torch_cuda=torch.version.cuda, matmul_precision=torch.get_float32_matmul_precision(),
                         matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                         frozen_floating_dtypes=floating_dtypes,
                         device=torch.cuda.get_device_name() if device == 'cuda' else device,
                         jax_backend=jax.default_backend()))
        return bridge, samplers, pool, frozen, cfg, binding
    except BaseException as error:
        exits, cleanup_errors = close_resources(None, pool)
        error.worker_exit_codes = exits
        if cleanup_errors:
            error.worker_cleanup_errors = cleanup_errors
        raise


def agent_class(config):
    if config.get('input_profile', 'online') == 'cached':
        bridge = load_bridge(config)
        path = Path(__file__).with_name('gciql.py')
        if bridge.sha256(path) != config['gciql_template_sha256']:
            raise ValueError('Versioned optional-goal GCIQL source changed')
        return bridge.load_alias('cached_official_gciql_template', path).GCIQLAgent
    from agents.gciql import GCIQLAgent
    return GCIQLAgent


def create_agent(config, cfg, observations=None, actions=None, goals=None):
    cached = config.get('input_profile', 'online') == 'cached'
    if observations is None:
        observations = np.zeros((1, 192+state_dim(config.get('state_profile', 'full28')) if cached else 192), np.float32)
        actions = np.zeros((1, 5), np.float32)
        goal_dim = 195 if config.get('goal_mode') == 'latent_object_xyz' else 3
        goals = np.zeros((1, goal_dim), np.float32) if cached else None
    kwargs = dict(ex_goals=goals) if cached else {}
    return agent_class(config).create(config['seed'], observations, actions, cfg, **kwargs)


def prepare_cached(config, device):
    if (config['algorithm'] != 'gciql' or config['goal_mode'] not in ('object_xyz', 'latent_object_xyz') or
        config['reward_mode'] != 'current_object_xyz' or config['workers'] not in (0, 2, 4)):
        raise ValueError('Explicit cached GCIQL input/reward/RAM-worker contract required')
    bridge = load_bridge(config)
    source_path = Path(config['source_config'])
    if bridge.sha256(source_path) != config['source_config_sha256']:
        raise ValueError('Frozen source config changed')
    source = json.loads(source_path.read_text())
    locks = bridge.setup(source)
    pin_run_imports()
    logger_path = Path(source['workspace']) / source['official'] / 'impls/utils/log_utils.py'
    if bridge.sha256(logger_path) != config['log_utils_sha256']:
        raise ValueError('Original logger changed')
    import jax
    import torch
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    if device == 'cuda':
        assert jax.default_backend() == 'gpu'
    compacts = {split: bridge.load_compact(source, split) for split in ('train', 'val')}
    assert len(compacts['train']['observations']) == 1001000
    assert len(compacts['val']['observations']) == 100100
    cfg = bridge.agent_config(config['batch_size'])
    assert cfg['gc_negative'] and cfg['encoder'] is None and cfg['alpha'] == 1.0
    samplers = {split: bridge.make_sampler(compact, cfg, ids=True) for split, compact in compacts.items()}
    cache = config['cache']
    for field in ('manifest', 'seal', 'terminal_manifest'):
        if bridge.sha256(cache[field]) != cache[field+'_sha256']:
            raise ValueError(f'Pinned cached {field} changed')
    lookups = {split: CanonicalLatentLookup(cache['manifest'], cache['terminal_manifest'], split)
               for split in ('train', 'val')}
    if any(lookup.binding['source_modality'] != cache['modality'] or
           lookup.binding['WM_checkpoint_sha256'] != source['locked_sha256'][source['routes'][cache['modality']]['checkpoint']]
           for lookup in lookups.values()):
        raise ValueError('Selected original WM/modality and cache producer differ')
    if Path(cache['seal']).resolve() != Path(cache['manifest']).with_name('cache_seal.json').resolve():
        raise ValueError('Explicit canonical cache seal differs')
    frozen = CachedLatents(lookups, compacts, config['visual_mode'], config['goal_mode'], config.get('state_profile', 'full28'), config['workers'])
    binding = dict(config=config, agent_config=cfg, source_locks=locks, sampler=sampler_binding(samplers),
        frozen_full_state=frozen_hash(frozen), frozen_full_state_semantics='cached_source_binding_not_runtime_WM',
        cache_sources=frozen.source_binding,
        original_state={split: digest_array(compact['observations']) for split, compact in compacts.items()},
        code={p.name: bridge.sha256(p) for p in Path(__file__).parent.iterdir()
              if p.name in ('train.py', 'reader.py', 'launch.py', 'test_formal.py', 'config.json', 'gciql.py')},
        versions={key: importlib.metadata.version(key) for key in ('torch', 'jax', 'jaxlib', 'flax', 'optax', 'numpy')},
        runtime=dict(jax_backend=jax.default_backend(), workers=config['workers'], runtime_encoder_forwards=0,
                     cached_feature_loading='readonly_RAM', torch_cuda_initialized=torch.cuda.is_initialized()))
    return bridge, samplers, frozen if config['workers'] else None, frozen, cfg, binding


def batch_for(frozen, sampler, split, config):
    frozen.reader.split = split
    raw = sampler.sample(config['batch_size'])
    return raw, encode_batch(frozen, raw, config['encoder_microbatch'])


def update(agent, batch, frozen):
    frozen_guard(frozen)
    # Original update already computes gradients and their statistics.
    started = time.perf_counter()
    agent, info = agent.update(batch)
    dispatched = time.perf_counter()
    metrics = {key: float(value) for key, value in info.items()}
    synchronized = time.perf_counter()
    frozen.last_update_timing = dict(jax_dispatch_seconds=dispatched-started,
                                    loss_sync_seconds=synchronized-dispatched)
    if not all(np.isfinite(x) for x in metrics.values()) or metrics['grad/norm'] <= 0:
        raise ValueError('Nonfinite update/loss/gradient or zero gradient norm')
    frozen_guard(frozen)
    return agent, metrics


def events_complete(counter, validation_count, config):
    return dict(committed_updates=counter, validation_count=validation_count,
                next_stage='sample_train', last_validation=counter - counter % config['validation_interval'],
                checkpoint_event_complete=True, next_update=counter + 1)


def save_boundary(agent, frozen, binding, output, counter, validation_count, budget_total):
    from utils.flax_utils import save_agent
    config = binding['config']; utility = helpers(config)
    assert validation_count == counter // config['validation_interval'], 'Validation event is not committed'
    frozen_guard(frozen)
    if frozen_hash(frozen) != binding['frozen_full_state']:
        raise ValueError('Full frozen WM bytes changed')
    parent = output / 'checkpoints'; created = not parent.exists(); parent.mkdir(exist_ok=True)
    if created: fsync_directory(output)
    destination = parent / f'update_{counter:09d}'
    stage = parent / f'.update_{counter:09d}.tmp'
    if destination.exists() or stage.exists():
        raise FileExistsError('Checkpoint transaction already exists')
    stage.mkdir()
    save_agent(agent, str(stage), counter)  # The only serialized full agent state.
    sidecar = dict(binding=binding, rng=host_rng(), budget_total=budget_total,
                   events=events_complete(counter, validation_count, config),
                   network_step=int(agent.network.step), adam_count=int(agent.network.opt_state[0].count))
    assert sidecar['network_step'] == counter + 1 and sidecar['adam_count'] == counter
    with (stage / 'run_state.pkl').open('xb') as f:
        pickle.dump(sidecar, f); f.flush(); os.fsync(f.fileno())
    agent_path = stage / f'params_{counter}.pkl'
    with agent_path.open('rb') as f:
        os.fsync(f.fileno())
    utility.atomic_json(stage / 'manifest.json', dict(counter=counter,
        agent_sha256=utility.sha(agent_path), sidecar_sha256=utility.sha(stage / 'run_state.pkl')))
    fsync_directory(stage)
    os.rename(stage, destination)
    fsync_directory(parent)
    return destination


def restore_boundary(template, snapshot, binding, total_updates):
    from utils.flax_utils import restore_agent
    utility = helpers(binding['config'])
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    counter = manifest['counter']
    assert snapshot.name == f'update_{counter:09d}', 'Only a committed transaction directory can resume'
    assert utility.sha(snapshot / f'params_{counter}.pkl') == manifest['agent_sha256']
    assert utility.sha(snapshot / 'run_state.pkl') == manifest['sidecar_sha256']
    with (snapshot / 'run_state.pkl').open('rb') as f:
        sidecar = pickle.load(f)
    assert sidecar['binding'] == binding, 'Only the CLI total-update budget may be extended'
    assert total_updates >= sidecar['budget_total'] and total_updates > counter
    assert sidecar['events'] == events_complete(counter, sidecar['events']['validation_count'], binding['config'])
    assert sidecar['events']['validation_count'] == counter // binding['config']['validation_interval']
    restored = restore_agent(template, str(snapshot), counter)
    assert int(restored.network.step) == sidecar['network_step'] == counter + 1
    assert int(restored.network.opt_state[0].count) == sidecar['adam_count'] == counter
    restore_host_rng(sidecar['rng'])
    return restored, counter, sidecar['events']['validation_count']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--output', required=True)
    parser.add_argument('--total-updates', type=int)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    total = args.total_updates or config['total_updates']
    visual_goal = config.get('input_profile') == 'cached' and config.get('goal_mode') == 'latent_object_xyz'
    assert total >= config['total_updates'] == (1000000 if visual_goal else 100000)
    assert config['batch_size'] == (1024 if visual_goal else 128) and config['seed'] == 0 and config['evaluation_episodes'] == 0
    assert config['encoder_microbatch'] == 64 and config['workers'] in ((0, 2, 4) if config.get('input_profile') == 'cached' else (4, 8))
    assert config['validation_interval'] == 5000 and config['save_interval'] == (100000 if visual_goal else 1000)
    assert os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE') == 'false'
    utility = helpers(config)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    fsync_directory(output.parent)
    status = dict(stage='preparing', started_at=utility.now(), updated_at=utility.now(), exit_code=None,
                  completed_updates=0, failed_stages=0, target_updates=total, evaluation_episodes=0)
    utility.atomic_json(output / 'status.json', status)
    pool = None; logger = None; code = 1
    try:
        bridge, samplers, pool, frozen, cfg, binding = prepare(config)
        from utils.log_utils import CsvLogger
        logger = CsvLogger(str(output / 'train.csv'))
        if args.resume:
            template = create_agent(config, cfg)
            agent, counter, validations = restore_boundary(template, args.resume, binding, total)
        else:
            random.seed(config['seed']); np.random.seed(config['seed'])
            import torch
            torch.manual_seed(config['seed'])
            frozen.reader.split = 'train'
            example = encode_batch(frozen, samplers['train'].sample(1), config['encoder_microbatch'])
            agent = create_agent(config, cfg, example['observations'], example['actions'], example['actor_goals'])
            counter = validations = 0
        utility.atomic_json(output / 'resolved.json', dict(binding=binding, total_updates=total,
                            resumed_from=str(args.resume) if args.resume else None,
                            initial_counter=counter, initial_network_step=int(agent.network.step), argv=list(os.sys.argv)))
        last_time = time.time(); started = last_time
        while counter < total:
            _, batch = batch_for(frozen, samplers['train'], 'train', config)
            agent, info = update(agent, batch, frozen); counter += 1
            timings = dict(frozen.last_batch_timing, **frozen.last_update_timing)
            if counter % config['validation_interval'] == 0:
                _, val_batch = batch_for(frozen, samplers['val'], 'val', config)
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                val_info = {key: float(value) for key, value in val_info.items()}
                assert all(np.isfinite(x) for x in val_info.values())
                assert frozen_hash(frozen) == binding['frozen_full_state']
                validations += 1
                now = time.time()
                metrics = {f'training/{k}': v for k, v in info.items()}
                metrics.update({f'validation/{k}': v for k, v in val_info.items()})
                metrics.update({'time/epoch_time': (now-last_time)/config['validation_interval'], 'time/total_time': now-started})
                metrics.update({f'timing/{key}': value for key, value in timings.items()})
                logger.log(metrics, counter); last_time = now
            if counter % config['save_interval'] == 0 or counter == total:
                checkpoint = save_boundary(agent, frozen, binding, output, counter, validations, total)
                status['last_checkpoint'] = str(checkpoint)
            if counter % config['heartbeat_interval'] == 0 or counter == total:
                status.update(stage='training', completed_updates=counter, network_step=int(agent.network.step),
                              validation_count=validations, updated_at=utility.now(), last_update_info=info)
                status['last_update_timings'] = timings
                utility.atomic_json(output / 'status.json', status)
        assert int(agent.network.step) == total + 1 and int(agent.network.opt_state[0].count) == total
        assert validations == total // config['validation_interval']
        status.update(stage='training_exit0_pending_independent_reload', exit_code=0, adam_count=total)
        code = 0
    except BaseException as error:
        status.update(stage='failed', failed_stages=1, error=traceback.format_exc(), exit_code=1)
        status['worker_exit_codes'] = getattr(error, 'worker_exit_codes', {})
        status['prepare_cleanup_errors'] = getattr(error, 'worker_cleanup_errors', [])
        raise
    finally:
        active_error = sys.exc_info()[1]
        exits, cleanup_errors = close_resources(logger, pool)
        if pool is not None: status['worker_exit_codes'] = exits
        if cleanup_errors:
            status.update(stage='resource_cleanup_failed', cleanup_errors=cleanup_errors, exit_code=1,
                          failed_stages=max(1, status['failed_stages']))
            code = 1
        status['updated_at'] = utility.now()
        utility.atomic_json(output / 'status.json', status)
        (output / 'trainer_exit_code').write_text(str(code) + '\n')
        if cleanup_errors and active_error is None:
            raise RuntimeError('Resource cleanup failed; see persisted cleanup_errors')


if __name__ == '__main__':
    main()
