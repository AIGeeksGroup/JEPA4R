"""Six true-width inputs around the preserved Single GCIQL update/save code."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent/'vendor'
HERE = Path(__file__).resolve().parent
SOURCE = ROOT
CORE = ROOT
IMPL = Path(__file__).resolve().parents[3]/'third_party/ogbench/impls'
HELPER = ROOT/'io.py'
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('JAX_THREEFRY_PARTITIONABLE', 'false')
sys.path[:0] = [str(HERE), str(IMPL), str(ROOT), str(CORE)]

import flax
import jax
import numpy as np
import ogbench
import torch
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from utils.log_utils import CsvLogger
from .sampler import DistanceInputs, create_agent, file_stat, load_cached_features, read_json


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


utility = load_module('cube_policy_io', HELPER)
core = load_module('cube_policy_core', CORE/'core.py')
model = load_module('cube_policy_gciql', CORE/'gciql.py')
from ._source import verify_sources
verify_sources()


class InputResources:
    def __init__(self, samplers, sources):
        self.samplers = samplers
        self.binding_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        self.exit_codes = {}
        self.last_batch_timing = {}

    def guard(self):
        for sampler in self.samplers.values():
            assert not sampler.dataset['observations'].flags.writeable
            assert sampler.features is None or not sampler.features.flags.writeable

    def close(self):
        for sampler in self.samplers.values():
            sampler.close()


# Adapt only the resource/provenance hooks of the original cached save/update
# helpers. The original update, checkpoint transaction and host RNG code stay intact.
core.helpers = lambda config: utility
core.frozen_hash = lambda resources: resources.binding_hash
core.frozen_guard = lambda resources: resources.guard()


def agent_config(config):
    cfg = dict(config['agent_config'])
    # JSON stores sequences as lists; restore the original factory's tuple types.
    for key in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[key] = tuple(cfg[key])
    return cfg


def prepare(config, shared_datasets=None):
    assert config['seed'] == 0 and config['batch_size'] == 1024 and config['total_updates'] == 1000000
    assert config['resume_from'] is None and config['input_contract']['zero_padding'] is False
    assert config['validation_interval'] == 5000 and config['save_interval'] == 100000
    assert config['agent_config']['lr'] == 3e-4 and config['agent_config']['alpha'] == 1.0
    assert jax.config.jax_threefry_partitionable is False
    assert jax.config.jax_default_matmul_precision is None and jax.config.jax_enable_x64 is False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    samplers, source_bindings = {}, {}
    try:
        for split in ('train', 'val'):
            source = config['data_binding']['splits'][split]
            path = source['path']
            assert file_stat(path) == source['accepted_stat']
            raw = (ogbench.load_dataset(path, compact_dataset=True, add_info=False)
                   if shared_datasets is None else shared_datasets[split]._dict)
            assert file_stat(path) == source['accepted_stat']
            assert list(raw['observations'].shape) == source['observation_shape']
            assert list(raw['actions'].shape) == source['action_shape']
            assert raw['observations'].dtype == raw['actions'].dtype == np.float32
            for value in raw.values():
                if isinstance(value, np.ndarray):
                    if np.issubdtype(value.dtype, np.floating):
                        assert np.isfinite(value).all()
                    value.setflags(write=False)
            dataset = Dataset.create(**raw) if shared_datasets is None else shared_datasets[split]
            features, feature_binding = None, None
            if config['modality'] != 'state':
                args = config['feature_loader_arguments'][split]
                assert args['expected_rows'] == len(raw['observations'])
                features, feature_binding = load_cached_features(**args)
            samplers[split] = DistanceInputs(dataset, agent_config(config), cubes=config['cube_count'],
                mode=config['modality'], features=features, workers=config['feature_lookup_workers'])
            sampler = samplers[split]
            assert sampler.current_dim == config['input_contract']['current_dim']
            assert sampler.goal_dim == config['input_contract']['goal_dim']
            source_bindings[split] = dict(source=source, feature=feature_binding,
                                         input_contract=sampler.input_metadata)
        resources = InputResources(samplers, source_bindings)
        binding = dict(config=config, agent_config=agent_config(config), source_bindings=source_bindings,
            sampler=core.sampler_binding(samplers), frozen_full_state=resources.binding_hash,
            frozen_full_state_semantics='readonly data/cache binding; no runtime WM',
            code={str(p):utility.sha(p) for p in [Path(__file__), HERE/'sampler.py', ROOT/'cube_inputs.py',
                CORE/'core.py', CORE/'gciql.py', HELPER, IMPL/'utils/datasets.py',
                IMPL/'utils/flax_utils.py', IMPL/'utils/networks.py', IMPL/'utils/encoders.py']},
            versions={name:importlib.metadata.version(name) for name in ('jax','jaxlib','flax','optax','numpy','torch')},
            runtime=dict(JAX_backend=jax.default_backend(), JAX_threefry_partitionable=jax.config.jax_threefry_partitionable,
                JAX_default_prng_impl=str(jax.config.jax_default_prng_impl),
                JAX_default_matmul_precision=str(jax.config.jax_default_matmul_precision),
                JAX_enable_x64=jax.config.jax_enable_x64, XLA_FLAGS=os.environ.get('XLA_FLAGS'),
                lookup_workers=config['feature_lookup_workers'], readonly_RAM=True, runtime_encoder_calls=0,
                no_cross_version_bitwise_claim=True))
        return samplers, resources, binding
    except BaseException:
        for sampler in samplers.values():
            sampler.close()
        raise


def initial_agent(config, samplers):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    # Preserve Single's real sample(1) host RNG consumption before factory init.
    batch = samplers['train'].sample(1)
    agent = create_agent(model.GCIQLAgent, config['seed'], batch, agent_config(config))
    assert int(agent.network.step) == 1 and int(agent.network.opt_state[0].count) == 0
    return agent, batch


def get_batch(samplers, resources, split, config):
    started = time.perf_counter()
    batch = samplers[split].sample(config['batch_size'])
    resources.last_batch_timing = dict(input_seconds=time.perf_counter()-started)
    return batch


def fingerprint(agent):
    leaves = jax.tree_util.tree_leaves(flax.serialization.to_state_dict(agent))
    h = hashlib.sha256()
    for leaf in leaves:
        array = np.asarray(leaf)
        assert np.isfinite(array).all()
        h.update(core.digest_array(array).encode())
    return dict(leaves=len(leaves), sha256=h.hexdigest())


def run(config_path):
    config = json.loads(Path(config_path).read_text())
    assert jax.default_backend() == 'gpu'
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    status = dict(stage='preparing', role=config['role'], started_at=utility.now(), updated_at=utility.now(),
        completed_updates=0, target_updates=1000000, failed_stages=0, exit_code=None)
    utility.atomic_json(output/'status.json', status)
    resources = logger = None
    code = 1
    try:
        samplers, resources, binding = prepare(config)
        logger = CsvLogger(str(output/'train.csv'))
        agent, example = initial_agent(config, samplers)
        utility.atomic_json(output/'resolved.json', dict(binding=binding, argv=sys.argv,
            resumed_from=None, initial_counter=0, initial_network_step=int(agent.network.step),
            initial_adam_count=int(agent.network.opt_state[0].count),
            actual_current_dim=example['observations'].shape[-1], actual_goal_dim=example['actor_goals'].shape[-1],
            synthetic_zero_padding=False))
        counter = validations = 0
        started = last_time = time.time()
        while counter < config['total_updates']:
            batch = get_batch(samplers, resources, 'train', config)
            agent, info = core.update(agent, batch, resources)
            counter += 1
            timings = dict(resources.last_batch_timing, **resources.last_update_timing)
            if counter % config['validation_interval'] == 0:
                val_batch = get_batch(samplers, resources, 'val', config)
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                val_info = {key:float(value) for key,value in val_info.items()}
                assert all(np.isfinite(x) for x in val_info.values())
                assert core.frozen_hash(resources) == binding['frozen_full_state']
                validations += 1
                now = time.time()
                metrics = {f'training/{key}':value for key,value in info.items()}
                metrics.update({f'validation/{key}':value for key,value in val_info.items()})
                metrics.update({'time/epoch_time':(now-last_time)/config['validation_interval'],
                                'time/total_time':now-started})
                metrics.update({f'timing/{key}':value for key,value in timings.items()})
                logger.log(metrics, counter)
                last_time = now
            if ((counter == 1 and config.get('save_step1', False)) or
                    counter % config['save_interval'] == 0 or counter == config['total_updates']):
                saved = core.save_boundary(agent, resources, binding, output, counter, validations, config['total_updates'])
                status['last_checkpoint'] = str(saved)
            if counter % config['heartbeat_interval'] == 0 or counter == config['total_updates']:
                status.update(stage='training', completed_updates=counter, network_step=int(agent.network.step),
                    validation_count=validations, updated_at=utility.now(), last_update_info=info,
                    last_update_timings=timings)
                utility.atomic_json(output/'status.json', status)
        assert int(agent.network.step) == 1000001 and int(agent.network.opt_state[0].count) == 1000000
        assert validations == 200
        np.savez(output/'final_val_probe.npz', **val_batch)
        utility.atomic_json(output/'final_state_fingerprint.json', fingerprint(agent))
        status.update(stage='training_exit0_pending_independent_reload', exit_code=0, adam_count=1000000)
        code = 0
    except BaseException:
        status.update(stage='failed', failed_stages=1, error=traceback.format_exc(), exit_code=1)
        raise
    finally:
        exits, errors = core.close_resources(logger, resources)
        if errors:
            status.update(stage='cleanup_failed', exit_code=1, cleanup_errors=errors)
            code = 1
        status.update(updated_at=utility.now(), lookup_process_exit_codes=exits)
        utility.atomic_json(output/'status.json', status)
        (output/'trainer_exit_code').write_text(str(code)+'\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.config))
