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

def digest_array(value):
    value = np.asarray(value)
    h = hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()

def close_resources(logger, pool):
    errors = []
    for resource in (logger, pool):
        if resource is not None:
            try:
                resource.close()
            except BaseException:
                errors.append(traceback.format_exc())
    return (getattr(pool, 'exit_codes', {}), errors)

def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def host_rng():
    import torch
    return dict(numpy=np.random.get_state(), python=random.getstate(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())

def sampler_binding(samplers):
    return {split: dict(size=s.size, config=s.config, arrays={key: digest_array(getattr(s, key)) for key in ('terminal_locs', 'initial_locs')}, valid_idxs=digest_array(s.dataset.valid_idxs)) for split, s in samplers.items()}

def update(agent, batch, frozen):
    frozen_guard(frozen)
    started = time.perf_counter()
    agent, info = agent.update(batch)
    dispatched = time.perf_counter()
    metrics = {key: float(value) for key, value in info.items()}
    synchronized = time.perf_counter()
    frozen.last_update_timing = dict(jax_dispatch_seconds=dispatched - started, loss_sync_seconds=synchronized - dispatched)
    if not all((np.isfinite(x) for x in metrics.values())) or metrics['grad/norm'] <= 0:
        raise ValueError('Nonfinite update/loss/gradient or zero gradient norm')
    frozen_guard(frozen)
    return (agent, metrics)

def events_complete(counter, validation_count, config):
    return dict(committed_updates=counter, validation_count=validation_count, next_stage='sample_train', last_validation=counter - counter % config['validation_interval'], checkpoint_event_complete=True, next_update=counter + 1)

def save_boundary(agent, frozen, binding, output, counter, validation_count, budget_total):
    from utils.flax_utils import save_agent
    config = binding['config']
    utility = helpers(config)
    assert validation_count == counter // config['validation_interval'], 'Validation event is not committed'
    frozen_guard(frozen)
    if frozen_hash(frozen) != binding['frozen_full_state']:
        raise ValueError('Full frozen WM bytes changed')
    parent = output / 'checkpoints'
    created = not parent.exists()
    parent.mkdir(exist_ok=True)
    if created:
        fsync_directory(output)
    destination = parent / f'update_{counter:09d}'
    stage = parent / f'.update_{counter:09d}.tmp'
    if destination.exists() or stage.exists():
        raise FileExistsError('Checkpoint transaction already exists')
    stage.mkdir()
    save_agent(agent, str(stage), counter)
    sidecar = dict(binding=binding, rng=host_rng(), budget_total=budget_total, events=events_complete(counter, validation_count, config), network_step=int(agent.network.step), adam_count=int(agent.network.opt_state[0].count))
    assert sidecar['network_step'] == counter + 1 and sidecar['adam_count'] == counter
    with (stage / 'run_state.pkl').open('xb') as f:
        pickle.dump(sidecar, f)
        f.flush()
        os.fsync(f.fileno())
    agent_path = stage / f'params_{counter}.pkl'
    with agent_path.open('rb') as f:
        os.fsync(f.fileno())
    utility.atomic_json(stage / 'manifest.json', dict(counter=counter, agent_sha256=utility.sha(agent_path), sidecar_sha256=utility.sha(stage / 'run_state.pkl')))
    fsync_directory(stage)
    os.rename(stage, destination)
    fsync_directory(parent)
    return destination
