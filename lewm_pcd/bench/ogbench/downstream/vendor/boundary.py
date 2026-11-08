import argparse
import csv
import json
from pathlib import Path
import pickle
import random
import time
import traceback
from trainer_runtime import agent_config, core, create_agent, fingerprint, get_batch, initial_agent, jax, model, np, prepare, restore_agent, torch, utility

def finite_metrics(info):
    values = {key: float(value) for key, value in info.items()}
    assert all((np.isfinite(value) for value in values.values()))
    return values

def kernel_shapes(agent, config):
    import flax
    state = flax.serialization.to_state_dict(agent)
    widths = {config['input_contract']['current_dim'] + config['input_contract']['goal_dim']}
    widths.add(next(iter(widths)) + 5)
    found = {}

    def walk(value, path=()):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, path + (key,))
        elif path[-1:] == ('kernel',):
            shape = np.asarray(value).shape
            if len(shape) > 1 and shape[-2] in widths:
                found['/'.join(path)] = list(shape)
    walk(state['network']['params'])
    assert found and all(('512' in str(shape) for shape in found.values()))
    assert {shape[-2] for shape in found.values()} == widths
    return found

def gate(config, output):
    assert jax.default_backend() == 'gpu'
    output.mkdir(parents=True, exist_ok=False)
    samplers = resources = None
    try:
        samplers, resources, binding = prepare(config)
        agent, example = initial_agent(config, samplers)
        shape_report = kernel_shapes(agent, config)
        initial = fingerprint(agent)
        assert initial['leaves'] == 153
        records = []
        for index in range(2):
            batch = get_batch(samplers, resources, 'train', config)
            agent, info = core.update(agent, batch, resources)
            records.append(finite_metrics(info))
            if index == 0 and config.get('save_step1', False):
                core.save_boundary(agent, resources, binding, output, 1, 0, config['total_updates'])
        val = get_batch(samplers, resources, 'val', config)
        validation = finite_metrics(agent.total_loss(val, grad_params=None)[1])
        saved = core.save_boundary(agent, resources, binding, output, 2, 0, config['total_updates'])
        expected = fingerprint(agent)
        assert int(agent.network.step) == 3 and int(agent.network.opt_state[0].count) == 2
        next_batch = get_batch(samplers, resources, 'train', config)
        np.savez(output / 'next_batch.npz', **next_batch)
        np.savez(output / 'next_ids.npz', **samplers['train'].last_ids)
        report = dict(actual_exit_code=0, role=config['role'], completed=time.time(), batch_size=1024, updates=2, validation_probes=1, formal_validation_events=0, actual_backend=jax.default_backend(), input_metadata=samplers['train'].input_metadata, first_layer_kernels=shape_report, initial_state=initial, saved_state=expected, checkpoint=str(saved), update_metrics=records, validation=validation, network_step=3, adam_count=2, synthetic_zero_padding=False, encoder_calls=0, WM_dynamics_calls=0, formal_training_started=False, actual_source_binding=binding)
        utility.atomic_json(output / 'report.json', report)
    finally:
        if resources:
            resources.close()

def restore_saved(config, output, counter):
    checkpoint = output / 'checkpoints' / f'update_{counter:09d}'
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    assert manifest['counter'] == counter
    assert utility.sha(checkpoint / f'params_{counter}.pkl') == manifest['agent_sha256']
    assert utility.sha(checkpoint / 'run_state.pkl') == manifest['sidecar_sha256']
    with (checkpoint / 'run_state.pkl').open('rb') as handle:
        sidecar = pickle.load(handle)
    assert sidecar['network_step'] == counter + 1 and sidecar['adam_count'] == counter
    assert sidecar['binding']['config'] == config
    for path, expected in sidecar['binding']['code'].items():
        assert utility.sha(path) == expected, path
    return (checkpoint, sidecar)

def reload_gate(config, output, report_path):
    assert jax.default_backend() == 'cpu'
    expected = json.loads((output / 'report.json').read_text())
    assert expected['actual_exit_code'] == 0 and expected['role'] == config['role']
    checkpoint, sidecar = restore_saved(config, output, 2)
    resources = None
    try:
        samplers, resources, binding = prepare(config)
        assert binding['source_bindings'] == sidecar['binding']['source_bindings']
        _, example = initial_agent(config, samplers)
        template = create_agent(model.GCIQLAgent, config['seed'], example, agent_config(config))
        restored = restore_agent(template, str(checkpoint), 2)
        actual = fingerprint(restored)
        assert actual == expected['saved_state']
        np.random.set_state(sidecar['rng']['numpy'])
        random.setstate(sidecar['rng']['python'])
        torch.set_rng_state(sidecar['rng']['torch'])
        batch = get_batch(samplers, resources, 'train', config)
        with np.load(output / 'next_batch.npz', allow_pickle=False) as wanted:
            assert set(batch) == set(wanted.files)
            for key, value in batch.items():
                np.testing.assert_array_equal(value, wanted[key], err_msg=key)
        with np.load(output / 'next_ids.npz', allow_pickle=False) as wanted:
            for key, value in samplers['train'].last_ids.items():
                np.testing.assert_array_equal(value, wanted[key], err_msg=key)
        actions = np.asarray(restored.sample_actions(batch['observations'], batch['actor_goals'], seed=jax.random.PRNGKey(0), temperature=0))
        assert actions.shape == (1024, 5) and np.isfinite(actions).all()
        utility.atomic_json(report_path, dict(actual_exit_code=0, role=config['role'], completed=time.time(), state=actual, fresh_process=True, backend='cpu', full153_state_exact=True, next_batch_and_rawIDs_exact=True, applied_host_RNG=['numpy', 'python', 'torchCPU'], cuda_RNG_payload_preserved_not_applied=True, finite_actions=True, CPU_action_forward=True, no_updates=True))
    finally:
        if resources:
            resources.close()

def accept(config, output, report_path):
    assert jax.default_backend() == 'cpu'
    status = json.loads((output / 'status.json').read_text())
    assert status['exit_code'] == 0 and status['completed_updates'] == 1000000
    assert status['validation_count'] == 200 and status['adam_count'] == 1000000
    assert (output / 'trainer_exit_code').read_text().strip() == '0'
    resolved = json.loads((output / 'resolved.json').read_text())
    assert resolved['resumed_from'] is None and resolved['initial_counter'] == 0
    assert resolved['initial_network_step'] == 1 and resolved['initial_adam_count'] == 0
    assert resolved['actual_current_dim'] == config['input_contract']['current_dim']
    assert resolved['actual_goal_dim'] == config['input_contract']['goal_dim']
    assert resolved['synthetic_zero_padding'] is False
    with (output / 'train.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 200 and [int(row['step']) for row in rows] == list(range(5000, 1000001, 5000))
    for row in rows:
        assert all((np.isfinite(float(value)) for value in row.values()))
        assert float(row['training/grad/norm']) > 0
    ledger = []
    counters = ([1] if config.get('save_step1', False) else []) + list(range(100000, 1000001, 100000))
    for counter in counters:
        checkpoint, sidecar = restore_saved(config, output, counter)
        assert sidecar['events'] == core.events_complete(counter, counter // 5000, config)
        ledger.append(dict(counter=counter, manifest_sha256=utility.sha(checkpoint / 'manifest.json')))
    with np.load(output / 'final_val_probe.npz', allow_pickle=False) as data:
        batch = dict(data)
    template = create_agent(model.GCIQLAgent, config['seed'], batch, agent_config(config))
    restored = restore_agent(template, str(checkpoint), 1000000)
    actual = fingerprint(restored)
    assert actual == json.loads((output / 'final_state_fingerprint.json').read_text())
    assert actual['leaves'] == 153
    assert int(restored.network.step) == 1000001 and int(restored.network.opt_state[0].count) == 1000000
    for leaf in jax.tree_util.tree_leaves(restored):
        assert np.isfinite(np.asarray(leaf)).all()
    actions = np.asarray(restored.sample_actions(batch['observations'], batch['actor_goals'], seed=jax.random.PRNGKey(0), temperature=0))
    assert actions.shape == (1024, 5) and np.isfinite(actions).all()
    utility.atomic_json(report_path, dict(actual_exit_code=0, role=config['role'], completed=time.time(), full153_state_exact=True, state=actual, Adam_count=1000000, network_step=1000001, CSV_validation_records=200, checkpoint_transactions=ledger, finite_actions=True, actual_current_dim=resolved['actual_current_dim'], actual_goal_dim=resolved['actual_goal_dim'], synthetic_zero_padding=False, CPU_action_forward=True, no_optimizer_updates=True, entire_1M_host_sampler_replay_performed=False, SR_performed=False, final_checkpoint=str(checkpoint), final_manifest_sha256=utility.sha(checkpoint / 'manifest.json')))
