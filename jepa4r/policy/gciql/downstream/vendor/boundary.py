# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/downstream/vendor/boundary.py (verbatim except: imports / paths
# adapted to the package; lines wrapped at 120 columns, unused imports removed)
import copy
import csv
import json
from pathlib import Path
import pickle
import random
import time
from trainer_runtime import (
    agent_config,
    core,
    create_agent,
    fingerprint,
    get_batch,
    initial_agent,
    jax,
    model,
    np,
    prepare,
    restore_agent,
    torch,
    utility,
)

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

def _save_full_state(agent, path):
    """Retain every named agent/Adam/RNG leaf for the numeric recovery gate."""
    import flax
    arrays = {}
    def visit(value, names=()):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, names + (str(key),))
        elif isinstance(value, (tuple, list)):
            for key, item in enumerate(value):
                visit(item, names + (str(key),))
        else:
            arrays['/'.join(names)] = np.asarray(value)
    visit(flax.serialization.to_state_dict(agent))
    np.savez(path, **arrays)


def parameter_fingerprint(agent):
    """Hash only complete named parameters, excluding counters/Adam/JAX RNG."""
    import flax
    import hashlib
    state = flax.serialization.to_state_dict(agent)['network']['params']
    digest = hashlib.sha256()
    counts = dict(tensors=0, elements=0)
    def visit(value, names=()):
        if isinstance(value, dict):
            for key in sorted(value):visit(value[key], names+(str(key),))
        else:
            array=np.asarray(value)
            if not np.isfinite(array).all():raise ValueError('nonfinite policy parameters')
            digest.update('/'.join(names).encode())
            digest.update(core.digest_array(array).encode())
            counts['tensors']+=1;counts['elements']+=array.size
    visit(state)
    return dict(sha256=digest.hexdigest(),**counts)


def deterministic_action_reference(agent, batch):
    """Original t0/key0 action forward; prove no model or host RNG mutation."""
    from jepa4r.policy.gciql.gates.arrays import compare_nested
    from jepa4r.policy.gciql.gate_checks import _state_cpu_arrays
    before=fingerprint(agent)
    host=_state_cpu_arrays(copy.deepcopy(core.host_rng()))
    actions=np.asarray(agent.sample_actions(batch['observations'],batch['actor_goals'],
                       seed=jax.random.PRNGKey(0),temperature=0))
    if actions.shape!=(1024,5)or actions.dtype!=np.float32 or not np.isfinite(actions).all():
        raise ValueError('complete deterministic action reference shape/dtype/finite failed')
    unchanged=fingerprint(agent)==before
    rng_comparison=compare_nested(_state_cpu_arrays(core.host_rng()),host,exact=True)
    if not unchanged or not rng_comparison['passed']:
        raise ValueError('action reference forward mutated agent/host RNG')
    return actions,dict(full_agent_state_unchanged=True,host_rng_unchanged=True,
                        key=0,temperature=0,complete_action_elements=actions.size)


def compare_reload_actions(actions, reference_path):
    from jepa4r.policy.gciql.gates.arrays import compare_array
    reference=np.load(reference_path,allow_pickle=False)
    if reference.shape!=(1024,5)or reference.dtype!=np.float32:
        raise ValueError('training-ended action reference shape/dtype differs')
    return compare_array(np.asarray(actions),reference,atol=1e-6,rtol=1e-4)


def gate(config, output, *, stop_after=2, resume_root=None):
    """Original two-update gate, optionally stopped/resumed at update one."""
    if type(stop_after) is not int or stop_after not in (1, 2):
        raise ValueError('smoke gate permits only one or two total updates')
    if resume_root is not None and stop_after != 2:
        raise ValueError('smoke recovery must complete update two')
    assert jax.default_backend() == 'gpu'
    output.mkdir(parents=True, exist_ok=False)
    samplers = resources = None
    try:
        samplers, resources, binding = prepare(config)
        agent, example = initial_agent(config, samplers)
        initial_counter = 0
        if resume_root is not None:
            checkpoint, sidecar = restore_saved(config, Path(resume_root), 1)
            if binding['source_bindings'] != sidecar['binding']['source_bindings']:
                raise ValueError('resumed policy data/cache binding differs')
            agent = restore_agent(agent, str(checkpoint), 1)
            np.random.set_state(sidecar['rng']['numpy'])
            random.setstate(sidecar['rng']['python'])
            torch.set_rng_state(sidecar['rng']['torch'])
            if sidecar['rng']['cuda']:
                torch.cuda.set_rng_state_all(sidecar['rng']['cuda'])
            initial_counter = 1
        shape_report = kernel_shapes(agent, config)
        initial = fingerprint(agent)
        initial_parameters=parameter_fingerprint(agent)
        assert initial['leaves'] == 153
        records = []
        for index in range(initial_counter, stop_after):
            batch = get_batch(samplers, resources, 'train', config)
            if resume_root is not None and index == 1:
                with np.load(Path(resume_root) / 'next_batch.npz', allow_pickle=False) as wanted:
                    if set(wanted.files) != set(batch):
                        raise ValueError('resumed batch fields differ')
                    for key, value in batch.items():
                        np.testing.assert_array_equal(value, wanted[key], err_msg=key)
                with np.load(Path(resume_root) / 'next_ids.npz', allow_pickle=False) as wanted:
                    if set(wanted.files) != set(samplers['train'].last_ids):
                        raise ValueError('resumed batch raw ID fields differ')
                    for key, value in samplers['train'].last_ids.items():
                        np.testing.assert_array_equal(value, wanted[key], err_msg=key)
            agent, info = core.update(agent, batch, resources)
            records.append(finite_metrics(info))
            if index == 0 and stop_after == 2 and config.get('save_step1', False):
                core.save_boundary(agent, resources, binding, output, 1, 0, config['total_updates'])
        validation = {}
        if stop_after == 2:
            val = get_batch(samplers, resources, 'val', config)
            validation = finite_metrics(agent.total_loss(val, grad_params=None)[1])
        saved = core.save_boundary(agent, resources, binding, output, stop_after, 0, config['total_updates'])
        expected = fingerprint(agent)
        final_parameters=parameter_fingerprint(agent)
        if initial_parameters==final_parameters:
            raise ValueError('GCIQL optimizer calls did not change actual parameters')
        assert int(agent.network.step) == stop_after + 1 and int(agent.network.opt_state[0].count) == stop_after
        _save_full_state(agent, output / 'full_state.npz')
        next_batch = get_batch(samplers, resources, 'train', config)
        np.savez(output / 'next_batch.npz', **next_batch)
        np.savez(output / 'next_ids.npz', **samplers['train'].last_ids)
        actions,forward_proof=deterministic_action_reference(agent,next_batch)
        np.save(output/'next_actions.npy',actions,allow_pickle=False)
        report = dict(
            actual_exit_code=0,
            role=config['role'],
            completed=time.time(),
            batch_size=1024,
            updates=2,
            validation_probes=1,
            formal_validation_events=0,
            actual_backend=jax.default_backend(),
            input_metadata=samplers['train'].input_metadata,
            first_layer_kernels=shape_report,
            initial_state=initial,
            saved_state=expected,
            checkpoint=str(saved),
            update_metrics=records,
            validation=validation,
            network_step=3,
            adam_count=2,
            synthetic_zero_padding=False,
            encoder_calls=0,
            WM_dynamics_calls=0,
            formal_training_started=False,
            actual_source_binding=binding,
        )
        report.update(updates=stop_after, updates_this_process=stop_after-initial_counter,
                      validation_probes=int(stop_after == 2), network_step=stop_after+1,
                      adam_count=stop_after, initial_counter=initial_counter,
                      resumed_from=str(resume_root) if resume_root is not None else None,
                      full_state_sha256=utility.sha(output / 'full_state.npz'),
                      initial_parameters=initial_parameters,final_parameters=final_parameters,
                      actual_parameters_changed=True,
                      action_reference_sha256=utility.sha(output/'next_actions.npy'),
                      action_reference_forward=forward_proof)
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

def reload_gate(config, output, report_path, *, cpu_reference=None, require_cpu_reference=False):
    assert jax.default_backend() == 'cpu'
    if require_cpu_reference and cpu_reference is None:
        raise ValueError('guide CPU reload requires an independent raw-state CPU endpoint reference')
    expected = json.loads((output / 'report.json').read_text())
    assert expected['actual_exit_code'] == 0 and expected['role'] == config['role']
    counter=expected.get('updates',2)
    if type(counter)is not int or counter not in (1,2):
        raise ValueError('reload requires the actual one/two-update gate boundary')
    checkpoint, sidecar = restore_saved(config, output, counter)
    resources = None
    try:
        samplers, resources, binding = prepare(config)
        assert binding['source_bindings'] == sidecar['binding']['source_bindings']
        _, example = initial_agent(config, samplers)
        template = create_agent(model.GCIQLAgent, config['seed'], example, agent_config(config))
        restored = restore_agent(template, str(checkpoint), counter)
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
        actions,forward_proof=deterministic_action_reference(restored,batch)
        cross_backend=None
        reference=output/'next_actions.npy'
        if expected.get('action_reference_sha256') is not None:
            if utility.sha(reference)!=expected['action_reference_sha256']:
                raise ValueError('original GPU action reference checksum differs')
            cross_backend=compare_reload_actions(actions,reference)
        action_comparison=None
        if cpu_reference is not None:
            from jepa4r.policy.gciql.policy_reference import validate_cpu_reference
            from jepa4r.policy.gciql.gates.arrays import compare_array
            cpu_actions,_=validate_cpu_reference(cpu_reference,config,output,expected)
            action_comparison=compare_array(actions,cpu_actions,atol=1e-6,rtol=1e-4)
        np.save(Path(report_path).with_suffix('.actions.npy'),actions,allow_pickle=False)
        if action_comparison is not None and not action_comparison['passed']:
            utility.atomic_json(report_path,dict(actual_exit_code=1,role=config['role'],
                completed=time.time(),action_comparison=action_comparison,error='reloaded actions differ'))
            raise ValueError('independent same-CPU reload action arrays exceed FP32 tolerance')
        utility.atomic_json(
            report_path,
            dict(
                actual_exit_code=0,
                role=config['role'],
                completed=time.time(),
                state=actual,
                fresh_process=True,
                backend='cpu',
                full153_state_exact=True,
                next_batch_and_rawIDs_exact=True,
                applied_host_RNG=['numpy', 'python', 'torchCPU'],
                cuda_RNG_payload_preserved_not_applied=True,
                finite_actions=True,
                CPU_action_forward=True,
                no_updates=True,
                completed_updates=counter,
                action_comparison=action_comparison,
                action_reference_forward=forward_proof,
                same_cpu_action_consistency_verified=action_comparison is not None,
                cpu_reference=cpu_reference,
                cross_backend_diagnostic=cross_backend,
                action_reference_sha256=expected.get('action_reference_sha256'),
            ),
        )
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
    actions = np.asarray(
        restored.sample_actions(batch['observations'], batch['actor_goals'], seed=jax.random.PRNGKey(0), temperature=0)
    )
    assert actions.shape == (1024, 5) and np.isfinite(actions).all()
    utility.atomic_json(
        report_path,
        dict(
            actual_exit_code=0,
            role=config['role'],
            completed=time.time(),
            full153_state_exact=True,
            state=actual,
            Adam_count=1000000,
            network_step=1000001,
            CSV_validation_records=200,
            checkpoint_transactions=ledger,
            finite_actions=True,
            actual_current_dim=resolved['actual_current_dim'],
            actual_goal_dim=resolved['actual_goal_dim'],
            synthetic_zero_padding=False,
            CPU_action_forward=True,
            no_optimizer_updates=True,
            entire_1M_host_sampler_replay_performed=False,
            SR_performed=False,
            final_checkpoint=str(checkpoint),
            final_manifest_sha256=utility.sha(checkpoint / 'manifest.json'),
        ),
    )
