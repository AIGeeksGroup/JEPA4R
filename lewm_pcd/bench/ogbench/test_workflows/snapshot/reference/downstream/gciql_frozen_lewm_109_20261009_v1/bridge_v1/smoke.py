"""Two optimizer updates per route, then a separately labelled resume proof."""
import argparse
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
from bridge import (FrozenLatents, OBS_FIELDS, RawRows, agent_config, load_compact,
                    make_sampler, policy_batch, sample_ids, setup, sha256)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def rng_state():
    import torch
    return dict(numpy=np.random.get_state(), python=random.getstate(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    import torch
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def assert_tree_equal(a, b):
    import jax
    left, right = jax.tree_util.tree_flatten(a), jax.tree_util.tree_flatten(b)
    assert left[1] == right[1], 'State structure mismatch'
    for x, y in zip(left[0], right[0]):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def state_dict(agent):
    import flax
    import jax
    return jax.device_get(flax.serialization.to_state_dict(agent))


def action(agent, batch):
    import jax
    value = np.asarray(agent.sample_actions(batch['observations'], batch['actor_goals'],
                                          seed=jax.random.PRNGKey(17), temperature=0.0))
    assert value.shape == (len(batch['actions']), 5) and np.isfinite(value).all()
    assert np.max(np.abs(value)) <= 1.0
    return value


def checked_update(agent, batch):
    import jax
    # Use the precise original update RNG and original loss, without copied math.
    _, loss_rng = jax.random.split(agent.rng)
    (_, loss_info), grads = jax.value_and_grad(agent.total_loss, argnums=1, has_aux=True)(
        batch, agent.network.params, rng=loss_rng)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(grads))
    assert all(np.isfinite(np.asarray(x)).all() for x in loss_info.values())
    updated, info = agent.update(batch)
    assert all(np.isfinite(np.asarray(x)).all() for x in info.values())
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state_dict(updated)))
    for name in ('actor', 'value', 'critic'):
        before = jax.tree_util.tree_leaves(agent.network.params['modules_' + name])
        after = jax.tree_util.tree_leaves(updated.network.params['modules_' + name])
        assert any(not np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(before, after)), name
    assert int(updated.network.step) == int(agent.network.step) + 1
    return updated, {k: float(v) for k, v in info.items()}


def run(args, config, out):
    lock = setup(config)
    import jax
    import torch
    from agents.gciql import GCIQLAgent
    from utils.flax_utils import save_agent, restore_agent
    if args.device == 'cuda':
        assert torch.cuda.is_available() and jax.default_backend() == 'gpu'
        assert float((torch.ones(2, device='cuda') * 3).sum()) == 6.0
        import jax.numpy as jnp
        assert float(jnp.dot(jnp.ones(2), jnp.ones(2))) == 2.0
    runtime = dict(python=sys.version, torch_cuda=torch.version.cuda,
                   jax_backend=jax.default_backend(), jax_devices=[str(x) for x in jax.devices()],
                   torch_device=torch.cuda.get_device_name() if args.device == 'cuda' else args.device)
    cfg = agent_config(config['batch_size'])
    assert all(cfg[k] == v for k, v in config['agent_overrides'].items())
    compact = load_compact(config)
    sampler = make_sampler(compact, cfg, ids=args.route != 'state')
    torch.manual_seed(config['seed'])
    bridge = None
    if args.route != 'state':
        bridge = FrozenLatents(config, args.route, RawRows(config, args.route, compact), args.device)
    before = bridge.snapshot() if bridge else None
    def convert(raw):
        return bridge.batch(raw) if bridge else policy_batch(raw)
    # Match main.py: seed the sampler immediately before its single sample(1).
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    example = convert(sampler.sample(1))
    encoder_parity = []
    if bridge:
        # These are independent real forwards, with no retained latent cache.
        encoder_parity.append(bridge.verify_direct(np.array([[0]], dtype=np.int64), 'observations'))
        encoder_parity.append(bridge.verify_direct(np.array([[0], [1001]], dtype=np.int64), 'observations'))
        bridge.check_frozen(before)
    agent = GCIQLAgent.create(config['seed'], example['observations'], example['actions'], cfg)
    fingerprint = dict(config=config, agent_config=cfg, locks=lock, route=args.route,
                       code={p.name: sha256(p) for p in Path(__file__).parent.iterdir() if p.name in
                             ('bridge.py', 'smoke.py', 'test_bridge.py', 'config.json')},
                       versions={x: importlib.metadata.version(x) for x in
                                 ('torch', 'jax', 'jaxlib', 'flax', 'optax', 'numpy', 'h5py', 'hdf5plugin')},
                       runtime=runtime)
    if args.restore:
        with open(out / 'resume.pkl', 'rb') as f:
            saved = pickle.load(f)
        assert_tree_equal(fingerprint, saved['fingerprint'])
        agent = restore_agent(agent, str(out), config['updates'])
        assert_tree_equal(state_dict(agent), saved['state'])
        np.testing.assert_array_equal(action(agent, saved['probe']), saved['action'])
        restore_rng(saved['rng'])
        raw = sampler.sample(config['batch_size'])
        batch = convert(raw)
        updated, info = checked_update(agent, batch)
        with open(out / 'expected_continuation.pkl', 'rb') as f:
            expected = pickle.load(f)
        assert_tree_equal(raw, expected['raw'])
        assert_tree_equal(batch, expected['batch'])
        assert_tree_equal(state_dict(updated), expected['state'])
        assert_tree_equal(rng_state(), expected['rng'])
        if bridge:
            bridge.check_frozen(before)
        write_json(out / 'restore_result.json', dict(status='passed', actual_exit_code=0,
                   restored_step=int(agent.network.step), continuation_step=int(updated.network.step), info=info))
        return
    write_json(out / 'resolved.json', fingerprint)
    initial_step = int(agent.network.step)
    metrics, sampled = [], []
    for i in range(config['updates']):
        raw = sampler.sample(config['batch_size'])
        batch = convert(raw)
        # The bridge's projected batch is handed directly to the untouched agent.
        reference, reference_info = agent.update({k: np.array(v, copy=True) for k, v in batch.items()})
        agent, info = checked_update(agent, batch)
        assert_tree_equal(state_dict(agent), state_dict(reference))
        assert_tree_equal(info, {k: float(v) for k, v in reference_info.items()})
        metrics.append(info)
        sampled.append({k: raw[k].tolist() for k in OBS_FIELDS} if bridge else {})
        if bridge:
            bridge.check_frozen(before)
        write_json(out / 'status.json', dict(stage='updates', completed=i + 1, failed=0,
                                            updated=time.time(), network_step=int(agent.network.step)))
    probe = batch
    deterministic = action(agent, probe)
    save_agent(agent, str(out), config['updates'])
    with open(out / 'resume.pkl', 'wb') as f:
        pickle.dump(dict(fingerprint=fingerprint, state=state_dict(agent), rng=rng_state(),
                         probe=probe, action=deterministic, optimizer_updates=config['updates']), f)
    raw = sampler.sample(config['batch_size'])
    batch = convert(raw)
    continuation, continuation_info = checked_update(agent, batch)
    if bridge:
        bridge.check_frozen(before)
    with open(out / 'expected_continuation.pkl', 'wb') as f:
        pickle.dump(dict(raw=raw, batch=batch, state=state_dict(continuation), rng=rng_state()), f)
    write_json(out / 'update_result.json', dict(status='passed', initial_network_step=initial_step,
               final_network_step=int(agent.network.step), optimizer_updates=config['updates'],
               continuation_proof_updates=1, metrics=metrics, sampled_ids=sampled,
               actions=deterministic.tolist(), frozen_full_state_equal=bool(bridge),
               module_sources=bridge.sources if bridge else {}, schedule=config['schedule'],
               direct_encoder_parity=encoder_parity,
               evaluation_episodes=0, source_label='official state' if not bridge else
               'fixed5-trained frozen LeWM on official Cube-state-derived rendered transitions'))
    # Release both runtimes before the separate fresh restore process owns the GPU.
    print('UPDATE_STAGE_PASSED; fresh restore must be launched separately', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--route', choices=['state', 'rgb', 'pcd'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    assert config['updates'] == 2 and config['batch_size'] in (2, 4)
    assert config['continuation_updates'] == 1 and config['schedule']['evaluation_episodes'] == 0
    out = Path(args.output)
    if not args.restore:
        out.mkdir(parents=True, exist_ok=False)
    elif not out.is_dir():
        raise FileNotFoundError(out)
    status = out / ('restore_status.json' if args.restore else 'status.json')
    write_json(status, dict(stage='starting', started=time.time(), completed=0, failed=0, exit_code=None))
    try:
        # Allocator settings must be supplied before any JAX import.
        assert os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE') == 'false'
        run(args, config, out)
    except Exception:
        write_json(status, dict(stage='failed', updated=time.time(), failed=1, exit_code=1,
                               error=traceback.format_exc()))
        raise
    write_json(status, dict(stage='restore_passed' if args.restore else 'updates_passed_restore_pending',
                           updated=time.time(), completed=1 if args.restore else 2, failed=0, exit_code=0))


if __name__ == '__main__':
    main()
