"""CPU contract gates; explicit separate CLI for root-authorized GPU gates."""
import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from reader import ReadPool, encode_batch, load_bridge
from train import (batch_for, digest_array, events_complete, frozen_guard, frozen_hash,
                   host_rng, prepare, restore_boundary, save_boundary, update)


def config_file():
    return Path(__file__).with_name('config.json')


def rng_equal(left, right):
    assert left[0] == right[0] and left[2:] == right[2:]
    np.testing.assert_array_equal(left[1], right[1])


def test_visual_xyz_B1024_four_role_mapping_and_current_reward():
    import reader
    lookup = reader.CanonicalLatentLookup.__new__(reader.CanonicalLatentLookup)
    valid_ids = np.concatenate([np.arange(1000), np.arange(1001, 2001)])
    lookup.latents = np.repeat(valid_ids[:, None], 192, axis=1).astype(np.float32)
    lookup.terminals = np.repeat(np.array([[1000], [2001]], np.float32), 192, axis=1)
    lookup.latents.flags.writeable = lookup.terminals.flags.writeable = False
    lookup.size = 2002; lookup.binding = {'fixture': 'same_raw_row'}
    states = np.zeros((2002, 28), np.float32)
    states[:, :19] = np.arange(2002, dtype=np.float32)[:, None]
    states[[999, 1001], 19] = np.float32(.4)
    states[2000, 19] = np.float32(.40001)
    states[[1, 1000], 19] = 2.
    def ids(values): return np.tile(np.array(values, np.int64), 256)[:, None]
    raw = dict(observations=ids([0, 999, 1001, 2000]), next_observations=ids([1, 1000, 1002, 2001]),
               value_goals=ids([1001, 0, 999, 0]), actor_goals=ids([0, 1001, 2000, 2000]),
               actions=np.zeros((1024, 5), np.float32), rewards=-np.ones(1024, np.float32), masks=np.ones(1024, np.float32))
    before = np.random.get_state()
    frozen = reader.CachedLatents({'train': lookup}, {'train': {'observations': states}}, 'cache', 'latent_object_xyz')
    batch = frozen.batch(raw)
    for role in ('observations', 'next_observations', 'value_goals', 'actor_goals'):
        assert batch[role].shape == (1024, 220 if 'observations' in role else 195)
        np.testing.assert_array_equal(batch[role][:, :192], np.repeat(raw[role], 192, axis=1).astype(np.float32))
        expected = states[raw[role][:, 0]]
        np.testing.assert_array_equal(batch[role][:, 192:], expected if 'observations' in role else expected[:, 19:22])
    np.testing.assert_array_equal(batch['rewards'], np.tile([0., 0., 0., -1.], 256))
    np.testing.assert_array_equal(batch['masks'], np.tile([0., 0., 0., 1.], 256))
    assert all(v.dtype == np.float32 and np.isfinite(v).all() for v in batch.values())
    assert not np.array_equal(batch['value_goals'], batch['actor_goals'])
    np.testing.assert_array_equal(raw['rewards'], -np.ones(1024, np.float32))
    frozen.guard(); rng_equal(before, np.random.get_state())
    for role in ('value_goals', 'actor_goals'):
        bad = dict(raw); bad[role] = raw[role].copy(); bad[role][0, 0] = 1000
        try: frozen.batch(bad)
        except ValueError: pass
        else: raise AssertionError('Terminal goal must fail')
    bad = dict(raw); bad['next_observations'] = raw['next_observations'].copy(); bad['next_observations'][0, 0] = 1001
    try: frozen.batch(bad)
    except ValueError: pass
    else: raise AssertionError('Cross-episode next must fail')


def test_zero_visual_195_matches_visual_shape_state_reward_and_rng():
    import importlib.util
    import reader
    spec = importlib.util.spec_from_file_location('prior_visual195_reader', Path(__file__).parent/'original_bytes/reader.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    class Lookup:
        binding = {'fixture': 'same_source'}
        def lookup(self, role, ids, **kwargs): return np.repeat(ids+7, 192, axis=1).astype(np.float32)
    states = np.arange(6*28, dtype=np.float32).reshape(6, 28)/100
    states[:, 19:22] = 0.; states[0, 19] = .4; states[1, 19] = 2.; states[3, 19] = .40001
    def ids(value): return np.full((1024, 1), value, np.int64)
    raw = dict(observations=ids(0), next_observations=ids(1), value_goals=ids(2), actor_goals=ids(3),
               actions=np.ones((1024, 5), np.float32), rewards=-np.ones(1024, np.float32), masks=np.ones(1024, np.float32))
    before = np.random.get_state()
    visual = reader.CachedLatents({'train': Lookup()}, {'train': {'observations': states}}, 'cache', 'latent_object_xyz')
    previous = old.CachedLatents({'train': Lookup()}, {'train': {'observations': states}}, 'cache', 'latent_object_xyz')
    v, p = visual.batch(raw), previous.batch(raw)
    assert visual.source_binding == previous.source_binding and visual.binding_hash == previous.binding_hash
    for key in v: np.testing.assert_array_equal(v[key], p[key])
    zero = reader.CachedLatents({'train': Lookup()}, {'train': {'observations': states}}, 'zero192', 'latent_object_xyz')
    z = zero.batch(raw)
    for role in ('observations', 'next_observations', 'value_goals', 'actor_goals'):
        assert z[role].shape == v[role].shape == (1024, 220 if 'observations' in role else 195)
        assert z[role].dtype == np.float32 and np.count_nonzero(z[role][:, :192]) == 0
        np.testing.assert_array_equal(z[role][:, 192:], v[role][:, 192:])
    for key in ('actions', 'rewards', 'masks'): np.testing.assert_array_equal(z[key], v[key])
    np.testing.assert_array_equal(z['rewards'], np.zeros(1024, np.float32))
    np.testing.assert_array_equal(z['masks'], np.zeros(1024, np.float32))
    assert not np.array_equal(z['actor_goals'][:, -3:], z['value_goals'][:, -3:])
    rng_equal(before, np.random.get_state())


def test_visual_xyz_template_and_original_default_unchanged():
    import importlib.util
    import reader
    import train
    from unittest.mock import patch
    path = Path(__file__).parent/'original_bytes/reader.py'
    spec = importlib.util.spec_from_file_location('previous_cached_reader', path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    class Lookup:
        binding = {'fixture': True}
        def lookup(self, role, ids, **kwargs): return np.repeat(ids, 192, axis=1).astype(np.float32)
    states = np.arange(4*28, dtype=np.float32).reshape(4, 28)
    raw = dict(observations=np.array([[0]], np.int64), next_observations=np.array([[1]], np.int64),
               value_goals=np.array([[2]], np.int64), actor_goals=np.array([[3]], np.int64),
               actions=np.zeros((1, 5), np.float32), rewards=np.zeros(1, np.float32), masks=np.ones(1, np.float32))
    for mode in ('cache', 'zero192'):
        previous = old.CachedLatents({'train': Lookup()}, {'train': {'observations': states}}, mode)
        current = reader.CachedLatents({'train': Lookup()}, {'train': {'observations': states}}, mode)
        assert previous.source_binding == current.source_binding and previous.binding_hash == current.binding_hash
        a, b = previous.batch(raw), current.batch(raw)
        for key in a: np.testing.assert_array_equal(a[key], b[key])
    class Agent:
        @staticmethod
        def create(seed, observations, actions, cfg, **kwargs): return observations, actions, kwargs['ex_goals']
    with patch.object(train, 'agent_class', return_value=Agent):
        for mode, dim in [('object_xyz', 3), ('latent_object_xyz', 195)]:
            obs, action, goals = train.create_agent(dict(input_profile='cached', goal_mode=mode, seed=0), {})
            assert obs.shape == (1, 220) and action.shape == (1, 5) and goals.shape == (1, dim)
            assert obs.dtype == action.dtype == goals.dtype == np.float32


def test_cached_raw_lookup_and_terminal_role_without_rng(tmp_path):
    import hashlib
    import reader
    assert hasattr(reader, 'CanonicalLatentLookup'), 'Missing canonical raw-ID cache interface'
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    records = []
    for episode in range(2):
        path = tmp_path / f'episode_{episode}.npy'
        values = np.repeat((episode*1001 + np.arange(1000))[:, None], 192, axis=1).astype(np.float32)
        np.save(path, values)
        records.append(dict(episode=episode, path=str(path), shape=[1000, 192], dtype='float32',
                            source_row_start=episode*1001, source_row_end_exclusive=episode*1001+1000))
    manifest_path = tmp_path/'cache_manifest.json'
    manifest_path.write_text(json.dumps(dict(status='passed', exit_code=0, source_modality='pcd',
        source_checkpoint_sha256='example-WM', preprocess_source_sha256='example-preprocess',
        encoder_hashes={'example': 'fixture'}, precision={'dtype': 'float32'}, splits={'train': records})))
    seal_path = tmp_path/'cache_seal.json'
    seal_path.write_text(json.dumps(dict(status='passed', exit_code=0, manifest_sha256=sha(manifest_path),
        cache_stats={record['path']: reader.file_stat(record['path']) for record in records})))
    terminal_path = tmp_path/'train_terminal_next.npz'
    raw_ids = np.array([1000, 2001], np.int64)
    np.savez(terminal_path, raw_ids=raw_ids, latents=np.repeat(raw_ids[:, None], 192, axis=1).astype(np.float32))
    terminal_manifest = tmp_path/'terminal_manifest.json'
    terminal_manifest.write_text(json.dumps(dict(status='passed', exit_code=0, source_modality='pcd',
        cache_manifest_sha256=sha(manifest_path), cache_seal_sha256=sha(seal_path),
        source_checkpoint_sha256='example-WM', preprocess_source_sha256='example-preprocess',
        encoder_hashes={'example': 'fixture'}, precision={'dtype': 'float32'}, raw_id_mapping='episode*1001+1000',
        splits={'train': dict(path=str(terminal_path), sha256=sha(terminal_path), shape=[2, 192], dtype='float32',
                             raw_ids_shape=[2], raw_ids_dtype='int64', raw_ids_sha256=hashlib.sha256(raw_ids.tobytes()).hexdigest())})))
    before = np.random.get_state()
    lookup = reader.CanonicalLatentLookup(manifest_path, terminal_manifest, 'train')
    ids = np.array([[999], [1001], [0], [999]], np.int64)
    actual = lookup.lookup('observations', ids)
    np.testing.assert_array_equal(actual[:, 0], ids[:, 0])
    next_ids = ids+1
    np.testing.assert_array_equal(lookup.lookup('next_observations', next_ids, paired_current_ids=ids)[:, 0], next_ids[:, 0])
    # Goals retain cross-episode/random/duplicate addresses; no BC trajectory restriction.
    np.testing.assert_array_equal(lookup.lookup('value_goals', ids[::-1])[:, 0], ids[::-1, 0])
    assert actual.flags.c_contiguous and actual.dtype == np.float32
    assert not lookup.latents.flags.writeable and not lookup.terminals.flags.writeable
    for role in ('observations', 'value_goals', 'actor_goals'):
        with pytest.raises(ValueError, match='terminal'): lookup.lookup(role, raw_ids)
    with pytest.raises(ValueError, match='paired'): lookup.lookup('next_observations', next_ids)
    with pytest.raises(ValueError, match='next'): lookup.lookup('next_observations', ids+2, paired_current_ids=ids)
    with pytest.raises(ValueError, match='outside'): lookup.lookup('observations', np.array([-1], np.int64))
    with pytest.raises(ValueError, match='int64'): lookup.lookup('observations', ids.astype(np.int32))
    rng_equal(before, np.random.get_state())


def test_legacy_rgb_manifest_positive_and_negative():
    import reader
    assert hasattr(reader, 'cache_modality'), 'Missing explicit historical RGB schema adapter'
    assert reader.cache_modality({'source_rgb_index': '/actual/RGB/index'}) == 'rgb'
    assert reader.cache_modality({'source_modality': 'pixels'}) == 'rgb'
    assert reader.cache_modality({'source_modality': 'pcd'}) == 'pcd'
    for record in ({}, {'source_pcd_index': '/PCD/index'},
                   {'source_rgb_index': '/RGB/index', 'source_pcd_index': '/PCD/index'},
                   {'source_modality': 'unknown'}):
        with pytest.raises(ValueError, match='modality'):
            reader.cache_modality(record)


def test_original_update_target_save_and_loss_ast_unchanged():
    import ast
    directory = Path(__file__).parent
    def functions(path):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    original = functions(directory/'original_bytes/train.py')
    current = functions(directory/'train.py')
    for name in ('update', 'save_boundary', 'restore_boundary', 'host_rng', 'restore_host_rng', 'events_complete'):
        assert current[name] == original[name], name
    def loop(path):
        return ast.dump(next(node for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.While)),
                        include_attributes=False)
    assert loop(directory/'train.py') == loop(directory/'original_bytes/train.py')
    original = functions(directory/'original_bytes/gciql.py')
    current = functions(directory/'gciql.py')
    for name in original.keys()-{'create'}:
        assert current[name] == original[name], name
    ctor = next(node for node in ast.walk(ast.parse((directory/'gciql.py').read_text()))
                if isinstance(node, ast.FunctionDef) and node.name=='create')
    assert 'ex_goals' in [arg.arg for arg in ctor.args.args], 'Missing asymmetric goal init template'


def test_cached_object_reward_current_time_and_zero_control():
    import reader
    assert hasattr(reader, 'CachedLatents'), 'Missing explicit cached input/reward adapter'
    class Lookup:
        binding = {'fixture': True}
        latents = np.ones((4, 192), np.float32)
        terminals = np.ones((1, 192), np.float32)
        latents.flags.writeable = terminals.flags.writeable = False
        def lookup(self, role, ids, **kwargs):
            return np.repeat(np.asarray(ids).reshape(-1, 1), 192, axis=1).astype(np.float32)
    # Row0 and row2 have identical object positions but different robots/source IDs.
    states = np.zeros((4, 28), np.float32)
    states[1, 19] = 1.0
    states[2, :19] = 17.0
    raw = dict(observations=np.array([[0], [1]], np.int64),
               next_observations=np.array([[1], [2]], np.int64),
               value_goals=np.array([[2], [2]], np.int64), actor_goals=np.array([[3], [0]], np.int64),
               actions=np.zeros((2, 5), np.float32), rewards=np.array([-1., -1.], np.float32),
               masks=np.array([1., 1.], np.float32))
    before = np.random.get_state()
    frozen = reader.CachedLatents({'train': Lookup()}, {'train': {'observations': states}}, 'cache')
    batch = frozen.batch(raw)
    assert batch['observations'].shape == batch['next_observations'].shape == (2, 220)
    assert batch['actor_goals'].shape == batch['value_goals'].shape == (2, 3)
    np.testing.assert_array_equal(batch['rewards'], [0., -1.])
    np.testing.assert_array_equal(batch['masks'], [0., 1.])
    np.testing.assert_array_equal(batch['value_goals'], states[[2, 2], 19:22])
    np.testing.assert_array_equal(batch['observations'][:, 192:], states[[0, 1]])
    np.testing.assert_array_equal(raw['rewards'], [-1., -1.])
    control = reader.CachedLatents({'train': Lookup()}, {'train': {'observations': states}}, 'zero192').batch(raw)
    np.testing.assert_array_equal(control['observations'][:, :192], np.zeros((2, 192), np.float32))
    for key in ('actor_goals', 'value_goals', 'actions', 'rewards', 'masks'):
        np.testing.assert_array_equal(batch[key], control[key])
    rng_equal(before, np.random.get_state())


def test_default_and_asymmetric_original_gciql_cpu():
    import importlib.util
    import flax
    import jax
    directory = Path(__file__).parent
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    # Source namespace is provided by the root-authorized CPU invocation, no bridge/WM load.
    original = load('gciql_original_cpu', directory/'original_bytes/gciql.py')
    current = load('gciql_ex_goals_cpu', directory/'gciql.py')
    cfg = original.get_config().to_dict()
    cfg.update(alpha=1.0, encoder=None, frame_stack=None, p_aug=0.0, batch_size=128)
    obs = np.zeros((1, 192), np.float32); actions = np.zeros((1, 5), np.float32)
    def leaves(agent):
        return jax.tree_util.tree_leaves(flax.serialization.to_state_dict(agent))
    legacy = leaves(original.GCIQLAgent.create(0, obs, actions, cfg))
    default = leaves(current.GCIQLAgent.create(0, obs, actions, cfg))
    assert len(legacy) == len(default)
    for left, right in zip(legacy, default): np.testing.assert_array_equal(left, right)
    asym = current.GCIQLAgent.create(0, np.zeros((1, 220), np.float32), actions, cfg,
                                     ex_goals=np.zeros((1, 3), np.float32))
    action = asym.sample_actions(np.zeros((1, 220), np.float32), np.zeros((1, 3), np.float32),
                                seed=jax.random.PRNGKey(17), temperature=0.0)
    assert action.shape == (1, 5) and np.isfinite(action).all()
    assert int(asym.network.step) == 1 and int(asym.network.opt_state[0].count) == 0


def test_cached_original_raw_sampler_rng_and_terminal_next(context):
    config, bridge, _, compacts = context
    cfg = bridge.agent_config(128)
    for split, compact in compacts.items():
        original = bridge.make_sampler(compact, cfg)
        ids = bridge.make_sampler(compact, cfg, ids=True)
        idxs = np.tile(np.array([0, 999, 1001, 2000], np.int64), 32)
        np.random.seed(0)
        before = np.random.get_state()
        expected = original.sample(128, idxs=idxs)
        after = np.random.get_state()
        np.random.set_state(before)
        actual = ids.sample(128, idxs=idxs)
        rng_equal(after, np.random.get_state())
        for role in bridge.OBS_FIELDS:
            np.testing.assert_array_equal(expected[role], compact['observations'][actual[role][:, 0]])
        np.testing.assert_array_equal(actual['next_observations'][:, 0], idxs+1)
        assert 1000 in actual['next_observations'] and not np.any(actual['observations'][:, 0] % 1001 == 1000)
        for field in ('actions', 'rewards', 'masks'):
            np.testing.assert_array_equal(actual[field], expected[field])


def spawn_import_sources(queue):
    import train
    import reader
    import utils
    import jax
    import torch
    queue.put(dict(train=train.__file__, reader=reader.__file__, utils=utils.__file__,
                   jax_backend=jax.default_backend(), torch_cuda_initialized=torch.cuda.is_initialized(),
                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   config=json.loads(config_file().read_text())))


def test_real_spawn_import_namespace(context, tmp_path):
    import multiprocessing as mp
    import train
    config, bridge, source, _ = context
    bridge.setup(source)
    train.pin_run_imports()
    ctx = mp.get_context('spawn'); queue = ctx.Queue()
    child = ctx.Process(target=spawn_import_sources, args=(queue,))
    child.start()
    try:
        record = queue.get(timeout=60)
        child.join(timeout=10)
        assert child.exitcode == 0
        directory = Path(__file__).resolve().parent
        assert Path(record['train']).resolve() == directory/'train.py'
        assert Path(record['reader']).resolve() == directory/'reader.py'
        expected_utils = Path(source['workspace'])/source['official']/'impls/utils/__init__.py'
        assert Path(record['utils']).resolve() == expected_utils.resolve()
        assert record['jax_backend'] == 'cpu' and not record['torch_cuda_initialized']
        assert record['cuda_visible_devices'] == '' and record['config'] == config
        record['actual_child_exit_code'] = child.exitcode
        destination = Path(os.environ.get('FORMAL_CPU_AUX', tmp_path))/'spawn_import_report.json'
        destination.write_text(json.dumps(record, indent=2))
    finally:
        if child.is_alive(): child.terminate(); child.join()
        queue.close(); queue.join_thread()


@pytest.fixture(scope='module')
def context():
    config = json.loads(config_file().read_text()); bridge = load_bridge(config)
    source = json.loads(Path(config['source_config']).read_text()); bridge.setup(source)
    compacts = {split: bridge.load_compact(source, split) for split in ('train', 'val')}
    return config, bridge, source, compacts


def test_original_baseline_sampler_config_and_rng(context):
    config, bridge, _, compacts = context
    from agents.gciql import get_config
    cfg = bridge.agent_config(config['batch_size']); expected = get_config().to_dict()
    expected.update(alpha=1.0, encoder=None, frame_stack=None, batch_size=config['batch_size'])
    assert cfg == expected and cfg['p_aug'] == 0.0
    for split, compact in compacts.items():
        state, ids = bridge.make_sampler(compact, cfg), bridge.make_sampler(compact, cfg, True)
        np.random.seed(0); before = np.random.get_state(); a = state.sample(config['batch_size']); after = np.random.get_state()
        np.random.set_state(before); b = ids.sample(config['batch_size']); rng_equal(after, np.random.get_state())
        for key in bridge.OBS_FIELDS:
            np.testing.assert_array_equal(a[key], compact['observations'][b[key][:, 0]])
        for key in ('actions', 'masks', 'rewards'): np.testing.assert_array_equal(a[key], b[key])


def test_workers_4_vs_8_order_and_parent_rng(context, tmp_path):
    config, bridge, source, compacts = context
    sampler = bridge.make_sampler(compacts['train'], bridge.agent_config(config['batch_size']), True)
    np.random.seed(374); batch = sampler.sample(64); before = np.random.get_state()
    baseline = bridge.RawRows(source, 'pcd', compacts['train'])
    expected = {role: baseline.read(batch[role], role) for role in bridge.OBS_FIELDS}
    records = []
    for workers in (4, 8):
        started = time.perf_counter(); pool = ReadPool(config, source, compacts, workers)
        created = time.perf_counter() - started
        try:
            durations = []
            for repetition in range(2):
                started = time.perf_counter()
                for role in bridge.OBS_FIELDS:
                    actual = pool.read('train', batch[role], role)
                    for key in actual: np.testing.assert_array_equal(actual[key], expected[role][key])
                durations.append(time.perf_counter() - started)
            rng_equal(before, np.random.get_state())
            records.append(dict(workers=workers, pool_constructor_seconds=created,
                                first_read_includes_worker_initialization_seconds=durations[0],
                                second_read_seconds=durations[1], batches=2, rows_per_role=64, roles=4))
        finally:
            pool.close()
    report = dict(records=records, fixed_ids={k: digest_array(batch[k]) for k in bridge.OBS_FIELDS},
                  ordering_bias='4 workers measured before8; second repetitions use warm filesystem cache',
                  parent_numpy_rng_unchanged=True, encoder_forward_run=False)
    destination = Path(os.environ.get('FORMAL_CPU_AUX', tmp_path)) / 'worker_comparison.json'
    destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(json.dumps(report, indent=2))


def test_encoder_chunk_preserves_four_fields_without_rl_accumulation(context):
    import torch
    config, bridge, _, _ = context
    raw = {key: np.array([[4], [2], [4], [1]], np.int64) for key in bridge.OBS_FIELDS}
    raw.update(actions=np.zeros((4, 5), np.float32), rewards=-np.ones(4), masks=np.ones(4))
    calls = []
    def encoder(x):
        calls.append(len(x)); return SimpleNamespace(logits=x)
    fake = SimpleNamespace(reader=SimpleNamespace(io_seconds=0.0), encoder=encoder,
                           inputs=lambda ids, role: torch.tensor(np.repeat(ids.astype(np.float32), 192, axis=1)))
    result = encode_batch(fake, raw, 2)
    assert calls == [2] * 8
    for key in bridge.OBS_FIELDS:
        np.testing.assert_array_equal(result[key][:, 0], raw[key][:, 0])
        assert result[key].shape == (4, 192) and result[key].dtype == np.float32
    for key in ('actions', 'masks', 'rewards'): assert result[key] is raw[key]


def test_event_boundaries():
    config = json.loads(config_file().read_text())
    assert events_complete(1000, 0, config)['last_validation'] == 0
    assert events_complete(5000, 1, config)['next_update'] == 5001
    assert events_complete(100000, 20, config) == dict(committed_updates=100000, validation_count=20,
        next_stage='sample_train', last_validation=100000, checkpoint_event_complete=True, next_update=100001)


def test_worker_cleanup_rejects_replaced_failed_process():
    for failed in (False, True):
        original = SimpleNamespace(pid=1, exitcode=-6 if failed else 0)
        replacement = SimpleNamespace(pid=2, exitcode=0)
        pool = ReadPool.__new__(ReadPool)
        pool.worker_processes = (original,)
        pool.pool = SimpleNamespace(_pool=[replacement], close=lambda: None, join=lambda: None)
        if failed:
            with pytest.raises(RuntimeError, match='exit failure'): pool.close()
            assert pool.exit_codes == {1: -6, 2: 0}
        else:
            assert pool.close() == {1: 0, 2: 0}


def test_current_role_worker_death_and_normal_order(monkeypatch):
    import multiprocessing as mp
    for death in (False, True):
        original = SimpleNamespace(pid=1, exitcode=None)
        replacement = SimpleNamespace(pid=2, exitcode=None)
        operations = []
        pool = ReadPool.__new__(ReadPool); pool.workers = 2; pool.worker_processes = (original,)
        def terminate():
            operations.append('terminate_owned_pool'); replacement.exitcode = -15
        def join(): operations.append('join_owned_pool')
        underlying = SimpleNamespace(_pool=[original], terminate=terminate, join=join)
        def submit(function, jobs):
            operations.append(('submitted_current_role_only', len(jobs)))
            def get(timeout):
                assert timeout == .25
                if death:
                    original.exitcode = -6; underlying._pool = [replacement]
                    raise mp.TimeoutError()  # Simulate the lost task that old map waited for forever.
                return [{'pcd_xyz': ids.copy()} for _, ids, _ in jobs]
            return SimpleNamespace(get=get)
        underlying.map_async = submit; pool.pool = underlying
        ids = np.array([[4], [2], [4], [1]], np.int64)
        before = np.random.get_state()
        if death:
            with pytest.raises(RuntimeError, match='during current role'): pool.read('train', ids, 'observations')
            assert operations[-2:] == ['terminate_owned_pool', 'join_owned_pool']
            assert pool.exit_codes == {1: -6, 2: -15}
        else:
            np.testing.assert_array_equal(pool.read('train', ids, 'observations')['pcd_xyz'], ids)
        rng_equal(before, np.random.get_state())


def test_fsync_commit_order_and_uncommitted_negative(context, tmp_path, monkeypatch):
    import train
    import torch
    from utils import flax_utils
    config = context[0]
    fake_agent = SimpleNamespace(network=SimpleNamespace(step=1, opt_state=[SimpleNamespace(count=0)]))
    frozen = SimpleNamespace(world_model=torch.nn.Linear(1, 1).requires_grad_(False).eval())
    binding = dict(config=config, frozen_full_state=frozen_hash(frozen))
    def save_stub(agent, directory, counter):
        (Path(directory)/f'params_{counter}.pkl').write_bytes(b'CPU fsync-order fixture, not a trained agent')
    monkeypatch.setattr(flax_utils, 'save_agent', save_stub)
    events = []; real_sync = train.fsync_directory; real_rename = train.os.rename
    def sync(path): events.append(('directory_fsync', Path(path).name)); real_sync(path)
    def rename(source, destination): events.append(('rename', Path(source).name)); real_rename(source, destination)
    monkeypatch.setattr(train, 'fsync_directory', sync); monkeypatch.setattr(train.os, 'rename', rename)
    snapshot = save_boundary(fake_agent, frozen, binding, tmp_path, 0, 0, 100000)
    assert events == [('directory_fsync', tmp_path.name), ('directory_fsync', '.update_000000000.tmp'),
                      ('rename', '.update_000000000.tmp'), ('directory_fsync', 'checkpoints')]
    failed_output = tmp_path/'failed'; failed_output.mkdir(); events.clear()
    def fail_stage_sync(path):
        events.append(('directory_fsync', Path(path).name))
        if Path(path).name.endswith('.tmp'): raise OSError('injected stage fsync failure')
        real_sync(path)
    monkeypatch.setattr(train, 'fsync_directory', fail_stage_sync)
    with pytest.raises(OSError, match='stage fsync'): save_boundary(fake_agent, frozen, binding, failed_output, 0, 0, 100000)
    assert not (failed_output/'checkpoints/update_000000000').exists()
    stage = failed_output/'checkpoints/.update_000000000.tmp'
    with pytest.raises(AssertionError, match='committed'): restore_boundary(fake_agent, stage, binding, 100000)


def test_prepare_and_logger_failure_cleanup(context, tmp_path, monkeypatch):
    import train
    import torch
    from agents.gciql import GCIQLAgent
    from utils import log_utils
    config, bridge, _, _ = context
    operations = []
    class FakePool:
        exit_codes = {19: 0}
        def close(self): operations.append('pool_closed'); return self.exit_codes
    fake_model = torch.nn.Linear(1, 1).requires_grad_(False).eval()
    fake_bridge = SimpleNamespace(sha256=bridge.sha256, setup=lambda source: {},
        load_compact=lambda source, split: {'observations': np.zeros((1001000 if split=='train' else 100100, 1), np.int8)},
        agent_config=lambda size: {}, make_sampler=lambda compact, cfg, ids: None,
        FrozenLatents=lambda *args: SimpleNamespace(world_model=fake_model))
    with monkeypatch.context() as patch:
        patch.setattr(train, 'load_bridge', lambda config: fake_bridge)
        patch.setattr(train, 'ReadPool', lambda *args: FakePool())
        def fail_binding(samplers): raise ValueError('injected binding failure')
        patch.setattr(train, 'sampler_binding', fail_binding)
        with pytest.raises(ValueError, match='binding failure') as error: train.prepare(config, 'cpu')
        assert operations == ['pool_closed'] and error.value.worker_exit_codes == {19: 0}
    operations.clear()
    class BadLogger:
        def __init__(self, path): pass
        def close(self): operations.append('logger_close_failed'); raise OSError('injected logger close failure')
    fake_frozen = SimpleNamespace(reader=SimpleNamespace(split='train'))
    sampler = SimpleNamespace(sample=lambda size: {})
    fake_agent = SimpleNamespace(network=SimpleNamespace(step=1))
    output = tmp_path/'main_failure'
    with monkeypatch.context() as patch:
        patch.setattr(train, 'prepare', lambda config: (bridge, {'train': sampler}, FakePool(), fake_frozen, {}, {}))
        patch.setattr(train, 'encode_batch', lambda *args: {'observations': np.zeros((1,192),np.float32), 'actions': np.zeros((1,5),np.float32)})
        patch.setattr(GCIQLAgent, 'create', lambda *args: fake_agent)
        patch.setattr(log_utils, 'CsvLogger', BadLogger)
        def fail_batch(*args): raise ValueError('original loop failure')
        patch.setattr(train, 'batch_for', fail_batch)
        patch.setattr(train.sys, 'argv', ['train.py', '--config', str(config_file()), '--output', str(output)])
        patch.setenv('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        with pytest.raises(ValueError, match='original loop failure'): train.main()
    status = json.loads((output/'status.json').read_text())
    assert status['exit_code'] == 1 and 'original loop failure' in status['error']
    assert 'logger close failure' in status['cleanup_errors'][0] and status['worker_exit_codes'] == {'19': 0}
    assert operations == ['logger_close_failed', 'pool_closed'] and (output/'trainer_exit_code').read_text().strip() == '1'


def test_atomic_single_agent_resume_and_wrong_binding(context, tmp_path):
    import torch
    from agents.gciql import GCIQLAgent
    config, bridge, _, compacts = context
    cfg = bridge.agent_config(config['batch_size'])
    agent = GCIQLAgent.create(0, np.zeros((1, 192), np.float32), np.zeros((1, 5), np.float32), cfg)
    model = torch.nn.Linear(1, 1).requires_grad_(False).eval(); frozen = SimpleNamespace(world_model=model)
    binding = dict(config=config, frozen_full_state=frozen_hash(frozen), contract='CPU transaction test only')
    sampler = bridge.make_sampler(compacts['train'], cfg, True)
    np.random.seed(0); sampler.sample(1)
    snapshot = save_boundary(agent, frozen, binding, tmp_path, 0, 0, 100000)
    assert len(list(snapshot.glob('params_*.pkl'))) == 1
    with (snapshot / 'run_state.pkl').open('rb') as f: sidecar = pickle.load(f)
    assert 'state' not in sidecar and 'agent' not in sidecar and sidecar['events']['next_update'] == 1
    expected = sampler.sample(config['batch_size']); expected_rng = np.random.get_state()
    restored, counter, validations = restore_boundary(agent, snapshot, binding, 200000)
    assert counter == validations == 0 and int(restored.network.step) == 1
    actual = sampler.sample(config['batch_size'])
    for key in actual: np.testing.assert_array_equal(actual[key], expected[key])
    rng_equal(expected_rng, np.random.get_state())
    with pytest.raises(AssertionError, match='budget'):
        restore_boundary(agent, snapshot, dict(binding, contract='wrong source'), 200000)
    with pytest.raises(FileExistsError): save_boundary(agent, frozen, binding, tmp_path, 0, 0, 100000)
    with pytest.raises(AssertionError, match='Validation event'):
        save_boundary(agent, frozen, binding, tmp_path, 0, 1, 100000)
    with pytest.raises(AssertionError, match='committed'):
        uncommitted = tmp_path / '.incomplete'; uncommitted.mkdir()
        (uncommitted/'manifest.json').write_text((snapshot/'manifest.json').read_text())
        restore_boundary(agent, uncommitted, binding, 200000)
    with torch.no_grad(): model.weight.add_(1)
    with pytest.raises(ValueError, match='bytes'): save_boundary(agent, frozen, binding, tmp_path, 0, 0, 100000)


def agent_digest(agent):
    import flax
    import jax
    state = flax.serialization.to_state_dict(agent)
    leaves, tree = jax.tree_util.tree_flatten(state)
    return dict(tree=str(tree), leaves=[digest_array(x) for x in leaves])


def gpu_gate(args):
    """Called only by a separately authorized root invocation; never by CPU pytest."""
    import jax
    import torch
    from train import create_agent, helpers
    formal_path = Path(args.config)
    formal = json.loads(formal_path.read_text())
    # One artificial boundary covers val-before-save and fresh next-batch resume.
    gate_updates = 2 if formal.get('goal_mode') == 'latent_object_xyz' else 1
    config = dict(formal, validation_interval=gate_updates, save_interval=gate_updates)
    output = Path(args.output)
    if not args.restore_check: output.mkdir(parents=True, exist_ok=False)
    utility = helpers(config); pool = None; code = 1; report = None
    status_path = output / ('restore_status.json' if args.restore_check else 'status.json')
    utility.atomic_json(status_path, dict(stage='gate_starting', exit_code=None, actual_rl_batch=config['batch_size']))
    try:
        bridge, samplers, pool, frozen, cfg, binding = prepare(config)
        if args.restore_check:
            template = create_agent(config, cfg)
            agent, counter, validations = restore_boundary(template, output/f'checkpoints/update_{gate_updates:09d}', binding, formal['total_updates'])
            assert counter == gate_updates and validations == 1
            raw, batch = batch_for(frozen, samplers['train'], 'train', config)
            agent, info = update(agent, batch, frozen)
            with (output/'expected.pkl').open('rb') as f: expected = pickle.load(f)
            assert {k: digest_array(v) for k, v in raw.items()} == expected['raw']
            assert {k: digest_array(v) for k, v in batch.items()} == expected['batch']
            assert agent_digest(agent) == expected['agent']
            assert digest_array(host_rng()['numpy'][1]) == expected['numpy_rng_array']
            rng_equal(host_rng()['numpy'], expected['numpy_rng'])
            actual_rng = host_rng()
            assert actual_rng['python'] == expected['host_rng']['python']
            assert torch.equal(actual_rng['torch'], expected['host_rng']['torch'])
            assert len(actual_rng['cuda']) == len(expected['host_rng']['cuda'])
            assert all(torch.equal(x, y) for x, y in zip(actual_rng['cuda'], expected['host_rng']['cuda']))
            assert frozen_hash(frozen) == binding['frozen_full_state']
            report = json.loads((output/'gate_update_report.json').read_text())
            report.update(status='passed', exit_code=0, fresh_next_batch_full_state_exact=True,
                           restored_network_step=gate_updates+1, continuation_network_step=int(agent.network.step))
        else:
            cached = config.get('input_profile') == 'cached'
            if cached:
                chunk_measure = dict(runtime_encoder_gate_skipped=True, cache_producer_binding=frozen.source_binding)
            else:
                torch.cuda.reset_peak_memory_stats()
                inputs = frozen.inputs(np.arange(64, dtype=np.int64)[:, None], 'observations')
                direct = frozen.encoder(inputs).logits.cpu().numpy()
                chunks = np.concatenate([frozen.encoder(part).logits.cpu().numpy() for part in inputs.split(32)])
                np.testing.assert_allclose(direct, chunks, atol=1e-5, rtol=1e-5)
                assert direct.shape == (64, 192) and direct.dtype == chunks.dtype == np.float32
                chunk_measure = dict(direct64_vs_chunk32_max_abs_error=float(np.max(np.abs(direct-chunks))),
                                     declared_atol=1e-5, declared_rtol=1e-5)
            random.seed(0); np.random.seed(0); torch.manual_seed(0)
            example = encode_batch(frozen, samplers['train'].sample(1), 64)
            agent = create_agent(config, cfg, example['observations'], example['actions'], example['actor_goals'])
            initial = agent_digest(agent)
            initial_modules = {name: [digest_array(x) for x in jax.tree_util.tree_leaves(agent.network.params['modules_'+name])]
                               for name in ('actor', 'value', 'critic')}
            for _ in range(gate_updates):
                raw, batch = batch_for(frozen, samplers['train'], 'train', config)
                agent, info = update(agent, batch, frozen)
            training_timing = dict(frozen.last_batch_timing, **frozen.last_update_timing)
            assert agent_digest(agent) != initial and int(agent.network.step) == gate_updates+1
            assert int(agent.network.opt_state[0].count) == gate_updates
            assert all(initial_modules[name] != [digest_array(x) for x in jax.tree_util.tree_leaves(agent.network.params['modules_'+name])]
                       for name in initial_modules)
            _, val_batch = batch_for(frozen, samplers['val'], 'val', config)
            _, val_info = agent.total_loss(val_batch, grad_params=None)
            assert all(np.isfinite(float(x)) for x in val_info.values())
            save_boundary(agent, frozen, binding, output, gate_updates, 1, formal['total_updates'])
            raw, batch = batch_for(frozen, samplers['train'], 'train', config)
            continuation, continuation_info = update(agent, batch, frozen)
            expected_rng = host_rng()
            with (output/'expected.pkl').open('xb') as f:
                pickle.dump(dict(raw={k: digest_array(v) for k, v in raw.items()},
                    batch={k: digest_array(v) for k, v in batch.items()}, agent=agent_digest(continuation),
                    numpy_rng=expected_rng['numpy'], numpy_rng_array=digest_array(expected_rng['numpy'][1]),
                    host_rng=expected_rng), f)
            assert frozen_hash(frozen) == binding['frozen_full_state']
            report = dict(status='updates_passed_restore_pending', exit_code=None, code=binding['code'],
                formal_config_sha256=utility.sha(formal_path), actual_rl_batch=config['batch_size'], train_updates=gate_updates,
                validation_batches=1, validation_batch=config['batch_size'], diagnostic_continuation_updates=1,
                artificial_validation_checkpoint_boundary=gate_updates, process_evaluation_episodes=0,
                batch_shapes={key: list(value.shape) for key, value in batch.items()},
                source_binding=binding, **chunk_measure, update_info=info, training_timing=training_timing,
                torch_peak_allocated_bytes=None if cached else torch.cuda.max_memory_allocated(),
                torch_peak_reserved_bytes=None if cached else torch.cuda.max_memory_reserved(),
                jax_memory_stats=jax.devices()[0].memory_stats(), frozen_full_state_equal=True)
        code = 0
    finally:
        try:
            exits = pool.close() if pool else {}
        except BaseException:
            code = 1
            raise
        finally:
            exits = getattr(pool, 'exit_codes', {})
            if report is not None and code == 0:
                report['worker_exit_codes'] = exits
                utility.atomic_json(output/('report.json' if args.restore_check else 'gate_update_report.json'), report)
            utility.atomic_json(status_path, dict(stage='gate_stage_exit0' if code==0 else 'gate_failed',
                                                  exit_code=code, worker_exit_codes=exits))
            (output/('restore_exit_code' if args.restore_check else 'update_exit_code')).write_text(str(code)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--gpu-gate', action='store_true', required=True)
    parser.add_argument('--restore-check', action='store_true'); parser.add_argument('--output', required=True)
    parser.add_argument('--config', default=str(config_file()))
    gpu_gate(parser.parse_args())



def test_rl3_proprio_drop_object_gt_keep_raw_reward_and_default_exact():
    import importlib.util
    from reader import CachedLatents
    old_path = Path(__file__).parent/'rl3_parent_bytes/reader.py'
    spec = importlib.util.spec_from_file_location('rl3_parent_reader', old_path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    class Lookup:
        binding = {'fixture': 'same_raw_row_cached192'}
        latents = np.zeros((6,192), np.float32); terminals = np.zeros((1,192), np.float32)
        latents.flags.writeable = terminals.flags.writeable = False
        def lookup(self, role, ids, **kwargs): return np.repeat(ids+7, 192, axis=1).astype(np.float32)
    base = np.arange(6*28, dtype=np.float32).reshape(6, 28)/100
    base[:, 19:22] = 0.; base[0, 19] = .4; base[1, 19] = 2.
    def ids(value): return np.full((1024, 1), value, np.int64)
    raw = dict(observations=ids(0), next_observations=ids(1), value_goals=ids(2), actor_goals=ids(3),
               actions=np.ones((1024, 5), np.float32), rewards=-np.ones(1024, np.float32), masks=np.ones(1024, np.float32))
    before = np.random.get_state()
    def make(states, mode, profile='full28'):
        return CachedLatents({'train': Lookup()}, {'train': {'observations': states.copy()}}, mode, 'latent_object_xyz', profile)
    for mode in ('cache', 'zero192'):
        default = make(base, mode); previous = old.CachedLatents({'train': Lookup()}, {'train': {'observations': base.copy()}}, mode, 'latent_object_xyz')
        assert default.source_binding == previous.source_binding and default.binding_hash == previous.binding_hash
        actual, expected = default.batch(raw), previous.batch(raw)
        for key in actual: np.testing.assert_array_equal(actual[key], expected[key])
        profile = make(base, mode, 'robot_proprio'); batch = profile.batch(raw)
        for role in ('observations', 'next_observations'):
            assert batch[role].shape == (1024, 211)
            np.testing.assert_array_equal(batch[role][:, 192:], base[raw[role][:,0], :19])
        for role in ('value_goals', 'actor_goals'):
            assert batch[role].shape == (1024, 195)
            np.testing.assert_array_equal(batch[role][:, -3:], base[raw[role][:,0], 19:22])
        changed = base.copy(); changed[:, 22:28] += 12345
        no_object_orientation = make(changed, mode, 'robot_proprio').batch(raw)
        for key in batch: np.testing.assert_array_equal(no_object_orientation[key], batch[key])
        changed = base.copy(); changed[0,19] = .40001
        changed_reward = make(changed, mode, 'robot_proprio').batch(raw)
        for key in ('observations', 'next_observations', 'actor_goals', 'value_goals', 'actions'):
            np.testing.assert_array_equal(changed_reward[key], batch[key])
        np.testing.assert_array_equal(batch['rewards'], np.zeros(1024, np.float32))
        np.testing.assert_array_equal(changed_reward['rewards'], -np.ones(1024, np.float32))
        np.testing.assert_array_equal(changed_reward['masks'], np.ones(1024, np.float32))
        assert profile.source_binding['excluded_object_state_indices'] == list(range(19,28))
        profile.guard()
    rng_equal(before, np.random.get_state())
    with pytest.raises(ValueError, match='state profile'): make(base, 'cache', 'typo')


def test_rl3_factory_template_uses_profile_shape_only(monkeypatch):
    import train
    calls = []
    class Agent:
        @staticmethod
        def create(seed, observations, actions, cfg, **kwargs):
            calls.append((seed, observations, actions, cfg, kwargs)); return 'fixture'
    monkeypatch.setattr(train, 'agent_class', lambda config: Agent)
    for profile, dim in [('full28',220), ('robot_proprio',211)]:
        config = dict(input_profile='cached', goal_mode='latent_object_xyz', seed=0, state_profile=profile)
        assert train.create_agent(config, {'fixture':'untouched'}) == 'fixture'
        seed, current, actions, cfg, kwargs = calls[-1]
        assert seed == 0 and current.shape == (1,dim) and actions.shape == (1,5)
        assert kwargs['ex_goals'].shape == (1,195) and not current.any()
    with pytest.raises(ValueError, match='state profile'):
        train.create_agent(dict(input_profile='cached',seed=0,state_profile='typo'), {})



def test_cached_RAM_workers_exact_current_batch_only_and_closed_threads():
    import importlib.util
    import threading
    from reader import CachedLatents
    import train
    spec=importlib.util.spec_from_file_location('zero_worker_parent_reader',Path(__file__).parent/'worker_parent_bytes/reader.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    class Lookup:
        binding={'fixture':'same_parent_sampled_rawIDs'}
        latents=np.zeros((6,192),np.float32);terminals=np.zeros((1,192),np.float32)
        latents.flags.writeable=terminals.flags.writeable=False
        def __init__(self,workers=0,fail=False):
            self.barrier=threading.Barrier(workers) if workers else None
            self.calls=[];self.lock=threading.Lock();self.fail=fail
        def lookup(self,role,ids,**kwargs):
            if self.barrier is not None:self.barrier.wait(timeout=5)
            with self.lock:self.calls.append((role,threading.get_ident(),ids.copy()))
            if self.fail and role=='value_goals':raise ValueError('owned cached role failure')
            return np.repeat(ids+7,192,axis=1).astype(np.float32)
    states=np.arange(6*28,dtype=np.float32).reshape(6,28)/100
    states[:,19:22]=0.;states[0,19]=.4
    def ids(value):return np.full((1024,1),value,np.int64)
    raw=dict(observations=ids(0),next_observations=ids(1),value_goals=ids(2),actor_goals=ids(3),actions=np.ones((1024,5),np.float32),rewards=-np.ones(1024,np.float32),masks=np.ones(1024,np.float32))
    before=np.random.get_state()
    for mode,profile in [('zero192','full28'),('cache','robot_proprio')]:
        expected=old.CachedLatents({'train':Lookup()},{'train':{'observations':states.copy()}},mode,'latent_object_xyz',profile)
        baseline=expected.batch(raw)
        current=CachedLatents({'train':Lookup()},{'train':{'observations':states.copy()}},mode,'latent_object_xyz',profile)
        assert current.source_binding==expected.source_binding and current.binding_hash==expected.binding_hash
        for key,value in current.batch(raw).items():np.testing.assert_array_equal(value,baseline[key])
        for workers in (2,4):
            lookup=Lookup(workers);frozen=CachedLatents({'train':lookup},{'train':{'observations':states.copy()}},mode,'latent_object_xyz',profile,workers)
            actual=frozen.batch(raw)
            assert list(actual)==list(baseline)
            for key,value in actual.items():
                assert value.dtype==baseline[key].dtype;np.testing.assert_array_equal(value,baseline[key])
            assert [name for name in ('observations','next_observations','value_goals','actor_goals') if name in {c[0] for c in lookup.calls}]==list(raw)[:4]
            assert len(lookup.calls)==4 and len({c[1] for c in lookup.calls})==workers
            assert all(np.array_equal(ids,raw[role]) for role,thread,ids in lookup.calls)
            threads=list(frozen.executor._threads);assert len(threads)==workers and all(t.is_alive() for t in threads)
            exits,errors=train.close_resources(None,frozen);assert exits=={} and errors==[] and all(not t.is_alive() for t in threads)
            assert frozen.source_binding['cache_lookup_workers']==workers and frozen.source_binding['ahead_sampling'] is False
    lookup=Lookup(2,True);frozen=CachedLatents({'train':lookup},{'train':{'observations':states.copy()}},'cache','latent_object_xyz','robot_proprio',2)
    try:
        with pytest.raises(ValueError,match='owned cached role failure'):frozen.batch(raw)
        assert len(lookup.calls)==4  # All submitted tasks joined before propagating the worker error.
    finally:frozen.close()
    assert all(not t.is_alive() for t in frozen.executor._threads)
    rng_equal(before,np.random.get_state())
    with pytest.raises(ValueError,match='lookup workers'):
        CachedLatents({'train':Lookup()},{'train':{'observations':states.copy()}},'cache','latent_object_xyz','robot_proprio',8)
