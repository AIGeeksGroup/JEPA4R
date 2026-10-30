"""Bounded official state GCIQL smoke; the eight-row fixture is not a dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMPL = ROOT / 'source/ogbench/impls'
sys.path.insert(0, str(IMPL))
sys.path.insert(1, str(ROOT / 'source/ogbench'))
import flax
import jax
import ogbench
from agents.gciql import GCIQLAgent, get_config
from utils.datasets import Dataset, GCDataset
from utils.flax_utils import restore_agent, save_agent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def phase(out, name, **fields):
    (out / 'phase.json').write_text(json.dumps(dict(phase=name, time=time.time(), **fields)) + '\n')


def fingerprint(agent):
    digest = hashlib.sha256()
    leaves = jax.tree_util.tree_leaves(flax.serialization.to_state_dict(agent))
    assert len(leaves) == 153
    for value in leaves:
        array = np.asarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return dict(sha256=digest.hexdigest(), leaves=len(leaves))


def metrics(info):
    result = {key: float(value) for key, value in info.items()}
    assert all(np.isfinite(value) for value in result.values())
    return result


def config():
    cfg = get_config()
    cfg.alpha = 1.0  # Official cube-triple-play-v0 task override.
    assert cfg.batch_size == 1024 and cfg.encoder is None and cfg.frame_stack is None
    return cfg


def check_batch(batch):
    for key in ('observations', 'next_observations', 'value_goals', 'actor_goals'):
        assert batch[key].shape == (1024, 46) and batch[key].dtype == np.float32
    assert batch['actions'].shape == (1024, 5) and batch['actions'].dtype == np.float32


def dataset(out, cfg):
    with np.load(ROOT / 'smoke/features.npz') as source:
        observations = source['observations']
        actions = source['actions']
    assert observations.shape == (8, 46) and actions.shape == (8, 5)
    terminals = np.zeros(8, np.float32)
    terminals[-1] = 1  # Artificial fixture boundary; not a true episode terminal.
    fixture = out / 'fixture_state.npz'
    assert not fixture.exists()
    np.savez(fixture, observations=observations, actions=actions, terminals=terminals)
    raw = ogbench.load_dataset(str(fixture), compact_dataset=True)
    return GCDataset(Dataset.create(**raw), cfg)


def main(args):
    assert jax.default_backend() == 'gpu'
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = config()
    source_files = ('agents/gciql.py', 'utils/datasets.py', 'utils/flax_utils.py', 'main.py')
    identity = dict(
        source_sha256={name: sha(IMPL / name) for name in source_files},
        imported_agent=str(Path(sys.modules[GCIQLAgent.__module__].__file__).resolve()),
        official_config=cfg.to_dict(), JAX=jax.__version__, NumPy=np.__version__,
        JAX_default_matmul_precision=str(jax.config.jax_default_matmul_precision),
        fixture_sha256=sha(ROOT / 'smoke/features.npz'),
        current_dim=46, goal_dim=46, action_dim=5, batch_size=1024,
        official_agent_sampler_save_restore_unmodified=True,
        zero_padding=False, visual_mask=False, encoder_calls=0,
        fixture_only=True, fixture_cut_not_true_terminal=True,
        full_official_dataset_tested=False, SR_started=False,
        formal_training_started=False,
    )
    np.random.seed(0)
    if args.fresh:
        phase(out, 'fresh_restore')
        with np.load(out / 'reload_batch.npz') as stored:
            batch = dict(stored)
        check_batch(batch)
        agent = GCIQLAgent.create(0, batch['observations'][:1], batch['actions'][:1], cfg)
        agent = restore_agent(agent, str(out / 'checkpoint'), 2)
        actual = fingerprint(agent)
        baseline = json.loads((out / 'report.json').read_text())
        assert baseline['source_sha256'] == identity['source_sha256']
        assert baseline['official_config'] == json.loads(json.dumps(identity['official_config']))
        expected = baseline['saved_state']
        assert actual == expected
        assert int(agent.network.step) == 3 and int(agent.network.opt_state[0].count) == 2
        actions = np.asarray(agent.sample_actions(batch['observations'], batch['actor_goals'],
                             seed=jax.random.PRNGKey(0), temperature=0))
        assert actions.shape == (1024, 5) and np.isfinite(actions).all()
        write_report(out / 'fresh_report.json', dict(**identity, actual_exit_code=0,
            full_state_exact=actual, finite_actions=True, network_step=3, Adam_count=2,
            restored_JAX_agent_RNG=True, host_NumPy_RNG_restore_claimed=False))
        phase(out, 'fresh_restore_complete')
        return
    data = dataset(out, cfg)
    example = data.sample(1)
    agent = GCIQLAgent.create(0, example['observations'], example['actions'], cfg)
    assert int(agent.network.step) == 1 and int(agent.network.opt_state[0].count) == 0
    if args.bench:
        phase(out, 'benchmark_warmup', updates=64)
        for _ in range(64):
            agent, info = agent.update(data.sample(1024))
        jax.block_until_ready((agent, info))
        windows = []
        for index in range(3):
            phase(out, 'benchmark_timed_window', window=index + 1, updates=256)
            host_seconds = 0.0
            wall_started = time.time()
            started = time.perf_counter()
            for _ in range(256):
                sample_start = time.perf_counter()
                batch = data.sample(1024)
                host_seconds += time.perf_counter() - sample_start
                agent, info = agent.update(batch)
            jax.block_until_ready((agent, info))
            elapsed = time.perf_counter() - started
            wall_finished = time.time()
            observed = metrics(info)
            assert observed['grad/norm'] > 0
            windows.append(dict(window=index + 1, updates=256, seconds=elapsed,
                wall_started=wall_started, wall_finished=wall_finished,
                updates_per_second=256 / elapsed, host_sample_seconds=host_seconds,
                last_metrics=observed))
        assert int(agent.network.step) == 833 and int(agent.network.opt_state[0].count) == 832
        write_report(out / 'benchmark_report.json', dict(**identity, actual_exit_code=0,
            warmup_updates=64, timed_updates=768, total_updates=832,
            network_step=833, Adam_count=832, windows=windows,
            timed_windows_synchronize_all_agent_and_info_leaves=True,
            GPU_utilization='external supervisor samples; not inferred from throughput',
            scope='Repeated eight-row fixture; not full-data throughput or SR evidence'))
        phase(out, 'benchmark_complete')
        return
    losses = []
    phase(out, 'two_updates')
    for _ in range(2):
        batch = data.sample(1024)
        check_batch(batch)
        agent, info = agent.update(batch)
        observed = metrics(info)
        assert observed['grad/norm'] > 0
        losses.append(observed)
    assert int(agent.network.step) == 3 and int(agent.network.opt_state[0].count) == 2
    phase(out, 'validation_same_fixture')
    validation = metrics(agent.total_loss(data.sample(1024), grad_params=None)[1])
    saved = fingerprint(agent)
    (out / 'checkpoint').mkdir(exist_ok=False)
    save_agent(agent, str(out / 'checkpoint'), 2)
    np.savez(out / 'reload_batch.npz', **batch)
    write_report(out / 'report.json', dict(**identity, actual_exit_code=0,
        two_update_metrics=losses, validation=validation, validation_source='same eight-row fixture',
        network_step=3, Adam_count=2, saved_state=saved,
        first_value_actor_input=92, first_critic_input=97,
        reward='official source_idx == value_goal_idx: success-1; mask=1-success'))
    phase(out, 'baseline_complete_waiting_fresh_restore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--fresh', action='store_true')
    mode.add_argument('--bench', action='store_true')
    main(parser.parse_args())
