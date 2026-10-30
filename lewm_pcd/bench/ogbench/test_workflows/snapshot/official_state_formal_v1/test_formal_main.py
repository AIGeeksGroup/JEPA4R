"""CPU-only integration checks of the unchanged official main function AST."""
import ast
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).parent
OLD = HERE.parent / 'official_state_workers_v1'
saved_argv = sys.argv
sys.argv = ['test_ordered_input.py', '--source-only']
try:
    spec = importlib.util.spec_from_file_location('_formal_fixture_reference', OLD / 'test_ordered_input.py')
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
finally:
    sys.argv = saved_argv

from formal_main import FormalInputBridge, MAIN_SHA, compile_deferred_main


def rng_bytes():
    state = np.random.get_state()
    return (state[0], state[1].tobytes(), *state[2:])


def batch_bytes(batch):
    return [(key, value.shape, value.dtype.str, value.tobytes()) for key, value in batch.items()]


def make_official(output, events):
    source = (HERE / 'source_reference/main.py').read_bytes()
    assert hashlib.sha256(source).hexdigest() == MAIN_SHA
    fixture = reference.fixture()
    config = dict(fixture.config, agent_name='fake', dataset_class='GCDataset', batch_size=1024, discrete=False)
    flags = types.SimpleNamespace(seed=0, run_group='test', save_dir=str(output), env_name='cube-triple-play-v0',
        agent=config, restore_path=None, train_steps=6, log_interval=2, eval_interval=3,
        save_interval=3, eval_on_cpu=1, eval_tasks=None, eval_episodes=50, video_episodes=0,
        video_frame_skip=3, eval_temperature=0., eval_gaussian=None)

    class FakeAgent:
        def __init__(self):
            self.network = types.SimpleNamespace(step=1, opt_state=[types.SimpleNamespace(count=0)])

        @classmethod
        def create(cls, seed, observations, actions, config):
            events.append(('example', observations.tobytes(), actions.tobytes(), rng_bytes()))
            return cls()

        def update(self, batch):
            events.append(('train', batch_bytes(batch)))
            self.network.step += 1
            self.network.opt_state[0].count += 1
            return self, {}

        def total_loss(self, batch, grad_params):
            events.append(('val', batch_bytes(batch), rng_bytes()))
            return 0., {}

    env = types.SimpleNamespace(task_infos=[dict(task_name='one'), dict(task_name='two')])
    env.unwrapped = env

    def load_raw(name, **kwargs):
        events.append(('data_route', name, kwargs.get('compact_dataset'), kwargs.get('dataset_path')))
        raw = {key: value.copy() for key, value in fixture.dataset._dict.items()}
        return env, raw, {key: value.copy() for key, value in raw.items()}

    ogbench = types.SimpleNamespace(make_env_and_datasets=load_raw)

    def make_env(name, frame_stack=None):
        return ogbench.make_env_and_datasets(name, compact_dataset=True)

    def evaluate(**kwargs):
        events.append(('eval', kwargs['agent'].network.opt_state[0].count,
                       kwargs['task_id'], kwargs['num_eval_episodes'],
                       int(np.random.randint(0, 2**32, dtype=np.int64)), rng_bytes()))
        return {'success': 0.}, [], []

    def save_agent(agent, path, epoch):
        events.append(('save', epoch, rng_bytes()))
        (Path(path) / f'params_{epoch}.pkl').write_text(json.dumps(dict(count=epoch)))

    def restore_agent(agent, path, epoch):
        value = FakeAgent()
        count = json.loads((Path(path) / f'params_{epoch}.pkl').read_text())['count']
        value.network.step = count + 1
        value.network.opt_state[0].count = count
        return value

    class Logger:
        def __init__(self, path):
            pass

        def log(self, metrics, step):
            pass

        def close(self):
            pass

    namespace = dict(json=json, os=__import__('os'), random=random, time=__import__('time'),
        defaultdict=defaultdict, np=np, FLAGS=flags, get_exp_name=lambda seed: 'fixed',
        setup_wandb=lambda **kwargs: events.append(('wandb_mode', kwargs.get('mode', 'online'))),
        wandb=types.SimpleNamespace(run=types.SimpleNamespace(project='fixture'),
                                   log=lambda *a, **k: None, finish=lambda: None),
        get_flag_dict=lambda: {}, make_env_and_datasets=make_env,
        Dataset=reference.OFFICIAL.Dataset, GCDataset=reference.OFFICIAL.GCDataset,
        HGCDataset=object, agents={'fake': FakeAgent}, restore_agent=restore_agent,
        CsvLogger=Logger, evaluate=evaluate, save_agent=save_agent,
        tqdm=types.SimpleNamespace(tqdm=lambda items, **kwargs: items, trange=range),
        jax=types.SimpleNamespace(device_put=lambda agent, device: agent, devices=lambda name: ['CPU']))
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    # main.__globals__ must be the same namespace that runtime patches update.
    module = types.ModuleType('_formal_original_main_ast')
    module.__dict__.update(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(HERE / 'source_reference/main.py'), 'exec'), module.__dict__)
    return module, ogbench


class FormalBoundaryContract(unittest.TestCase):
    def test_original_main_sequence_example_validation_eval_save_rng(self):
        sequences = []
        original_randint = np.random.randint

        def linux_randint(low, high=None, *args, **kwargs):
            if high == 2**32:
                kwargs['dtype'] = np.int64  # Windows default int is int32; target Linux is int64.
            return original_randint(low, high, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            for candidate in (False, True):
                events = []
                root = Path(folder) / ('candidate' if candidate else 'original')
                official, ogbench = make_official(root, events)
                with mock.patch.dict(sys.modules, ogbench=ogbench), mock.patch.object(np.random, 'randint', linux_randint):
                    if candidate:
                        with FormalInputBridge(official, Path('/accepted/data'), root / 'bridge') as bridge:
                            code, proof = compile_deferred_main(official,
                                (HERE / 'source_reference/main.py').read_bytes(), bridge.defer_evaluation_node)
                            self.assertEqual(proof['changed_AST_sites'], 2)
                            self.assertTrue(proof['all_other_AST_equal'])
                            official.main([])
                            self.assertEqual(bridge.step, 6)
                            self.assertFalse(bridge.inputs.pending)
                            self.assertEqual(len([event for event in events if event[0] == 'eval']), 0)
                            bridge.run_deferred_evaluations(code)
                    else:
                        official.main([])
                final_rng = rng_bytes()
                # Only the explicit local data-directory forwarding and offline
                # logging differ. Original main's sample/eval/save RNG is exact.
                sequences.append(([event for event in events if event[0] in ('train', 'val', 'example')],
                                  [event for event in events if event[0] == 'eval'], final_rng))
                if candidate:
                    self.assertIn(('wandb_mode', 'offline'), events)
                    self.assertIn(('data_route', 'cube-triple-play-v0', True,
                                   str(Path('/accepted/data') / 'cube-triple-play-v0.npz')), events)
                    saved = list(root.rglob('checkpoint_6_sidecar.json'))
                    self.assertEqual(len(saved), 1)
                    record = json.loads(saved[0].read_text())
                    self.assertEqual(record['network_step'], 7)
                    self.assertEqual(record['adam_count'], 6)
                    self.assertTrue(Path(record['host_numpy_rng_sidecar']).is_file())
                    self.assertEqual(sorted(int(path.stem.split('_')[1]) for path in root.rglob('params_*.pkl')),
                                     [1, 3, 6])
        self.assertEqual(sequences[0], sequences[1])

    def test_actual_boundary_steps_sample1_and_final_have_no_lookahead(self):
        events = []
        with tempfile.TemporaryDirectory() as folder:
            official, _ = make_official(Path(folder), events)
            official.FLAGS.train_steps = 1000000
            official.FLAGS.log_interval = 5000
            official.FLAGS.eval_interval = 100000
            official.FLAGS.save_interval = 100000
            with FormalInputBridge(official, Path('/accepted/data'), Path(folder) / 'bridge') as bridge:
                data = official.GCDataset(reference.fixture().dataset, official.FLAGS.agent)
                np.random.seed(77)
                expected = reference.OFFICIAL.GCDataset(data.dataset, data.config).sample(1)
                expected_rng = rng_bytes()
                np.random.seed(77)
                self.assertEqual(batch_bytes(expected), batch_bytes(data.sample(1)))
                self.assertEqual(expected_rng, rng_bytes())
                self.assertFalse(bridge.inputs.pending)
                self.assertEqual(bridge.step, 0)
                for step in (1, 5000, 100000, 1000000):
                    bridge.inputs.barrier()
                    bridge.step = step - 1
                    before = np.random.get_state()
                    expected = reference.OFFICIAL.GCDataset(data.dataset, data.config).sample(1024)
                    expected_rng = rng_bytes()
                    np.random.set_state(before)
                    self.assertEqual(batch_bytes(expected), batch_bytes(data.sample(1024)), step)
                    self.assertEqual(expected_rng, rng_bytes(), step)
                    self.assertFalse(bridge.inputs.pending, step)


if __name__ == '__main__':
    unittest.main()
