"""Focused CPU fakes: original sampler, real widths, terminal and online 2N calls."""
import ast
import dataclasses
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import types
from typing import Any
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
REFERENCE = Path(os.environ.get('P3_REFERENCE_ROOT', str(HERE / 'reference_source')))


class Mapping(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dict = self


def tree_map(function, value):
    return {key: tree_map(function, item) for key, item in value.items()} if isinstance(value, dict) else function(value)


raw = (REFERENCE / 'source/ogbench/impls/utils/datasets.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == '52c373922a87e1ba8ad8b3b96c3474d5e6385e2ea2c7206930f7a8a315816609'
nodes = [n for n in ast.parse(raw).body if
         isinstance(n, ast.FunctionDef) and n.name == 'get_size' or
         isinstance(n, ast.ClassDef) and n.name in ('Dataset', 'GCDataset')]
official = types.ModuleType('utils.datasets')
official.__dict__.update(dataclasses=dataclasses, Any=Any, np=np, FrozenDict=Mapping,
    jax=types.SimpleNamespace(tree_util=types.SimpleNamespace(tree_map=tree_map, tree_leaves=lambda x: list(x.values()))))
sys.modules.setdefault('utils', types.ModuleType('utils'))
sys.modules['utils.datasets'] = official
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'original_datasets', 'exec'), official.__dict__)
spec = importlib.util.spec_from_file_location('cube_inputs', REFERENCE / 'cube_inputs.py')
cube = importlib.util.module_from_spec(spec)
sys.modules['cube_inputs'] = cube
spec.loader.exec_module(cube)

import pldm_inputs
import native_sr


def config():
    return dict(frame_stack=None, p_aug=0., discount=.99, gc_negative=True,
        value_p_curgoal=.2, value_p_trajgoal=.5, value_p_randomgoal=.3, value_geom_sample=True,
        actor_p_curgoal=0., actor_p_trajgoal=1., actor_p_randomgoal=0., actor_geom_sample=False)


class ReferenceSampler(official.GCDataset):
    def __post_init__(self):
        super().__post_init__()
        self.ids = []

    def sample_goals(self, *args, **kwargs):
        ids = super().sample_goals(*args, **kwargs)
        self.ids.append(ids.copy())
        return ids


class FunctionalTests(unittest.TestCase):
    def test_original_sampler_rng_ids_terminal_and_real512_projection(self):
        module = pldm_inputs.load_inputs(REFERENCE / 'cube23_distance_gciql_v1/distance_inputs.py')
        rows = 2002
        states = np.arange(rows * 37, dtype=np.float32).reshape(rows, 37) / np.float32(10000)
        actions = np.zeros((rows, 5), np.float32)
        terminal = np.zeros(rows, np.float32)
        terminal[[1000, 2001]] = 1
        dataset = official.Dataset.create(observations=states, actions=actions,
            terminals=np.minimum(terminal + np.r_[terminal[1:], 0], 1), valids=1-terminal)
        features = np.arange(rows * 512, dtype=np.float32).reshape(rows, 512) / np.float32(10000)
        features.setflags(write=False)
        reference = ReferenceSampler(dataset, config())
        np.random.seed(0)
        original = reference.sample(2, idxs=np.array([999, 1001]))
        expected_rng = np.random.get_state()
        sampler = module.VisualInputs(dataset, config(), cubes=2, mode='rgb', features=features, workers=4)
        try:
            np.random.seed(0)
            batch = sampler.sample(2, idxs=np.array([999, 1001]), prefetch_next=True)
            actual_rng = np.random.get_state()
            self.assertEqual(expected_rng[0], actual_rng[0])
            np.testing.assert_array_equal(expected_rng[1], actual_rng[1])
            self.assertEqual(expected_rng[2:], actual_rng[2:])
            self.assertEqual(batch['observations'].shape, (2, 531))
            self.assertEqual(batch['actor_goals'].shape, (2, 518))
            np.testing.assert_array_equal(batch['next_observations'][:, :512], features[[1000, 1002]])
            np.testing.assert_array_equal(batch['observations'][:, 512:], states[[999, 1001], :19])
            np.testing.assert_array_equal(sampler.last_ids['value_goal'], reference.ids[0])
            np.testing.assert_array_equal(sampler.last_ids['actor_goal'], reference.ids[1])
            success = np.all(np.linalg.norm(module.object_xyz(original['observations'], 2)-module.object_xyz(original['value_goals'], 2), axis=-1) <= .4, axis=-1)
            np.testing.assert_array_equal(batch['rewards'], success.astype(original['rewards'].dtype)-1.)
            np.testing.assert_array_equal(batch['masks'], 1.-success.astype(original['masks'].dtype))
            with self.assertRaises(ValueError):
                sampler.sample(1, idxs=np.array([1000]))
            sampler.barrier()
            self.assertFalse(sampler.pending)
        finally:
            sampler.close()

    def test_native_goal_then_current_every_call_without_goal_feature_cache(self):
        source = REFERENCE / 'cube23_distance_gciql_v1/native_sr_visual_v1/visual_sr.py'
        group = dict(cube_count=2, state_dim=37, current_dim=531, goal_dim=518, max_episode_steps=500, mode='rgb')
        core = native_sr.install_core(source, group)
        class Env:
            def __init__(self):
                self.unwrapped = self
                self.np_random_seed = 42
                self.events = []
            def reset(self, *, seed, options):
                self.np_random_seed = seed
                assert options['render_goal'] is True
                return np.arange(37, dtype=np.float32), dict(goal_rendered=np.full((224,224,3), 51, np.uint8))
            def render(self, *, camera):
                self.events.append('render-current')
                return np.full((224,224,3), 93, np.uint8)
            def close(self): pass
        class Encoder:
            calls = 0
            def encode(self, image):
                self.calls += 1
                env.events.append('encode-goal' if image[0,0,0] == 51 else 'encode-current')
                return np.full(512, self.calls, np.float32)
        class Agent:
            def sample_actions(self, observations, goals, seed, temperature):
                self.current, self.goal = observations, goals
                return np.zeros(5, np.float32)
        env, encoder, agent = Env(), Encoder(), Agent()
        wrapped = core.OnlineVisualEnv(env, native_sr.GROUP, encoder, 42)
        observation, _ = wrapped.reset(options={'task_id': 1})
        self.assertEqual(encoder.calls, 0)
        policy = core.OnlineGCIQLPolicy(agent, wrapped)
        for _ in range(2):
            policy.sample_actions(observation, observation+1, seed=None, temperature=0.)
        self.assertEqual(env.events, ['encode-goal','render-current','encode-current']*2)
        self.assertEqual((encoder.calls, wrapped.goal_encoder_calls), (4, 2))
        self.assertEqual((agent.current.shape, agent.goal.shape), ((531,), (518,)))
        np.testing.assert_array_equal(agent.current[512:], observation[:19])
        np.testing.assert_array_equal(agent.goal[512:], core.goal_xyz(observation+1))
        with self.assertRaises(ValueError):
            core.validate_latent(np.zeros(192, np.float32), 'wrong-family')


if __name__ == '__main__':
    unittest.main()
