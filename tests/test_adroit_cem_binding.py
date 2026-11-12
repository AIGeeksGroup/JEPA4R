"""CEM source identity and fragment selection must fail before planning."""
from pathlib import Path
import importlib.util
import json
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cem_binding_entry', ROOT / 'lewm_pcd/scripts/checks/run_adroit_fragment_cem.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class FragmentSelectionTests(unittest.TestCase):
    def test_original_torch_dtype_cannot_be_hidden_by_numpy_cast(self):
        reference = [dict(role='encoder', torch_dtypes={'output':'torch.float32'}, shapes={'output':[2,4]})]
        self.assertTrue(entry.validate_model_trace_records(reference, reference))
        changed = [dict(reference[0], torch_dtypes={'output':'torch.bfloat16'})]
        with self.assertRaisesRegex(ValueError, 'torch_dtypes'):
            entry.validate_model_trace_records(changed, reference)

    def test_changed_episode_or_start_is_rejected(self):
        expected = {'episodes': [0], 'start_steps': [10], 'goal_offset': 25}
        self.assertEqual(entry.validate_fragment_selection([0], [10], 25, expected), expected)
        for episodes, steps, offset in [([1], [10], 25), ([0], [11], 25), ([0], [10], 24)]:
            with self.assertRaises(ValueError):
                entry.validate_fragment_selection(episodes, steps, offset, expected)

    def test_selection_requires_actual_integer_indices(self):
        for episodes, steps, offset in [([.9], [10], 25), ([0], [10.1], 25),
                                         ([False], [10], 25), ([0], [10], 25.1)]:
            with self.assertRaises(ValueError):
                entry.validate_fragment_selection(episodes, steps, offset)


class CEMTraceTests(unittest.TestCase):
    def test_original_solver_actions_and_rng_are_unchanged_by_callback(self):
        try:
            import torch
            from gymnasium.spaces import Box
            from stable_worldmodel.planning.solver.cem import CEMSolver
        except ImportError:
            self.skipTest('original solver neutrality check requires the pinned runtime')
        import numpy as np
        from types import SimpleNamespace
        class Cost:
            def parameters(self): return iter(())
            def get_cost(self, infos, candidates): return candidates.square().mean(dim=(2, 3))
        def solver(callbacks):
            result = CEMSolver(Cost(), num_samples=30, topk=5, n_steps=1,
                               device='cpu', seed=42, callbacks=callbacks)
            result.configure(action_space=Box(-1., 1., (1, 30), np.float32), n_envs=1,
                             config=SimpleNamespace(horizon=5, action_block=5))
            return result
        with tempfile.TemporaryDirectory() as directory:
            trace = entry.FullCEMTrace(Path(directory) / 'trace')
            original, observed = solver([]), solver([trace])
            info = {'observation': torch.zeros((1, 39))}
            first, second = original.solve(info), observed.solve(info)
            self.assertTrue(torch.equal(first['actions'], second['actions']))
            self.assertEqual(first['costs'], second['costs'])
            self.assertTrue(torch.equal(original.torch_gen.get_state(), observed.torch_gen.get_state()))
            self.assertEqual(len(trace.records), 1)

    def test_callback_retains_complete_arrays_without_changing_inputs_or_rng(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            trace = entry.FullCEMTrace(Path(directory) / 'trace')
            state = {'candidates': np.arange(60, dtype=np.float32).reshape(1, 3, 2, 10),
                     'costs': np.array([[3, 1, 2]], np.float32),
                     'topk_vals': np.array([[1, 2]], np.float32),
                     'topk_inds': np.array([[1, 2]], np.int64),
                     'topk_candidates': np.ones((1, 2, 2, 10), np.float32),
                     'mean': np.ones((1, 2, 10), np.float32),
                     'var': np.zeros((1, 2, 10), np.float32),
                     'prev_mean': np.zeros((1, 2, 10), np.float32),
                     'prev_var': np.ones((1, 2, 10), np.float32)}
            before = {k: v.copy() for k, v in state.items()}
            rng = np.random.get_state()
            trace.reset(); trace.start_batch(); trace(step=0, **state); trace.end_solve()
            after = np.random.get_state()
            np.testing.assert_array_equal(rng[1], after[1]); self.assertEqual(rng[2:], after[2:])
            for name, value in state.items():
                np.testing.assert_array_equal(value, before[name])
            path = next((Path(directory) / 'trace').glob('*.npz'))
            with np.load(path, allow_pickle=False) as arrays:
                self.assertEqual(set(arrays.files), set(state) | {'first_action_block'})
                for name, value in state.items(): np.testing.assert_array_equal(arrays[name], value)
                np.testing.assert_array_equal(arrays['first_action_block'], state['mean'][:, 0])

    def test_trace_never_overwrites_a_previous_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace'; entry.FullCEMTrace(path)
            with self.assertRaises(FileExistsError): entry.FullCEMTrace(path)

    def test_world_observer_records_real_steps_and_restores_methods(self):
        import numpy as np
        class World:
            def reset(self):
                self.infos = {'qpos': np.zeros((1, 1, 36)), 'qvel': np.zeros((1, 1, 36))}
            def _run(self, *, on_step=None):
                for step in range(2):
                    self.infos['qpos'] += 1
                    self.infos['executed_action'] = np.ones((1, 1, 30)) * step
                    if on_step: on_step(self, np.ones(1, dtype=bool))
                return 2
        world = World(); before_reset, before_run = world.reset, world._run
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            with entry.observe_rollout(world, Path(directory) / 'rollout') as trace:
                world.reset()
                # World injects dataset goals after reset and before its rollout loop.
                goal_keys = ('goal', 'goal_pcd_xyz', 'goal_pcd_rgb', 'goal_native_observation',
                             'goal_observation', 'goal_qpos', 'goal_qvel')
                world.infos.update({name: np.array([7.], np.float32) for name in goal_keys})
                self.assertEqual(world._run(on_step=lambda w, mask: calls.append(1)), 2)
            self.assertEqual(world.reset, before_reset); self.assertEqual(world._run, before_run)
            self.assertEqual(calls, [1, 1])
            self.assertEqual([r['kind'] for r in trace.records], ['initial', 'planner_initial', 'step', 'step'])
            with np.load(Path(directory) / 'rollout' / trace.records[1]['arrays_file']) as arrays:
                for name in goal_keys: np.testing.assert_array_equal(arrays[name], world.infos[name])
            with np.load(Path(directory) / 'rollout' / trace.records[-1]['arrays_file']) as arrays:
                np.testing.assert_array_equal(arrays['qpos'], np.full((1, 1, 36), 2))
                np.testing.assert_array_equal(arrays['executed_action'], np.ones((1, 1, 30)))


@unittest.skipIf(importlib.util.find_spec('torch') is None, 'model tracing requires the pinned Torch runtime')
class ModelTraceTests(unittest.TestCase):
    def model(self):
        import torch
        model = torch.nn.Module()
        model.encoder = torch.nn.Linear(3, 4)
        model.projector = torch.nn.Identity()
        model.action_encoder = torch.nn.Linear(2, 4)
        model.predictor = torch.nn.Linear(4, 4)
        model.pred_proj = torch.nn.Identity()
        return model.eval().requires_grad_(False)

    def test_full_model_hooks_preserve_outputs_rng_and_state(self):
        import torch
        torch.manual_seed(211)
        model = self.model()
        x, actions = torch.randn(2, 3), torch.randn(2, 2)
        expected = model.pred_proj(model.predictor(model.projector(model.encoder(x))+model.action_encoder(actions)))
        rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            with entry.observe_model(model, Path(directory)/'model') as trace:
                actual = model.pred_proj(model.predictor(model.projector(model.encoder(x))+model.action_encoder(actions)))
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(trace.summary['frozen_unchanged'])
            self.assertEqual(trace.summary['calls'], {role:1 for role in entry.ModelTrace.roles})
            for module in (getattr(model, role) for role in entry.ModelTrace.roles):
                self.assertFalse(module._forward_hooks)
            self.assertEqual(len(list((Path(directory)/'model').glob('*.npz'))), 5)

    def test_model_hooks_are_removed_after_original_error(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'original failure'):
                with entry.observe_model(model, Path(directory)/'model') as trace:
                    raise RuntimeError('original failure')
            self.assertFalse(trace.summary['completed'])
            for module in (getattr(model, role) for role in entry.ModelTrace.roles):
                self.assertFalse(module._forward_hooks)


class RawNativeStepTests(unittest.TestCase):
    def test_step_only_native_info_is_recorded_without_modifying_pool_buffers(self):
        import numpy as np
        try:
            import gymnasium as gym
            from stable_worldmodel.world.env_pool import EnvPool
        except ImportError:
            self.skipTest('actual EnvPool requires the pinned runtime')
        class Native(gym.Env):
            action_space = gym.spaces.Box(-1., 1., (30,), np.float32)
            observation_space = gym.spaces.Box(-np.inf, np.inf, (39,), np.float64)
            def reset(self, **kwargs):
                self.n = 0
                return np.zeros(39), dict(qpos=np.zeros(36), qvel=np.zeros(36))
            def step(self, action):
                self.n += 1
                return np.zeros(39), 0., False, False, dict(qpos=np.full(36,self.n), qvel=np.zeros(36),
                                                           executed_action=np.clip(action,-1.,1.))
        class World:
            def __init__(self): self.envs = EnvPool([Native])
            def reset(self): _,self.infos = self.envs.reset()
            def _run(self, *, on_step):
                for i in range(2):
                    _,self.rewards,self.terminateds,self.truncateds,self.infos = self.envs.step(np.full((1,30),2.5))
                    on_step(self,np.ones(1,dtype=bool))
        world = World(); base=world.envs.envs[0]; original_step=base.step
        with tempfile.TemporaryDirectory() as directory:
            with entry.observe_rollout(world, Path(directory)/'trace') as trace:
                world.reset();world._run()
            self.assertNotIn('executed_action',world.infos)
            self.assertEqual(base.step,original_step)
            self.assertEqual([r['kind'] for r in trace.records],['initial','planner_initial','step','step'])
            with np.load(Path(directory)/'trace'/trace.records[-1]['arrays_file']) as arrays:
                np.testing.assert_array_equal(arrays['executed_action'],np.ones((1,1,30)))
                np.testing.assert_array_equal(arrays['qpos'],np.full((1,1,36),2))


try:
    import h5py
    import numpy as np
except ImportError:
    h5py = None


@unittest.skipIf(h5py is None, 'real HDF binding checks require the pinned HDF environment')
class HDFInputBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.backing = self.root / 'backing.h5'
        with h5py.File(self.backing, 'w') as handle:
            handle['points'] = np.arange(12, dtype=np.float32).reshape(2, 6)
        self.test = self.root / 'test.h5'
        with h5py.File(self.test, 'w') as handle:
            handle['points'] = h5py.ExternalLink('backing.h5', 'points')
        self.train = self.root / 'train.h5'
        with h5py.File(self.train, 'w') as handle:
            handle['action'] = np.zeros((2, 30), dtype=np.float32)
        self.checkpoint = self.root / 'model.pt'
        self.checkpoint.write_bytes(b'fixed model identity')
        self.initials = self.root / 'initials.npz'
        np.savez(self.initials, seed=np.array([42]))

    def binding(self, directory=None):
        directory = self.root if directory is None else directory
        return entry.build_input_binding(directory / 'train.h5', directory / 'test.h5',
                                         directory / 'model.pt', directory / 'initials.npz', 'pcd', 'smoke')

    def test_external_backing_change_rejected_when_header_unchanged(self):
        expected = self.binding()
        header = self.test.read_bytes()
        with h5py.File(self.backing, 'r+') as handle:
            handle['points'][0, 0] = 99
        self.assertEqual(self.test.read_bytes(), header)
        with self.assertRaisesRegex(ValueError, 'binding'):
            entry.validate_input_binding(self.binding(), expected)

    def test_same_content_directory_move_is_accepted(self):
        expected = self.binding()
        moved = self.root / 'moved'
        moved.mkdir()
        for name in ['train.h5', 'test.h5', 'backing.h5', 'model.pt', 'initials.npz']:
            shutil.copyfile(self.root / name, moved / name)
        self.assertTrue(entry.validate_input_binding(self.binding(moved), expected))

    def test_same_shape_other_model_is_rejected(self):
        expected = self.binding()
        self.checkpoint.write_bytes(b'other model identity')
        with self.assertRaisesRegex(ValueError, 'binding'):
            entry.validate_input_binding(self.binding(), expected)

    def test_virtual_backing_change_is_bound(self):
        layout = h5py.VirtualLayout(shape=(2, 6), dtype=np.float32)
        layout[:] = h5py.VirtualSource('backing.h5', 'points', shape=(2, 6))
        with h5py.File(self.test, 'w', libver='latest') as handle:
            handle.create_virtual_dataset('points', layout)
        expected = self.binding()
        header = self.test.read_bytes()
        with h5py.File(self.backing, 'r+') as handle:
            handle['points'][1, 0] = 77
        self.assertEqual(header, self.test.read_bytes())
        with self.assertRaises(ValueError):
            entry.validate_input_binding(self.binding(), expected)


if __name__ == '__main__':
    unittest.main()
