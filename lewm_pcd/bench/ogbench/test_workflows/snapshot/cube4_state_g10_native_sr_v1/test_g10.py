"""Two narrow CPU tests; no model, JAX, simulator or SR calls."""
import unittest
import numpy as np
from bridge import evaluator, auditor, serial


class G10Tests(unittest.TestCase):
    def test_N4_projection_and_original_policy_passthrough(self):
        module = evaluator()
        class Agent:
            def sample_actions(self, **kwargs):
                self.received = kwargs
                return 'original'
        observation, goal = np.arange(55, dtype=np.float64), np.arange(55, dtype=np.float64) + 100
        expected = goal[[19, 20, 21, 28, 29, 30, 37, 38, 39, 46, 47, 48]]
        agent = Agent(); policy = module.GoalXYZPolicy(agent, 4)
        key = np.array([12, 34], np.uint32)
        before = np.random.get_state()
        self.assertEqual(policy.sample_actions(observation, goal, seed=key, temperature=0), 'original')
        self.assertIs(agent.received['observations'], observation)
        self.assertIs(agent.received['seed'], key)
        np.testing.assert_array_equal(agent.received['goals'], expected)
        self.assertEqual(agent.received['temperature'], 0)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0]);np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        for module in (module, auditor(), serial()):
            self.assertTrue(module.parameter_proof['inverse_exact'])

    def test_N4_reset_offsets_and_wrong_shape_rejected(self):
        module = evaluator()
        class Env:
            def __init__(self):self.unwrapped = self;self.calls = 0
            def reset(self, **kwargs):
                self.calls += 1;self.kwargs = kwargs;self.np_random_seed = kwargs['seed']
                return np.arange(55), {'goal': np.arange(55, dtype=np.float64)}
        env = Env(); wrapped = module.SeedReset(env, 970500049, 4)
        wrapped.reset(options=dict(task_id=5, render_goal=False))
        self.assertEqual(env.calls, 1)
        self.assertEqual(env.kwargs, dict(seed=970500049, options=dict(task_id=5, render_goal=False)))
        self.assertEqual(wrapped.reset_record['goal_XYZ_offsets'], [19, 28, 37, 46])
        self.assertEqual(wrapped.reset_record['goal_XYZ_shape'], [12])
        with self.assertRaises(AssertionError):module.project_goal(np.arange(46), 4)
        with self.assertRaises(AssertionError):wrapped.reset(options=dict(task_id=5))


if __name__ == '__main__':unittest.main()
