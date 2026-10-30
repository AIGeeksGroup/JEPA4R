"""Two narrow CPU adapter tests; no simulator, JAX model or evaluator run."""
import unittest
import numpy as np
import evaluate_state_xyz as adapter


class AdapterTests(unittest.TestCase):
    def test_actual_width_goal_projection_and_policy_passthrough(self):
        class Agent:
            def sample_actions(self,**kwargs):
                self.received=kwargs
                return 'original-policy-result'
        for cubes in (1,2,3):
            state=np.arange(19+9*cubes,dtype=np.float64)
            goal=state+100.
            expected=np.concatenate([goal[19+9*i:22+9*i] for i in range(cubes)])
            np.testing.assert_array_equal(adapter.project_goal(goal,cubes),expected)
            original=Agent();policy=adapter.GoalXYZPolicy(original,cubes)
            key=np.array([12,34],np.uint32)
            self.assertEqual(policy.sample_actions(state,goal,seed=key,temperature=0),'original-policy-result')
            self.assertIs(original.received['observations'],state)
            self.assertIs(original.received['seed'],key)
            self.assertEqual(original.received['temperature'],0)
            np.testing.assert_array_equal(original.received['goals'],expected)
            self.assertEqual(original.received['goals'].shape,(3*cubes,))
            with self.assertRaises(AssertionError):adapter.project_goal(np.arange(19+9*cubes+192),cubes)

    def test_reset_forwards_only_explicit_seed_and_records_real_goal(self):
        class Env:
            unwrapped=None
            def __init__(self,cubes):
                self.unwrapped=self;self.cubes=cubes;self.calls=0
            def reset(self,*args,**kwargs):
                self.calls+=1;self.arguments=kwargs;self.np_random_seed=kwargs['seed']
                return np.arange(19+9*self.cubes),{'goal':np.arange(19+9*self.cubes,dtype=np.float64)}
        env=Env(2);wrapped=adapter.SeedReset(env,970400023,2)
        observation,info=wrapped.reset(options=dict(task_id=4,render_goal=False))
        self.assertEqual(env.calls,1)
        self.assertEqual(env.arguments,dict(seed=970400023,options=dict(task_id=4,render_goal=False)))
        self.assertEqual(wrapped.reset_record['actual_env_seed'],970400023)
        self.assertEqual(wrapped.reset_record['goal_XYZ_offsets'],[19,28])
        np.testing.assert_array_equal(wrapped.reset_record['goal_XYZ'],[19,20,21,28,29,30])
        self.assertFalse(wrapped.reset_record['legacy_visual_scene_exact_pairing_claimed'])
        with self.assertRaises(AssertionError):wrapped.reset(options=dict(task_id=4))


if __name__=='__main__':
    unittest.main()
