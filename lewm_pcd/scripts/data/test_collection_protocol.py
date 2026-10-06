import unittest
import collection_protocol as protocol
import json
from pathlib import Path
import tempfile

import numpy as np

from collection_protocol import (
    STRATEGY_EXPERT_HIGH_NOISE,
    STRATEGY_EXPERT_LOW_NOISE,
    STRATEGY_EXPERT_RANDOM,
    STRATEGY_EXPERT_RANDOM_BRANCH100,
    STRATEGY_EXPERT_RANDOM_BRANCH45,
    STRATEGY_RANDOM,
    FixedTaskPlanPolicyMixin,
    action_source,
    reset_options_for_task,
    strategy_counts,
    validate_fixed_paired_schema,
)


class CollectionProtocolTests(unittest.TestCase):
    def _paired_episode(self):
        return {
            'step_idx': np.arange(3),
            'action': np.r_[np.zeros((2, 2)), np.full((1, 2), np.nan)],
            'qpos': np.zeros((3, 2)),
            'qvel': np.zeros((3, 2)),
            'collection_reset_seed': np.full(3, 23),
            'privileged/target_block_pos': np.tile([0.5, 0.2, 0.02], (3, 1)),
            'privileged/target_block_yaw': np.zeros((3, 1)),
            'terminated': np.zeros(3, dtype=bool),
            'truncated': np.array([False, False, True]),
            'success': np.zeros(3, dtype=bool),
            'pixels': np.zeros((3, 2, 3, 3), dtype=np.uint8),
            'pcd_xyz': np.zeros((3, 4, 3), dtype=np.float32),
            'pcd_rgb': np.zeros((3, 4, 3), dtype=np.uint8),
        }

    def _validate_paired(self, episode):
        return validate_fixed_paired_schema(
            episode,
            expected_action_dim=2,
            expected_rgb_shape=(2, 3, 3),
            expected_pcd_shape=(4, 3),
        )

    def test_fixed_schema_rejects_missing_modality(self):
        episode = self._paired_episode()
        del episode['pixels']
        with self.assertRaisesRegex(ValueError, 'pixels'):
            self._validate_paired(episode)

    def test_fixed_schema_rejects_invalid_modality_dimensions(self):
        episode = self._paired_episode()
        episode['pcd_xyz'] = np.zeros((3, 4, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'pcd_xyz.*shape'):
            self._validate_paired(episode)

    def test_fixed_schema_rejects_invalid_action_dimension(self):
        episode = self._paired_episode()
        episode['action'] = np.r_[np.zeros((2, 3)), np.full((1, 3), np.nan)]
        with self.assertRaisesRegex(ValueError, 'action shape'):
            self._validate_paired(episode)

    def test_fixed_schema_rejects_invalid_modality_dtypes(self):
        episode = self._paired_episode()
        episode['pcd_rgb'] = np.zeros((3, 4, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'pcd_rgb.*uint8'):
            self._validate_paired(episode)

    def test_mixed_strategy_switches_to_random_at_step_100(self):
        self.assertEqual(action_source('expert_random', 99), 'expert')
        self.assertEqual(action_source('expert_random', 100), 'random')

    def test_branch100_alias_switches_at_step_100(self):
        self.assertEqual(action_source(STRATEGY_EXPERT_RANDOM_BRANCH100, 99), 'expert')
        self.assertEqual(action_source(STRATEGY_EXPERT_RANDOM_BRANCH100, 100), 'random')

    def test_branch45_alias_switches_at_step_45_and_keeps_default_noise(self):
        self.assertEqual(action_source(STRATEGY_EXPERT_RANDOM_BRANCH45, 44), 'expert')
        self.assertEqual(action_source(STRATEGY_EXPERT_RANDOM_BRANCH45, 45), 'random')
        self.assertEqual(protocol.STRATEGY_NOISE[STRATEGY_EXPERT_RANDOM_BRANCH45], 0.05)

    def test_strategy_counts_match_requested_training_and_validation_quotas(self):
        self.assertEqual(
            strategy_counts(2000),
            {
                STRATEGY_EXPERT_LOW_NOISE: 400,
                STRATEGY_EXPERT_HIGH_NOISE: 800,
                STRATEGY_EXPERT_RANDOM: 400,
                STRATEGY_RANDOM: 400,
            },
        )
        self.assertEqual(
            strategy_counts(200),
            {
                STRATEGY_EXPERT_LOW_NOISE: 40,
                STRATEGY_EXPERT_HIGH_NOISE: 80,
                STRATEGY_EXPERT_RANDOM: 40,
                STRATEGY_RANDOM: 40,
            },
        )
        self.assertNotIn(STRATEGY_EXPERT_RANDOM_BRANCH100, strategy_counts(200))
        self.assertNotIn(STRATEGY_EXPERT_RANDOM_BRANCH45, strategy_counts(200))

    def test_fixed_task_policy_keeps_selecting_the_same_oracle_after_done(self):
        class ActionSpace:
            shape = (5,)
            low = -np.ones(5)
            high = np.ones(5)

            def __init__(self):
                self.sampled = 0

            def seed(self, seed):
                pass

            def sample(self):
                self.sampled += 1
                return np.full(5, 0.75, dtype=np.float32)

        class Env:
            def __init__(self):
                self.action_space = ActionSpace()
                self.retarget_calls = 0

            def set_new_target(self, **kwargs):
                self.retarget_calls += 1

        class Agent:
            done = True

            def __init__(self):
                self.reset_calls = 0
                self.select_calls = 0

            def reset(self, ob, info):
                self.reset_calls += 1

            def select_action(self, ob, info):
                self.select_calls += 1
                return np.array([0.25, -0.25, 0.5, -0.5, 0.0])

        class EnvPool:
            num_envs = 1
            action_space = type('BatchedSpace', (), {'shape': (1, 5)})()

            def __init__(self, env):
                self.envs = [env]

        class Policy(FixedTaskPlanPolicyMixin):
            pass

        env = Env()
        agent = Agent()
        policy = Policy.__new__(Policy)
        policy.strategy = STRATEGY_EXPERT_LOW_NOISE
        policy.env = EnvPool(env)
        policy._agents = [agent]
        policy._oracle_agents = {'cube': [agent]}
        policy.rng = np.random.default_rng(5)

        info = {
            'step_idx': np.array([[0]]),
            'privileged/target_task': [['cube']],
        }
        first = policy.get_action(info)
        info['step_idx'] = np.array([[1]])
        second = policy.get_action(info)

        np.testing.assert_allclose(first[0], second[0])
        self.assertEqual(agent.reset_calls, 1)
        self.assertEqual(agent.select_calls, 2)
        self.assertEqual(env.retarget_calls, 0)

    def test_mixed_policy_uses_random_actions_starting_at_step_100(self):
        class ActionSpace:
            shape = (5,)
            low = -np.ones(5)
            high = np.ones(5)

            def __init__(self):
                self.sampled = 0

            def seed(self, seed):
                pass

            def sample(self):
                self.sampled += 1
                return np.full(5, 0.75, dtype=np.float32)

        class Env:
            def __init__(self):
                self.action_space = ActionSpace()

        class Agent:
            def __init__(self):
                self.select_calls = 0

            def select_action(self, ob, info):
                self.select_calls += 1
                return np.full(5, -0.25, dtype=np.float32)

        class EnvPool:
            num_envs = 1
            action_space = type('BatchedSpace', (), {'shape': (1, 5)})()

            def __init__(self, env):
                self.envs = [env]

        class Policy(FixedTaskPlanPolicyMixin):
            pass

        env = Env()
        agent = Agent()
        policy = Policy.__new__(Policy)
        policy.strategy = STRATEGY_EXPERT_RANDOM
        policy.env = EnvPool(env)
        policy._agents = [agent]
        policy._oracle_agents = {'cube': [agent]}
        policy.rng = np.random.default_rng(9)

        info = {
            'step_idx': np.array([[99]]),
            'privileged/target_task': [['cube']],
        }
        expert_action = policy.get_action(info)
        info['step_idx'] = np.array([[100]])
        random_action = policy.get_action(info)

        np.testing.assert_allclose(expert_action[0], np.full(5, -0.25))
        np.testing.assert_allclose(
            random_action[0], np.random.default_rng(9).uniform(-1.0, 1.0, 5),
            rtol=1e-6,
        )
        self.assertEqual(agent.select_calls, 1)
        self.assertEqual(env.action_space.sampled, 0)

    def test_random_policy_is_reproducible_with_fresh_action_spaces(self):
        class ActionSpace:
            shape = (5,)
            low = np.array([-0.5, -1.0, -0.25, -1.0, -0.75])
            high = np.array([0.5, 1.0, 0.25, 1.0, 0.75])

            def seed(self, seed):
                self.rng = np.random.default_rng(seed)

            def sample(self):
                return np.random.default_rng().uniform(self.low, self.high)

        class Env:
            @property
            def action_space(self):
                return ActionSpace()

        class Agent:
            def reset(self, ob, info):
                self.reset_random = np.random.uniform(size=8)

            def select_action(self, ob, info):
                raise AssertionError('random strategy must not select oracle actions')

        class Policy(FixedTaskPlanPolicyMixin):
            pass

        def rollout(seed):
            policy = Policy.__new__(Policy)
            policy.strategy = STRATEGY_RANDOM
            policy.env = type('Pool', (), {
                'envs': [Env()], 'num_envs': 1,
                'action_space': type('Space', (), {'shape': (1, 5)})(),
            })()
            policy._agents = [None]
            policy._oracle_agents = {'cube': [Agent()]}
            policy.rng = np.random.default_rng(seed)
            actions = []
            for step in range(200):
                actions.append(policy.get_action({
                    'step_idx': np.array([[step]]),
                    'privileged/target_task': [['cube']],
                })[0])
            return np.asarray(actions), policy._agents[0].reset_random

        actions, oracle_random = rollout(3072)
        repeat, repeat_oracle_random = rollout(3072)
        different, _ = rollout(3073)
        np.testing.assert_array_equal(actions, repeat)
        np.testing.assert_array_equal(oracle_random, repeat_oracle_random)
        self.assertFalse(np.array_equal(actions, different))
        self.assertTrue(np.all(actions >= ActionSpace.low))
        self.assertTrue(np.all(actions <= ActionSpace.high))

    def test_task_reset_uses_task_positions_random_start_yaw_and_upright_goal(self):
        options = reset_options_for_task(
            {
                'init_xyzs': np.array([[0.425, 0.1, 0.02]]),
                'goal_xyzs': np.array([[0.425, -0.1, 0.02]]),
            }
        )

        self.assertEqual(options['variation'], ['cube.start_yaw'])
        values = options['variation_values']
        np.testing.assert_allclose(values['cube.start_position'], [[0.425, 0.1]])
        np.testing.assert_allclose(values['cube.goal_position'], [[0.425, -0.1]])
        np.testing.assert_allclose(values['cube.goal_yaw'], [0.0])

    def test_seeded_reset_reproduces_full_initial_state_and_forwards_seed(self):
        self.assertTrue(hasattr(protocol, 'seeded_fixed_task_reset'))

        class Env:
            variation_space = object()

        def variations(space, *, seed, options, default_variations):
            env.yaw = np.random.default_rng(seed).uniform(0, 2 * np.pi)

        def parent_reset(obj, *, seed, options):
            obj.actual_seed = seed
            obj.qpos = np.r_[np.random.default_rng(seed).normal(size=6), obj.yaw]
            obj.qvel = np.zeros(7)
            return obj.qpos.copy(), {'qpos': obj.qpos.copy(), 'qvel': obj.qvel.copy()}

        env = Env()
        kwargs = dict(
            options={'variation': ['cube.start_yaw']},
            reset_variation_space=variations,
            base_reset=parent_reset,
            default_variations=('cube.start_yaw',),
        )
        _, first = protocol.seeded_fixed_task_reset(env, seed=12, **kwargs)
        _, repeat = protocol.seeded_fixed_task_reset(env, seed=12, **kwargs)
        _, different = protocol.seeded_fixed_task_reset(env, seed=13, **kwargs)
        np.testing.assert_array_equal(first['qpos'], repeat['qpos'])
        np.testing.assert_array_equal(first['qvel'], repeat['qvel'])
        self.assertFalse(np.array_equal(first['qpos'], different['qpos']))
        self.assertEqual(first['collection_reset_seed'], 12)
        self.assertEqual(env.actual_seed, 13)

    def test_episode_record_rejects_short_or_misaligned_paired_columns(self):
        self.assertTrue(hasattr(protocol, 'episode_record'))
        episode = {
            'step_idx': np.arange(201),
            'action': np.r_[np.zeros((200, 5)), np.full((1, 5), np.nan)],
            'qpos': np.zeros((201, 13)),
            'qvel': np.zeros((201, 12)),
            'collection_reset_seed': np.full(201, 23),
            'privileged/target_block_pos': np.tile([0.50, 0.20, 0.02], (201, 1)),
            'privileged/target_block_yaw': np.zeros((201, 1)),
            'truncated': np.r_[np.zeros(200, dtype=bool), True],
            'terminated': np.zeros(201, dtype=bool),
            'success': np.zeros(201, dtype=bool),
        }
        record = protocol.episode_record(episode, expected_actions=200)
        self.assertEqual(record['actions'], 200)
        self.assertEqual(record['frames'], 201)
        self.assertEqual(record['reset_seed'], 23)
        self.assertEqual(record['termination_reason'], 'horizon')
        self.assertEqual(len(record['initial_state_sha256']), 64)
        with self.assertRaisesRegex(ValueError, 'length'):
            protocol.episode_record(
                {**episode, 'qvel': np.zeros((200, 12))}, expected_actions=200,
            )
        with self.assertRaisesRegex(ValueError, 'horizon'):
            protocol.episode_record(
                {key: value[:101] for key, value in episode.items()}, expected_actions=200,
            )
        target = episode['privileged/target_block_pos'].copy()
        target[150] += 0.1
        with self.assertRaisesRegex(ValueError, 'target'):
            protocol.episode_record(
                {**episode, 'privileged/target_block_pos': target}, expected_actions=200,
            )

    def test_manifest_preserves_failure_counts_and_rejects_existing_output(self):
        self.assertTrue(hasattr(protocol, 'CollectionManifest'))
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('pix.h5.manifest.json', 'pcd.h5.manifest.json')]
            manifest = protocol.CollectionManifest(paths, {'strategy': 'random', 'actual_noise': 0.0})
            manifest.update('failed', exit_code=1, error='pcd write failed', pix_written=1, pcd_written=0)
            for path in paths:
                state = json.loads(path.read_text())
                self.assertEqual(state['stage'], 'failed')
                self.assertEqual(state['exit_code'], 1)
                self.assertEqual(state['pix_episodes_written'], 1)
                self.assertEqual(state['pcd_episodes_written'], 0)
                self.assertEqual(state['completed'], 0)
            before = paths[0].read_bytes()
            with self.assertRaises(FileExistsError):
                protocol.CollectionManifest(paths, {'strategy': 'random'})
            self.assertEqual(paths[0].read_bytes(), before)

    def test_fixed_task_metadata_initializes_definitions_without_changing_mode(self):
        self.assertTrue(hasattr(protocol, 'fixed_task_info'))

        class Env:
            _mode = 'data_collection'
            calls = 0

            def set_tasks(self):
                self.calls += 1
                self.task_infos = [{'task_name': f'task{i}'} for i in range(1, 6)]

        env = Env()
        self.assertEqual(protocol.fixed_task_info(env, 5)['task_name'], 'task5')
        self.assertEqual(protocol.fixed_task_info(env, 1)['task_name'], 'task1')
        self.assertEqual(env.calls, 1)
        self.assertEqual(env._mode, 'data_collection')

    def test_fixed_goal_accepts_float32_rounding_but_rejects_changed_target(self):
        self.assertTrue(hasattr(protocol, 'goal_matches'))
        goal = np.array([0.425, -0.1, 0.02], dtype=np.float64)
        self.assertTrue(protocol.goal_matches(goal.astype(np.float32), goal))
        self.assertFalse(protocol.goal_matches(goal + 1e-4, goal))


if __name__ == '__main__':
    unittest.main()
