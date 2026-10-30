"""Small artifact fixtures only; no policy, JAX, or simulation."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import audit_state_xyz_sr as auditor


class ArtifactBoundaries(unittest.TestCase):
    def fixture(self, folder, cubes=1, steps=200, success=0):
        config = dict(group_id={1: 'G01', 2: 'G04', 3: 'G07'}[cubes], cube_count=cubes,
                      max_episode_steps={1: 200, 2: 500, 3: 1000}[cubes])
        full = np.arange(19 + 9 * cubes, dtype=np.float64)
        indices = {1: [19, 20, 21], 2: [19, 20, 21, 28, 29, 30],
                   3: [19, 20, 21, 28, 29, 30, 37, 38, 39]}[cubes]
        goal = full[indices]
        current_sha = 'a' * 64
        reset = dict(seed=970100000, actual_env_seed=970100000,
                     options=dict(task_id=1, render_goal=False), task_id=1, episode=0, cube_count=cubes,
                     full_goal=full.tolist(), full_goal_dtype='float64', full_goal_sha256=auditor.array_sha(full),
                     goal_XYZ=goal.tolist(), goal_XYZ_shape=[3 * cubes], goal_XYZ_sha256=auditor.array_sha(goal),
                     goal_XYZ_offsets=[19, 28, 37][:cubes], current_shape=[19 + 9 * cubes],
                     current_sha256=current_sha, goal_XYZ_is_official_scaled10=True, extra_policy_warmup=0,
                     action_space_seed_explicitly_changed=False, legacy_visual_scene_exact_pairing_claimed=False)
        inputs = dict(current_dim=19 + 9 * cubes, goal_dim=3 * cubes, all_inputs_finite=True,
                      synthetic_zero_padding=False, encoder_calls=0, perstep_raw_observations_persisted=False,
                      goal_XYZ_sha256=auditor.array_sha(goal), goal_dtype='float64', policy_calls=steps,
                      observations_sha256=[current_sha] * steps)
        (folder / 'reset.json').write_text(json.dumps(reset))
        (folder / 'policy_inputs.json').write_text(json.dumps(inputs))
        dones = np.zeros(steps, dtype=bool); dones[-1] = True
        successes = np.zeros(steps, dtype=bool); successes[-1] = bool(success)
        np.savez(folder / 'steps.npz', actions=np.zeros((steps, 5), np.float32),
                 rewards=successes.astype(np.float32), dones=dones, successes=successes,
                 actor_keys=np.zeros((steps, 2), np.uint32))
        result = dict(actual_exit_code=0, group_id=config['group_id'], task_id=1, episode=0, seed=970100000,
                      diagnostic=False, environment_closed=True, extra_policy_warmup=0,
                      original_evaluate_reused=True, steps=steps, success=success, policy_calls=steps)
        self.seal(folder, result)
        (folder / 'attempt_exit_code').write_text('0\n')
        return config

    def seal(self, folder, result):
        result['files'] = {n: dict(bytes=(folder / n).stat().st_size, sha256=auditor.sha(folder / n))
                           for n in ('reset.json', 'policy_inputs.json', 'steps.npz')}
        (folder / 'result.json').write_text(json.dumps(result))

    def test_actual_widths_and_normal_time_limits(self):
        for cubes, limit in [(1, 200), (2, 500), (3, 1000)]:
            with tempfile.TemporaryDirectory() as name:
                folder = Path(name)
                config = self.fixture(folder, cubes, limit)
                self.assertEqual(auditor.episode(folder, config, 1, 0, False)['steps'], limit)

    def test_early_native_success_is_valid(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name); config = self.fixture(folder, steps=2, success=1)
            self.assertEqual(auditor.episode(folder, config, 1, 0, False)['success'], 1)

    def test_arm_fields_cannot_replace_object_goal_even_with_updated_file_SHA(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name); config = self.fixture(folder, cubes=2, steps=500)
            reset = auditor.read(folder / 'reset.json'); reset['goal_XYZ'] = [0, 1, 2, 3, 4, 5]
            (folder / 'reset.json').write_text(json.dumps(reset)); self.seal(folder, auditor.read(folder / 'result.json'))
            with self.assertRaises(AssertionError): auditor.episode(folder, config, 1, 0, False)

    def test_early_failure_cannot_masquerade_as_full_episode(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name); config = self.fixture(folder, steps=4, success=0)
            with self.assertRaises(AssertionError): auditor.episode(folder, config, 1, 0, False)


if __name__ == '__main__':
    unittest.main()
