# Moved from OGBench-cube@4fbb0c6:lewm_pcd/scripts/checks/test_official_rgb_preservation.py (verbatim except: module
# names / paths of the moved code adapted; lines wrapped at 120 columns)
"""D5 RGB conversion keeps original1536D cache values; only images are replaced."""
import importlib.util
from pathlib import Path
import pickle
import unittest
import numpy as np

FILE=Path(__file__).resolve().parents[2]/'jepa4r/data/relocate/adroit_data.py'
class RGBPreservationTests(unittest.TestCase):
    def test_original_observations1536_shape_dtype_and_bytes_are_preserved(self):
        spec = importlib.util.spec_from_file_location('_rgb_preservation', FILE)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        states = np.arange(200 * 39, dtype=np.float32).reshape(200, 39) / 1000
        rewards = np.arange(200, dtype=np.float64) / 100
        flags = np.arange(200) % 7 == 0
        original = np.arange(200 * 1536, dtype=np.float64).reshape(200, 1536) / 17
        demo = {
            'actions': np.linspace(-2, 2, 200 * 30, dtype=np.float32).reshape(200, 30),
            'state_obs': states,
            'observations': original,
            'proprio': states[:, :30].copy(),
            'rewards': rewards,
            'init_state_dict': {'qpos': np.array([0.0])},
            'env_infos': {'goal_achieved': flags},
            'custom_field': {'source': 'retained original'},
        }
        before = pickle.dumps(demo, protocol=4)

        class Env:
            def set_env_state(self, value):
                self.t = 0

            def get_obs(self):
                return states[self.t].copy()

            def get_env_state(self):
                return {'qpos': np.array([float(self.t)])}

            def step(self, action):
                i = self.t
                np.testing.assert_array_equal(action, demo['actions'][i])
                self.t += 1
                return np.zeros(39), rewards[i], False, {'goal_achieved': flags[i]}

        frame = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (256, 1, 3))
        result = m.rgb_demo(demo, Env(), lambda: frame.copy())
        self.assertEqual(result['observations'].shape, (200, 1536))
        self.assertEqual(result['observations'].dtype, original.dtype)
        self.assertEqual(result['observations'].tobytes(), original.tobytes())
        self.assertEqual(result['images'].shape, (200, 256, 256, 3))
        self.assertEqual(result['images'].dtype, np.dtype('uint8'))
        self.assertEqual(pickle.dumps(demo, protocol=4), before)
        for key in demo:
            self.assertEqual(pickle.dumps(result[key], protocol=4), pickle.dumps(demo[key], protocol=4), key)
if __name__=='__main__':unittest.main()
