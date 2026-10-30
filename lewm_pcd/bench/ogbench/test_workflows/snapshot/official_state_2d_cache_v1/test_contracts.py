"""Narrow CPU contracts only; no simulator, model, or CUDA execution."""
import copy
import unittest

import numpy as np

from produce import canonical_ids, protocol_digest


class ProducerContracts(unittest.TestCase):
    def test_raw_episode_and_terminal_identity_includes_real_last_row(self):
        first, second = canonical_ids(0), canonical_ids(1)
        np.testing.assert_array_equal(first, np.arange(1001, dtype=np.int64))
        self.assertEqual(first[999], 999)
        self.assertEqual(first[1000], 1000)
        self.assertEqual(second[0], 1001)
        self.assertEqual(second[-1], 2001)

    def test_protocol_ignores_location_and_run_limit_but_binds_camera_and_model(self):
        plan = dict(checkpoint_sha256='model', data_acceptance_sha256='data',
            source_stat_freeze_sha256='stat', source_pins={'source.py': 'source'},
            camera={'local_x_up_degrees': 5, 'offsamples': 4}, renderer_workers=2,
            encoder_batch=32, torch_cpu_threads=1, split_episodes={'train': 3000, 'val': 300},
            raw_rows_per_episode=1001, cache_dir='pilot', shared_state_root='shared')
        expected = protocol_digest(plan, 'producer', 'renderer')
        changed = copy.deepcopy(plan)
        changed.update(cache_dir='full', shared_state_root='another', pilot=True, limit_episodes=1)
        self.assertEqual(expected, protocol_digest(changed, 'producer', 'renderer'))
        changed['camera']['local_x_up_degrees'] = 10
        self.assertNotEqual(expected, protocol_digest(changed, 'producer', 'renderer'))
        self.assertNotEqual(expected, protocol_digest(plan, 'new_producer', 'renderer'))
        changed = dict(plan, pixel_preprocess_device='auto_validated')
        self.assertNotEqual(expected, protocol_digest(changed, 'producer', 'renderer'))


if __name__ == '__main__':
    unittest.main()
