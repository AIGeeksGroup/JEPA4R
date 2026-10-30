"""Narrow CPU-only contract checks, without simulator/model/CUDA imports."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from common import episode_seed, object_sha, protocol_payload


class Contracts(unittest.TestCase):
    def test_ep_seed_is_private_stable_and_split_specific(self):
        before = np.random.get_state()
        first = episode_seed('train', 0)
        self.assertEqual(first, episode_seed('train', 0))
        self.assertTrue(0 <= first < 2**32)
        self.assertEqual(len({episode_seed(split, episode) for split in ('train','val')
                              for episode in (0,1,299)}), 6)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        all_seeds = [episode_seed(split, episode) for split, count in (('train',3000),('val',300))
                     for episode in range(count)]
        self.assertEqual(len(set(all_seeds)), 3300)

    def test_protocol_binds_pcd_crop_seed_and_unchanged_source_array_origin(self):
        plan = json.loads((Path(__file__).parent/'producer_plan.json').read_text())
        digest = object_sha(protocol_payload(plan))
        changed = copy.deepcopy(plan)
        changed['cache_dir'] = '/different/location'
        self.assertEqual(digest, object_sha(protocol_payload(changed)))
        changed['pcd']['crop_box'][1][2] = .6
        self.assertNotEqual(digest, object_sha(protocol_payload(changed)))
        changed = copy.deepcopy(plan)
        changed['source_arrays_receipt_sha256'] = 'changed'
        self.assertNotEqual(digest, object_sha(protocol_payload(changed)))
        np.testing.assert_array_equal(np.arange(1001)[-2:], [999,1000])
        self.assertEqual(np.arange(1001,2002)[0],1001)


if __name__ == '__main__':
    unittest.main()
