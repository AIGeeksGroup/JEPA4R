# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tests/test_collection_hdf_resume.py (verbatim)
"""Real HDF/VDS recovery mechanics on synthetic rows; no simulator or policy."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

try:
    from . import test_collection_resume as fixture
except ImportError:
    import test_collection_resume as fixture
collection = fixture.collection


@unittest.skipUnless(importlib.util.find_spec('h5py'), 'requires the existing HDF environment')
class HdfPairInitialStateTests(unittest.TestCase):
    def pair(self, *, distinct_initials):
        import h5py
        import numpy as np
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        paths = [Path(temporary.name) / name for name in ('rgb.h5', 'pcd.h5')]
        for index, path in enumerate(paths):
            with h5py.File(path, 'x') as file:
                file['ep_len'] = np.array([3, 3], np.int32)
                file['ep_offset'] = np.array([0, 3], np.int64)
                action = np.zeros((6, 5), np.float32)
                action[[2, 5]] = np.nan
                file['action'] = action
                for key, width in (('observation', 28), ('qpos', 21), ('qvel', 20)):
                    values = np.zeros((6, width), np.float32)
                    if distinct_initials:
                        values[3, 0] = 1.
                    file[key] = values
                file['collection_reset_seed'] = np.repeat([1000000, 1000001], 3)
                file['step_idx'] = np.tile(np.arange(3), 2)
                if index == 0:
                    file.create_dataset('pixels', shape=(6, 224, 224, 3), dtype='u1')
                else:
                    file.create_dataset('pcd_xyz', shape=(6, 1200, 3), dtype='f4')
                    file.create_dataset('pcd_rgb', shape=(6, 1200, 3), dtype='u1')
        return paths

    def test_rejects_duplicate_physical_initial_states_within_one_shard(self):
        rgb, pcd = self.pair(distinct_initials=False)
        with self.assertRaisesRegex(ValueError, 'duplicate physical initial'):
            collection.validate_pair(rgb, pcd, episodes=2, horizon=2,
                                     cube_count=1, world_seed=1000000)

    def test_accepts_distinct_physical_initial_states_within_one_shard(self):
        rgb, pcd = self.pair(distinct_initials=True)
        result = collection.validate_pair(rgb, pcd, episodes=2, horizon=2,
                                          cube_count=1, world_seed=1000000)
        self.assertEqual(result['actual_exit_code'], 0)
        self.assertEqual(result['reset_seeds'], [1000000, 1000001])
        self.assertEqual(len(set(result['initial_state_sha256'])), 2)


@unittest.skipUnless(importlib.util.find_spec('h5py'), 'requires the existing HDF environment')
class HdfResumeTests(unittest.TestCase):
    materialize = fixture.CollectionResumeTests.materialize
    qc = fixture.CollectionResumeTests.qc

    def setUp(self):
        fixture.CollectionResumeTests.setUp(self)
        self.patches[-1].stop()  # Use the actual validate_pair and merge_raw.

    def child(self, command, **kwargs):
        result = fixture.CollectionResumeTests.child(self, command, **kwargs)
        if result.returncode:
            return result
        import h5py
        import numpy as np
        values = dict(word.split('=', 1) for word in command if '=' in word)
        directory = Path(values['cache_dir'])
        horizon, seed = int(values['world.max_episode_steps']), int(values['world_seed'])
        rows = horizon + 1
        for name in ('pair_pix.h5', 'pair_pcd.h5'):
            path = directory / 'datasets' / name
            # Replace only the synthetic bytes made by this test's fake child.
            self.assertEqual(path.read_text(), str(seed))
            path.unlink()
            with h5py.File(path, 'x') as file:
                file['ep_len'] = np.array([rows], np.int32)
                file['ep_offset'] = np.array([0], np.int64)
                actions = np.zeros((rows, 5), np.float32)
                actions[-1] = np.nan
                file['action'] = actions
                for key, width in (('observation', 28), ('qpos', 21), ('qvel', 20)):
                    file[key] = np.full((rows, width), seed, np.float32)
                file['collection_reset_seed'] = np.full(rows, seed, np.int64)
                file['step_idx'] = np.arange(rows, dtype=np.int64)
                if name == 'pair_pix.h5':
                    file.create_dataset('pixels', shape=(rows, 224, 224, 3), dtype='u1')
                else:
                    file.create_dataset('pcd_xyz', shape=(rows, 1200, 3), dtype='f4')
                    file.create_dataset('pcd_rgb', shape=(rows, 1200, 3), dtype='u1')
        return result

    def test_actual_hdf_qc_vds_merge_and_completed_resume(self):
        import h5py
        import numpy as np
        collection.run(self.plan, part_ids=[0])
        failed_seed = self.plan['parts'][1]['world_seed']
        self.failed.add(failed_seed)
        with self.assertRaises(RuntimeError):
            collection.run(self.plan, resume=True)
        self.failed.clear()
        result = collection.run(self.plan, resume=True)
        self.assertEqual(result['actual_exit_code'], 0)
        with h5py.File(result['paths']['train_pcd'], 'r') as file:
            self.assertTrue(file['pcd_xyz'].is_virtual)
            np.testing.assert_array_equal(file['ep_len'][:], [201, 201])
            np.testing.assert_array_equal(file['collection_reset_seed'][::201],
                                          [self.plan['parts'][i]['world_seed'] for i in (0, 1)])
            self.assertTrue(np.isnan(file['action'][[200, 401]]).all())
        count = len(self.calls)
        self.assertEqual(collection.run(self.plan, resume=True), result)
        self.assertEqual(len(self.calls), count)
        marker = self.root / 'COLLECTION.json'
        changed = json.loads(marker.read_text())
        changed['paths']['train_pcd'] = str(self.root / 'unverified.h5')
        marker.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'manifest checksum'):
            collection.run(self.plan, resume=True)


if __name__ == '__main__':
    unittest.main()
