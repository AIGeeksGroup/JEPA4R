"""LeWM sampling and source identity without importing tensor libraries."""
import importlib
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class LeWMDataTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).parents[1] / 'lewm_data.py'
        self.assertTrue(path.is_file(), 'LeWM data seam is missing')
        return importlib.import_module('bench.ogbench.lewm_data')

    def test_plan_has_exact_wm_sampling_without_opening_data(self):
        module = self.module()
        with patch.object(Path, 'open', side_effect=AssertionError('plan opened data')):
            for profile, rows in (('double500', 500), ('triple_front600', 600), ('quadruple600', 600)):
                for mode in ('2d', '3d'):
                    value = module.describe(profile, mode, '/absent')
                    self.assertEqual((value['num_steps'], value['frameskip'], value['span']), (4, 5, 20))
                    self.assertEqual((value['action_dim'], value['action_chunk_dim']), (5, 25))
                    self.assertEqual(value['clips_per_episode'], rows - 19)
                    self.assertEqual(value['keys_to_cache'], ['action'])
                    self.assertEqual(value['keys_to_load'], ['pixels', 'action'] if mode == '2d'
                                     else ['pcd_xyz', 'pcd_rgb', 'action'])
                    self.assertIn(mode + '_train_action_rows', value['train_path'])
                    self.assertIn(mode + '_validation_action_rows', value['validation_path'])

    def test_single_describes_original_data_and_official_profiles_fail(self):
        module = self.module()
        for mode in ('2d', '3d'):
            value = module.describe('single200', mode, '/absent')
            self.assertEqual(value['protocol'], 'single_original_random_split_v1')
            self.assertEqual(value['seed'], 3072)
            self.assertNotIn('validation_path', value)
        with self.assertRaisesRegex(ValueError, 'official|collected'):
            module.describe('official2', '2d', '/absent')
        with self.assertRaisesRegex(ValueError, 'mode'):
            module.describe('double500', 'rgb', '/absent')

    def test_manifest_checksum_rejects_before_hdf5_or_torch(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'data_views'
            folder.mkdir()
            (folder / 'training_view_index.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'manifest checksum'):
                module.validate_binding('double500', '3d', directory)

    def test_pcd_identity_comes_from_exact_sealed_manifest_entry(self):
        module = self.module()
        profile = module.describe('double500', '3d', '/absent')
        train_name = Path(profile['train_path']).name
        manifest = {'views': [dict(path='/old/' + train_name, keys=['pcd_xyz', 'pcd_rgb', 'action'],
                                   episodes=10000, rows=5000000, sha256='a' * 64, bytes=123)]}
        row = module._manifest_row(manifest, profile, 'train')
        self.assertEqual(row['sha256'], 'a' * 64)
        manifest['views'][0]['keys'] = ['pixels', 'action']
        with self.assertRaisesRegex(ValueError, 'modality|manifest'):
            module._manifest_row(manifest, profile, 'train')

    def test_manifest_may_not_substitute_split_or_missing_pcd_hash(self):
        module = self.module()
        profile = module.describe('double500', '3d', '/absent')
        row = dict(path='/old/' + Path(profile['train_path']).name,
                   keys=['pcd_xyz', 'pcd_rgb', 'action'], episodes=1000, rows=500000,
                   sha256='a' * 64, bytes=123)
        with self.assertRaisesRegex(ValueError, 'quota|split'):
            module._manifest_row({'views': [row]}, profile, 'train')
        row.update(episodes=10000, rows=5000000, sha256=None)
        with self.assertRaisesRegex(ValueError, 'sha256'):
            module._manifest_row({'views': [row]}, profile, 'train')

    def test_reader_rejects_protocol_override_before_tensor_import(self):
        module = self.module()
        for kwargs in ({'num_steps': 16}, {'frameskip': 1}, {'keys_to_cache': []},
                       {'keys_to_load': ['pixels', 'action']}, {'allow_fixture': True},
                       {'compact_clip_indices': False}, {'raw_chunk_cache_nbytes': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                module.make_reader('double500', '3d', 'train', '/absent', **kwargs)
        with self.assertRaisesRegex(ValueError, 'split'):
            module.make_reader('double500', '3d', 'test', '/absent')

    def _pcd_fixture(self, directory):
        """Real virtual columns with production metadata and no raw allocations."""
        import h5py
        import numpy as np
        module = self.module()
        profile = module.contracts.get_profile('double500')
        folder = Path(directory) / 'data_views'
        folder.mkdir()
        source = Path(directory) / 'raw.h5'
        with h5py.File(source, 'w') as handle:
            for key, shape, dtype in (('pcd_xyz', (5000000, 1200, 3), 'float32'),
                                      ('pcd_rgb', (5000000, 1200, 3), 'uint8'),
                                      ('action', (5000000, 5), 'float32')):
                handle.create_dataset(key, shape=shape, dtype=dtype, chunks=(1, *shape[1:]))
        manifest = dict(actual_exit_code=0, raw_data_unchanged=True, raw_rows_per_episode=501,
                        view_rows_per_episode=500, split_counts={'train': 10000, 'validation': 1000},
                        no_GT_model_input=True, no_physical_visual_merge=True, no_feature_cache=True,
                        canonical_index={'sha256': 'b' * 64}, independent_seal={'sha256': 'c' * 64},
                        window=dict(clips_per_episode=481, frameskip=5, num_steps=4, span=20, starts=[0, 480]),
                        raw_data_files={str(source): dict(bytes=source.stat().st_size)}, views=[])
        for split, count in (('train', 10000), ('validation', 1000)):
            rows = count * 500
            path = folder / ('3d_' + split + '_action_rows500.h5')
            with h5py.File(path, 'w', libver='latest') as handle:
                for key, tail, dtype in (('pcd_xyz', (1200, 3), 'float32'),
                                         ('pcd_rgb', (1200, 3), 'uint8'), ('action', (5,), 'float32')):
                    layout = h5py.VirtualLayout(shape=(rows, *tail), dtype=dtype)
                    layout[:] = h5py.VirtualSource(str(source), key, shape=(5000000, *tail))[:rows]
                    handle.create_virtual_dataset(key, layout)
                handle['ep_len'] = np.full(count, 500, dtype=np.int32)
                handle['ep_offset'] = np.arange(count, dtype=np.int64) * 500
                handle.attrs.update(view_recipe='all_tasks_H500_action_bearing_rows500_v1',
                                    canonical_index_sha256='b' * 64, independent_seal_sha256='c' * 64,
                                    raw_terminal_observation_row500='preserved_in_raw_archive')
            manifest['views'].append(dict(path=str(path), keys=['pcd_xyz', 'pcd_rgb', 'action'],
                                          episodes=count, rows=rows, bytes=path.stat().st_size,
                                          sha256=module.contracts._sha(path)))
            manifest['views'].append(dict(path='/original/' + profile[split + '_view'],
                                          keys=['pixels', 'action'], episodes=count, rows=rows,
                                          bytes=1, sha256=profile[split + '_sha256']))
        raw = json.dumps(manifest).encode()
        (folder / 'training_view_index.json').write_bytes(raw)
        profile['manifest_sha256'] = hashlib.sha256(raw).hexdigest()
        return profile, source, manifest

    def test_real_pcd_vds_accepts_and_missing_source_rejects(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            profile, source, _ = self._pcd_fixture(directory)
            with patch.dict(module.contracts._PROFILES, double500=profile):
                value = module.validate_binding('double500', '3d', directory)
                self.assertEqual(len(value['train_sha256']), 64)
                self.assertNotEqual(value['train_sha256'], profile['train_sha256'])
                source.rename(source.with_suffix('.missing'))
                with self.assertRaises((ValueError, OSError)):
                    module.validate_binding('double500', '3d', directory)

    def test_real_pcd_guard_rejects_wrong_terminal_attrs_even_with_new_hash(self):
        import h5py
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            profile, _, manifest = self._pcd_fixture(directory)
            path = Path(directory) / 'data_views/3d_train_action_rows500.h5'
            with h5py.File(path, 'r+') as handle:
                handle.attrs['raw_terminal_observation_row500'] = 'included'
            manifest['views'][0]['sha256'] = module.contracts._sha(path)
            raw = json.dumps(manifest).encode()
            (path.parent / 'training_view_index.json').write_bytes(raw)
            profile['manifest_sha256'] = hashlib.sha256(raw).hexdigest()
            with patch.dict(module.contracts._PROFILES, double500=profile):
                with self.assertRaisesRegex(ValueError, 'producer provenance'):
                    module.validate_binding('double500', '3d', directory)

    def test_native_scalar_skip5_observation_and_action_boundaries(self):
        module = self.module()
        try:
            import h5py
            import numpy as np
            import torch
        except ImportError as exc:
            self.skipTest(str(exc))
        import os
        from bench.ogbench.reader import reader_class
        cls = reader_class()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scalar.h5'
            actions = np.arange(40 * 5, dtype=np.float32).reshape(40, 5)
            xyz = np.arange(40 * 2 * 3, dtype=np.float32).reshape(40, 2, 3)
            with h5py.File(path, 'w') as handle:
                handle['pcd_xyz'] = xyz
            reader = cls.__new__(cls)
            reader.h5_file = h5py.File(path, 'r', swmr=True)
            reader._owner_pid = os.getpid()
            reader.h5_path = path
            reader._cache = {'action': actions}
            reader._keys = ['pcd_xyz', 'action']
            reader.offsets, reader.lengths = np.array([0, 20]), np.array([20, 20])
            reader.frameskip, reader.num_steps, reader.span = 5, 4, 20
            reader.clip_indices = [(0, 0), (1, 0)]
            reader.transform = None
            for index, start in enumerate((0, 20)):
                result = reader[index]
                self.assertTrue(torch.equal(result['pcd_xyz'], torch.from_numpy(xyz[start:start + 20:5])))
                self.assertTrue(torch.equal(result['action'], torch.from_numpy(actions[start:start + 20].reshape(4, 25))))
            reader.close()

    def test_shared_reader_still_delegates_scalar_math(self):
        self.module()
        text = (Path(__file__).parents[1] / 'reader.py').read_text()
        self.assertNotIn('def _load_slice', text)
        self.assertNotIn('def __getitem__(self, idx)', text)


if __name__ == '__main__':
    unittest.main()
