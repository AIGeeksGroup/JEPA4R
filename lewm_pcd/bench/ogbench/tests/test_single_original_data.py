"""Single's historical recipe and unbounded original HDF5 route."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class SingleOriginalDataTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).parents[1] / 'single_original.py'
        self.assertTrue(path.is_file(), 'original Single data route is missing')
        return importlib.import_module('bench.ogbench.single_original')

    def test_describe_has_original_windows_and_one_source(self):
        module = self.module()
        for mode, filename in (('2d', 'fixed5_mixed_pix_train_chunk5.h5'),
                               ('3d', 'fixed5_mixed_pcd_train.h5')):
            value = module.describe_single_original(mode, '/original/ogbench')
            self.assertEqual(value['profile_name'], 'single200')
            self.assertEqual(value['protocol'], 'single_original_random_split_v1')
            self.assertEqual(Path(value['dataset_path']).name, filename)
            self.assertNotIn('validation_path', value)
            self.assertEqual((value['seed'], value['train_split']), (3072, 0.9))
            self.assertEqual(value['split_scope'], 'window')
            self.assertEqual(value['validation_fraction'], 1 - 0.9)
            self.assertEqual(value['manager_seed_argument'], 'omitted')
            self.assertFalse(value['global_seed_before_model'])
            self.assertEqual((value['raw_rows_per_episode'], value['clips_per_episode']), (201, 182))
            self.assertEqual(value['starts'], [0, 181])
            self.assertEqual(value['keys_to_cache'], ['action', 'observation'])
            self.assertEqual(value['keys_to_load'][-2:], ['action', 'observation'])
            self.assertTrue(value['terminal_nan_action_window_preserved'])

    def test_recipe_preserves_original_sampling_normalization_and_workers(self):
        module = self.module()
        for mode, workers, prefetch in (('2d', 6, 3), ('3d', 2, 2)):
            cfg = module.load_single_recipe(mode)
            self.assertEqual((cfg['seed'], cfg['train_split']), (3072, 0.9))
            self.assertNotIn('fresh_seed', cfg)
            self.assertNotIn('validation_dataset', cfg['data'])
            self.assertEqual(cfg['data']['dataset']['num_steps'], "${eval:'${num_preds} + ${history_size}'}")
            self.assertEqual(cfg['data']['dataset']['frameskip'], 5)
            self.assertEqual(cfg['data']['dataset']['keys_to_cache'], ['action', 'observation'])
            self.assertEqual((cfg['loader']['num_workers'], cfg['loader']['prefetch_factor']), (workers, prefetch))
            self.assertTrue(cfg['loader']['persistent_workers'])
            self.assertEqual(cfg['trainer']['max_epochs'], 10)
        self.assertEqual(module.load_single_recipe('2d')['data']['dataset']['keys_to_merge'], {'proprio': 'proprio'})

    def test_recipe_override_must_equal_frozen_original(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'recipe.json'
            cfg = module.load_single_recipe('2d')
            path.write_text(json.dumps(cfg))
            self.assertEqual(module.load_single_recipe('2d', path), cfg)
            cfg['seed'] = 0
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'original.*recipe|recipe.*original'):
                module.load_single_recipe('2d', path)

    def test_invalid_mode_and_remote_root_fail(self):
        module = self.module()
        with self.assertRaisesRegex(ValueError, 'mode'):
            module.describe_single_original('rgb', '/original')
        with self.assertRaisesRegex(ValueError, 'local'):
            module.describe_single_original('2d', 's3://bucket')

    def test_reader_delegates_original_loader_and_exact_rgb_options(self):
        module = self.module()
        sentinel, transform = object(), object()
        calls = []
        def original_loader(name, **kwargs):
            calls.append((name, kwargs))
            return sentinel
        description = module.describe_single_original('2d', '/original')
        with patch.object(module, 'validate_single_original_binding', return_value=description) as validate:
            result = module.make_single_original_reader('2d', '/original', loader=original_loader,
                                                       transform=transform, cache_dir='/cache')
        self.assertIs(result, sentinel)
        validate.assert_called_once_with('2d', '/original')
        self.assertEqual(calls[0][0], description['dataset_path'])
        self.assertEqual(calls[0][1], dict(num_steps=4, frameskip=5,
            keys_to_load=['pixels', 'action', 'observation'], keys_to_cache=['action', 'observation'],
            keys_to_merge={'proprio': 'proprio'}, transform=transform, cache_dir='/cache'))

    def test_reader_rejects_new_sampling_and_cache_protocol(self):
        module = self.module()
        for options in ({'num_steps': 16}, {'frameskip': 1}, {'keys_to_cache': ['action']},
                        {'compact_clip_indices': True}, {'format': 'bench_lewm_sealed'},
                        {'keys_to_merge': {}}, {'allow_fixture': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                module.make_single_original_reader('2d', '/absent', loader=lambda *a, **k: None, **options)

    def fixture(self, directory, mode):
        import h5py
        import numpy as np
        module = self.module()
        bindings = json.loads(json.dumps(module._BINDINGS))
        selected = bindings[mode]
        path = Path(directory) / selected['filename']
        source = Path(directory) / 'fixed5_mixed_all_pcd.h5'
        selected['episodes'] = 1
        selected['rows'] = 201
        selected['columns'] = {key: dict(value, shape=[201, *value['shape'][1:]])
                               for key, value in selected['columns'].items()}
        keys = selected['columns']
        if mode == '3d':
            with h5py.File(source, 'w') as handle:
                for key, info in keys.items():
                    handle.create_dataset(key, shape=tuple(info['shape']), dtype=info['dtype'],
                                          chunks=(1, *info['shape'][1:]))
                handle['action'][200] = np.nan
            selected['backing']['stat'] = dict(bytes=source.stat().st_size, mtime_ns=source.stat().st_mtime_ns)
            selected['backing']['path'] = str(source.resolve())
            selected['backing']['columns'] = {key: dict(value, virtual=False) for key, value in keys.items()}
        with h5py.File(path, 'w', libver='latest') as handle:
            for key, info in keys.items():
                if mode == '3d':
                    layout = h5py.VirtualLayout(shape=tuple(info['shape']), dtype=info['dtype'])
                    layout[:] = h5py.VirtualSource(str(source), key, shape=tuple(info['shape']))[:]
                    handle.create_virtual_dataset(key, layout)
                else:
                    handle.create_dataset(key, shape=tuple(info['shape']), dtype=info['dtype'],
                                          chunks=(1, *info['shape'][1:]))
            if mode == '2d':
                handle['action'][200] = np.nan
            lengths = np.array([201], dtype=np.int32)
            offsets = np.array([0], dtype=np.int64)
            handle['ep_len'], handle['ep_offset'] = lengths, offsets
        selected['stat'] = dict(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
        selected['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        selected['index'] = {key: dict(shape=list(value.shape), dtype=str(value.dtype),
                                      sha256=hashlib.sha256(value.tobytes()).hexdigest())
                             for key, value in [('ep_len', lengths), ('ep_offset', offsets)]}
        if mode == '3d':
            with h5py.File(path, 'r') as handle:
                selected['virtual_sources'] = module._virtual_sources(handle, keys)
        return bindings, path, source

    @unittest.skipUnless(importlib.util.find_spec('h5py'), 'h5py unavailable in local interpreter')
    def test_original_schema_allows_terminal_nan_and_avoids_large_rehash(self):
        module = self.module()
        for mode in ('2d', '3d'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                bindings, path, _ = self.fixture(directory, mode)
                with patch.object(module, '_BINDINGS', bindings), patch.object(module, '_sha', wraps=module._sha) as sha:
                    value = module.validate_single_original_binding(mode, directory)
                self.assertEqual(value['dataset_path'], str(path.resolve()))
                self.assertEqual(value['clips_per_episode'], 182)
                self.assertEqual([call.args[0].resolve() for call in sha.call_args_list],
                                 [path.resolve()] if mode == '3d' else [])

    @unittest.skipUnless(importlib.util.find_spec('h5py'), 'h5py unavailable in local interpreter')
    def test_changed_stat_and_missing_backing_are_rejected(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            bindings, path, source = self.fixture(directory, '3d')
            with patch.object(module, '_BINDINGS', bindings):
                source.rename(source.with_suffix('.missing'))
                with self.assertRaises((OSError, ValueError)):
                    module.validate_single_original_binding('3d', directory)
            bindings, path, _ = self.fixture(directory, '2d')
            bindings['2d']['stat']['mtime_ns'] -= 1
            with patch.object(module, '_BINDINGS', bindings), self.assertRaisesRegex(ValueError, 'stat'):
                module.validate_single_original_binding('2d', directory)

    @unittest.skipUnless(importlib.util.find_spec('h5py'), 'h5py unavailable in local interpreter')
    def test_changed_schema_and_episode_index_reject_after_matching_stat(self):
        import h5py
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            bindings, path, _ = self.fixture(directory, '2d')
            with h5py.File(path, 'r+') as handle:
                handle['ep_offset'][0] = 1
            bindings['2d']['stat']['mtime_ns'] = path.stat().st_mtime_ns
            with patch.object(module, '_BINDINGS', bindings), self.assertRaisesRegex(ValueError, 'episode index'):
                module.validate_single_original_binding('2d', directory)
            bindings, path, _ = self.fixture(directory, '2d')
            bindings['2d']['columns']['observation']['dtype'] = 'float32'
            with patch.object(module, '_BINDINGS', bindings), self.assertRaisesRegex(ValueError, 'column schema'):
                module.validate_single_original_binding('2d', directory)


if __name__ == '__main__':
    unittest.main()
