"""Single original-protocol runner contracts; no tensors or training."""
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class SingleOriginalRunnerTests(unittest.TestCase):
    def setUp(self):
        self.runner = importlib.import_module('bench.ogbench.lewm')

    def test_single_requires_explicit_child_protocol_and_other_profiles_reject_it(self):
        with self.assertRaisesRegex(ValueError, 'Single|single|protocol'):
            self.runner.build_command('single200', 'lewm_rgb', 'data', 'out')
        for profile in ('double500', 'triple_front600', 'quadruple600'):
            with self.assertRaisesRegex(ValueError, 'Single|single|protocol'):
                self.runner.build_command(profile, 'lewm_rgb', 'data', 'out', single_original=True)

    def test_single_command_is_lazy_and_carries_explicit_selector(self):
        for method in ('lewm_rgb', 'lewm_pcd'):
            command = self.runner.build_command('single200', method, 'data', 'out',
                                                python='PY', single_original=True)
            self.assertIn('--single-original-protocol', command)
            self.assertEqual(command[:3], ['PY', '-m', 'bench.ogbench.lewm'])
        result = subprocess.run([sys.executable, '-c',
            'import sys; from bench.ogbench.lewm import build_command; '
            'build_command("single200","lewm_rgb","d","o",single_original=True); '
            'assert not any(n in sys.modules for n in '
            '["torch","hydra","omegaconf","lightning","stable_pretraining"])'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_single_recipe_keeps_original_split_normalizers_and_seed(self):
        for mode, keys in (('2d', ['pixels', 'action', 'observation']),
                           ('3d', ['pcd_xyz', 'pcd_rgb', 'action', 'observation'])):
            cfg = self.runner.build_config('single200', mode, Path('raw'), single_original=True)
            self.assertEqual((cfg['seed'], cfg['train_split']), (3072, .9))
            self.assertNotIn('fresh_seed', cfg)
            self.assertNotIn('validation_dataset', cfg['data'])
            self.assertEqual(cfg['data']['dataset']['keys_to_load'], keys)
            self.assertEqual(cfg['data']['dataset']['keys_to_cache'], ['action', 'observation'])
            self.assertEqual(cfg['data']['dataset']['num_steps'],
                             self.runner.load_recipe(mode, single_original=True)['data']['dataset']['num_steps'])
            self.assertEqual(cfg['data']['dataset']['frameskip'], 5)
            self.assertIn('3072', cfg['subdir'])
            self.assertEqual(cfg['subdir'], cfg['output_model_name'])
            self.assertNotIn('seed0', cfg['subdir'])

    def test_single_resource_flags_only_change_loader_resources(self):
        original = self.runner.build_config('single200', '2d', 'data', single_original=True)
        adapted = self.runner.build_config('single200', '2d', 'data', workers=0, prefetch=2,
                                           single_original=True)
        self.assertEqual(adapted['loader']['num_workers'], 0)
        self.assertNotIn('prefetch_factor', adapted['loader'])
        self.assertNotIn('persistent_workers', adapted['loader'])
        for key in ('model', 'optimizer', 'loss', 'trainer', 'seed', 'train_split', 'data'):
            self.assertEqual(adapted[key], original[key])

    def test_single_router_passes_original_loader_without_recursion_and_restores(self):
        original = lambda *args, **kwargs: ('stock', args, kwargs)
        native = types.SimpleNamespace(swm=types.SimpleNamespace(data=types.SimpleNamespace(load_dataset=original)))
        binding = dict(profile_name='single200', mode='2d',
                       protocol='single_original_random_split_v1',
                       dataset_path=str(Path('data/raw.h5').resolve()))
        with patch('bench.ogbench.single_original.make_single_original_reader', return_value='raw') as reader:
            with self.runner.route_datasets(native, binding, Path('data')):
                self.assertEqual(native.swm.data.load_dataset(binding['dataset_path'], transform=None), 'raw')
                with self.assertRaisesRegex(ValueError, 'unbound|original|path'):
                    native.swm.data.load_dataset('data/another.h5')
            self.assertIs(native.swm.data.load_dataset, original)
            reader.assert_called_once_with('2d', Path('data'), loader=original, transform=None)

    def test_single_native_selection_preserves_both_training_sources(self):
        loaded = []
        def fake_load(name, path, package=False):
            loaded.append(Path(path))
            return types.SimpleNamespace()
        with patch.object(self.runner, '_load', side_effect=fake_load):
            with self.runner.pinned_native(single_original=True):
                pass
        self.assertEqual(loaded[-1], self.runner.CODE / 'train.py')
        self.assertEqual(hashlib.sha256(loaded[-1].read_bytes()).hexdigest(),
                         '733564c824ce4023abdcdcf0a65ad72f5b71efe9d21d7958a466f87c7cdfe6e5')
        self.assertEqual(hashlib.sha256(self.runner.NATIVE_TRAIN.read_bytes()).hexdigest(),
                         '5a91153791f2d3a46eb6d463ebf97cc20d6777b6fdd83b6ee64c25646dc1c699')

    def test_single_existing_output_is_rejected_without_touching_contents(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'existing'
            output.mkdir()
            sentinel = output / 'sentinel'
            sentinel.write_text('unchanged')
            with self.assertRaises(FileExistsError):
                self.runner.run_job('single200', '2d', 'data', output, single_original=True)
            self.assertEqual(sentinel.read_text(), 'unchanged')
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_child_parser_forwards_explicit_selector(self):
        with patch.object(self.runner, 'run_job', return_value=0) as run:
            self.assertEqual(self.runner.main(['--profile', 'single200', '--mode', '2d',
                '--data-root', 'raw', '--output', 'fresh', '--single-original-protocol']), 0)
        self.assertTrue(run.call_args.kwargs['single_original'])


if __name__ == '__main__':
    unittest.main()
