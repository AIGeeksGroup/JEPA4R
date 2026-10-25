"""Contract tests for the sealed thin LeWM entry; no model training."""
import ast
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class LeWMRunnerTests(unittest.TestCase):
    def runner(self):
        self.assertIsNotNone(importlib.util.find_spec('bench.ogbench.lewm'),
                             'thin LeWM runner has not been implemented')
        return importlib.import_module('bench.ogbench.lewm')

    def test_command_is_lazy_and_selects_explicit_mode(self):
        runner = self.runner()
        for method, mode in [('lewm_rgb', '2d'), ('lewm_pcd', '3d')]:
            command = runner.build_command('double500', method, 'sealed', 'fresh',
                                           workers=2, prefetch=3, python='python-test')
            self.assertEqual(command[:3], ['python-test', '-m', 'bench.ogbench.lewm'])
            self.assertEqual(command[command.index('--mode') + 1], mode)
            self.assertEqual(command[command.index('--workers') + 1], '2')
            self.assertEqual(command[command.index('--prefetch') + 1], '3')
            self.assertTrue(Path(command[command.index('--output') + 1]).is_absolute())
        result = subprocess.run([sys.executable, '-c',
            'import sys; from bench.ogbench.lewm import build_command; '
            'build_command("double500","lewm_rgb","d","o"); '
            'assert not any(n in sys.modules for n in '
            '["torch","hydra","omegaconf","lightning","stable_pretraining"])'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_closed_profiles_and_resources_are_checked(self):
        runner = self.runner()
        for method in ('lewm_rgb', 'lewm_pcd'):
            for profile in ('double500', 'triple_front600', 'quadruple600'):
                self.assertIn('--mode', runner.build_command(profile, method, 'd', 'o'))
            with self.assertRaisesRegex(ValueError, 'Single|single|closed|producer'):
                runner.build_command('single200', method, 'd', 'o')
        for kwargs in ({'workers': -1}, {'workers': True}, {'prefetch': 0}):
            with self.assertRaises(ValueError):
                runner.build_command('double500', 'lewm_rgb', 'd', 'o', **kwargs)
        with self.assertRaises(ValueError):
            runner.build_command('double500', 'lejepa', 'd', 'o')

    def test_packaged_recipes_preserve_math_and_explicit_validation(self):
        runner = self.runner()
        for mode, encoder in [('2d', 'encoders.ViTCLS'), ('3d', 'encoders.DexArtPointNet')]:
            cfg = runner.build_config('double500', mode, Path('sealed'), workers=0, prefetch=2)
            self.assertEqual(cfg['model']['encoder']['_target_'], encoder)
            self.assertEqual(cfg['embed_dim'], 192)
            self.assertEqual(cfg['model']['action_encoder']['input_dim'], 25)
            self.assertEqual(cfg['optimizer'], {'type': 'AdamW', 'lr': 5e-5, 'weight_decay': 1e-3})
            self.assertEqual(cfg['loss']['sigreg'], {'weight': .09, 'kwargs': {'knots': 17, 'num_proj': 1024}})
            self.assertEqual(cfg['trainer']['precision'], 'bf16-mixed')
            self.assertEqual(cfg['trainer']['max_epochs'], 10)
            self.assertEqual((cfg['seed'], cfg['fresh_seed']), (0, 0))
            self.assertNotIn('resume_checkpoint', cfg)
            self.assertEqual(cfg['loader']['num_workers'], 0)
            self.assertNotIn('prefetch_factor', cfg['loader'])
            self.assertNotIn('persistent_workers', cfg['loader'])
            self.assertNotEqual(cfg['data']['dataset']['name'], cfg['data']['validation_dataset']['name'])
            for role in ('dataset', 'validation_dataset'):
                self.assertEqual(cfg['data'][role]['format'], 'bench_lewm_sealed')
                self.assertEqual((cfg['data'][role]['num_steps'], cfg['data'][role]['frameskip']), (4, 5))

    def test_custom_config_cannot_change_encoder_or_loss(self):
        runner = self.runner()
        recipe = runner.load_recipe('2d')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(recipe), encoding='utf8')
            self.assertIn('--config', runner.build_command('double500', 'lewm_rgb', 'd', 'o', config_path=path))
            wrong = copy.deepcopy(recipe)
            wrong['model']['encoder']['size'] = 'base'
            path.write_text(json.dumps(wrong), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'recipe|configuration'):
                runner.build_command('double500', 'lewm_rgb', 'd', 'o', config_path=path)

    def test_changed_packaged_recipe_is_rejected(self):
        runner = self.runner()
        with tempfile.TemporaryDirectory() as folder:
            recipe = runner.load_recipe('2d')
            recipe['loss']['sigreg']['weight'] = .5
            (Path(folder) / '2d.json').write_text(json.dumps(recipe), encoding='utf8')
            with patch.object(runner, 'NATIVE_DIRECTORY', Path(folder)):
                with self.assertRaisesRegex(ValueError, 'recipe.*(source|SHA|differ)'):
                    runner.load_recipe('2d')

    def test_native_entry_is_byte_preserved_and_has_sealed_split(self):
        runner = self.runner()
        path = runner.NATIVE_TRAIN
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         '5a91153791f2d3a46eb6d463ebf97cc20d6777b6fdd83b6ee64c25646dc1c699')
        tree = ast.parse(path.read_text(encoding='utf8'))
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertTrue({'_prepare_fresh_run', '_split_datasets', 'lejepa_forward', 'run'} <= names)
        text = path.read_text(encoding='utf8')
        self.assertIn('LinearWarmupCosineAnnealingLR', text)
        self.assertIn('"interval": "epoch"', text)

    def test_dataset_router_rejects_unbound_path_and_restores_loader(self):
        runner = self.runner()
        original = lambda *args, **kwargs: None
        native = types.SimpleNamespace(swm=types.SimpleNamespace(data=types.SimpleNamespace(load_dataset=original)))
        binding = {'profile_name': 'double500', 'mode': '2d', 'train_path': str(Path('d/train.h5').resolve()),
                   'validation_path': str(Path('d/val.h5').resolve())}
        calls = []
        def reader(profile, mode, split, root, **kwargs):
            calls.append((profile, mode, split, kwargs))
            return split
        with patch('bench.ogbench.lewm_data.make_reader', reader):
            with runner.route_datasets(native, binding, Path('d')):
                self.assertEqual(native.swm.data.load_dataset(binding['train_path'], transform='original'), 'train')
                self.assertEqual(native.swm.data.load_dataset(binding['validation_path']), 'validation')
                with self.assertRaisesRegex(ValueError, 'unbound|sealed'):
                    native.swm.data.load_dataset('other.h5')
            self.assertIs(native.swm.data.load_dataset, original)
        self.assertEqual([c[2] for c in calls], ['train', 'validation'])
        self.assertEqual(calls[0][3]['transform'], 'original')

    def test_child_failure_keeps_state_log_config_and_exit_code(self):
        runner = self.runner()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'fresh'
            with patch('bench.ogbench.lewm_data.validate_binding', side_effect=ValueError('sealed hash mismatch')):
                code = runner.run_job('double500', '2d', Path(folder) / 'data', output)
            self.assertEqual(code, 1)
            state = json.loads((output / 'status.json').read_text(encoding='utf8'))
            self.assertEqual(state['stage'], 'lewm_failed')
            self.assertEqual(state['actual_exit_code'], 1)
            self.assertIn('sealed hash mismatch', state['error'])
            self.assertTrue((output / 'stdout_stderr.log').is_file())
            self.assertTrue((output / 'config.json').is_file())
            self.assertEqual((output / 'actual_exit_code').read_text().strip(), '1')
            before = (output / 'status.json').read_bytes()
            with self.assertRaises(FileExistsError):
                runner.run_job('double500', '2d', Path(folder) / 'data', output)
            self.assertEqual((output / 'status.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
