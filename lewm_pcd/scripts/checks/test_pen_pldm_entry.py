"""CPU preparation/dispatch tests; no framework training is executed."""
import contextlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


LEWM = Path(__file__).resolve().parents[2]
ENTRY = LEWM / 'scripts/baselines/pen/train_pldm.py'


class PenPLDMEntryTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(ENTRY.is_file(), 'Pen PLDM thin entry is not implemented')
        spec = importlib.util.spec_from_file_location('_pen_pldm_entry_test', ENTRY)
        self.entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.entry)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.framework = self.root / 'framework'
        script = self.framework / 'scripts/train/pldm.py'
        script.parent.mkdir(parents=True)
        script.write_text('import logging\nlogger=logging.getLogger(__name__)\n'
                          'def run(cfg):\n    logger.info("Run ID: %s", cfg)\n'
                          '    logger=None\n    return cfg\n', encoding='utf-8')
        (script.parent / 'config').mkdir()
        (script.parent / 'config/pldm.yaml').write_text('trainer: {max_epochs: 100}\n')
        self.dataset = self.root / 'train_rgb.h5'
        self.dataset.write_bytes(b'fixture; never opened as a dataset')

    def argv(self, *extra):
        return ['--framework-root', str(self.framework), '--dataset', str(self.dataset),
                '--output', str(self.root / 'fresh-output'), *extra]

    def prepare(self, *extra):
        with patch.object(self.entry, 'execute', side_effect=AssertionError('must not train')):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(self.entry.main(self.argv(*extra)), 0)
        return json.loads(output.getvalue())

    def test_default_and_explicit_dry_run_are_preparation_only(self):
        for extra in ((), ('--mode', 'formal', '--dry-run')):
            plan = self.prepare(*extra)
            self.assertEqual(plan['execution'], 'PREPARED_ONLY')
            self.assertEqual(plan['recipe']['selected_frames'], 4)
            self.assertEqual(plan['recipe']['frameskip'], 5)
            self.assertEqual(plan['recipe']['raw_span'], 20)
            self.assertEqual(plan['recipe']['batch_size'], 128)
            self.assertEqual(plan['recipe']['action_block_dim'], 120)
            self.assertIn('trainer.max_epochs=10', plan['hydra_overrides'])
            self.assertIn('~data.dataset.keys_to_merge', plan['hydra_overrides'])
            self.assertIn('all windows preserved', plan['data_boundary'])
            self.assertFalse((self.root / 'fresh-output').exists())

    def test_help_and_no_arguments_do_not_import_learning_runtime(self):
        with patch.object(self.entry.importlib, 'import_module', side_effect=AssertionError('heavy import')):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(self.entry.main([]), 0)
                with self.assertRaises(SystemExit) as exited:
                    self.entry.main(['--help'])
        self.assertEqual(exited.exception.code, 0)

    def test_relocate_switch_preserves_recipe_and_uses_native_action_width(self):
        plan = self.prepare('--task', 'adroit_relocate', '--workers', '24', '--prefetch-factor', '2')
        self.assertEqual(plan['task'], 'adroit_relocate')
        self.assertEqual(plan['name'], 'pldm_adroit_relocate')
        self.assertEqual(plan['recipe']['raw_action_dim'], 30)
        self.assertEqual(plan['recipe']['action_block_dim'], 150)
        self.assertEqual(plan['expected_visual_frames'], 2000000)
        self.assertEqual(plan['source_row_schema'], 'pre_action_rows_with_outgoing_action')
        self.assertEqual(plan['workers'], 24)
        self.assertEqual(plan['prefetch_factor'], 2)
        for key in ('selected_frames', 'frameskip', 'batch_size', 'epochs', 'loss'):
            self.assertEqual(plan['recipe'][key], self.prepare()['recipe'][key])
        self.assertEqual(self.prepare(), self.prepare('--task', 'adroit_pen'))

    def test_relocate_checkpoint_selector_uses_existing_loader(self):
        path = LEWM / 'bench/cortexbench/conf/model/pldm_relocate.yaml'
        self.assertTrue(path.is_file())
        values = dict(line.split(':', 1) for line in path.read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith('#'))
        self.assertEqual({k.strip(): v.strip() for k, v in values.items()},
                         {'_target_': 'jepa_baselines.load_pldm',
                          'checkpoint_dir': '${oc.env:RELOCATE_PLDM_CHECKPOINT}', 'epoch': '10'})

    def test_smoke_retains_ten_epoch_schedule_budget_and_caps_actual_steps(self):
        plan = self.prepare('--mode', 'smoke', '--dry-run')
        self.assertIn('trainer.max_epochs=10', plan['hydra_overrides'])
        self.assertIn('+trainer.max_steps=2', plan['hydra_overrides'])
        self.assertIn('+trainer.limit_train_batches=2', plan['hydra_overrides'])
        self.assertIn('+trainer.limit_val_batches=2', plan['hydra_overrides'])
        self.assertIn('pending', plan['acceptance'])

    def test_invalid_budgets_and_names_fail_before_runtime(self):
        for extra in (('--mode', 'formal', '--max-steps', '2'), ('--prefetch-factor', '4'),
                      ('--workers', '-1'), ('--mode', 'smoke', '--max-steps', '0'),
                      ('--task', 'rh56_relocate'),
                      ('--name', '../escape')):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.entry.main(self.argv(*extra))

    def test_logger_fix_only_uses_an_isolated_memory_module(self):
        source_path = self.framework / 'scripts/train/pldm.py'
        before = source_path.read_bytes()
        source = before.decode()
        original = {'__name__': '_original_pldm_fixture'}
        exec(compile(source, str(source_path), 'exec'), original)
        with self.assertRaises(UnboundLocalError):
            original['run']('example')
        tree, receipt = self.entry.logger_compatible_tree(source, str(source_path))
        isolated = {'__name__': '_test_pldm_compat'}
        with patch.object(logging.Logger, 'info') as log:
            exec(compile(tree, str(source_path), 'exec'), isolated)
            self.assertEqual(isolated['run']('example'), 'example')
            log.assert_called_once_with('Run ID: %s', 'example')
        self.assertTrue(receipt['applied'])
        self.assertEqual(receipt['disk_source_changed'], False)
        self.assertEqual(source_path.read_bytes(), before)

    def test_cuda_unavailable_never_creates_output_or_loads_native(self):
        plan = self.prepare('--mode', 'formal', '--dry-run')
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.object(self.entry.importlib, 'import_module', return_value=torch):
            with patch.object(self.entry, 'load_native', side_effect=AssertionError('must not load')):
                with self.assertRaises(RuntimeError):
                    self.entry.execute(plan)
        self.assertFalse(Path(plan['output']).exists())

    def test_explicit_execution_delegates_to_native_run_and_restores_environment(self):
        plan = self.prepare('--mode', 'smoke', '--dry-run')
        called, hooks, cache = [], [], []
        cfg = object()
        clips = [(0, 0), (0, 80)]  # The native terminal-touching selection is retained.
        native = SimpleNamespace(swm=SimpleNamespace(clip_indices=clips.copy()),
                                 spt=SimpleNamespace(set=lambda **kw: cache.append(kw)),
                                 run=SimpleNamespace(__wrapped__=lambda config: called.append(config)))
        @contextlib.contextmanager
        def handle(swm, path, recorder):
            hooks.append((swm, path))
            recorder({'parent_handle_closed': True})
            yield
        helpers = SimpleNamespace(dataset_handle_hook=handle)
        hydra = SimpleNamespace(initialize_config_dir=lambda **kw: contextlib.nullcontext(),
                                compose=lambda **kw: cfg)
        runtime = {'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
                   'hydra': hydra, 'scripts.baselines.pen.data': helpers}
        before = {k: os.environ.get(k) for k in ('STABLEWM_HOME', 'SPT_CACHE_DIR')}
        with patch.object(self.entry.importlib, 'import_module', side_effect=runtime.__getitem__):
            with patch.object(self.entry, 'framework_revision', return_value=self.entry.FRAMEWORK_COMMIT):
                with patch.object(self.entry, 'load_native', return_value=(native, {'applied': True})):
                    self.entry.execute(plan)
        self.assertEqual(called, [cfg])
        self.assertEqual(hooks, [(native.swm, Path(plan['dataset']))])
        self.assertEqual(native.swm.clip_indices, clips)
        self.assertTrue(cache)
        self.assertEqual(before, {k: os.environ.get(k) for k in before})
        state = json.loads((Path(plan['output']) / 'state.json').read_text())
        self.assertEqual(state['phase'], 'native_returned_acceptance_pending')
        self.assertFalse(state['formal_acceptance_performed'])
        self.assertEqual(state['data_receipts'], [{'parent_handle_closed': True}])
        self.assertEqual((Path(plan['output']) / 'actual_exit_code').read_text(), '0\n')


if __name__ == '__main__':
    unittest.main()
