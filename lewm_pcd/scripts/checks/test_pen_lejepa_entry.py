"""CPU-only preparation and dispatch tests; never run a learning backend."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


LEWM = Path(__file__).resolve().parents[2]
ENTRY = LEWM / 'scripts/baselines/pen/train_lejepa.py'


class PenLejepaEntryTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(ENTRY.is_file(), 'Pen LeJEPA thin entry is not implemented')
        spec = importlib.util.spec_from_file_location('_pen_lejepa_entry_test', ENTRY)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def prepared(self, extra=()):
        with patch.object(self.module, 'execute', side_effect=AssertionError('must not train')):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(self.module.main(['--dataset', 'train_rgb.h5', '--output', 'new-run', *extra]), 0)
        return json.loads(output.getvalue())

    def test_default_is_preparation_only_and_keeps_native_recipe(self):
        result = self.prepared()
        self.assertEqual(result['execution'], 'PREPARED_ONLY')
        self.assertEqual(result['recipe']['backbone'], 'vit_hf_tiny_patch14')
        self.assertEqual(result['recipe']['views'], 4)
        self.assertEqual(result['recipe']['batch_size'], 256)
        self.assertEqual(result['recipe']['downstream_dim'], 512)
        self.assertEqual(result['data_boundary'], 'native_all_visual_frames')
        self.assertEqual(result['expected_pen_visual_frames'], 914082)
        self.assertIn('--epochs', result['native_argv'])
        self.assertEqual(result['native_argv'][result['native_argv'].index('--epochs') + 1], '10')

    def test_help_and_no_arguments_do_not_import_backends(self):
        with patch.object(self.module.importlib, 'import_module', side_effect=AssertionError('heavy import')):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(self.module.main([]), 0)
                with self.assertRaises(SystemExit) as exited:
                    self.module.main(['--help'])
        self.assertEqual(exited.exception.code, 0)

    def test_explicit_dry_run_never_executes(self):
        result = self.prepared(['--mode', 'formal', '--dry-run'])
        self.assertEqual(result['execution'], 'PREPARED_ONLY')
        self.assertEqual(result['prefetch_factor'], 2)
        self.assertNotIn('--max-steps', result['native_argv'])

    def test_smoke_retains_native_step_limit_and_no_checkpoint_claim(self):
        result = self.prepared(['--mode', 'smoke', '--max-steps', '2', '--dry-run'])
        self.assertEqual(result['native_argv'][result['native_argv'].index('--max-steps') + 1], '2')
        self.assertEqual(result['checkpoint_scope'], 'native_smoke_does_not_save_a_checkpoint')

    def test_formal_cannot_hide_a_smoke_limit_and_prefetch_is_bounded(self):
        for extra in (['--mode', 'formal', '--max-steps', '2'], ['--prefetch-factor', '4'],
                      ['--mode', 'smoke', '--max-steps', '0'], ['--name', '../escape'],
                      ['--exclude-terminal-frames']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.module.main(['--dataset', 'input.h5', '--output', 'new', *extra])

    def test_loader_changes_only_scheduling_and_closes_parent_hdf(self):
        calls, closed = [], []
        original = lambda *a, **kw: calls.append((a, kw)) or 'loader'
        raw = object()
        dataset = SimpleNamespace(ds=raw)
        loader = self.module.loader_factory(original, 8, 2, closed.append)
        self.assertEqual(loader(dataset, batch_size=256, shuffle=True, drop_last=True,
                                persistent_workers=True, prefetch_factor=4), 'loader')
        self.assertEqual(closed, [raw])
        self.assertEqual(calls[0][1], dict(batch_size=256, shuffle=True, drop_last=True,
                                          persistent_workers=True, prefetch_factor=2, num_workers=8))
        self.module.loader_factory(original, 0, 3, closed.append)(dataset, prefetch_factor=4, persistent_workers=True)
        self.assertNotIn('prefetch_factor', calls[-1][1])
        self.assertFalse(calls[-1][1]['persistent_workers'])

    def test_no_cuda_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'input.h5'
            data.write_bytes(b'not opened by this test')
            output = Path(directory) / 'not-created'
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            with patch.dict(sys.modules, {'torch': fake_torch}):
                with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                    self.module.main(['--dataset', str(data), '--output', str(output), '--mode', 'formal'])
            self.assertFalse(output.exists())

    def test_explicit_execution_delegates_native_main_and_restores_globals(self):
        with tempfile.TemporaryDirectory() as directory:
            self.check_native_dispatch(Path(directory))

    def check_native_dispatch(self, directory):
        # A fake native backend verifies dispatch and restoration without tensor work.
        data = Path(directory) / 'input.h5'
        data.write_bytes(b'not opened by the fake native implementation')
        output = Path(directory) / 'fresh'
        seen = []
        original_loader = object()
        indices = [(0, 0), (0, 1), (0, 2)]
        raw_dataset = SimpleNamespace(clip_indices=indices, lengths=[3])
        @contextlib.contextmanager
        def handle_hook(module, path, recorder=None):
            seen.append(('data', module, path))
            original = module.load_dataset
            def load(*args, **kwargs):
                result = original(*args, **kwargs)
                seen.append(('handle', result))
                return result
            module.load_dataset = load
            try:
                yield
            finally:
                module.load_dataset = original
                seen.append(('data_restored',))
        original_dataset_loader = lambda *args, **kwargs: raw_dataset
        native = SimpleNamespace(swm=SimpleNamespace(load_dataset=original_dataset_loader),
                                 DataLoader=original_loader)
        def native_main():
            loaded = native.swm.load_dataset(data)
            self.assertIs(loaded.clip_indices, indices)
            self.assertEqual(loaded.clip_indices, [(0, 0), (0, 1), (0, 2)])
            seen.append(('main', list(sys.argv), os.environ['STABLEWM_HOME']))
        native.main = native_main
        helpers = SimpleNamespace(dataset_handle_hook=handle_hook, close_parent_hdf=lambda ds: None)
        previous_argv, previous_env = sys.argv, os.environ.get('STABLEWM_HOME')
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        with patch.dict(sys.modules, {'torch': fake_torch}):
            with patch.object(self.module, '_load_native', return_value=native):
                with patch.object(self.module, '_load_data_helpers', return_value=helpers):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(self.module.main(['--dataset', str(data), '--output', str(output),
                                                           '--mode', 'smoke', '--max-steps', '2']), 0)
        self.assertEqual([item[0] for item in seen], ['data', 'handle', 'main', 'data_restored'])
        self.assertIs(seen[1][1], raw_dataset)
        self.assertEqual(raw_dataset.clip_indices, [(0, 0), (0, 1), (0, 2)])
        main_call = next(item for item in seen if item[0] == 'main')
        self.assertEqual(main_call[2], str(output.resolve()))
        self.assertIs(native.swm.load_dataset, original_dataset_loader)
        self.assertIs(native.DataLoader, original_loader)
        self.assertIs(sys.argv, previous_argv)
        self.assertEqual(os.environ.get('STABLEWM_HOME'), previous_env)
        state = json.loads((output / 'state.json').read_text())
        self.assertEqual(state['phase'], 'native_returned_acceptance_pending')
        self.assertFalse(state['formal_acceptance_performed'])
        with self.assertRaises(FileExistsError):
            self.module.main(['--dataset', str(data), '--output', str(output), '--mode', 'formal'])

if __name__ == '__main__':
    unittest.main()
