"""Behavioral tests for the opt-in dispatcher; no simulator/learning imports."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / 'jepa4r/policy/bc/relocate/run_rh56.py'


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(ENTRY.is_file(), 'explicit RH56 dispatcher has not been implemented')
        spec = importlib.util.spec_from_file_location('_rh56_public_entry_test', ENTRY)
        self.entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.entry)

    def test_no_stage_does_not_execute(self):
        with patch.object(self.entry, 'invoke', side_effect=AssertionError('must not run')):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(self.entry.main([]), 0)
        self.assertIn('RH56', output.getvalue())

    def test_help_does_not_import_backend(self):
        with patch.object(self.entry.importlib, 'import_module', side_effect=AssertionError('heavy import')):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as exited:
                    self.entry.main(['wm', '--help'])
        self.assertEqual(exited.exception.code, 0)

    def test_dry_run_route_and_unknown_backend_arguments(self):
        before = dict(os.environ)
        with patch.object(self.entry, 'invoke', side_effect=AssertionError('must not run')):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                result = self.entry.main(['wm', '--route', 'pcd', '--dataset', 'paired.h5',
                                          '--workers', '24', '--dry-run'])
        self.assertEqual(result, 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan['module'], 'jepa4r.envs.twins.rh56.wm_bridge')
        self.assertIn('pcd', plan['argv'])
        self.assertIn('paired.h5', plan['argv'])
        self.assertIn('24', plan['argv'])
        self.assertNotIn('--uint8-gpu-preprocess', plan['argv'])
        self.assertEqual(before, dict(os.environ))

    def test_learning_route_must_be_explicit(self):
        with patch.object(self.entry, 'invoke', side_effect=AssertionError('must not run')):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exited:
                    self.entry.main(['wm', '--dry-run'])
        self.assertEqual(exited.exception.code, 2)

    def test_collect_is_paired_and_uses_approved_camera(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.entry.main(['collect', '--output', 'fresh', '--dry-run'])
        plan = json.loads(output.getvalue())
        self.assertEqual(plan['module'], 'jepa4r.envs.twins.rh56.collect')
        self.assertEqual(plan['argv'][plan['argv'].index('--camera') + 1], 'opposite')
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.entry.main(['collect', '--route', 'rgb', '--dry-run'])

    def test_actual_wm_dispatch_and_environment_restoration(self):
        received = []
        before = os.environ.get('RH56_BASELINE_ROOT')
        def selected(module, argv):
            received.append((module, argv, os.environ['RH56_BASELINE_ROOT']))
        with patch.object(self.entry, 'invoke', side_effect=selected):
            self.assertEqual(self.entry.main(['wm', '--route', 'rgb', '--mode', 'smoke']), 0)
        self.assertEqual(received[0][0], 'jepa4r.envs.twins.rh56.wm_bridge')
        self.assertEqual(received[0][2], str(ROOT / 'jepa4r'))   # the moved entry's code root
        self.assertIn('--mode', received[0][1])
        self.assertEqual(os.environ.get('RH56_BASELINE_ROOT'), before)

    def test_missing_expert_rejected_before_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            fresh = Path(directory) / 'not-created'
            with patch.dict(os.environ, {'RH56_EXPERT_ROOT': ''}):
                with patch.object(self.entry, 'invoke', side_effect=AssertionError('must not run')):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        self.entry.main(['collect', '--output', str(fresh)])
            self.assertFalse(fresh.exists())

    def test_data_stages_delegate_to_shared_data_module(self):
        for stage in ('validate', 'publish'):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.entry.main([stage, '--batch', 'input', '--dry-run'])
            plan = json.loads(output.getvalue())
            self.assertEqual(plan['module'], 'jepa4r.envs.twins.rh56.data_tools')
            self.assertEqual(plan['argv'][0], stage)


if __name__ == '__main__':
    unittest.main()
