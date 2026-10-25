"""The switch describes a command unless execution is explicitly requested."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

LEWM = Path(__file__).resolve().parents[3]
ENTRY = LEWM / 'scripts/run_ogbench.py'


class SwitchTests(unittest.TestCase):
    def call(self, *args):
        return subprocess.run([sys.executable, str(ENTRY), *args], capture_output=True, text=True)

    def test_entry_exists(self):
        self.assertTrue(ENTRY.is_file(), 'new unified switch is missing')

    def test_list_does_not_import_training_frameworks(self):
        result = self.call('--list')
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertIn('single200', value['profiles'])
        self.assertIn('lejepa', value['methods'])
        self.assertNotIn('lejepa_vitb', value['methods'])

    def test_default_is_plan_only_and_no_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'never-created'
            result = self.call('--profile', 'quadruple600', '--method', 'lejepa', '--stage', 'train',
                               '--data-root', temporary, '--output-root', str(target))
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(result.stdout)
            self.assertFalse(value['execute'])
            self.assertEqual(value['profile']['cube_count'], 4)
            self.assertFalse(target.exists())
            self.assertNotIn('ssh', value['command'])

    def test_cube_mismatch_rejected_before_launch(self):
        result = self.call('--profile', 'double500', '--cube-count', '3', '--method', 'pldm',
                           '--stage', 'train', '--data-root', '.', '--output-root', 'not-created')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('cube', result.stderr.lower())

    def test_vitb_is_not_registered(self):
        result = self.call('--profile', 'single200', '--method', 'lejepa_vitb', '--stage', 'train',
                           '--data-root', '.', '--output-root', 'not-created')
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
