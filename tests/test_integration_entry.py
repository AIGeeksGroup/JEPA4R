"""The integrated entry must preserve both original command interfaces."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
import unittest
from unittest.mock import patch
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


class IntegrationEntryTests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([sys.executable, '-B', str(ROOT / 'run.py'), *args],
                              cwd=ROOT, capture_output=True, text=True, encoding='utf-8')

    def output(self, *args):
        result = self.invoke(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_root_list_exposes_both_suites(self):
        self.assertEqual(set(self.output('--list')['suites']), {'cube', 'relocate'})

    def test_explicit_cube_list_retains_original_contract(self):
        value = self.output('--suite', 'cube', '--list')
        self.assertEqual(value['cube_counts'], [1, 2, 3, 4])
        self.assertEqual(len(value['models']), 6)

    def test_legacy_cube_flags_select_cube(self):
        value = self.output('--cube-count', '1', '--list')
        self.assertEqual(value['cube_counts'], [1, 2, 3, 4])

    def test_legacy_relocate_flags_select_relocate(self):
        value = self.output('--env=adroit-relocate', '--list')
        self.assertEqual(len(value['adroit-relocate']), 6)
        self.assertEqual(value['rh56-relocate'], ['jepa4r', 'lewm'])

    def test_explicit_relocate_list_retains_original_contract(self):
        self.assertIn('rh56-relocate', self.output('--suite', 'relocate', '--list'))

    def test_common_world_model_python_flag_does_not_change_suite(self):
        value = self.output('--env', 'adroit-relocate', '--wm-python', sys.executable, '--list')
        self.assertIn('adroit-relocate', value)

    def test_conflicting_suite_flags_fail_before_execution(self):
        result = self.invoke('--cube-count', '1', '--env', 'adroit-relocate', '--list')
        self.assertEqual(result.returncode, 2)
        self.assertIn('conflict', result.stderr.lower())

    def test_explicit_suite_cannot_silently_override_other_suite(self):
        result = self.invoke('--suite', 'cube', '--env', 'rh56-relocate', '--list')
        self.assertEqual(result.returncode, 2)
        self.assertIn('conflict', result.stderr.lower())

    def test_original_verification_records_remain_byte_preserved(self):
        expected = {
            'cube': '2d93733da5e3412b8449f7b4f56c6f10a7da6de237c70c1bc40b2b72a20e334c',
            'adroit': '1f8338ab713ec281a38427ad948354108a6f50b80cda4a4cd0ae2cee482400f6',
        }
        for name, digest in expected.items():
            path = ROOT / 'provenance/baselines' / name / 'SOURCE_MANIFEST.json'
            self.assertTrue(path.is_file(), str(path))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_legacy_environment_exports_and_registrations_are_retained(self):
        registrations = []
        registration = ModuleType('gymnasium.envs.registration')
        registration.registry = {}
        registration.register = lambda **values: registrations.append(values['id'])
        spec = importlib.util.spec_from_file_location('_integration_envs', ROOT / 'lewm_pcd/envs/__init__.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'gymnasium.envs.registration': registration}):
            spec.loader.exec_module(module)
        self.assertIn('PCDExpertPolicy', module.__all__)
        self.assertIn('PointCloudWrapper', module.__all__)
        self.assertEqual(set(registrations), {'swm/OGBCubePCD-v0', 'swm/OGBCubeRM75-v0', 'swm/OGBCubeRM75PCD-v0'})


if __name__ == '__main__':
    unittest.main()
