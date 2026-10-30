"""Code/catalog acceptance only: never instantiate an encoder or run a native job."""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


class TestWorkflowCatalog(unittest.TestCase):
    def api(self):
        name = 'bench.ogbench.test_workflows.catalog'
        package = importlib.util.find_spec('bench.ogbench.test_workflows')
        self.assertIsNotNone(package, 'The Test comparison workflow catalog is not included yet')
        return importlib.import_module(name)

    def test_matrix_is_24_visual_groups_with_tiny_replacing_lejepa(self):
        api = self.api()
        groups = api.load_catalog()['groups']
        comparison = [g for g in groups if g['matrix'] == 'test24']
        baseline = [g for g in groups if g['matrix'] == 'state_baseline']
        tiny = [g for g in comparison if g['model'] == 'lejepa']
        self.assertEqual(len(comparison), 24)
        self.assertEqual(len(baseline), 4)
        self.assertEqual({g['model'] for g in baseline}, {'state'})
        self.assertEqual(len(tiny), 4)
        self.assertEqual({g['model'] for g in comparison},
                         {'lewm_rgb', 'lewm_pcd', 'pldm', 'lejepa', 'visreg', 'vjepa'})
        self.assertEqual({g['cube_count'] for g in tiny}, {1, 2, 3, 4})
        self.assertEqual(len({g['id'] for g in groups}), 28)

    def test_only_tiny_lejepa_is_published(self):
        api = self.api()
        for logical, record in api.load_sources()['files'].items():
            for name in (logical, record['repository_path']):
                lower = name.lower()
                self.assertFalse('lejepa' in lower and any(part in lower for part in ('vitb', 'vit_b', 'vit-b', '768')), name)
        groups = [g for g in api.load_catalog()['groups'] if g['model'] == 'lejepa']
        self.assertEqual(len(groups), 4)
        for group in groups:
            self.assertEqual(group['matrix'], 'test24')
            self.assertEqual(group['input_contract']['latent_dim'], 512)

    def test_aliases_and_invalid_group_are_explicit(self):
        api = self.api()
        self.assertEqual(api.get_group('lewm2d', 2), api.get_group('lewm_rgb', 2))
        self.assertEqual(api.get_group('lewm3d', 4), api.get_group('lewm_pcd', 4))
        self.assertEqual(api.get_group('vjepa21_vitl1024', 3), api.get_group('vjepa', 3))
        with self.assertRaises(ValueError):
            api.get_group('unknown', 1)
        with self.assertRaises(ValueError):
            api.get_group('state', 0)

    def test_native_contracts_and_real_configs_exist(self):
        api = self.api()
        dimensions = {'state': 0, 'lewm_rgb': 192, 'lewm_pcd': 192,
                      'pldm': 512, 'visreg': 768, 'vjepa': 1024, 'lejepa': 512}
        for group in api.load_catalog()['groups']:
            with self.subTest(group=group['id']):
                n, d = group['cube_count'], dimensions[group['model']]
                contract = group['input_contract']
                self.assertEqual(contract['latent_dim'], d)
                self.assertEqual(contract['current_dim'], 19 + 9*n if d == 0 else d + 19)
                self.assertEqual(contract['goal_dim'], 3*n if d == 0 else d + 3*n)
                self.assertFalse(contract['zero_padding'])
                self.assertTrue(group['config_sources'])
                for source in group['config_sources']:
                    self.assertIsInstance(json.loads(api.read_source(source)), dict)
                self.assertFalse(group['formal_result_revalidated'])

    def test_source_snapshot_hashes_and_deduplication(self):
        api = self.api()
        report = api.verify_sources()
        self.assertEqual(report['hash_mismatches'], [])
        self.assertGreater(report['native_files'], 100)
        self.assertGreater(report['reused_v3_files'], 0)
        for forbidden in ('scratch', 'test_scratch', '__pycache__', 'node_modules'):
            self.assertFalse(any(forbidden in Path(name).parts for name in api.load_sources()['files']))

    def test_encoder_specs_and_bound_source_metadata_are_included(self):
        api = self.api()
        prefix = api.load_sources()['native_root']+'/'
        checked = 0
        for group in api.load_catalog()['groups']:
            config = json.loads(api.read_source(group['config_sources'][0]))
            binding = config.get('encoder_binding', {})
            if not binding.get('spec_path', '').startswith(prefix):
                continue
            logical = binding['spec_path'][len(prefix):]
            data = api.read_source(logical)
            self.assertEqual(hashlib.sha256(data).hexdigest(), binding['spec_sha256'])
            spec = json.loads(data)
            for path_key, hash_key in (('original_loader', 'original_loader_sha256'),
                                       ('original_config', 'original_config_sha256'),
                                       ('asset_manifest_path', 'asset_manifest_sha256')):
                if spec.get(path_key, '').startswith(prefix):
                    source = api.read_source(spec[path_key][len(prefix):])
                    self.assertEqual(hashlib.sha256(source).hexdigest(), spec[hash_key])
            checked += 1
        self.assertGreaterEqual(checked, 12)

    def test_plan_keeps_exact_recorded_native_phases(self):
        api = self.api()
        phase_count = 0
        for group in api.load_catalog()['groups']:
            plan = api.build_plan(group['model'], group['cube_count'])
            self.assertFalse(plan['executed'])
            self.assertEqual(plan['group'], group['id'])
            for stage in plan['stages'].values():
                self.assertIn(stage['status'], ('recorded', 'entrypoint_only', 'not_applicable'))
                for item in stage['recorded_phases']:
                    native = json.loads(api.read_source(item['source_plan']))
                    self.assertEqual(item['phase'], native['phases'][item['phase_index']])
                    phase_count += 1
        self.assertGreater(phase_count, 80)
        for model in ('state', 'vjepa', 'visreg', 'lewm_pcd', 'pldm', 'lejepa'):
            self.assertIn('gciql', api.build_plan(model, 4)['stages'])
        self.assertEqual(api.build_plan('state', 1)['stages']['encode']['status'], 'not_applicable')

    def test_materialization_is_exact_and_refuses_overwrite(self):
        api = self.api()
        materializer = importlib.import_module('bench.ogbench.test_workflows.materialize')
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)/'new_workspace'
            report = materializer.materialize(destination)
            self.assertEqual(report['files'], api.verify_sources()['native_files'])
            sources = api.load_sources()['files']
            for logical, record in sources.items():
                actual = hashlib.sha256(destination.joinpath(*logical.split('/')).read_bytes()).hexdigest()
                self.assertEqual(actual, record['sha256'], logical)
            with self.assertRaises(FileExistsError):
                materializer.materialize(destination)

    def test_source_path_traversal_and_mutation_fail_closed(self):
        api = self.api()
        for logical in ('../escape.py', '/absolute.py', 'C:/escape.py', r'..\escape.py'):
            with self.subTest(path=logical), self.assertRaises(ValueError):
                api.read_source(logical)
        with self.assertRaises(ValueError):
            api.build_plan('vjepa', 2, 'unknown-stage')
        sources = api.load_sources()
        logical = next(iter(sources['files']))
        sources['files'][logical]['sha256'] = '0'*64
        with mock.patch.object(api, 'load_sources', return_value=sources):
            with self.assertRaises(ValueError):
                api.read_source(logical)

    def test_inspector_does_not_import_torch_or_execute_native_commands(self):
        api = self.api()
        script = api.REPO/'lewm_pcd/scripts/inspect_test_workflows.py'
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-B', str(script), '--model', 'vjepa',
            '--cube-count', '4', '--stage', 'gciql'], capture_output=True, text=True, env=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertFalse(plan['executed'])
        self.assertEqual(plan['group'], 'vjepa_cube4')
        listed = subprocess.run([sys.executable, '-B', str(script), '--list'],
            capture_output=True, text=True, env=environment)
        self.assertEqual(listed.returncode, 0, listed.stderr)
        menu = json.loads(listed.stdout)
        self.assertEqual(len(menu['groups']), 24)
        self.assertEqual(len(menu['baselines']), 4)
        self.assertTrue(all(group['model'] == 'state' for group in menu['baselines']))
        check = subprocess.run([sys.executable, '-B', '-c',
            'import sys; from bench.ogbench.test_workflows.catalog import build_plan; '
            'build_plan("visreg", 3); assert "torch" not in sys.modules and "jax" not in sys.modules'],
            capture_output=True, text=True, env=environment)
        self.assertEqual(check.returncode, 0, check.stderr)
        rejection = subprocess.run([sys.executable, '-B', str(script), '--execute'],
            capture_output=True, text=True, env=environment)
        self.assertNotEqual(rejection.returncode, 0)


if __name__ == '__main__':
    unittest.main()
