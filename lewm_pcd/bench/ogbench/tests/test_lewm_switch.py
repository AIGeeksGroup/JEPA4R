"""LeWM aliases extend command planning without changing accepted old commands."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[4]
LEWM = REPO / 'lewm_pcd'
ENTRY = LEWM / 'scripts/run_ogbench.py'
ALIASES = {'lewm2d': 'lewm_rgb', 'lewm3d': 'lewm_pcd'}
CLOSED_PROFILES = ('double500', 'triple_front600', 'quadruple600')


def reference_code_root(repo):
    sibling = repo.with_name('JEPA4R-OGBench-Unified-20261025-v1') / 'lewm_pcd'
    return sibling if sibling.is_dir() else repo / 'evidence/unified_v1/command_source/lewm_pcd'


V1 = reference_code_root(REPO)

# Fail immediately if a plan attempts to load a numerical or training framework.
PLAN_SCRIPT = r'''
import importlib.abc
import runpy
import sys
class NoFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'hydra', 'lightning', 'stable_pretraining', 'stable_worldmodel'}:
            raise AssertionError('plan imported training framework: ' + fullname)
sys.meta_path.insert(0, NoFrameworks())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''


def isolated(code, root, *arguments):
    return subprocess.run([sys.executable, '-I', '-B', '-c', code, str(root), *arguments],
                          capture_output=True, text=True)


class LeWMSwitchTests(unittest.TestCase):
    def call(self, *arguments):
        return isolated(PLAN_SCRIPT, ENTRY, *map(str, arguments))

    def plan(self, *arguments):
        result = self.call(*arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_list_keeps_old_methods_and_adds_public_aliases(self):
        value = self.plan('--list')
        self.assertTrue({'pldm', 'lejepa', 'state', 'lewm_rgb', 'lewm_pcd',
                         'lewm2d', 'lewm3d'} <= set(value['methods']))
        self.assertNotIn('lejepa_vitb', value['methods'])

    def test_closed_lewm_train_plans_are_framework_free_and_create_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            for profile in CLOSED_PROFILES:
                for requested, canonical in ALIASES.items():
                    with self.subTest(profile=profile, method=requested):
                        output = Path(temporary) / (profile + '-' + requested)
                        value = self.plan('--profile', profile, '--method', requested,
                                          '--stage', 'train', '--data-root', temporary,
                                          '--output-root', output, '--workers', '2', '--prefetch', '1')
                        self.assertFalse(value['execute'])
                        self.assertEqual(value['requested_method'], requested)
                        self.assertEqual(value['method'], canonical)
                        command = value['command']
                        self.assertEqual(command[1:3], ['-m', 'bench.ogbench.lewm'])
                        self.assertEqual(command[command.index('--profile') + 1], profile)
                        self.assertEqual(command[command.index('--mode') + 1],
                                         '3d' if canonical == 'lewm_pcd' else '2d')
                        self.assertEqual(command[command.index('--data-root') + 1],
                                         str(Path(temporary).resolve()))
                        self.assertEqual(command[command.index('--output') + 1], str(output.resolve()))
                        self.assertEqual(command[command.index('--workers') + 1], '2')
                        self.assertEqual(command[command.index('--prefetch') + 1], '1')
                        self.assertFalse(output.exists())
                        self.assertFalse(output.with_name(output.name + '.dispatch').exists())

    def test_old_lewm_names_produce_the_same_training_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            for requested, canonical in ALIASES.items():
                common = ('--profile', 'double500', '--stage', 'train', '--data-root', temporary,
                          '--output-root', Path(temporary) / 'uncreated')
                alias = self.plan(*common, '--method', requested)
                old_name = self.plan(*common, '--method', canonical)
                self.assertEqual(alias['command'], old_name['command'])
                self.assertEqual(old_name['requested_method'], canonical)
                self.assertEqual(alias['environment'], old_name['environment'])

    def test_aliases_are_canonical_for_existing_downstream_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            for profile in ('single200', *CLOSED_PROFILES):
                for requested, canonical in ALIASES.items():
                    for stage in ('gciql', 'native-sr'):
                        with self.subTest(profile=profile, method=requested, stage=stage):
                            output = Path(temporary) / 'uncreated'
                            value = self.plan('--profile', profile, '--method', requested, '--stage', stage,
                                              '--data-root', temporary, '--output-root', output,
                                              '--config', Path(temporary) / 'frozen-recipe.json')
                            self.assertEqual(value['requested_method'], requested)
                            self.assertEqual(value['method'], canonical)
                            command = value['command']
                            self.assertEqual(command[command.index('--method') + 1], canonical)
                            self.assertFalse(output.exists())

    def test_downstream_recipe_does_not_claim_a_fixed_training_encoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            for requested in ALIASES:
                for stage in ('gciql', 'native-sr'):
                    with self.subTest(method=requested, stage=stage):
                        value = self.plan('--profile', 'double500', '--method', requested,
                                          '--stage', stage, '--data-root', temporary,
                                          '--output-root', Path(temporary) / 'uncreated',
                                          '--config', Path(temporary) / 'checkpoint-bound-recipe.json')
                        self.assertIsNone(value['lewm_recipe'],
                                          'downstream encoder identity comes from its frozen configuration')

    def test_3d_alias_reaches_the_existing_pcd_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            value = self.plan('--profile', 'double500', '--method', 'lewm3d', '--stage', 'native-sr',
                              '--data-root', temporary, '--output-root', Path(temporary) / 'uncreated',
                              '--config', Path(temporary) / 'recipe.json')
        code = r'''
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from bench.ogbench.contracts import get_profile
from bench.ogbench.downstream import bind_recipe
path = Path(sys.argv[1]) / 'bench/ogbench/tests/test_downstream.py'
spec = importlib.util.spec_from_file_location('_switch_binding_fixture', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
bound = bind_recipe(get_profile('double500'), sys.argv[2], module.recipe(2, 'lewm_pcd'))
print(json.dumps({key: bound[key] for key in ('method', 'encoder_family', 'latent_dim', 'modality', 'input_contract')}))
'''
        result = isolated(code, LEWM, value['method'])
        self.assertEqual(result.returncode, 0, result.stderr)
        bound = json.loads(result.stdout)
        self.assertEqual(bound['method'], 'lewm_pcd')
        self.assertEqual(bound['modality'], 'pcd')
        self.assertEqual(bound['encoder_family'], 'lewm')
        self.assertEqual(bound['latent_dim'], 192)
        self.assertEqual(bound['input_contract'], dict(current_dim=211, goal_dim=198,
                                                      latent_dim=192, zero_padding=False))

    def test_single_train_automatically_selects_original_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'uncreated'
            for method in (*ALIASES, *ALIASES.values()):
                with self.subTest(method=method):
                    value = self.plan('--profile', 'single200', '--method', method,
                                      '--stage', 'train', '--data-root', temporary,
                                      '--output-root', output)
                    self.assertIn('--single-original-protocol', value['command'])
                    self.assertFalse(value['execute'])
                    self.assertFalse(output.exists())
                    self.assertFalse(output.with_name(output.name + '.dispatch').exists())

    def test_official_source_and_unknown_profile_fail_before_launch(self):
        profiles = json.loads((LEWM / 'bench/ogbench/profiles.json').read_text(encoding='utf8'))
        official = [name for name, profile in profiles.items() if profile['source'] != 'self_collected']
        self.assertTrue(official, 'the separate official Play profiles must remain registered')
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'uncreated'
            for profile in (*official, 'unregistered_cube'):
                for method in (*ALIASES, *ALIASES.values()):
                    with self.subTest(profile=profile, method=method):
                        result = self.call('--profile', profile, '--method', method, '--stage', 'train',
                                           '--data-root', temporary, '--output-root', output)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertNotIn('plan imported training framework', result.stderr)
                        self.assertNotIn('invalid choice', result.stderr)
                        self.assertFalse(output.exists())
                        self.assertFalse(output.with_name(output.name + '.dispatch').exists())

    def test_cube_mismatch_rejected_for_lewm_before_launch(self):
        result = self.call('--profile', 'double500', '--cube-count', '3', '--method', 'lewm3d',
                           '--stage', 'train', '--data-root', '.', '--output-root', 'uncreated')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('cube count disagrees', result.stderr)

    @unittest.skipUnless(V1.is_dir(), 'immutable v1 reference is required for historical command comparison')
    def test_packaged_reference_fallback_imports_without_the_large_v1_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / 'unpacked-v2'
            packaged = repo / 'evidence/unified_v1/command_source/lewm_pcd'
            destination = packaged / 'bench/ogbench'
            destination.mkdir(parents=True)
            for name in ('training.py', 'contracts.py', 'profiles.json', '__init__.py'):
                shutil.copyfile(V1 / 'bench/ogbench' / name, destination / name)
            self.assertFalse(repo.with_name('JEPA4R-OGBench-Unified-20261025-v1').exists())
            self.assertEqual(reference_code_root(repo), packaged)
            code = r'''
import json
import sys
sys.path.insert(0, sys.argv[1])
from bench.ogbench import training
print(json.dumps(training.build_command('double500', 'pldm', 'sealed', 'fresh', python='PY')))
'''
            result = isolated(code, reference_code_root(repo))
            self.assertEqual(result.returncode, 0, result.stderr)
            command = json.loads(result.stdout)
            self.assertEqual(command[:3], ['PY', '-m', 'bench.ogbench.pldm.train'])
            self.assertEqual(command[command.index('--profile') + 1], 'double500')

    @unittest.skipUnless(V1.is_dir(), 'immutable v1 reference is required for historical command comparison')
    def test_pldm_tiny_commands_and_environment_match_v1_in_separate_processes(self):
        code = r'''
import json
import os
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from bench.ogbench import training
commands = []
for profile in ('single200', 'double500', 'triple_front600', 'quadruple600'):
    for method in ('pldm', 'lejepa'):
        for workers, prefetch in ((None, None), (2, 1)):
            commands.append(training.build_command(profile, method, 'sealed', 'fresh',
                                                    workers=workers, prefetch=prefetch, python='PY'))
environment = training.build_environment('sealed', 'fresh', environ={'PYTHONPATH': 'prior', 'KEEP': 'unchanged'})
environment['PYTHONPATH'] = os.pathsep.join(path.replace(str(root), '<CODE>')
                                           for path in environment['PYTHONPATH'].split(os.pathsep))
print(json.dumps(dict(commands=commands, environment=environment)))
'''
        old = isolated(code, V1)
        current = isolated(code, LEWM)
        self.assertEqual(old.returncode, 0, old.stderr)
        self.assertEqual(current.returncode, 0, current.stderr)
        self.assertEqual(json.loads(old.stdout), json.loads(current.stdout))


if __name__ == '__main__':
    unittest.main()
