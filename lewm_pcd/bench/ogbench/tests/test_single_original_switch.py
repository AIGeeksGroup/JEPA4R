"""Single original-protocol planning and immutable v2 command regressions."""
import hashlib
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
REFERENCE_FILES = (
    '__init__.py', 'training.py', 'contracts.py', 'profiles.json', 'lewm.py',
    'lewm_data.py', 'lewm_data_bindings.json', 'lewm_native/2d.json', 'lewm_native/3d.json',
)


def reference_code_root(repo):
    sibling = repo.with_name('JEPA4R-OGBench-Unified-20261025-v2') / 'lewm_pcd'
    return sibling if sibling.is_dir() else repo / 'evidence/unified_v2/command_source/lewm_pcd'


V2 = reference_code_root(REPO)
PACKAGED_V2 = REPO / 'evidence/unified_v2/command_source/lewm_pcd'

PLAN_SCRIPT = r'''
import builtins
import importlib.abc
import io
from pathlib import Path
import runpy
import sys
class NoTrainingOrDataLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'h5py', 'hydra', 'omegaconf',
                                    'lightning', 'stable_pretraining', 'stable_worldmodel'}:
            raise AssertionError('plan imported data/training library: ' + fullname)
sys.meta_path.insert(0, NoTrainingOrDataLibraries())
def guard(original):
    def checked(file, *args, **kwargs):
        if isinstance(file, (str, bytes, Path)) and str(file).lower().endswith(('.h5', '.hdf5', '.npz')):
            raise AssertionError('plan opened input dataset: ' + str(file))
        return original(file, *args, **kwargs)
    return checked
builtins.open, io.open = guard(builtins.open), guard(io.open)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''

COMMAND_SCRIPT = r'''
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
for profile in ('double500', 'triple_front600', 'quadruple600'):
    for method in ('lewm_rgb', 'lewm_pcd'):
        for workers, prefetch in ((None, None), (0, None), (2, 1)):
            commands.append(training.build_command(profile, method, 'sealed', 'fresh',
                            workers=workers, prefetch=prefetch, python='PY'))
environment = training.build_environment('sealed', 'fresh',
                                        environ={'PYTHONPATH': 'prior', 'KEEP': 'unchanged'})
environment['PYTHONPATH'] = os.pathsep.join(path.replace(str(root), '<CODE>')
                                           for path in environment['PYTHONPATH'].split(os.pathsep))
print(json.dumps(dict(commands=commands, environment=environment)))
'''


def isolated(code, root, *arguments):
    return subprocess.run([sys.executable, '-I', '-B', '-c', code, str(root), *map(str, arguments)],
                          capture_output=True, text=True)


class SingleOriginalSwitchTests(unittest.TestCase):
    def plan(self, *arguments):
        result = isolated(PLAN_SCRIPT, ENTRY, *arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def command_matrix(self, root):
        result = isolated(COMMAND_SCRIPT, root)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_single_aliases_and_canonical_names_plan_without_input_io_or_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data = base / 'ogbench-train-20261003/datasets/ogbench'
            output = base / 'uncreated'
            for requested, canonical in ALIASES.items():
                with self.subTest(method=requested):
                    common = ('--profile', 'single200', '--cube-count', '1', '--stage', 'train',
                              '--data-root', data, '--output-root', output,
                              '--workers', '2', '--prefetch', '1')
                    alias = self.plan(*common, '--method', requested)
                    canonical_plan = self.plan(*common, '--method', canonical)
                    self.assertEqual(alias['command'], canonical_plan['command'])
                    self.assertEqual(alias['environment'], canonical_plan['environment'])
                    self.assertEqual(alias['requested_method'], requested)
                    self.assertEqual(alias['method'], canonical)
                    command = alias['command']
                    self.assertEqual(command[1:3], ['-m', 'bench.ogbench.lewm'])
                    self.assertEqual(command.count('--single-original-protocol'), 1)
                    self.assertEqual(command[command.index('--mode') + 1],
                                     '2d' if canonical == 'lewm_rgb' else '3d')
                    self.assertEqual(command[command.index('--data-root') + 1], str(data.resolve()))
                    self.assertEqual(command[command.index('--output') + 1], str(output.resolve()))
                    self.assertEqual(command[command.index('--workers') + 1], '2')
                    self.assertEqual(command[command.index('--prefetch') + 1], '1')
                    self.assertFalse(alias['execute'])
                    self.assertEqual(list(base.iterdir()), [])

    def test_single_cube_mismatch_rejects_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'uncreated'
            result = isolated(PLAN_SCRIPT, ENTRY, '--profile', 'single200', '--cube-count', '2',
                              '--method', 'lewm3d', '--stage', 'train', '--data-root', temporary,
                              '--output-root', output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('cube count disagrees', result.stderr)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_plan_identifies_actual_wm_binding_separately_from_cube_task_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / 'ogbench-train-20261003/datasets/ogbench'
            for method in ALIASES:
                with self.subTest(method=method):
                    value = self.plan('--profile', 'single200', '--method', method, '--stage', 'train',
                                      '--data-root', data, '--output-root', Path(temporary) / 'uncreated')
                    binding = value['wm_training_data_binding']
                    self.assertEqual(value['profile']['name'], 'single200')
                    self.assertEqual(binding['protocol'], 'single_original_random_split_v1')
                    self.assertEqual(binding['seed'], 3072)
                    self.assertEqual(binding['split_scope'], 'window')
                    self.assertEqual(binding['train_fraction'], .9)
                    self.assertEqual(binding['validation_fraction'], 1 - .9)
                    self.assertEqual(binding['manager_seed_argument'], 'omitted')
                    self.assertIs(binding['global_seed_before_model'], False)
                    self.assertTrue(Path(binding['dataset_path']).is_relative_to(data.resolve()))
                    self.assertTrue({'action', 'observation'} <= set(binding['keys_to_load']))
                    self.assertNotIn('fresh_seed', binding)
                    self.assertNotIn('data_views', binding['dataset_path'])

    def test_single_original_protocol_does_not_leak_into_other_training_plans(self):
        with tempfile.TemporaryDirectory() as temporary:
            for profile, methods in [('single200', ('pldm', 'lejepa')),
                                     *((profile, tuple(ALIASES)) for profile in CLOSED_PROFILES)]:
                for method in methods:
                    with self.subTest(profile=profile, method=method):
                        value = self.plan('--profile', profile, '--method', method, '--stage', 'train',
                                          '--data-root', temporary,
                                          '--output-root', Path(temporary) / 'uncreated')
                        self.assertNotIn('--single-original-protocol', value['command'])
                        binding = value.get('wm_training_data_binding') or {}
                        self.assertNotEqual(binding.get('protocol'), 'single_original_random_split_v1')

    def test_existing_pldm_tiny_and_closed_lewm_commands_and_environment_match_v2(self):
        self.assertTrue(V2.is_dir(), 'immutable v2 command reference must be packaged')
        self.assertEqual(self.command_matrix(V2), self.command_matrix(LEWM))

    def test_v2_reference_fallback_works_without_a_sibling_checkout(self):
        self.assertTrue(PACKAGED_V2.is_dir(), 'immutable v2 command reference must be packaged')
        manifest = json.loads((REPO / 'evidence/unified_v2/UNIFIED_SOURCE_MANIFEST.json').read_text(encoding='utf8'))
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / 'unpacked-v3'
            packaged = repo / 'evidence/unified_v2/command_source/lewm_pcd'
            for name in REFERENCE_FILES:
                source = PACKAGED_V2 / 'bench/ogbench' / name
                self.assertTrue(source.is_file(), 'missing immutable v2 reference: ' + name)
                expected = manifest['files']['lewm_pcd/bench/ogbench/' + name]
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), expected,
                                 'packaged command reference differs from immutable v2: ' + name)
                destination = packaged / 'bench/ogbench' / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            self.assertFalse(repo.with_name('JEPA4R-OGBench-Unified-20261025-v2').exists())
            self.assertEqual(reference_code_root(repo), packaged)
            self.assertEqual(self.command_matrix(packaged), self.command_matrix(LEWM))


if __name__ == '__main__':
    unittest.main()
