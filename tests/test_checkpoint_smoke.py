"""Load-only release validation; no checkpoint, model or simulator is started."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/checkpoint_smoke.py'


class CheckpointSmokeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), 'load-only checkpoint interface is missing')
        spec = importlib.util.spec_from_file_location('checkpoint_smoke', SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        directory = self.root / 'cube_single/jepa4r'
        payload = directory / 'jepa4r/world_model/weights_epoch_10.pt'
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b'owned checkpoint fixture')
        item = dict(path='jepa4r/world_model/weights_epoch_10.pt', bytes=payload.stat().st_size,
                    sha256=hashlib.sha256(payload.read_bytes()).hexdigest())
        self.release = dict(task='cube_single', model='jepa4r', repo_id='AIGeeksGroup/JEPA4R',
                            revision='a' * 40, files=[item], world_model={'weights': item['path']}, policies=[])
        self.registry = self.root / 'registry.json'
        self.registry.write_text(json.dumps(dict(schema='jepa4r_released_checkpoints_v1', releases=[self.release])))
        self.bundle = dict(self.release, root=str(self.root), directory=str(directory), verified=True,
                           files=[dict(item, local_path=str(payload))])

    def verify(self, bundle=None):
        return self.module.verify_bundle(bundle or self.bundle, self.registry)

    def test_help_uses_only_standard_library(self):
        result = subprocess.run([sys.executable, '-S', str(SCRIPT), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--component', result.stdout)
        self.assertIn('--bundle', result.stdout)

    def test_valid_bundle_is_rehashed(self):
        release, files = self.verify()
        self.assertEqual(release['revision'], 'a' * 40)
        self.assertEqual(files[self.release['files'][0]['path']].read_bytes(), b'owned checkpoint fixture')

    def test_verified_true_cannot_hide_corruption(self):
        Path(self.bundle['files'][0]['local_path']).write_bytes(b'x' * self.bundle['files'][0]['bytes'])
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.verify()

    def test_bundle_cannot_replace_pinned_revision_or_task(self):
        for key, value in [('revision', 'b' * 40), ('task', 'cube_double'), ('model', 'lewm')]:
            with self.subTest(key=key):
                bundle = copy.deepcopy(self.bundle)
                bundle[key] = value
                with self.assertRaises(ValueError):
                    self.verify(bundle)

    def test_bundle_cannot_replace_hash_or_omit_files(self):
        for files in [[], [dict(self.bundle['files'][0], sha256='b' * 64)]]:
            bundle = dict(self.bundle, files=files)
            with self.assertRaises(ValueError):
                self.verify(bundle)

    def test_paths_must_match_task_model_directory(self):
        bundle = copy.deepcopy(self.bundle)
        bundle['files'][0]['local_path'] = str(self.root / 'elsewhere.pt')
        with self.assertRaisesRegex(ValueError, 'path'):
            self.verify(bundle)

    def test_registry_rejects_traversal_and_mutable_revision(self):
        for key, value in [('revision', 'main'), ('files', [dict(self.release['files'][0], path='../escape.pt')])]:
            changed = dict(self.release, **{key: value})
            self.registry.write_text(json.dumps(dict(schema='jepa4r_released_checkpoints_v1', releases=[changed])))
            bundle = dict(self.bundle, **{key: value})
            with self.assertRaises(ValueError):
                self.verify(bundle)

    def test_main_rejects_corruption_before_loading_and_records_failure(self):
        bundle = self.root / 'bundle.json'
        bundle.write_text(json.dumps(self.bundle))
        Path(self.bundle['files'][0]['local_path']).write_bytes(b'corrupt')
        output = self.root / 'failure.json'
        with mock.patch.object(self.module, 'REGISTRY', self.registry), mock.patch.object(self.module, 'load_wm') as loader:
            code = self.module.main(['--bundle', str(bundle), '--component', 'wm', '--output', str(output)])
        self.assertEqual(code, 1)
        loader.assert_not_called()
        report = json.loads(output.read_text())
        self.assertEqual(report['actual_exit_code'], 1)
        self.assertEqual(report['environment_steps'], 0)
        self.assertEqual(report['optimizer_updates'], 0)

    def test_existing_output_is_preserved_before_loading(self):
        output = self.root / 'old.json'
        output.write_text('historical receipt')
        with self.assertRaises(FileExistsError):
            self.module.main(['--bundle', str(self.root / 'missing.json'), '--component', 'wm', '--output', str(output)])
        self.assertEqual(output.read_text(), 'historical receipt')

    def test_all_four_original_policy_metadata_shapes(self):
        self.assertTrue(hasattr(self.module, 'policy_config'), 'original policy metadata adapter is missing')
        for cubes in range(1, 5):
            config = dict(cube_count=cubes, modality='pcd', seed=0, total_updates=1000000,
                          input_contract=dict(current_dim=211, goal_dim=192 + 3 * cubes, zero_padding=False))
            binding = dict(config=config, agent_config={'agent_name': 'gciql'})
            metadata = dict(binding=binding) if cubes < 4 else dict(config, agent_config=binding['agent_config'])
            if cubes == 4:
                binding = dict(config=metadata, agent_config=metadata['agent_config'])
            sidecar = dict(network_step=1000001, adam_count=1000000, binding=binding)
            self.assertEqual(self.module.policy_config(metadata, sidecar, cubes), {'agent_name': 'gciql'})

    def test_policy_metadata_rejects_wrong_cube_shape_and_counter(self):
        self.assertTrue(hasattr(self.module, 'policy_config'), 'original policy metadata adapter is missing')
        config = dict(cube_count=4, modality='pcd', seed=0, total_updates=1000000,
                      input_contract=dict(current_dim=211, goal_dim=204, zero_padding=False))
        binding = dict(config=config, agent_config={'agent_name': 'gciql'})
        metadata = dict(binding=binding)
        sidecar = dict(network_step=1000001, adam_count=1000000, binding=binding)
        with self.assertRaisesRegex(ValueError, 'contract'):
            self.module.policy_config(metadata, sidecar, 3)
        sidecar['adam_count'] = 999999
        with self.assertRaisesRegex(ValueError, 'counter'):
            self.module.policy_config(metadata, sidecar, 4)


if __name__ == '__main__':
    unittest.main()
