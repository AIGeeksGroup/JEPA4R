# Moved from OGBench-cube@4fbb0c6:lewm_pcd/scripts/checks/test_asset_identity.py (verbatim except: module names / paths
# of the moved code adapted; lines wrapped at 120 columns)
"""Trusted RH56 bundle identity; pure stdlib, no simulator or Torch import."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

FILE=Path(__file__).resolve().parents[2]/'jepa4r/envs/twins/rh56/asset_identity.py'
class BundleIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s = importlib.util.spec_from_file_location('tested_bundle_identity', FILE)
        cls.api = importlib.util.module_from_spec(s)
        s.loader.exec_module(cls.api)
    def fixture(self,root):
        repo = root / 'repo'
        bundle = root / 'bundle'
        manifest = repo / 'provenance/bundle.json'
        manifest.parent.mkdir(parents=True)
        bundle.mkdir()
        files=[]
        for name, raw in [
            ('env.py', b'environment'),
            ('ppo.py', b'policy'),
            ('experts/relocate_expert.pt', b'weights123'),
            ('assets/model.xml', b'<mujoco/>'),
        ]:
            p = bundle / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
            files.append({'relative_path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        manifest.write_text(json.dumps({'schema':'jepa4r-rh56-native-trusted-bundle-v1','files':files}))
        deps = repo / 'DEPENDENCIES.json'
        deps.write_text(
            json.dumps(
                {
                    'rh56_native_bundle': {
                        'manifest': 'provenance/bundle.json',
                        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        'schema': 'jepa4r-rh56-native-trusted-bundle-v1',
                        'file_count': 4,
                    }
                }
            )
        )
        return repo,bundle,deps,manifest
    def test_frozen_full_manifest_and_relocated_identical_bundle_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            repo, bundle, deps, manifest = self.fixture(root)
            a = self.api.verify_rh56_bundle(bundle, dependencies_path=deps)
            shutil.copytree(bundle, root / 'moved')
            b = self.api.verify_rh56_bundle(root / 'moved', dependencies_path=deps)
            self.assertEqual(a['verified_files'], 4)
            self.assertEqual(a['content_identity_sha256'], b['content_identity_sha256'])
    def test_same_size_changed_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            repo, bundle, deps, manifest = self.fixture(Path(d))
            (bundle / 'experts/relocate_expert.pt').write_bytes(b'weights456')
            with self.assertRaisesRegex(ValueError,'SHA'):self.api.verify_rh56_bundle(bundle,dependencies_path=deps)
    def test_changed_environment_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            repo,bundle,deps,manifest=self.fixture(Path(d));(bundle/'env.py').write_bytes(b'environmenX')
            with self.assertRaisesRegex(ValueError,'SHA'):self.api.verify_rh56_bundle(bundle,dependencies_path=deps)
    def test_manifest_self_tampering_cannot_supply_new_expected_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            repo, bundle, deps, manifest = self.fixture(Path(d))
            value = json.loads(manifest.read_text())
            value['files'][0]['sha256'] = '0' * 64
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'manifest SHA'):
                self.api.verify_rh56_bundle(bundle, dependencies_path=deps)
    def test_missing_asset_and_untrusted_manifest_path_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            repo,bundle,deps,manifest=self.fixture(Path(d));(bundle/'assets/model.xml').unlink()
            with self.assertRaises(FileNotFoundError):self.api.verify_rh56_bundle(bundle,dependencies_path=deps)
            x = json.loads(deps.read_text())
            x['rh56_native_bundle']['manifest'] = '../outside.json'
            deps.write_text(json.dumps(x))
            with self.assertRaisesRegex(ValueError, 'relative'):
                self.api.verify_rh56_bundle(bundle, dependencies_path=deps)
if __name__=='__main__':unittest.main()
