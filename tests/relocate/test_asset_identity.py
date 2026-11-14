"""RH56 bundle check against its file list; pure stdlib, no simulator or Torch import."""

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

FILE = Path(__file__).resolve().parents[2] / "jepa4r/envs/twins/rh56/asset_identity.py"


class BundleIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s = importlib.util.spec_from_file_location("tested_bundle_identity", FILE)
        cls.api = importlib.util.module_from_spec(s)
        s.loader.exec_module(cls.api)

    def fixture(self, root):
        bundle = root / "bundle"
        manifest = root / "bundle.json"
        bundle.mkdir()
        files = []
        for name, raw in [
            ("env.py", b"environment"),
            ("ppo.py", b"policy"),
            ("experts/relocate_expert.pt", b"weights123"),
            ("assets/model.xml", b"<mujoco/>"),
        ]:
            p = bundle / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
            files.append({"relative_path": name, "bytes": len(raw)})
        manifest.write_text(json.dumps({"schema": "jepa4r-rh56-native-bundle-v1", "file_count": 4, "files": files}))
        return bundle, manifest

    def test_full_list_and_relocated_identical_bundle_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bundle, manifest = self.fixture(root)
            a = self.api.verify_rh56_bundle(bundle, manifest_path=manifest)
            shutil.copytree(bundle, root / "moved")
            b = self.api.verify_rh56_bundle(root / "moved", manifest_path=manifest)
            self.assertEqual(a["verified_files"], 4)
            self.assertEqual(a["content_identity_sha256"], b["content_identity_sha256"])

    def test_changed_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            bundle, manifest = self.fixture(Path(d))
            (bundle / "experts/relocate_expert.pt").write_bytes(b"weights")
            with self.assertRaisesRegex(ValueError, "size"):
                self.api.verify_rh56_bundle(bundle, manifest_path=manifest)

    def test_missing_asset_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            bundle, manifest = self.fixture(Path(d))
            (bundle / "assets/model.xml").unlink()
            with self.assertRaises(FileNotFoundError):
                self.api.verify_rh56_bundle(bundle, manifest_path=manifest)

    def test_list_without_the_expert_or_with_a_wrong_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            bundle, manifest = self.fixture(Path(d))
            value = json.loads(manifest.read_text())
            value["file_count"] = 5
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "count"):
                self.api.verify_rh56_bundle(bundle, manifest_path=manifest)
            value["files"] = [f for f in value["files"] if f["relative_path"] != "ppo.py"]
            value["file_count"] = 3
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "lacks required"):
                self.api.verify_rh56_bundle(bundle, manifest_path=manifest)

    def test_unsafe_list_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            bundle, manifest = self.fixture(Path(d))
            value = json.loads(manifest.read_text())
            value["files"][0]["relative_path"] = "../outside.py"
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "relative"):
                self.api.verify_rh56_bundle(bundle, manifest_path=manifest)


if __name__ == "__main__":
    unittest.main()
