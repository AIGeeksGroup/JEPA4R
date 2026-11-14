import hashlib
from pathlib import Path
import tempfile
import unittest
from jepa4r.baselines.cube import workflow_pldm_encoder as adapter


class PLDMSourceTests(unittest.TestCase):
    def setUp(self):
        self.vendor = Path(adapter.__file__).resolve().parents[3] / "third_party/pldm"

    def test_all_import_sources_are_bound_by_their_bytes(self):
        bound = adapter.encoder_source_bindings(self.vendor)
        self.assertEqual(len(bound), len(adapter.IMPORT_FILES))
        for name, digest in bound.items():
            self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(), digest)

    def test_missing_import_source_rejects(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in adapter.IMPORT_FILES[:-1]:
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes((self.vendor / name).read_bytes())
            with self.assertRaises(FileNotFoundError):
                adapter.encoder_source_bindings(root)


if __name__ == "__main__":
    unittest.main()
