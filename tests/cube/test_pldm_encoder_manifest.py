# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tests/test_pldm_encoder_manifest.py (verbatim except: imports /
# paths adapted to the package; lines wrapped at 120 columns, unused imports removed)
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from jepa4r.baselines.cube import workflow_pldm_encoder as adapter


class PLDMManifestTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(adapter.__file__).parent/'pldm'
        self.vendor=Path(adapter.__file__).resolve().parents[3]/'third_party/pldm'
        manifest=Path(adapter.__file__).resolve().parents[3]/'configs/policy/gciql/pldm/source_manifest.json'
        self.manifest=json.loads(manifest.read_text())

    def test_current_real_upstream_manifest_and_all_import_sources_pass(self):
        bound=adapter.encoder_source_bindings(self.vendor)
        self.assertEqual(len(bound),len(adapter.IMPORT_FILES))
        for name,digest in bound.items():self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(),digest)

    def test_wrong_upstream_commit_rejects(self):
        wrong=copy.deepcopy(self.manifest);wrong['upstream_commit']='0'*40
        with patch.object(adapter.json,'loads',return_value=wrong),self.assertRaises(ValueError):
            adapter.encoder_source_bindings(self.vendor)

    def test_wrong_import_source_bytes_reject(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder)
            for name in adapter.IMPORT_FILES:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes((self.vendor/name).read_bytes())
            (root/adapter.IMPORT_FILES[-1]).write_bytes(b'# modified source\n')
            with self.assertRaises(ValueError):adapter.encoder_source_bindings(root)


if __name__=='__main__':unittest.main()
