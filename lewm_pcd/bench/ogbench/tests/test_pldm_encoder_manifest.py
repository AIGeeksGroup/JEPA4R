import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from bench.ogbench import workflow_pldm_encoder as adapter


class PLDMManifestTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(adapter.__file__).parent/'pldm'
        self.manifest=json.loads((self.root/'source_manifest.json').read_text())

    def test_current_real_upstream_manifest_and_all_import_sources_pass(self):
        bound=adapter.encoder_source_bindings(self.root/'vendor/source')
        self.assertEqual(len(bound),len(adapter.IMPORT_FILES))
        for name,digest in bound.items():self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(),digest)

    def test_wrong_upstream_commit_rejects(self):
        wrong=copy.deepcopy(self.manifest);wrong['upstream_commit']='0'*40
        with patch.object(adapter.json,'loads',return_value=wrong),self.assertRaises(ValueError):
            adapter.encoder_source_bindings(self.root/'vendor/source')

    def test_wrong_import_source_bytes_reject(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder)
            for name in adapter.IMPORT_FILES:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes((self.root/'vendor/source'/name).read_bytes())
            (root/adapter.IMPORT_FILES[-1]).write_bytes(b'# modified source\n')
            with self.assertRaises(ValueError):adapter.encoder_source_bindings(root)


if __name__=='__main__':unittest.main()
