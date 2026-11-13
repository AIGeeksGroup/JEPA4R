# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tests/test_retained_training.py (verbatim except: imports /
# paths adapted to the package; lines wrapped at 120 columns, unused imports removed)
"""Content aliases preserve original provenance and reject unmapped code."""
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from jepa4r.models.cube import retained_training as retained


class AliasTests(unittest.TestCase):
    def test_real_byte_aliases_retain_old_paths_and_missing_metadata(self):
        with tempfile.TemporaryDirectory()as folder:
            code=Path(folder)/'old_code.py';code.write_text('x=1\n')
            original={'/old/code.py':retained.sha(code),
                '/old/lewm_pcd/third_party/stable-worldmodel/LICENSE':retained.MISSING_METADATA['LICENSE']}
            mapping,missing=retained.content_aliases(original,[code])
            self.assertEqual(mapping['/old/code.py'],dict(path=str(code.resolve()),sha256=retained.sha(code)))
            self.assertEqual(
                missing,
                [
                    dict(
                        original_path='/old/lewm_pcd/third_party/stable-worldmodel/LICENSE',
                        sha256=retained.MISSING_METADATA['LICENSE'],
                        scope='unavailable original non-executable metadata',
                    )
                ],
            )
            self.assertIn('/old/code.py',original)

    def test_unavailable_python_or_unknown_metadata_is_rejected(self):
        for name in ['/old/missing.py','/old/other.json','/old/LICENSE']:
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'unmapped original producer'):
                retained.content_aliases({name:'a'*64},[])

    def test_wrong_expected_license_bytes_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'unmapped original producer'):
            retained.content_aliases({'/old/lewm_pcd/third_party/stable-worldmodel/LICENSE':'b'*64},[])


if __name__=='__main__':unittest.main()
