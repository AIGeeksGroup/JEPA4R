"""A Git source archive must include every runtime file named by its manifest."""
from pathlib import Path
import json
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SourcePackageTests(unittest.TestCase):
    def test_manifest_runtime_files_are_tracked(self):
        if not (ROOT / '.git').exists():
            self.skipTest('Git packaging check runs in the producing checkout')
        result = subprocess.run(['git', '-C', str(ROOT), 'ls-files'], capture_output=True,
                                text=True, encoding='utf-8', check=True)
        tracked = set(result.stdout.splitlines())
        manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
        missing = sorted(set(manifest['files']) - tracked)
        self.assertEqual(missing, [], 'Runtime source omitted from the Git archive')


if __name__ == '__main__':
    unittest.main()
