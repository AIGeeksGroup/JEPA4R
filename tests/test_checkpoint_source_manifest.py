"""The combined release must satisfy the existing top-level source gate."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lewm_pcd'))
from bench.ogbench.workflow import verify_sources


class CheckpointSourceManifestTests(unittest.TestCase):
    def test_current_release_passes_the_original_source_gate(self):
        self.assertGreater(verify_sources()['files'], 0)


if __name__ == '__main__':
    unittest.main()
