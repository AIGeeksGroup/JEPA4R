# Moved from OGBench-cube@4fbb0c6:tests/test_runtime_source_identity.py (verbatim except: module names / paths of the
# moved code adapted; lines wrapped at 120 columns, unused imports removed)
"""Release source identities must accept the checkout and reject changed bytes."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from jepa4r.models.cube import training_runtime
from jepa4r.policy.gciql.downstream import _source


class RuntimeSourceIdentityTests(unittest.TestCase):
    def test_downstream_runtime_accepts_current_release(self):
        self.assertTrue(_source.verify_sources()['files'])

    def test_world_model_runtime_accepts_current_release(self):
        self.assertEqual(training_runtime.verify_sources()['mismatches'], [])

    def test_downstream_runtime_rejects_changed_boundary(self):
        original = _source.source_path
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / 'boundary.py'
            changed.write_bytes(original('boundary.py').read_bytes() + b'\n# changed\n')
            with patch.object(
                _source, 'source_path', side_effect=lambda name: changed if name == 'boundary.py' else original(name)
            ):
                with self.assertRaisesRegex(ValueError, 'Current source changed: boundary.py'):
                    _source.verify_sources()


if __name__ == '__main__':
    unittest.main()
