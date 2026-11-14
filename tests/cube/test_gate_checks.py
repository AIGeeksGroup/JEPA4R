"""Smoke-only gate entry plans are read-only and bounded before heavy imports."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
import io

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa4r.policy.gciql import gate_checks


class GatePlanTests(unittest.TestCase):
    def test_plan_is_bounded_and_does_not_import_frameworks_or_make_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(dict(mode='smoke', model='lejepa', stop_after=2)))
            output = Path(folder) / 'new'
            plan = gate_checks.build_plan('tiny-step', path, output)
            self.assertEqual(plan['maximum_optimizer_updates'], 2)
            self.assertFalse(output.exists())
            self.assertNotIn('torch', gate_checks.__dict__)
            self.assertNotIn('jax', gate_checks.__dict__)

    def test_formal_or_unbounded_gate_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            for cfg in [dict(mode='formal', stop_after=2), dict(mode='smoke', stop_after=3)]:
                path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):
                    gate_checks.build_plan('tiny-step', path, Path(folder) / 'new')

    def test_module_entry_restores_original_source_runtime_paths_for_plan(self):
        from jepa4r.models.cube.training_runtime import code_root
        original = sys.path.copy()
        swm = str(code_root() / 'third_party/stable-worldmodel')
        sys.path[:] = [p for p in sys.path if p != swm]
        try:
            with tempfile.TemporaryDirectory() as folder:
                cfg = Path(folder) / 'config.json'
                cfg.write_text(json.dumps(dict(mode='smoke', model='lewm_pcd', stop_after=1)))
                output = Path(folder) / 'new'
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(gate_checks.main(['--stage','model-numerics','--config',str(cfg),
                                                       '--output',str(output)]), 0)
                self.assertIn(swm, sys.path)
                self.assertFalse(output.exists())
        finally:
            sys.path[:] = original


if __name__ == '__main__':
    unittest.main()
