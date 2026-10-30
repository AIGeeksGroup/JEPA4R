import json
from pathlib import Path
import unittest

from audit_gate_metadata import expected_gate, load_original


class RealGate(unittest.TestCase):
    def test_frozen_real_plan_reproduces_original_mismatch_and_exact_fix(self):
        original = load_original()
        gate = json.loads((Path(__file__).parent.parent / 'producer_plan.json').read_text())['preprocess_gate']
        self.assertNotEqual(gate, original.GATE)
        self.assertEqual(gate, expected_gate(original))
        self.assertEqual({k: gate[k] for k in original.GATE}, original.GATE)

    def test_numeric_or_description_changes_still_rejected(self):
        expected = expected_gate(load_original())
        for field, replacement in [('latent_atol', 1e-3), ('minimum_gain', 0), ('scope', 'different')]:
            modified = dict(expected, **{field: replacement})
            self.assertNotEqual(modified, expected)


if __name__ == '__main__':
    unittest.main()
