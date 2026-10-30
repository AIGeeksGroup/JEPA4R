"""Two CPU-only release checks; no model, renderer, benchmark or producer call."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_quad_release_tested', HERE / 'profile_policy.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class ProfilePolicyTests(unittest.TestCase):
    def inputs(self):
        decision = json.loads((HERE / 'release_decision.json').read_bytes())
        report_path = HERE / 'original_GPU_report_reference.json'
        self.assertEqual(hashlib.sha256(report_path.read_bytes()).hexdigest(), decision['report_sha256'])
        return decision, json.loads(report_path.read_bytes())

    def test_exact_positive_gain_retains_failed_old_gate_and_original_function_AST(self):
        decision, report = self.inputs()
        before = copy.deepcopy(report)
        policy.check_report(decision, report)
        self.assertEqual(report, before)
        self.assertFalse(report['accepted'])
        self.assertFalse(decision['original_engineering_5percent_gate_passed'])
        self.assertGreater(decision['positive_gain_fraction'], 0)
        self.assertLess(decision['positive_gain_fraction'], .05)
        plan = dict(source_pins={'official_state_3d_cache_v1/produce.py':
            '2dde0d8302061cba76edd384b9f1c825e41629c80ae327c4c962545e817aa43c'})
        profile = policy.profile_from_plan(plan, {'original_report_sha256': decision['report_sha256']})
        self.assertEqual((profile['renderer_workers'], profile['prefetch_jobs'], profile['affinity_cores']), (16, 32, 16))
        self.assertTrue(profile['GPU_preprocess'])
        default = HERE.parent / 'pcd_cache_adapter_v1/effective_sources_v1/official_state_3d_cache_v1/acceleration_v1/v4/run_profile.py'
        original = Path(os.environ.get('QUAD_V4_WRAPPER_SOURCE', str(default))).read_text()
        effective, edits = policy.policy_source(original)
        self.assertEqual(len(edits), 2)
        def functions(text):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body
                    if isinstance(n, ast.FunctionDef) and n.name != 'main'}
        self.assertEqual(functions(original), functions(effective))

    def test_wrong_report_identity_or_numeric_mismatch_is_rejected(self):
        decision, report = self.inputs()
        for key, value in [('plan_sha256', '0' * 64), ('prepared_bitexact', False),
                           ('latent_max_abs_diff', .001), ('accepted', True)]:
            changed = copy.deepcopy(report)
            changed[key] = value
            with self.subTest(field=key), self.assertRaises(AssertionError):
                policy.check_report(decision, changed)


if __name__ == '__main__':
    unittest.main()
