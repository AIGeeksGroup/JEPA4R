import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pcd_resume_only', HERE / 'run_pcd.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ResumeGuardTests(unittest.TestCase):
    def fixture(self, root):
        q, pcd = root / 'quad', root / 'pcd'
        here = q / 'control_v2/PCD'

        def put(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value if isinstance(value, str) else json.dumps(value))

        put(pcd / 'runtime_v1/outer_exit_code', '1')
        put(q / 'control_v1/PCD/wrapper_exit_code', '1')
        put(pcd / 'runtime_v1/status.json', {'actual_exit_code': 1})
        for name in ('00_wait_G11_acceptance', '01_reference', '02_fresh', '03_GPU_preprocess'):
            put(pcd / 'runtime_v1' / (name + '_exit.json'), dict(actual_exit_code=0, residual_owned_members=[]))
        report = HERE.parent / 'monitor_checks/20261020T2155_v1/PCD_GPU_preprocess_report.json'
        target = pcd / '3d/GPU_preprocess_report.json'
        target.parent.mkdir(parents=True)
        target.write_bytes(report.read_bytes())
        decision = q / 'decision.json'
        put(decision, dict(actual_exit_code=0, positive_gain_released=True,
            original_engineering_5percent_gate_passed=False, unchanged_numeric_tolerances=True,
            gate_report_sha256=module.GATE_SHA))
        put(here / 'launch_plan.json', dict(performance_release=dict(path=str(decision), sha256=module.sha(decision)),
            phases=[dict(name=name) for name in ('04_profile', '05_pilot', '06_full', '07_audit')]))
        put(q / 'control_v1/G11/runtime/outer_exit_code', '0')
        put(q / 'control_v1/G11/wrapper_exit_code', '0')
        put(q / 'acceptance/G11_full_acceptance.json', dict(actual_exit_code=0, full153_state_exact=True,
            Adam_count=1000000, network_step=1000001, CSV_validation_records=200,
            checkpoint_transactions=[dict(counter=n) for n in [1] + list(range(100000, 1000001, 100000))]))
        return q, pcd, here

    def test_closed_numeric_gate_keeps_old_failed_aggregate_and_only_remaining_stages(self):
        with tempfile.TemporaryDirectory() as name:
            paths = self.fixture(Path(name))
            module.prerequisites(*paths)
            self.assertEqual((paths[1] / 'runtime_v1/outer_exit_code').read_text(), '1')
            self.assertFalse((paths[1] / 'runtime_v2').exists())

    def test_real_old_subprocess_failure_cannot_be_relabelled_as_performance_exception(self):
        with tempfile.TemporaryDirectory() as name:
            paths = self.fixture(Path(name))
            path = paths[1] / 'runtime_v1/03_GPU_preprocess_exit.json'
            path.write_text(json.dumps(dict(actual_exit_code=1, residual_owned_members=[])))
            with self.assertRaises(AssertionError):
                module.prerequisites(*paths)


if __name__ == '__main__':
    unittest.main()
