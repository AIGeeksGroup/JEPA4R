"""Dependency boundaries only; no filesystem mutation, process or model execution."""
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('tested_waiter', HERE/'wait_then_run.py')
waiter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(waiter)


class MemoryPath:
    def __init__(self, files, name='quad'):
        self.files, self.name = files, name

    def __truediv__(self, child):
        return MemoryPath(self.files, self.name+'/'+child)

    def exists(self):
        return self.name in self.files

    def read_bytes(self):
        return self.files[self.name]

    def read_text(self):
        return self.read_bytes().decode()

    def __str__(self):
        return self.name


def fixture(complete):
    plan = HERE.parents[1]/'cube4_gciql_native_sr_103032Test_20261021_v1/launch_plan.json'
    if not plan.exists():
        plan = waiter.ROOT/'cube4_gciql_native_sr_v1/launch_plan.json'
    files = {'quad/launch_plan.json': plan.read_bytes()}
    if complete:
        files['quad/wrapper_exit_code'] = files['quad/runtime_v1/outer_exit_code'] = b'0\n'
        for phase in waiter.PHASES:
            files['quad/runtime_v1/'+phase+'_exit.json'] = json.dumps(dict(actual_exit_code=0, residual_owned_members=[])).encode()
        files['quad/runtime_v1/closure.json'] = json.dumps(dict(actual_exit_code=0, completed_phases=8, SR_episodes=500)).encode()
        for group in ('G11', 'G12'):
            for name in ('report.json', 'independent_audit.json'):
                files['quad/full/'+group+'/'+name] = json.dumps(dict(actual_exit_code=0, group_id=group,
                    episodes=250, unique_seeds=250, current_dim=211, goal_dim=204)).encode()
    return MemoryPath(files)


class WaiterTests(unittest.TestCase):
    def test_missing_then_complete_closure(self):
        self.assertFalse(waiter.ready(fixture(False)))
        self.assertTrue(waiter.ready(fixture(True)))

    def test_failed_exit_cannot_release_pipeline(self):
        root = fixture(True)
        root.files['quad/wrapper_exit_code'] = b'1\n'
        with self.assertRaises(AssertionError):
            waiter.ready(root)


if __name__ == '__main__':
    unittest.main()
