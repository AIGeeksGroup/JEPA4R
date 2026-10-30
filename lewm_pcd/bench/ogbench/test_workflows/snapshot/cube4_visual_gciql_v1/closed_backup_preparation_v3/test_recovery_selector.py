"""Two affected-boundary metadata fixtures; no payload, NN, archive or transport."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from closed_G12 import base, locations, verify_recovery, RECOVERY_SHA, BINDING_SHA


def normal():
    old = Path(__file__).resolve().parents[1]/'closed_backup_preparation_v2/test_selector_metadata.py'
    sys.path.insert(0, str(old.parent))
    spec = importlib.util.spec_from_file_location('_old_metadata_fixture', old)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    facts = module.normal()
    facts[0].update(group_id='G12', role='G12_cube4_pcd', modality='pcd')
    facts[1]['role'] = facts[2]['role'] = 'G12_cube4_pcd'
    recovery = [dict(actual_exit_code=0, PCD_recovery_chain_actual0=True),
        dict(actual_exit_code=0, publisher_sha256=RECOVERY_SHA, binding_sha256=BINDING_SHA,
            cube_count=4, data_id='cube-quadruple-play-v0', PCD_episodes=5500, PCD_rows=5505500,
            chain=dict(original_runtime_v1_actual_exit_code=1, original_GPU_gate_accepted=False,
                logical_completed_phases=8, new_runtime_completed_phases=4)),
        dict(actual_exit_code=0, stage='complete'), [dict(actual_exit_code=0) for _ in range(3)],
        ['0', '0', '1', '1', '1', '1']]
    return facts, recovery


class RecoverySelectorTests(unittest.TestCase):
    def test_actual_recovery_namespace_and_closed_metadata(self):
        facts, recovery = normal(); base.verify_metadata(*facts); verify_recovery(*recovery)
        at = locations(base.Q)
        self.assertTrue(at['config'].as_posix().endswith('/functional_v2/configs/G12_cube4_pcd.v3.json'))
        self.assertTrue(at['role'].as_posix().endswith('/control_v2/G12'))
        self.assertTrue(at['release'].as_posix().endswith('/control_v2/G12_wait_arm_v2/RELEASE_v2.json'))

    def test_missing_new_closure_refused(self):
        facts, recovery = normal(); recovery = copy.deepcopy(recovery)
        recovery[3][-1]['actual_exit_code'] = None
        with self.assertRaises(AssertionError): verify_recovery(*recovery)
        facts[4][-1]['actual_exit_code'] = None
        with self.assertRaises(AssertionError): base.verify_metadata(*facts)


if __name__ == '__main__': unittest.main()
