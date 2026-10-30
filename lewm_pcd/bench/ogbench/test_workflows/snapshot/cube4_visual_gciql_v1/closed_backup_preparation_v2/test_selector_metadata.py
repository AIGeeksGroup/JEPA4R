"""Two metadata-only cases; no model, NN, payload packaging or transport."""
import copy
import os
import unittest
from closed_role import COUNTERS, REFERENCE, utilities, verify_metadata


def normal():
    cfg = dict(group_id='G11', role='G11_cube4_rgb', executable=True, cube_count=4, modality='rgb', seed=0,
        batch_size=1024, total_updates=1000000, checkpoint_steps=COUNTERS, save_step1=True,
        input_contract=dict(current_dim=211, goal_dim=204, zero_padding=False))
    accepted = dict(actual_exit_code=0, role='G11_cube4_rgb', full153_state_exact=True, state=dict(leaves=153),
        Adam_count=1000000, network_step=1000001, CSV_validation_records=200,
        synthetic_zero_padding=False, actual_current_dim=211, actual_goal_dim=204,
        checkpoint_transactions=[dict(counter=x) for x in COUNTERS])
    status = dict(exit_code=0, role='G11_cube4_rgb', completed_updates=1000000, adam_count=1000000,
                  network_step=1000001, validation_count=200)
    controller = dict(actual_exit_code=0, completed_phases=4, trained_groups=1, accepted_groups=1)
    exits = [dict(actual_exit_code=0, residual_owned_members=[]) for _ in range(4)]
    return [cfg, accepted, status, controller, exits, ['0', '0']]


class ClosedMetadataTests(unittest.TestCase):
    def test_complete_metadata(self):
        utilities(os.environ.get('BACKUP_REFERENCE_SOURCE', str(REFERENCE)))
        verify_metadata(*normal())

    def test_missing_closure_refused(self):
        fixture = copy.deepcopy(normal())
        fixture[4][-1]['actual_exit_code'] = None
        with self.assertRaises(AssertionError):
            verify_metadata(*fixture)


if __name__ == '__main__':
    unittest.main()
