"""Same two metadata cases plus the exact old/unexecuted phase selection boundary."""
import copy
import unittest
from test_metadata import normal
from metadata_backup_v2 import verify_metadata, closed_plan_phases, OLD, NEW


class MetadataPhaseSelectionTests(unittest.TestCase):
    def test_closed_metadata_omits_unexecuted_old_phase_paths(self):
        verify_metadata(*normal())
        old_plan = {'phases': [{'name': name} for name in OLD+NEW]}
        new_plan = {'phases': [{'name': name} for name in NEW]}
        self.assertEqual([p['name'] for p in closed_plan_phases(1, old_plan)], list(OLD))
        self.assertEqual([p['name'] for p in closed_plan_phases(2, new_plan)], list(NEW))

    def test_missing_new_closure_refused(self):
        fixture = copy.deepcopy(normal()); fixture[4][-1]['actual_exit_code'] = None
        with self.assertRaises(AssertionError): verify_metadata(*fixture)


if __name__ == '__main__': unittest.main()
