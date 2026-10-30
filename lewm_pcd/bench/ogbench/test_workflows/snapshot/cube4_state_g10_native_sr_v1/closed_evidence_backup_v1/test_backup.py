"""Two CPU metadata boundaries; no real SR, models, archive or transport."""
import copy
import unittest
import backup


def facts():
    return [dict(stage='G10_final1M_SR_and_audits_complete',actual_exit_code=0,completed_phases=4,groups=['G10'],training_updates=0),
        [dict(actual_exit_code=0,residual_owned_members=[]) for _ in range(4)],['0','0']]


class BackupTests(unittest.TestCase):
    def test_complete_metadata_and_frozen_IO_inverse(self):
        backup.verify_closed(*facts())
        self.assertTrue(backup.load_pack().reuse_proof['inverse_entire_source_exact'])
        self.assertTrue(backup.load_copy().reuse_proof['inverse_entire_source_exact'])
    def test_missing_final_audit_exit_is_refused(self):
        v=copy.deepcopy(facts());v[1][-1]['actual_exit_code']=None
        with self.assertRaises(AssertionError):backup.verify_closed(*v)


if __name__=='__main__':unittest.main()
