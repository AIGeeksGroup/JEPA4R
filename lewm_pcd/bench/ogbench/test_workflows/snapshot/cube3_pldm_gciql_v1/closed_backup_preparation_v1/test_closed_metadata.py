"""Two real closed-metadata fixtures; never read a model or build/copy a package."""
import copy
import json
import os
from pathlib import Path
import unittest

import backup

HERE=Path(__file__).resolve().parent


def fixture():
    return json.loads((HERE/'CLOSED_METADATA_FIXTURE_v1.json').read_bytes())['input']


class ClosedMetadataTests(unittest.TestCase):
    def test_actual_P3_tenphase_final1M_and_tiny1_full250_are_accepted(self):
        values=fixture();backup.verify_closed(**values)
        base=HERE.parents[1]/'cube4_gciql_103032Test_20261020_v1/closed_backup_preparation_v2'
        packed=backup.load_pack(Path(os.environ.get('P3_PACK_SOURCE',str(base/'closed_role.py'))))
        copied=backup.load_copy(Path(os.environ.get('P3_COPY_SOURCE',str(base/'copy_verified.py'))))
        self.assertTrue(packed.parameter_proof['inverse_exact'])
        self.assertTrue(copied.parameter_proof['inverse_exact'])
        self.assertEqual(packed.parameter_proof['changes'],1)
        self.assertEqual(copied.parameter_proof['changes'],3)

    def test_missing_real_final_audit_exit_is_refused(self):
        values=copy.deepcopy(fixture())
        values['exits']['13_native_full_audit']['actual_exit_code']=None
        with self.assertRaises(AssertionError):backup.verify_closed(**values)


if __name__=='__main__':unittest.main()
