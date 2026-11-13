# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tests/test_tiny_resume_budget.py (verbatim except: imports /
# paths adapted to the package; lines wrapped at 120 columns, no lint exemptions, unused imports removed)
"""Pure CPU control checks before the original Tiny loop can forward/update."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
import sys
import unittest
from unittest.mock import patch
from jepa4r.policy.gciql.gate_checks import TinyFixedGate
from jepa4r.utils.training_state import FixedBatchCursor


class TinyResumeBudgetTests(unittest.TestCase):
    def gate(self,position=1,stop=2,entries=3):
        g=TinyFixedGate.__new__(TinyFixedGate)
        g.entries=[dict(id='b'+str(i),epoch=i,batch_index=0,array_key='b'+str(i))for i in range(entries)]
        g.cursor=FixedBatchCursor([v['id']for v in g.entries])
        g.cursor.load_state_dict(dict(batch_ids=[v['id']for v in g.entries],position=position,
                                     next_batch_id=None if position==entries else g.entries[position]['id']))
        g.config={'stop_after':stop,'resume_state':{'path':'owned-state','sha256':'a'*64}}
        g.identity={};return g

    def attach(self,g,progress):
        fake=ModuleType('torch');fake.nn=SimpleNamespace(ModuleDict=lambda values:values)
        with (
            patch.dict(sys.modules, {'torch': fake}),
            patch('jepa4r.policy.gciql.gate_checks.pinned', return_value=Path('owned-state')),
            patch('jepa4r.utils.training_state.restore_torch_state', return_value=progress),
        ):
            return g.attach(object(),object(),object(),object(),object())

    def test_step1_to2_remains_accepted(self):
        g=self.gate();self.assertEqual(self.attach(g,dict(step=1,epoch=1,batch_index=0)),1)
        self.assertEqual(g.next_batch_index(1),0)

    def test_completed_resume_is_rejected_before_original_loop(self):
        g=self.gate(position=2)
        with self.assertRaisesRegex(ValueError,'budget|complete|remaining'):
            self.attach(g,dict(step=2,epoch=2,batch_index=0))

    def test_progress_step_epoch_or_batch_mismatch_rejects(self):
        for values in [
            dict(step=0, epoch=1, batch_index=0),
            dict(step=1, epoch=0, batch_index=0),
            dict(step=1, epoch=1, batch_index=1),
        ]:
            with self.subTest(values=values),self.assertRaises(ValueError):self.attach(self.gate(),values)

    def test_no_next_entry_and_changed_next_id_reject(self):
        g=self.gate(entries=2)
        with self.assertRaises(ValueError):self.attach(g,dict(step=1,epoch=1,batch_index=0))
        g=self.gate();g.entries[1]['id']='different'
        with self.assertRaises(ValueError):self.attach(g,dict(step=1,epoch=1,batch_index=0))

    def test_next_batch_guard_refuses_completed_or_wrong_epoch(self):
        with self.assertRaises(ValueError):self.gate(position=2).next_batch_index(2)
        with self.assertRaises(ValueError):self.gate().next_batch_index(0)


if __name__=='__main__':unittest.main()
