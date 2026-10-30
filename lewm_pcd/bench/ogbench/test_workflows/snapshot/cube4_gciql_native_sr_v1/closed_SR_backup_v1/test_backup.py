"""Only new metadata closure and reversible asset-label bindings; no real payload IO."""
import copy
import os
from pathlib import Path
import unittest
from build_backup import verify_closed
from reuse import load

PROJECT=Path(__file__).resolve().parents[2]


def facts():
    runtime=dict(stage='Native_SR_pipeline_complete',actual_exit_code=0)
    closure=dict(actual_exit_code=0,completed_phases=8,cube_count=4,data_id='cube-quadruple-play-v0',training_updates=0,SR_episodes=500)
    exits=[dict(actual_exit_code=0,residual_owned_members=[]) for _ in range(8)]
    reports={g:(dict(actual_exit_code=0,group_id=g,mode='full',episodes=250,unique_seeds=250,successes=0,success_rate=0.),
        dict(actual_exit_code=0,group_id=g,mode='full',episodes=250,unique_seeds=250,successes=0,success_rate=0.,successes_by_task={str(t):0 for t in range(1,6)})) for g in ('G11','G12')}
    return [runtime,closure,exits,['0','0'],reports]


class BackupTests(unittest.TestCase):
    def test_closed_zero_SR_is_preserved_and_bridges_are_exact(self):
        verify_closed(*facts())
        pack=os.environ.get('QUAD_SR_PACK_REFERENCE',str(PROJECT/'gciql_m5_restore_103032Test_20261020_v1/closed_sr_backup_v1/build_closed_v2.py'))
        copier=os.environ.get('QUAD_SR_COPY_REFERENCE',str(PROJECT/'cube4_gciql_103032Test_20261020_v1/closed_backup_preparation_v2/copy_verified.py'))
        for kind,path in [('pack',pack),('copy',copier)]:self.assertTrue(load(kind,path).reuse_proof['inverse_entire_source_exact'])
    def test_unclosed_phase_refuses_backup(self):
        value=copy.deepcopy(facts());value[2][-1]['actual_exit_code']=None
        with self.assertRaises(AssertionError):verify_closed(*value)


if __name__=='__main__':unittest.main()
