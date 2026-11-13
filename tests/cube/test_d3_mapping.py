# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tests/test_d3_mapping.py (verbatim except: imports / paths
# adapted to the package; lines wrapped at 120 columns)
"""Complete row bindings and current/goal seam checks without a model."""
from pathlib import Path
import sys
import unittest
import copy
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from jepa4r.policy.gciql import d3_mapping as mapping


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.features=np.arange(3*192,dtype=np.float32).reshape(3,192)
        self.states=np.arange(3*28,dtype=np.float32).reshape(3,28)
        self.actions=np.arange(15,dtype=np.float32).reshape(3,5)
        self.terminals=np.array([False,False,True])
        self.arrays,self.meta=mapping.build_episode(self.features,self.states,self.actions,self.terminals,
            split='train',cube_count=1,episode=0,raw_start=0,source_sha256='a'*64)

    def test_complete_current_goal_action_mapping_passes(self):
        result=mapping.validate_episode(self.features,self.arrays,self.meta)
        self.assertTrue(result['passed'])
        self.assertEqual(self.arrays['current'].shape,(3,211))
        self.assertEqual(self.arrays['goal'].shape,(3,195))
        self.assertTrue(np.all(self.arrays['task_id']==-1))

    def test_swapped_features_wrong_split_and_wrong_task_are_rejected(self):
        with self.assertRaises(ValueError):
            mapping.validate_episode(self.features[[1,0,2]],self.arrays,self.meta)
        meta=copy.deepcopy(self.meta);meta['split']='val'
        with self.assertRaises(ValueError):
            mapping.validate_episode(self.features,self.arrays,meta)
        arrays={k:v.copy()for k,v in self.arrays.items()};arrays['task_id'][0]=1
        with self.assertRaises(ValueError):
            mapping.validate_episode(self.features,arrays,self.meta)

    def test_cross_split_or_task_goal_is_rejected(self):
        mapping.validate_goal_link(self.meta,self.meta)
        other=copy.deepcopy(self.meta);other['split']='val'
        with self.assertRaises(ValueError):mapping.validate_goal_link(self.meta,other)
        other=copy.deepcopy(self.meta);other['task_id']=1
        with self.assertRaises(ValueError):mapping.validate_goal_link(self.meta,other)


if __name__=='__main__':unittest.main()
