# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tests/test_state_replay.py (verbatim except: imports / paths
# adapted to the package; lines wrapped at 120 columns)
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from state_replay import state_from_arrays,flatten_state


class StateReplayTests(unittest.TestCase):
    def setUp(self):
        self.template={'network':{'params':{'w':np.zeros((2,3),np.float32)},'step':1},'rng':np.zeros(2,np.uint32)}
        self.arrays=flatten_state(self.template)
        self.arrays['network/step']=np.array(3,np.int32)

    def test_complete_paths_shapes_and_endpoint_counter_dtype_are_preserved(self):
        state,mapping=state_from_arrays(self.template,self.arrays)
        self.assertEqual(set(flatten_state(state)),set(self.arrays))
        self.assertEqual(state['network']['step'].dtype,np.int32)
        self.assertEqual(len(mapping),3)

    def test_missing_extra_shape_and_float_dtype_are_rejected(self):
        for change in ('missing','extra','shape','dtype'):
            arrays=self.arrays.copy()
            if change=='missing':del arrays['rng']
            elif change=='extra':arrays['hidden']=np.zeros(1,np.float32)
            elif change=='shape':arrays['network/params/w']=np.zeros((3,2),np.float32)
            else:arrays['network/params/w']=arrays['network/params/w'].astype(np.float64)
            with self.subTest(change=change),self.assertRaises(ValueError):state_from_arrays(self.template,arrays)


if __name__=='__main__':unittest.main()
