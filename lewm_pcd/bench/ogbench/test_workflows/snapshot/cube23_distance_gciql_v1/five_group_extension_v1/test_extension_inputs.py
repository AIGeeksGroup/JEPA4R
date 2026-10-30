"""Only new Cube4 and visual-to-existing-runtime interface boundaries, CPU only."""
import ast
import hashlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE=Path(__file__).resolve().parent
CASE=HERE.parent
sys.path.insert(0,str(CASE))
import test_distance_inputs as reference
import extension_inputs as inputs

ORDERED=Path(os.environ.get('ORDERED_INPUT_SOURCE',str(CASE.parent/'cube_downstream_101732Test_20261017_v1/official_state_workers_v1/ordered_input.py')))


class ExtensionTests(unittest.TestCase):
    def test_runtime_import_guard_matches_actual_preserved_files(self):
        tree=ast.parse((HERE/'extension_runtime.py').read_bytes())
        pins=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
                  and any(isinstance(name,ast.Name) and name.id=='PRESERVED' for name in node.targets))
        for rel,expected in pins.items():
            self.assertEqual(hashlib.sha256((CASE/rel).read_bytes()).hexdigest(),expected,rel)

    def test_cube4_singleton_actual55_12_never_visual_or_zero_oracle(self):
        data,_=reference.fixture(4)
        with (patch.object(reference.cube_module.CubeInputs,'current',side_effect=AssertionError('visual forbidden')),
              patch.object(reference.cube_module.CubeInputs,'goal',side_effect=AssertionError('visual forbidden'))):
            sampler=inputs.StateInputs(data,reference.config(),4,ordered_input_path=ORDERED)
            try:
                np.random.seed(0);batch=sampler.sample(1,prefetch_next=False)
                self.assertEqual(batch['observations'].shape,(1,55))
                self.assertEqual(batch['actor_goals'].shape,(1,12))
                self.assertIsNone(sampler.features)
                self.assertFalse(sampler.input_metadata['synthetic_zero_padding'])
                self.assertFalse(sampler.pending)
            finally:sampler.close()

    def test_fourth_cube_alone_can_fail_ALL_reward(self):
        data,_=reference.fixture(4);states=data['observations'].copy();states[:4]=0
        for cube in range(4):states[1,19+9*cube]=.4
        states[2,46]=np.nextafter(np.float32(.4),np.float32(1.))
        states[3,46]=10.
        data=reference.official.Dataset.create(**dict(data,observations=states))
        sampler=inputs.StateInputs(data,reference.config(),4,ordered_input_path=ORDERED)
        try:
            values=np.resize(np.array([1,2]),1024)
            with (patch.object(reference.official.Dataset,'get_random_idxs',return_value=np.zeros(1024,np.int64)),
                  patch.object(reference.official.GCDataset,'sample_goals',side_effect=[values,np.full(1024,3,np.int64)])):
                batch=sampler.sample(prefetch_next=False)
            np.testing.assert_array_equal(batch['rewards'],np.resize([0.,-1.],1024))
            np.testing.assert_array_equal(batch['masks'],np.resize([0.,1.],1024))
            self.assertEqual(batch['value_goals'].shape,(1024,12))
        finally:sampler.close()

    def test_cube4_prefetch_baseline_same_factory_batches_IDs_and_committed_rng(self):
        values=[]
        for mode in ('baseline','prefetch1'):
            data,_=reference.fixture(4)
            sampler=inputs.StateInputs(data,reference.config(),4,mode,ordered_input_path=ORDERED)
            try:
                np.random.seed(0);example=sampler.sample(1,prefetch_next=False)
                batches=[];ids=[]
                for index in range(3):
                    batches.append(sampler.sample(prefetch_next=index<2))
                    ids.append({k:v.copy() for k,v in sampler.last_ids.items()})
                sampler.barrier()
                values.append((example,batches,ids,np.random.get_state()))
            finally:sampler.close()
        for left,right in zip(values[0][:3],values[1][:3]):
            if isinstance(left,dict):
                for key in left:np.testing.assert_array_equal(left[key],right[key])
            else:
                for a,b in zip(left,right):
                    for key in a:np.testing.assert_array_equal(a[key],b[key])
        reference.same_rng(self,values[0][3],values[1][3])

    def test_visual_prefetch_keyword_is_noop_and_original_projection_IDs_rng_unchanged(self):
        for cubes in (2,3):
            data,features=reference.fixture(cubes);records=[]
            for cls in (reference.adapter.DistanceInputs,inputs.VisualInputs):
                sampler=cls(data,reference.config(),cubes=cubes,mode='rgb',features=features,workers=4)
                try:
                    np.random.seed(311)
                    kwargs={'prefetch_next':True} if cls is inputs.VisualInputs else {}
                    batch=sampler.sample(2,idxs=np.array([999,1001]),**kwargs)
                    records.append((batch,{k:v.copy() for k,v in sampler.last_ids.items()},np.random.get_state()))
                    if cls is inputs.VisualInputs:
                        sampler.barrier();self.assertFalse(sampler.pending)
                        self.assertEqual(batch['observations'].shape,(2,211))
                        self.assertEqual(batch['value_goals'].shape,(2,192+3*cubes))
                        np.testing.assert_array_equal(batch['next_observations'][:,:192],features[[1000,1002]])
                finally:sampler.close()
            for a,b in zip(records[0][:2],records[1][:2]):
                for key in a:np.testing.assert_array_equal(a[key],b[key])
            reference.same_rng(self,records[0][2],records[1][2])


if __name__=='__main__':unittest.main()
