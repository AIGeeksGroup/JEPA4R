"""CPU-only original sampler/prefetch sequence and actual-dimension checks."""
import copy
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
# Reuse the existing pinned official Dataset/GCDataset AST CPU fixture, not a sampler rewrite.
import test_distance_inputs as reference
import state_inputs as inputs

ORDERED=Path(os.environ.get('ORDERED_INPUT_SOURCE',str(CASE.parent/'cube_downstream_101732Test_20261017_v1/official_state_workers_v1/ordered_input.py')))


def snapshot(sampler):
    return {key:value.copy() for key,value in sampler.last_ids.items()}


class StatePriorityTests(unittest.TestCase):
    def test_all_cube_dimensions_no_oracle_and_original_factory(self):
        class Agent:
            @classmethod
            def create(cls,seed,observations,actions,config,ex_goals):
                return observations.shape,ex_goals.shape,config
        for cubes in (1,2,3):
            data,_=reference.fixture(cubes)
            with (patch.object(reference.cube_module.CubeInputs,'current',side_effect=AssertionError('oracle forbidden')),
                  patch.object(reference.cube_module.CubeInputs,'goal',side_effect=AssertionError('oracle forbidden'))):
                sampler=inputs.StateInputs(data,reference.config(),cubes,ordered_input_path=ORDERED)
                try:
                    batch=sampler.sample(prefetch_next=False)
                    self.assertEqual(batch['observations'].shape,(1024,19+9*cubes))
                    self.assertEqual(batch['actor_goals'].shape,(1024,3*cubes))
                    self.assertFalse(sampler.input_metadata['synthetic_zero_padding'])
                    self.assertFalse(any(key.startswith('_') for key in batch))
                    cfg=object();a=inputs.create_agent(Agent,0,batch,cfg)
                    self.assertEqual(a[:2],((1,19+9*cubes),(1,3*cubes)));self.assertIs(a[2],cfg)
                finally:sampler.close()

    def test_original_parent_sample_IDs_and_rng(self):
        for cubes in (1,2,3):
            data,_=reference.fixture(cubes);cfg=reference.config()
            original=reference.Reference(data,cfg)
            np.random.seed(220);raw=original.sample(1024);expected=np.random.get_state()
            sampler=inputs.StateInputs(data,cfg,cubes,ordered_input_path=ORDERED)
            try:
                np.random.seed(220);batch=sampler.sample(prefetch_next=False)
                reference.same_rng(self,expected,np.random.get_state())
                np.testing.assert_array_equal(sampler.last_ids['value_goal'],original.goals[0])
                np.testing.assert_array_equal(sampler.last_ids['actor_goal'],original.goals[1])
                np.testing.assert_array_equal(batch['observations'],raw['observations'])
                np.testing.assert_array_equal(batch['next_observations'],raw['next_observations'])
                np.testing.assert_array_equal(batch['actions'],raw['actions'])
            finally:sampler.close()

    def _sequence(self,cubes,mode):
        data,_=reference.fixture(cubes);cfg=reference.config()
        train=inputs.StateInputs(data,cfg,cubes,mode,ordered_input_path=ORDERED)
        val=inputs.StateInputs(data,cfg,cubes,'baseline',ordered_input_path=ORDERED)
        batches=[];ids=[];boundaries=[];validation=[]
        try:
            np.random.seed(0)
            example=train.sample(1,prefetch_next=False)
            self.assertEqual(example['actor_goals'].shape,(1,3*cubes))
            self.assertFalse(train.pending)
            before_factory=np.random.get_state()
            class Agent:
                @classmethod
                def create(cls,seed,observations,actions,config,ex_goals):
                    return observations.shape,ex_goals.shape
            self.assertEqual(inputs.create_agent(Agent,0,example,cfg),((1,19+9*cubes),(1,3*cubes)))
            reference.same_rng(self,before_factory,np.random.get_state())
            for step in range(1,5):
                boundary=step in (1,3,4)
                batches.append(train.sample(prefetch_next=not boundary));ids.append(snapshot(train))
                if boundary:
                    train.barrier();self.assertFalse(train.pending)
                    boundaries.append(np.random.get_state())
                    validation.append(val.sample(prefetch_next=False))
            train.close()
            following=val.sample(prefetch_next=False)
            return batches,ids,boundaries,validation,following,np.random.get_state()
        finally:train.close();val.close()

    def test_prefetch_preserves_batches_IDs_validation_and_nextbatch_rng(self):
        for cubes in (1,2,3):
            baseline=self._sequence(cubes,'baseline');prefetch=self._sequence(cubes,'prefetch1')
            for left,right in zip(baseline[:2],prefetch[:2]):
                for first,second in zip(left,right):
                    for key in first:np.testing.assert_array_equal(first[key],second[key])
            for a,b in zip(baseline[2],prefetch[2]):reference.same_rng(self,a,b)
            for first,second in zip(baseline[3],prefetch[3]):
                for key in first:np.testing.assert_array_equal(first[key],second[key])
            for key in baseline[4]:np.testing.assert_array_equal(baseline[4][key],prefetch[4][key])
            reference.same_rng(self,baseline[5],prefetch[5])

    def test_pending_barrier_restores_committed_rng_and_nextbatch(self):
        data,_=reference.fixture(1);cfg=reference.config()
        baseline=inputs.StateInputs(data,cfg,1,ordered_input_path=ORDERED)
        prefetch=inputs.PrefetchStateInputs(data,cfg,1,ordered_input_path=ORDERED)
        try:
            np.random.seed(21);first=baseline.sample();second=baseline.sample();expected=np.random.get_state()
            np.random.seed(21);actual=prefetch.sample();self.assertTrue(prefetch.pending)
            committed=prefetch.committed_rng_state
            prefetch.barrier();reference.same_rng(self,committed,np.random.get_state())
            next_batch=prefetch.sample(prefetch_next=False)
            for key in first:np.testing.assert_array_equal(first[key],actual[key])
            for key in second:np.testing.assert_array_equal(second[key],next_batch[key])
            reference.same_rng(self,expected,np.random.get_state())
        finally:baseline.close();prefetch.close()

    def test_singleton_factory_sample_matches_original_and_exposes_legacy_properties(self):
        data,_=reference.fixture(1);cfg=reference.config()
        parent=reference.Reference(data,cfg)
        np.random.seed(0);expected=parent.sample(1);state=np.random.get_state()
        wrapped=inputs.PrefetchStateInputs(data,cfg,1,ordered_input_path=ORDERED)
        try:
            np.random.seed(0);actual=wrapped.sample(1,prefetch_next=False)
            reference.same_rng(self,state,np.random.get_state())
            reference.same_rng(self,state,wrapped.committed_rng_state)
            self.assertFalse(wrapped.pending)
            np.testing.assert_array_equal(actual['observations'],expected['observations'])
            np.testing.assert_array_equal(wrapped.last_ids['value_goal'],parent.goals[0])
            self.assertIs(wrapped.dataset,data);self.assertIs(wrapped.config,cfg)
            self.assertEqual(wrapped.size,data.size)
            np.testing.assert_array_equal(wrapped.terminal_locs,parent.terminal_locs)
            np.testing.assert_array_equal(wrapped.initial_locs,parent.initial_locs)
            self.assertIsNone(wrapped.features);self.assertEqual((wrapped.current_dim,wrapped.goal_dim),(28,3))
        finally:wrapped.close()

    def test_all_object_distance_boundary_uses_current_and_value_goal(self):
        for cubes in (1,2,3):
            data,_=reference.fixture(cubes)
            states=data['observations'].copy();states[:5]=0
            for i in range(cubes):
                states[1,19+9*i]=.3;states[2,19+9*i]=.4;states[4,19+9*i]=10.
            states[3,19]=np.nextafter(np.float32(.4),np.float32(1.))
            controlled=reference.official.Dataset.create(**dict(data,observations=states))
            sampler=inputs.StateInputs(controlled,reference.config(),cubes,ordered_input_path=ORDERED)
            goal_ids=np.resize(np.array([1,2,3]),1024)
            try:
                with (patch.object(reference.official.Dataset,'get_random_idxs',return_value=np.zeros(1024,np.int64)),
                      patch.object(reference.official.GCDataset,'sample_goals',side_effect=[goal_ids,np.full(1024,4,np.int64)])):
                    batch=sampler.sample(prefetch_next=False)
                np.testing.assert_array_equal(batch['rewards'],np.resize(np.array([0.,0.,-1.]),1024))
                np.testing.assert_array_equal(batch['masks'],np.resize(np.array([0.,0.,1.]),1024))
            finally:sampler.close()


if __name__=='__main__':
    unittest.main()
