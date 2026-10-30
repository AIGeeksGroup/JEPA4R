"""CPU-only tests reuse the frozen official Dataset/GCDataset class AST.

Only Flax's mapping container and JAX tree traversal are substituted; sampling,
goal drawing, terminal logic and NumPy RNG are original source, not rewritten.
"""
import ast
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from typing import Any

import numpy as np

CASE=Path(__file__).resolve().parent
LOCAL=CASE.parent/'cube_downstream_101732Test_20261017_v1'
DATASET_SOURCE=Path(os.environ.get('OGBENCH_DATASETS_SOURCE',str(LOCAL/'official_state_workers_v1/evidence_backup_v1/extracted/protected_official_source/impls/utils/datasets.py')))
CUBE_SOURCE=Path(os.environ.get('CUBE_INPUTS_SOURCE',str(LOCAL/'cube_inputs.py')))
SOURCE_SHA='52c373922a87e1ba8ad8b3b96c3474d5e6385e2ea2c7206930f7a8a315816609'


def tree_map(function,value):
    return {key:tree_map(function,item) for key,item in value.items()} if isinstance(value,dict) else function(value)


class Mapping(dict):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._dict=self


raw=DATASET_SOURCE.read_bytes()
assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
nodes=[node for node in ast.parse(raw).body if
       (isinstance(node,ast.FunctionDef) and node.name=='get_size') or
       (isinstance(node,ast.ClassDef) and node.name in ('Dataset','GCDataset'))]
official=types.ModuleType('utils.datasets')
official.__dict__.update(dataclasses=dataclasses,Any=Any,np=np,FrozenDict=Mapping,
    jax=types.SimpleNamespace(tree_util=types.SimpleNamespace(tree_map=tree_map,tree_leaves=lambda x:list(x.values()))))
sys.modules.setdefault('utils',types.ModuleType('utils'))
sys.modules['utils.datasets']=official
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(DATASET_SOURCE),'exec'),official.__dict__)
spec=importlib.util.spec_from_file_location('cube_inputs',CUBE_SOURCE)
cube_module=importlib.util.module_from_spec(spec);sys.modules['cube_inputs']=cube_module;spec.loader.exec_module(cube_module)
import distance_inputs as adapter


def config():
    return dict(frame_stack=None,p_aug=0.,discount=.99,gc_negative=True,
        value_p_curgoal=.2,value_p_trajgoal=.5,value_p_randomgoal=.3,value_geom_sample=True,
        actor_p_curgoal=0.,actor_p_trajgoal=1.,actor_p_randomgoal=0.,actor_geom_sample=False)


def fixture(cubes):
    rows=2002
    states=np.arange(rows*(19+9*cubes),dtype=np.float32).reshape(rows,-1)*np.float32(.00001)
    actions=np.arange(rows*5,dtype=np.float32).reshape(rows,5)*np.float32(.00001)
    original_term=np.zeros(rows,np.float32);original_term[[1000,2001]]=1
    valids=1-original_term
    terminals=np.minimum(original_term+np.r_[original_term[1:],0],1)
    dataset=official.Dataset.create(observations=states,actions=actions,terminals=terminals,valids=valids)
    features=np.arange(rows*192,dtype=np.float32).reshape(rows,192)*np.float32(.00001)
    features.setflags(write=False)
    return dataset,features


class Reference(official.GCDataset):
    def __post_init__(self):
        super().__post_init__();self.goals=[]

    def sample_goals(self,*args,**kwargs):
        ids=super().sample_goals(*args,**kwargs);self.goals.append(ids.copy());return ids


def same_rng(test,left,right):
    test.assertEqual(left[0],right[0]);np.testing.assert_array_equal(left[1],right[1]);test.assertEqual(left[2:],right[2:])


class InputTests(unittest.TestCase):
    def test_sampler_IDs_rng_and_true_terminal_next(self):
        for cubes in (2,3):
            data,features=fixture(cubes);cfg=config()
            reference=Reference(data,cfg)
            np.random.seed(154);raw=reference.sample(2,idxs=np.array([999,1001]));expected_rng=np.random.get_state()
            adapted=adapter.DistanceInputs(data,cfg,cubes=cubes,mode='rgb',features=features,workers=4)
            try:
                np.random.seed(154);batch=adapted.sample(2,idxs=np.array([999,1001]));actual_rng=np.random.get_state()
                same_rng(self,expected_rng,actual_rng)
                np.testing.assert_array_equal(adapted.last_ids['next'],[1000,1002])
                np.testing.assert_array_equal(adapted.last_ids['value_goal'],reference.goals[0])
                np.testing.assert_array_equal(adapted.last_ids['actor_goal'],reference.goals[1])
                np.testing.assert_array_equal(batch['actions'],raw['actions'])
                np.testing.assert_array_equal(batch['next_observations'][:,:192],features[[1000,1002]])
                self.assertEqual(batch['observations'].shape,(2,211))
                self.assertEqual(batch['actor_goals'].shape,(2,192+3*cubes))
                self.assertEqual(batch['rewards'].dtype,raw['rewards'].dtype)
                with self.assertRaises(ValueError):adapted.sample(1,idxs=np.array([1000]))
            finally:adapted.close()

    def test_all_per_cube_distance_and_current_value_goal_boundary(self):
        for cubes in (2,3):
            data,_=fixture(cubes)
            states=data['observations'].copy();states[:5]=0
            for i in range(cubes):
                states[1,19+9*i]=.3
                states[2,19+9*i]=.4
                states[4,19+9*i]=10.
            states[3,19]=np.nextafter(np.float32(.4),np.float32(1.))
            new=official.Dataset.create(**dict(data,observations=states))
            adapted=adapter.DistanceInputs(new,config(),cubes=cubes,mode='state')
            with patch.object(official.GCDataset,'sample_goals',side_effect=[np.array([1,2,3]),np.array([4,4,4])]):
                batch=adapted.sample(3,idxs=np.array([0,0,0]))
            # All cubes at .3/.4 pass even though concatenated XYZ L2 exceeds .4.
            np.testing.assert_array_equal(batch['rewards'],[0.,0.,-1.])
            np.testing.assert_array_equal(batch['masks'],[0.,0.,1.])
            self.assertEqual(batch['value_goals'].shape,(3,3*cubes))

    def test_state_uses_actual_dimensions_and_never_zero_oracle(self):
        for cubes in (2,3):
            data,_=fixture(cubes)
            with (patch.object(cube_module.CubeInputs,'current',side_effect=AssertionError('visual/oracle current forbidden')),
                  patch.object(cube_module.CubeInputs,'goal',side_effect=AssertionError('visual/oracle goal forbidden'))):
                adapted=adapter.DistanceInputs(data,config(),cubes=cubes,mode='state',features=None,workers=0)
                batch=adapted.sample(4)
            self.assertEqual(batch['observations'].shape,(4,19+9*cubes))
            self.assertEqual(batch['actor_goals'].shape,(4,3*cubes))
            self.assertFalse(adapted.input_metadata['synthetic_zero_padding'])
            with self.assertRaises(ValueError):adapter.DistanceInputs(data,config(),cubes=cubes,mode='state',workers=4)

    def test_four_role_threads_equal_parent_lookup_bytes_and_rng(self):
        data,features=fixture(3);outputs=[];rngs=[]
        for workers in (0,4):
            sampler=adapter.DistanceInputs(data,config(),cubes=3,mode='pcd',features=features,workers=workers)
            try:
                np.random.seed(701);outputs.append(sampler.sample(16));rngs.append(np.random.get_state())
            finally:sampler.close()
        same_rng(self,*rngs)
        for key in outputs[0]:np.testing.assert_array_equal(outputs[0][key],outputs[1][key])

    def test_factory_receives_independent_actual_shapes(self):
        class FakeAgent:
            @classmethod
            def create(cls,seed,observations,actions,config,ex_goals=None):
                return seed,observations.shape,actions.shape,ex_goals.shape,config
        for cubes,mode in ((2,'state'),(3,'state'),(2,'rgb'),(3,'pcd')):
            data,features=fixture(cubes)
            sampler=adapter.DistanceInputs(data,config(),cubes=cubes,mode=mode,features=None if mode=='state' else features)
            batch=sampler.sample(2);cfg=object()
            result=adapter.create_agent(FakeAgent,0,batch,cfg)
            self.assertEqual(result[1],(1,19+9*cubes if mode=='state' else 211))
            self.assertEqual(result[3],(1,3*cubes if mode=='state' else 192+3*cubes))
            self.assertIs(result[4],cfg)

    def test_one_episode_loader_maps_real_terminal_and_freezes_one_RAM_array(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cache=root/'cache';episode=cache/'train/episode_000000'
            episode.mkdir(parents=True)
            source=root/'cube-double-play-v0.npz';source.write_bytes(b'CPU metadata fixture only')
            values=np.arange(1001*192,dtype=np.float32).reshape(1001,192)
            path=episode/'latent.npy'
            with path.open('wb') as stream:np.save(stream,values,allow_pickle=False)
            digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            checkpoint='a'*64
            record=dict(split='train',episode=0,actual_exit_code=0,raw_row_start=0,
                raw_row_count=1001,main_rows=1000,terminal_raw_id=1000,checkpoint_sha256=checkpoint,
                canonical_raw_ids_sha256=hashlib.sha256(np.arange(1001,dtype='<i8').tobytes()).hexdigest(),
                source_npz=dict(path=str(source),accepted_stat=adapter.file_stat(source)),
                latent=dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path),shape=[1001,192],dtype='float32'))
            receipt=episode/'receipt.json';receipt.write_text(json.dumps(record))
            manifest=cache/'cache_manifest.json'
            manifest.write_text(json.dumps(dict(actual_exit_code=0,checkpoint_sha256=checkpoint,
                splits=dict(train=[dict(episode=0,receipt_path=str(receipt),receipt_sha256=digest(receipt))]))))
            actual,binding=adapter.load_cached_features(cache,'train',digest(manifest),checkpoint,source,1001)
            np.testing.assert_array_equal(actual,values)
            np.testing.assert_array_equal(actual[1000],values[1000])
            self.assertFalse(actual.flags.writeable)
            self.assertTrue(binding['one_RAM_allocation'])


if __name__=='__main__':
    unittest.main()
