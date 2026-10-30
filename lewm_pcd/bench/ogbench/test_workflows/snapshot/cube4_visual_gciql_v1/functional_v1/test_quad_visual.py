"""Focused original-sampler CPU tests, without Torch/JAX model or GPU."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE=Path(__file__).resolve().parent
LEGACY=Path(os.environ.get('GCIQL_LEGACY_CASE',str(HERE.parents[1]/'cube23_gciql_distance_matched_101732Test_20261018_v1'))).resolve()
os.environ['GCIQL_LEGACY_CASE']=str(LEGACY)
sys.path.insert(0,str(LEGACY))
import test_distance_inputs as reference
import quad_visual as quad


class QuadTests(unittest.TestCase):
    def test_original_cube2_cube3_batches_IDs_and_NPbytes_unchanged(self):
        for cubes in (2,3):
            data,features=reference.fixture(cubes);records=[]
            for cls in (reference.adapter.DistanceInputs,quad.VisualInputs):
                sampler=cls(data,reference.config(),cubes=cubes,mode='rgb',features=features,workers=4)
                try:
                    np.random.seed(170);batch=sampler.sample(2,idxs=np.array([999,1001]))
                    records.append((batch,{k:v.copy() for k,v in sampler.last_ids.items()},np.random.get_state()))
                finally:sampler.close()
            for left,right in zip(records[0][:2],records[1][:2]):
                for key in left:np.testing.assert_array_equal(left[key],right[key])
            reference.same_rng(self,records[0][2],records[1][2])

    def test_cube4_true211_204_original_IDs_RNG_and_terminalnext(self):
        data,features=reference.fixture(4);cfg=reference.config();original=reference.Reference(data,cfg)
        np.random.seed(71);raw=original.sample(2,idxs=np.array([999,1001]));expected=np.random.get_state()
        sampler=quad.VisualInputs(data,cfg,cubes=4,mode='pcd',features=features,workers=4)
        try:
            np.random.seed(71);batch=sampler.sample(2,idxs=np.array([999,1001]),prefetch_next=True)
            reference.same_rng(self,expected,np.random.get_state())
            self.assertEqual(batch['observations'].shape,(2,211));self.assertEqual(batch['actor_goals'].shape,(2,204))
            np.testing.assert_array_equal(sampler.last_ids['next'],[1000,1002])
            np.testing.assert_array_equal(sampler.last_ids['value_goal'],original.goals[0])
            np.testing.assert_array_equal(sampler.last_ids['actor_goal'],original.goals[1])
            np.testing.assert_array_equal(batch['observations'][:,:192],features[[999,1001]])
            np.testing.assert_array_equal(batch['observations'][:,192:],raw['observations'][:,:19])
            np.testing.assert_array_equal(batch['next_observations'][:,:192],features[[1000,1002]])
            np.testing.assert_array_equal(batch['value_goals'][:,:192],features[original.goals[0]])
            np.testing.assert_array_equal(batch['actions'],raw['actions'])
            self.assertFalse(sampler.input_metadata['synthetic_zero_padding']);sampler.barrier();self.assertFalse(sampler.pending)
        finally:sampler.close()

    def test_each_cube_distance_ALL_includes_fourth_and_value_not_actor(self):
        data,features=reference.fixture(4);states=data['observations'].copy();states[:5]=0
        for cube in range(4):states[1,19+9*cube]=.3;states[2,19+9*cube]=.4
        states[3,46]=np.nextafter(np.float32(.4),np.float32(1.));states[4,46]=10.
        data=reference.official.Dataset.create(**dict(data,observations=states))
        sampler=quad.VisualInputs(data,reference.config(),cubes=4,mode='rgb',features=features,workers=4)
        try:
            with patch.object(reference.official.GCDataset,'sample_goals',side_effect=[np.array([1,2,3]),np.array([4,4,4])]):
                batch=sampler.sample(3,idxs=np.zeros(3,np.int64))
            np.testing.assert_array_equal(batch['rewards'],[0.,0.,-1.]);np.testing.assert_array_equal(batch['masks'],[0.,0.,1.])
            self.assertEqual(batch['value_goals'].shape,(3,204))
        finally:sampler.close()

    def test_singleton_actual_factory_and_only_membership_guard_delta(self):
        data,features=reference.fixture(4);sampler=quad.VisualInputs(data,reference.config(),cubes=4,mode='rgb',features=features,workers=4)
        class Agent:
            @classmethod
            def create(cls,seed,observations,actions,config,ex_goals):return observations.shape,ex_goals.shape
        try:
            np.random.seed(0);example=sampler.sample(1,prefetch_next=False)
            self.assertEqual(quad.distance_module().create_agent(Agent,0,example,reference.config()),((1,211),(1,204)))
            proof=quad.guard_proof();self.assertEqual(proof['replacement_count'],1)
            self.assertEqual(proof['original_source_sha256'],quad.DISTANCE_SHA)
        finally:sampler.close()

    def test_content_identical_stat_rebind_is_explicit_and_does_not_mutate_old_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cache=root/'cache';ep=cache/'train/episode_000000';ep.mkdir(parents=True)
            source=root/'cube-quadruple-play-v0.npz';source.write_bytes(b'explicit CPU migration fixture, not real NPZ')
            current=reference.adapter.file_stat(source);old=dict(current,inode=current['inode']+100,ctime_ns=current['ctime_ns']-1)
            values=np.arange(1001*192,dtype=np.float32).reshape(1001,192);latent=ep/'latent.npy';np.save(latent,values,allow_pickle=False)
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checkpoint='a'*64
            record=dict(split='train',episode=0,actual_exit_code=0,raw_row_start=0,raw_row_count=1001,main_rows=1000,terminal_raw_id=1000,
                checkpoint_sha256=checkpoint,canonical_raw_ids_sha256=hashlib.sha256(np.arange(1001,dtype='<i8').tobytes()).hexdigest(),
                source_npz=dict(path=str(source),accepted_stat=old),latent=dict(path=str(latent),bytes=latent.stat().st_size,sha256=sha(latent),shape=[1001,192],dtype='float32'))
            receipt=ep/'receipt.json';receipt.write_text(json.dumps(record));manifest=cache/'cache_manifest.json'
            manifest.write_text(json.dumps(dict(actual_exit_code=0,checkpoint_sha256=checkpoint,splits={'train':[dict(episode=0,receipt_path=str(receipt),receipt_sha256=sha(receipt))]})))
            audit=root/'prior_audit.json';audit.write_text(json.dumps(dict(actual_exit_code=0,manifest_sha256=sha(manifest),episodes_verified=1,raw_rows_verified=1001,latent_dim=192,dtype='float32')))
            sidecar=root/'rebind.json';proof=dict(actual_exit_code=0,cube_count=4,data_id='cube-quadruple-play-v0',modality='rgb',cache_dir=str(cache),cache_manifest_sha256=sha(manifest),checkpoint_sha256=checkpoint,raw_rows_per_episode=1001,split_episodes={'train':1,'val':0},
                splits={'train':dict(source_npz=str(source),original_accepted_stat=old,current_accepted_stat=current,original_raw_sha256=sha(source),current_raw_sha256=sha(source),current_source_hash_actual_exit_code=0,sourcebeforeafterstat={'before':current,'after':current})},
                cache_full_SHA_verification=dict(path=str(audit),sha256=sha(audit),scope='prior_producer_host'))
            sidecar.write_text(json.dumps(proof));args=dict(cache_dir=cache,split='train',manifest_sha256=sha(manifest),checkpoint_sha256=checkpoint,source_npz=source,expected_rows=1001)
            with self.assertRaises(AssertionError):quad.load_cached_features(**args)
            before={str(p):sha(p) for p in [receipt,manifest,latent,source]}
            actual,binding=quad.load_cached_features(**args,source_rebind={'path':str(sidecar),'sha256':sha(sidecar)})
            np.testing.assert_array_equal(actual,values);self.assertFalse(actual.flags.writeable)
            self.assertEqual(binding['source_rebind']['original_accepted_stat'],old)
            self.assertEqual(binding['source_rebind']['current_accepted_stat'],current)
            self.assertTrue(binding['vector_verification_this_loader']['current_bytes_SHA_and_finite_verified'])
            self.assertEqual(before,{str(p):sha(p) for p in [receipt,manifest,latent,source]})
            proof['splits']['train']['current_raw_sha256']='b'*64;sidecar.write_text(json.dumps(proof))
            with self.assertRaises(AssertionError):quad.load_cached_features(**args,source_rebind={'path':str(sidecar),'sha256':sha(sidecar)})


if __name__=='__main__':unittest.main()
