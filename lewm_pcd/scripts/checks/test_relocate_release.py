"""Permanent regression checks for replay shard IDs and the shared numerical contract."""
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def merger():
    spec = importlib.util.spec_from_file_location('replay_shard_merge',ROOT/'scripts/data/merge_h5_shards.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def shard(path, episode, value):
    with h5py.File(path,'w') as out:
        for key,data in [('ep_len',np.array([2],np.int32)),('ep_offset',np.array([0],np.int64)),
                         ('ep_idx',np.full(2,episode,np.int64)),('action',np.full((2,30),value,np.float32))]:
            out.create_dataset(key,data=data,maxshape=(None,*data.shape[1:]))


def test_replay_merge_preserves_global_ids_and_values(tmp_path):
    a,b=tmp_path/'a.h5',tmp_path/'b.h5';shard(a,7,.25);shard(b,8,-.5)
    with pytest.raises(ValueError,match='shard-local'):merger().append_shard(a,b)
    merger().append_shard(a,b,preserve_episode_indices=True)
    with h5py.File(a) as out:
        np.testing.assert_array_equal(out['ep_idx'][:],[7,7,8,8])
        np.testing.assert_array_equal(out['ep_offset'][:],[0,2])
        np.testing.assert_array_equal(out['action'][:2],np.full((2,30),.25,np.float32))
        np.testing.assert_array_equal(out['action'][2:],np.full((2,30),-.5,np.float32))
    with h5py.File(b) as original:assert len(original['action'])==2


def test_overlapping_replay_shards_are_rejected_before_append(tmp_path):
    a,b=tmp_path/'a.h5',tmp_path/'b.h5';shard(a,7,.25);shard(b,7,-.5)
    with pytest.raises(ValueError,match='overlapping'):merger().append_shard(a,b,True)
    with h5py.File(a) as out:assert len(out['action'])==2


def test_bc_replay_and_cem_use_same_sampling_values():
    direct=json.loads((ROOT/'config/adroit_relocate_sampling.json').read_text())
    cem=yaml.safe_load((ROOT/'config/eval/adroit_relocate_pcd_fragment.yaml').read_text())['world']['pre_wrappers'][0]
    cem=dict(cem);cem.pop('_target_');assert cem.pop('keep_rgb') is False
    cem['camera']=cem.pop('cameras')[0];cem['obj_fraction']=cem.pop('cube_fraction');cem['task']='relocate-v0'
    assert direct==cem
