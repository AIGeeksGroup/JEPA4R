"""Direct state inputs plus the previously selected one-batch ordered prefetch.

Sampling/RNG and deferred materialization reuse frozen implementations. This
adapter adds only IDs, per-object ALL-distance labels and actual goal XYZ.
"""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import types

import numpy as np
from utils.datasets import GCDataset

BASE_PATH=Path(__file__).resolve().parent.parent/'distance_inputs.py'
BASE_SHA='b9906dc2f4e7a38826f5ee05031713618a6ce408ee5555302801505452758f90'
ORDERED_SHA='4508ce847694976f8af924215d11985ac4d795093b22d62e8424928346fbdda2'
IDS=('_current_ids','_next_ids','_value_goal_ids','_actor_goal_ids')


def _base():
    assert hashlib.sha256(BASE_PATH.read_bytes()).hexdigest()==BASE_SHA
    name='_frozen_distance_math_state_priority'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,BASE_PATH)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


def ordered_class(path,state_dim):
    path=Path(path)
    source=path.read_bytes()
    assert hashlib.sha256(source).hexdigest()==ORDERED_SHA
    before="('observations', 46)"
    after=f"('observations', {state_dim})"
    text=source.decode();assert text.count(before)==1
    effective=text.replace(before,after)
    assert effective.count(after)==1 and effective.replace(after,before)==text
    name=f'_frozen_ordered_state_width_{state_dim}'
    if name not in sys.modules:
        module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
        exec(compile(effective,str(path),'exec'),module.__dict__)
    proof=dict(original_source_sha256=ORDERED_SHA,actual_state_dim=state_dim,
        effective_source_sha256=hashlib.sha256(effective.encode()).hexdigest(),
        change='only original observation-width validation; no sampling/materialization/barrier changes',
        original_source_unmodified=True)
    return sys.modules[name].OrderedInput,proof


class _StateSampler(GCDataset):
    def sample_goals(self,*args,**kwargs):
        ids=super().sample_goals(*args,**kwargs)
        self._goal_ids.append(ids.copy())
        return ids

    def sample(self,batch_size,idxs=None,evaluation=False):
        if idxs is None:
            idxs=self.dataset.get_random_idxs(batch_size)
        self._goal_ids=[]
        batch=super().sample(batch_size,idxs=idxs,evaluation=evaluation)
        value_ids,actor_ids=self._goal_ids
        # These small ID arrays need no deferred read or RNG in the worker.
        batch.update(zip(IDS,(idxs.copy(),idxs+1,value_ids,actor_ids)))
        return batch


class StateInputs:
    def __init__(self,dataset,config,cubes,mode='baseline',batch_size=1024,ordered_input_path=None):
        if cubes not in (1,2,3) or mode not in ('baseline','prefetch1') or batch_size!=1024:
            raise ValueError('Only actual Cube1/2/3 state and selected B1024 baseline/prefetch1')
        if config['frame_stack'] is not None or config['p_aug']!=0:
            raise ValueError('Original no-stack/no-augmentation profile required')
        states=dataset['observations'];state_dim=19+9*cubes
        if states.shape!=(dataset.size,state_dim) or states.dtype!=np.float32 or states.flags.writeable:
            raise ValueError('Original readonly fullstate28/37/46 required')
        self.cubes,self.batch_size=cubes,batch_size
        self.current_dim,self.goal_dim=state_dim,3*cubes
        self.features=self.feature=None
        self.sampler=_StateSampler(dataset,config)
        self._math=_base()
        path=ordered_input_path or os.environ.get('ORDERED_INPUT_SOURCE')
        if path is None:
            path=Path(__file__).resolve().parents[2]/'official_state_workers_v1/ordered_input.py'
        selected,proof=ordered_class(path,state_dim)
        self.inputs=selected(self.sampler,mode=mode,batch_size=batch_size)
        self.input_metadata=dict(cubes=cubes,mode='state',input_mode=mode,current_dim=state_dim,goal_dim=3*cubes,
            latent_dim=0,features=None,synthetic_zero_padding=False,lookup_workers=0,
            background_materializers=1 if mode=='prefetch1' else 0,ahead_batches=1 if mode=='prefetch1' else 0,
            sampler='originalGCDataset on caller; only deterministic array materialization in background',
            projection='original full current/next; goal XYZ only',
            reward='ALL(per-cube current-to-valuegoal L2<=.4 official scaled); r=success-1; mask=1-success',
            ordered_source=proof,distance_math_source_sha256=BASE_SHA)

    def sample(self,batch_size=None,*,prefetch_next=True):
        if batch_size==1:
            # Preserve the original one-example factory sampling and host RNG.
            self.inputs.barrier()
            batch=self.sampler.sample(1)
            self.inputs.barrier()
        else:
            batch=self.inputs.sample(batch_size,prefetch_next=prefetch_next)
        values=[batch.pop(key) for key in IDS]
        self.last_ids=dict(zip(('current','next','value_goal','actor_goal'),values))
        current,value_goal=batch['observations'],batch['value_goals']
        actual_batch_size=len(current)
        xyz=self._math.object_xyz
        success=np.all(np.linalg.norm(xyz(current,self.cubes)-xyz(value_goal,self.cubes),axis=-1)<=.4,axis=-1)
        batch['rewards']=success.astype(batch['rewards'].dtype)-1.
        batch['masks']=1.-success.astype(batch['masks'].dtype)
        batch['value_goals']=xyz(value_goal,self.cubes).reshape(actual_batch_size,3*self.cubes)
        batch['actor_goals']=xyz(batch['actor_goals'],self.cubes).reshape(actual_batch_size,3*self.cubes)
        return batch

    def __getattr__(self,name):
        return getattr(self.sampler,name)

    @property
    def pending(self):
        return self.inputs.pending

    @property
    def committed_rng_state(self):
        return self.inputs.committed_rng_state

    def barrier(self):
        self.inputs.barrier()

    def close(self):
        self.inputs.close()


class PrefetchStateInputs(StateInputs):
    def __init__(self,dataset,config,cubes,batch_size=1024,ordered_input_path=None):
        super().__init__(dataset,config,cubes,'prefetch1',batch_size,ordered_input_path)


def create_agent(agent_class,seed,batch,config):
    return _base().create_agent(agent_class,seed,batch,config)
