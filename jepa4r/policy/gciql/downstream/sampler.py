# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/downstream/sampler.py (verbatim except: DistanceInputs also accepts the 768-D VISReg and 1024-D V-JEPA 2.1 latents that this file's own load_cached_features contract lists; the source accepted only 192/512, which conflicts with the paper's Tab. 1 VISReg / V-JEPA 2.1 Cube rows, and the paper wins)
"""Thin multi-Cube input projection and per-object ALL-distance reward.

The original GCDataset owns all sampling and RNG. Features are one readonly
FP32 RAM array per split, shared by same-batch lookup threads, never workers
that sample future batches. State-only constructs no visual dimensions.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from ._source import install_impls
install_impls()
from utils.datasets import GCDataset
from .vendor.cube_inputs import CubeInputs


def file_stat(path):
    value=Path(path).stat()
    return dict(size=value.st_size,mtime_ns=value.st_mtime_ns,ctime_ns=value.st_ctime_ns,
                device=value.st_dev,inode=value.st_ino)


def read_json(path,expected_sha):
    before=file_stat(path)
    blob=Path(path).read_bytes()
    assert hashlib.sha256(blob).hexdigest()==expected_sha,str(path)
    assert file_stat(path)==before,str(path)
    return json.loads(blob)


def load_cached_features(cache_dir,split,manifest_sha256,checkpoint_sha256,source_npz,expected_rows,
                         feature_dim=192,encoder_family='lewm'):
    """One bounded per-episode read into one RAM allocation; no extra feature pass."""
    if split not in ('train','val') or expected_rows%1001:
        raise ValueError('Expected canonical raw1001 split')
    root=Path(cache_dir).resolve()
    manifest=read_json(root/'cache_manifest.json',manifest_sha256)
    assert manifest['actual_exit_code']==0 and manifest['checkpoint_sha256']==checkpoint_sha256
    if (encoder_family,feature_dim) not in (('lewm',192),('pldm',512),('lejepa',512),
                                          ('visreg',768),('vjepa',1024)):
        raise ValueError('Unsupported encoder family/output width')
    if encoder_family!='lewm':
        protocol=manifest['protocol']
        if protocol['encoder_family']!=encoder_family or protocol['latent_dim']!=feature_dim:
            raise ValueError('Feature cache encoder binding mismatch')
    entries=manifest['splits'][split]
    assert len(entries)*1001==expected_rows
    assert [entry['episode'] for entry in entries]==list(range(len(entries)))
    source=Path(source_npz).resolve()
    source_stat=file_stat(source)
    result=np.empty((expected_rows,feature_dim),dtype=np.float32)
    stream_hash=hashlib.sha256()
    for entry in entries:
        record=read_json(entry['receipt_path'],entry['receipt_sha256'])
        episode=entry['episode'];start=episode*1001
        assert record['split']==split and record['episode']==episode and record['actual_exit_code']==0
        assert record['raw_row_start']==start and record['raw_row_count']==1001 and record['main_rows']==1000
        assert record['terminal_raw_id']==start+1000 and record['checkpoint_sha256']==checkpoint_sha256
        expected_ids=np.arange(start,start+1001,dtype='<i8')
        assert record['canonical_raw_ids_sha256']==hashlib.sha256(expected_ids.tobytes()).hexdigest()
        assert Path(record['source_npz']['path']).resolve()==source
        assert record['source_npz']['accepted_stat']==source_stat
        latent=record['latent'];path=Path(latent['path']).resolve()
        assert path.is_relative_to(root) and latent['shape']==[1001,feature_dim] and latent['dtype']=='float32'
        before=file_stat(path);assert before['size']==latent['bytes']
        blob=path.read_bytes()
        assert hashlib.sha256(blob).hexdigest()==latent['sha256']
        assert file_stat(path)==before
        values=np.load(io.BytesIO(blob),allow_pickle=False)
        assert values.shape==(1001,feature_dim) and values.dtype==np.float32 and np.isfinite(values).all()
        result[start:start+1001]=values
        stream_hash.update(entry['receipt_sha256'].encode())
        stream_hash.update(latent['sha256'].encode())
    assert file_stat(source)==source_stat
    result.setflags(write=False)
    return result,dict(split=split,rows=expected_rows,shape=list(result.shape),dtype='float32',
        manifest_sha256=manifest_sha256,checkpoint_sha256=checkpoint_sha256,
        source_npz=str(source),source_stat=source_stat,canonical_rows_per_episode=1001,
        loaded_receipt_feature_stream_sha256=stream_hash.hexdigest(),
        readonly=True,one_RAM_allocation=True,encoder_calls=0,
        encoder_family=encoder_family,latent_dim=feature_dim)


def object_xyz(states,cubes):
    return np.stack([states[:,19+9*i:22+9*i] for i in range(cubes)],axis=1)


@dataclass
class DistanceInputs(GCDataset):
    cubes: int=2
    mode: str='state'
    features: object=None
    workers: int=0
    latent_dim: int=192

    def __post_init__(self):
        super().__post_init__()
        if self.cubes not in (1,2,3,4) or self.mode not in ('state','rgb','pcd'):
            raise ValueError('Explicit Cube1..4 and state/RGB/PCD mode required')
        if self.config['frame_stack'] is not None or self.config['p_aug']!=0:
            raise ValueError('Keep the original no-stack/no-augmentation state profile')
        states=self.dataset['observations']
        if states.shape!=(self.size,19+9*self.cubes) or states.dtype!=np.float32:
                raise ValueError('Original matching official 19+9N state required')
        if states.flags.writeable:
            raise ValueError('Original state must remain readonly')
        if self.mode=='state':
            if self.features is not None or self.workers!=0:
                raise ValueError('State-only uses no latent array or synthetic lookup workers')
            self.current_dim,self.goal_dim=19+9*self.cubes,3*self.cubes
            self.contract=None
        else:
            if self.latent_dim not in (192,512,768,1024):   # source: (192,512); paper Tab. 1 VISReg 768 / V-JEPA 2.1 1024
                raise ValueError('Selected encoder must produce actual192, actual512, actual768 or actual1024')
            if not isinstance(self.features,np.ndarray) or self.features.shape!=(self.size,self.latent_dim):
                raise ValueError('One aligned readonly FP32 feature array at the actual encoder width required')
            if self.features.dtype!=np.float32 or self.features.flags.writeable:
                raise ValueError('Features must be readonly FP32 RAM')
            if self.workers not in (0,4):
                raise ValueError('Use selected four role threads or zero-thread reference')
            self.current_dim,self.goal_dim=self.latent_dim+19,self.latent_dim+3*self.cubes
            self.contract=CubeInputs(self.cubes,self.latent_dim)
        self.executor=ThreadPoolExecutor(max_workers=self.workers,thread_name_prefix='cube-role-lookup') if self.workers else None
        self.input_metadata=dict(cubes=self.cubes,mode=self.mode,current_dim=self.current_dim,goal_dim=self.goal_dim,
            state_indices='full official state' if self.mode=='state' else 'originalstate[0:19]',
            goal_XYZ_offsets=[19+9*i for i in range(self.cubes)],
            latent_dim=0 if self.mode=='state' else self.latent_dim,synthetic_zero_padding=False,
            sampler='originalGCDataset once in parent',lookup_workers=self.workers,
            reward='ALL(each cube current-to-valuegoal L2<=0.4 scaled); r=success-1; mask=1-success')

    def sample_goals(self,*args,**kwargs):
        ids=super().sample_goals(*args,**kwargs)
        self._goal_ids.append(ids.copy())
        return ids

    def sample(self,batch_size,idxs=None,evaluation=False):
        if idxs is None:
            idxs=self.dataset.get_random_idxs(batch_size)
        idxs=np.asarray(idxs)
        if idxs.shape!=(batch_size,) or not np.issubdtype(idxs.dtype,np.integer):
            raise ValueError('Original integer current IDs required')
        if np.any(idxs<0) or np.any(idxs>=self.size-1):
            raise ValueError('Current IDs cannot be the final raw terminal')
        if 'valids' in self.dataset and not np.all(self.dataset['valids'][idxs]>0):
            raise ValueError('An invalid raw terminal cannot be sampled as current')
        self._goal_ids=[]
        batch=super().sample(batch_size,idxs=idxs,evaluation=evaluation)
        value_ids,actor_ids=self._goal_ids
        current=batch['observations'];value_goal=batch['value_goals']
        success=np.all(np.linalg.norm(object_xyz(current,self.cubes)-object_xyz(value_goal,self.cubes),axis=-1)<=.4,axis=-1)
        # Preserve the original sampler's reward/mask dtype; only labels change.
        batch['rewards']=success.astype(batch['rewards'].dtype)-1.
        batch['masks']=1.-success.astype(batch['masks'].dtype)
        if self.mode=='state':
            # Direct actual dimensions. Never call the zero192 oracle branch.
            batch['value_goals']=object_xyz(value_goal,self.cubes).reshape(batch_size,3*self.cubes)
            batch['actor_goals']=object_xyz(batch['actor_goals'],self.cubes).reshape(batch_size,3*self.cubes)
        else:
            roles=('observations','next_observations','value_goals','actor_goals')
            ids=(idxs,idxs+1,value_ids,actor_ids)
            if self.executor:
                futures=[self.executor.submit(self.features.__getitem__,item) for item in ids]
                visual=[future.result() for future in futures]
            else:
                visual=[self.features[item] for item in ids]
            for role,latent in zip(roles,visual):
                operation=self.contract.current if role in ('observations','next_observations') else self.contract.goal
                batch[role]=operation(latent,batch[role],oracle=False)
        self.last_ids=dict(current=idxs.copy(),next=idxs+1,value_goal=value_ids,actor_goal=actor_ids)
        return batch

    def close(self):
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor=None


def create_agent(agent_class,seed,batch,config):
    """Reuse the already-adapted original GCIQL factory at actual input widths."""
    return agent_class.create(seed,batch['observations'][:1],batch['actions'][:1],config,
                              ex_goals=batch['actor_goals'][:1])
