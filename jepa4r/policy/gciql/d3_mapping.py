# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/d3_mapping.py (verbatim except: imports / paths adapted to the
# package; lines wrapped at 120 columns)
"""Row-level R4 evidence using the original Cube current/goal input seam."""
import numpy as np
from jepa4r.policy.gciql.gates.arrays import array_identity
from .downstream.vendor.cube_inputs import CubeInputs


def build_episode(features, states, actions, terminals, *, split, cube_count, episode, raw_start, source_sha256):
    if split not in ('train','val'):
        raise ValueError('official train/val split is required')
    features,states,actions,terminals=map(np.asarray,(features,states,actions,terminals))
    rows=len(features);seam=CubeInputs(cube_count,features.shape[-1])
    arrays = dict(
        states=states.copy(),
        current=seam.current(features, states),
        goal=seam.goal(features, states),
        actions=actions.copy(),
        terminals=terminals.copy(),
        raw_row_id=np.arange(raw_start, raw_start + rows, dtype=np.int64),
        episode_id=np.full(rows, episode, np.int64),
        step_id=np.arange(rows, dtype=np.int64),
        split_id=np.full(rows, 0 if split == 'train' else 1, np.int8),
        task_id=np.full(rows, -1, np.int32),
    )
    meta = dict(
        schema='cube_r4_episode_mapping_v1',
        split=split,
        cube_count=cube_count,
        episode=episode,
        raw_start=raw_start,
        rows=rows,
        source_sha256=source_sha256,
        task_id=-1,
        task_semantics='official Play has no benchmark task label; -1 is explicitly unlabelled',
        feature_row_sha256=[array_identity(row) for row in features],
        source_arrays={
            k: array_identity(v) for k, v in dict(states=states, actions=actions, terminals=terminals).items()
        },
    )
    validate_episode(features,arrays,meta)
    return arrays,meta


def validate_episode(features,arrays,meta):
    features=np.asarray(features);rows=meta['rows'];split=meta['split']
    if meta.get('schema')!='cube_r4_episode_mapping_v1'or split not in ('train','val')or meta.get('task_id')!=-1:
        raise ValueError('R4 schema/split/unlabelled task identity differs')
    if (len(features)!=rows or features.dtype!=np.float32 or
            [array_identity(row)for row in features]!=meta['feature_row_sha256']):
        raise ValueError('R4 feature row order/identity differs')
    expected=dict(raw_row_id=np.arange(meta['raw_start'],meta['raw_start']+rows,dtype=np.int64),
                  episode_id=np.full(rows,meta['episode'],np.int64),step_id=np.arange(rows,dtype=np.int64),
                  split_id=np.full(rows,0 if split=='train'else 1,np.int8),task_id=np.full(rows,-1,np.int32))
    for name,value in expected.items():
        if not np.array_equal(arrays[name],value):
            raise ValueError('R4 raw row/episode/step/split/task alignment differs: '+name)
    for name,digest in meta['source_arrays'].items():
        if array_identity(arrays[name])!=digest:
            raise ValueError('R4 original source array differs: '+name)
    seam=CubeInputs(meta['cube_count'],features.shape[-1])
    if (not np.array_equal(arrays['current'],seam.current(features,arrays['states']))or
            not np.array_equal(arrays['goal'],seam.goal(features,arrays['states']))or
            arrays['actions'].shape!=(rows,5)or arrays['terminals'].shape!=(rows,)or
            not all(np.isfinite(value).all()for value in arrays.values())):
        raise ValueError('R4 complete current/goal/action/terminal arrays differ')
    return dict(passed=True,rows=rows,current_dim=arrays['current'].shape[1],goal_dim=arrays['goal'].shape[1])


def validate_goal_link(current,goal):
    if (
        any(current.get(k) != goal.get(k) for k in ('split', 'cube_count', 'source_sha256'))
        or current.get('task_id') != -1
        or goal.get('task_id') != -1
    ):
        raise ValueError('goal crosses its official Play split/source or invents a task label')
    return True
