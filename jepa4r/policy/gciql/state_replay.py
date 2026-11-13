# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/state_replay.py (verbatim except: imports / paths adapted to the package)
"""Restore every named numeric leaf of an independently exported state tree."""
import numpy as np
from jepa4r.policy.gciql.gates.arrays import array_identity


def flatten_state(state):
    arrays={}
    def visit(value,names=()):
        if isinstance(value,dict):
            for key,item in value.items():visit(item,names+(str(key),))
        elif isinstance(value,(tuple,list)):
            for key,item in enumerate(value):visit(item,names+(str(key),))
        else:arrays['/'.join(names)]=np.asarray(value)
    visit(state)
    return arrays


def state_from_arrays(template,arrays):
    expected=flatten_state(template)
    if set(expected)!=set(arrays):
        raise ValueError('raw-state leaf path set differs from the original factory template')
    mapping=[]
    def restore(value,names=()):
        if isinstance(value,dict):return {k:restore(v,names+(str(k),))for k,v in value.items()}
        if isinstance(value,(tuple,list)):return type(value)(restore(v,names+(str(k),))for k,v in enumerate(value))
        key='/'.join(names);reference=np.asarray(value);actual=np.asarray(arrays[key])
        if actual.shape!=reference.shape or actual.dtype.kind not in 'biuf' or not np.isfinite(actual).all():
            raise ValueError('raw-state shape/numeric/finite differs: '+key)
        # A fresh Python integer counter can serialize as int64 while the real
        # jitted endpoint is int32. Keep the endpoint dtype and verify it again
        # against the full endpoint fingerprint after framework reconstruction.
        if actual.dtype!=reference.dtype and not(actual.dtype.kind==reference.dtype.kind and actual.dtype.kind in 'iu'):
            raise ValueError('raw-state dtype differs: '+key)
        mapping.append(dict(path=key,shape=list(actual.shape),endpoint_dtype=str(actual.dtype),
                            template_dtype=str(reference.dtype),sha256=array_identity(actual)))
        return actual.copy()
    return restore(template),mapping
