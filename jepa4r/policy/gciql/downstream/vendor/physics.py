import hashlib
import sys
from pathlib import Path
import numpy as np
_BLAS_LIMITER=None

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def numpy_blas_paths(pools, numpy_file=None):
    owner = (Path(numpy_file or np.__file__).resolve().parent.parent / 'numpy.libs').resolve()
    paths = [p['filepath'] for p in pools if p['user_api'] == 'blas' and Path(p['filepath']).resolve().parent == owner]
    if len(paths) != 1:
        raise ValueError('Exactly one actual NumPy wheel-linked BLAS library required')
    return paths

def configure_numpy_physics_runtime():
    """Match original state evaluator IK math; keep Torch threads separately1."""
    global _BLAS_LIMITER
    from threadpoolctl import threadpool_info, ThreadpoolController
    before = threadpool_info()
    paths = numpy_blas_paths(before)
    _BLAS_LIMITER = ThreadpoolController().select(filepath=paths).limit(limits=4, user_api='blas')
    after = threadpool_info()
    pools = [p for p in after if p['filepath'] in paths]
    if len(pools) != 1 or pools[0]['num_threads'] != 4:
        raise ValueError('Original state NumPy-linked BLAS physics runtime requires4 threads')
    others = [p for p in after if p['filepath'] not in paths]
    if {p['filepath']: p['num_threads'] for p in before if p['filepath'] not in paths} != {p['filepath']: p['num_threads'] for p in others}:
        raise ValueError('Targeted NumPy limit unexpectedly changed other runtime pools')
    torch_threads = sys.modules['torch'].get_num_threads() if 'torch' in sys.modules else None
    if torch_threads is not None and torch_threads != 1:
        raise ValueError('Torch inference must remain1 thread')
    return {'numpy_version': np.__version__, 'numpy_path': np.__file__, 'BLAS_threads': 4, 'threadpools': pools, 'other_threadpools_record_only': others, 'library_sha256': {p['filepath']: sha(p['filepath']) for p in pools}, 'torch_num_threads': torch_threads, 'reference': 'official_state_sr NumPy BLAS4 physics; other BLAS/OpenMP pools are recorded, not constrained to4'}
