"""NumPy's BLAS thread count for the Cube evaluation: the state evaluator's IK math ran with 4 BLAS threads."""

from pathlib import Path
import sys

import numpy as np

_BLAS_LIMITER = None


def numpy_blas_paths(pools, numpy_file=None):
    owner = (Path(numpy_file or np.__file__).resolve().parent.parent / "numpy.libs").resolve()
    paths = [p["filepath"] for p in pools if p["user_api"] == "blas" and Path(p["filepath"]).resolve().parent == owner]
    if len(paths) != 1:
        raise ValueError("expected exactly one BLAS library linked into the NumPy wheel")
    return paths


def configure_numpy_physics_runtime():
    """Limit NumPy's BLAS to 4 threads; other thread pools stay as they are and Torch stays at 1 thread."""
    global _BLAS_LIMITER
    from threadpoolctl import ThreadpoolController, threadpool_info

    paths = numpy_blas_paths(threadpool_info())
    _BLAS_LIMITER = ThreadpoolController().select(filepath=paths).limit(limits=4, user_api="blas")
    if "torch" in sys.modules and sys.modules["torch"].get_num_threads() != 1:
        raise ValueError("Torch must run with 1 thread here")
