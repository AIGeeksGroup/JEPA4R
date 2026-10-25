"""Lightweight projection import; the original JAX sampler loads on demand."""
from .vendor.cube_inputs import CubeInputs


def __getattr__(name):
    if name in ('DistanceInputs', 'load_cached_features', 'create_agent', 'object_xyz'):
        from . import sampler
        return getattr(sampler, name)
    raise AttributeError(name)
