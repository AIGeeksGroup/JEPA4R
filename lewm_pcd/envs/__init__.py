"""Relocate environment package; selected backends import their simulators."""
from importlib import import_module

__all__ = ['PCDWorldModelPolicy', 'PointCloudScaler', 'WindowedWorldModelPolicy']


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module('.eval_policy', __name__), name)
    globals()[name] = value
    return value
