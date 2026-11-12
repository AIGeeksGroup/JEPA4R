"""Lazy native collection and independent fragment registrations."""
from importlib import import_module
from gymnasium.envs.registration import register, registry

if 'swm/AdroitRelocate-v0' not in registry:
    register(id='swm/AdroitRelocate-v0', entry_point='envs.adroit.env:AdroitEnv', kwargs={'env_name': 'relocate-v0'})
if 'swm/AdroitRelocateFragment-v0' not in registry:
    register(id='swm/AdroitRelocateFragment-v0', entry_point='envs.adroit.fragment_env:RelocateFragmentEnv')

_EXPORTS = {'AdroitEnv': 'env', 'AdroitFeedbackPolicy': 'policy',
            'SavedStateExpert': 'policy', 'AdroitPointCloudWrapper': 'pcd',
            'RelocateFragmentEnv': 'fragment_env'}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module('.' + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
