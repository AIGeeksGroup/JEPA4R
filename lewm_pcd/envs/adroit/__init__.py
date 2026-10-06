"""Lazy native collection and independent fragment registrations."""
from importlib import import_module
from gymnasium.envs.registration import register, registry

for task, env_id in [('pen-v0', 'swm/AdroitPen-v0'), ('relocate-v0', 'swm/AdroitRelocate-v0')]:
    if env_id not in registry:
        register(id=env_id, entry_point='envs.adroit.env:AdroitEnv', kwargs={'env_name': task})
if 'swm/AdroitPenFragment-v0' not in registry:
    register(id='swm/AdroitPenFragment-v0', entry_point='envs.adroit.fragment_env:PenFragmentEnv')

_EXPORTS = {'AdroitEnv': 'env', 'AdroitFeedbackPolicy': 'policy',
            'SavedStateExpert': 'policy', 'AdroitPointCloudWrapper': 'pcd',
            'PenFragmentEnv': 'fragment_env'}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module('.' + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
