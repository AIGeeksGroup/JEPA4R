"""Lazy native collection registration."""
from importlib import import_module
from gymnasium.envs.registration import register, registry

if 'swm/AdroitRelocate-v0' not in registry:
    register(
        id='swm/AdroitRelocate-v0', entry_point='jepa4r.envs.adroit.env:AdroitEnv', kwargs={'env_name': 'relocate-v0'}
    )

_EXPORTS = {'AdroitEnv': 'env', 'AdroitFeedbackPolicy': 'policy',
            'SavedStateExpert': 'policy', 'AdroitPointCloudWrapper': 'pcd'}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module('.' + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


# The env interface (configs/env/adroit_relocate.yaml impl): the native Adroit env of env_cfg.env_id.
def make(env_cfg, **kwargs):
    """gymnasium env ``env_cfg.env_id`` (registered above: AdroitEnv on mj_envs relocate-v0, paired rgb / depth /
    segmentation from env_cfg.camera.name); the collection, BC and evaluation run in the Relocate harness."""
    import gymnasium as gym
    return gym.make(str(env_cfg.env_id), **kwargs)
