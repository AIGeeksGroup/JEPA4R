"""Environment exports with simulator backends loaded only when selected."""
from importlib import import_module

_EXPORTS = {
    'PCDWorldModelPolicy': 'eval_policy',
    'PointCloudScaler': 'eval_policy',
    'WindowedWorldModelPolicy': 'eval_policy',
    'PCDExpertPolicy': 'expert_policy',
    'PointCloudWrapper': 'pcd_wrapper',
}
__all__ = list(_EXPORTS)

# Retain main's original environment IDs without importing their heavy backends.
try:
    from gymnasium.envs.registration import register, registry
except ModuleNotFoundError as error:
    if error.name != 'gymnasium':
        raise
else:
    for environment, entry in (
        ('swm/OGBCubePCD-v0', 'envs.cube_pcd_env:CubePCDEnv'),
        ('swm/OGBCubeRM75-v0', 'envs.cube_rm75_env:RM75CubeEnv'),
        ('swm/OGBCubeRM75PCD-v0', 'envs.cube_rm75_env:RM75CubePCDEnv'),
    ):
        if environment not in registry:
            register(id=environment, entry_point=entry)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module('.' + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
