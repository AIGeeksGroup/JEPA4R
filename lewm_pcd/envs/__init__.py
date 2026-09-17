from gymnasium.envs.registration import register

from .eval_policy import PCDWorldModelPolicy, PointCloudScaler, WindowedWorldModelPolicy
from .expert_policy import PCDExpertPolicy
from .pcd_wrapper import PointCloudWrapper

register(
    id='swm/OGBCubePCD-v0',
    entry_point='envs.cube_pcd_env:CubePCDEnv',
)

# Franka Emika Panda with OGBench's original 5-D task-space action.
register(
    id='swm/OGBCubeFranka-v0',
    entry_point='envs.cube_franka_env:FrankaCubeEnv',
)
register(
    id='swm/OGBCubeFrankaPCD-v0',
    entry_point='envs.cube_franka_env:FrankaCubePCDEnv',
)

__all__ = ['PointCloudWrapper', 'PCDExpertPolicy', 'PCDWorldModelPolicy', 'PointCloudScaler', 'WindowedWorldModelPolicy']
