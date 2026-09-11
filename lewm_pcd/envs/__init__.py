from gymnasium.envs.registration import register

from .eval_policy import PCDWorldModelPolicy, PointCloudScaler, WindowedWorldModelPolicy
from .expert_policy import PCDExpertPolicy
from .pcd_wrapper import PointCloudWrapper

register(
    id='swm/OGBCubePCD-v0',
    entry_point='envs.cube_pcd_env:CubePCDEnv',
)

__all__ = ['PointCloudWrapper', 'PCDExpertPolicy', 'PCDWorldModelPolicy', 'PointCloudScaler', 'WindowedWorldModelPolicy']
