from gymnasium.envs.registration import register
from .expert_policy import PCDExpertPolicy
from .pcd_wrapper import PointCloudWrapper

__all__ = ['PCDExpertPolicy', 'PointCloudWrapper']
register(id='swm/OGBCubePCD-v0', entry_point='envs.cube_pcd_env:CubePCDEnv')
