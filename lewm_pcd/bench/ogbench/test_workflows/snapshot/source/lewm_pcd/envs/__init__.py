from gymnasium.envs.registration import register

from .eval_policy import PCDWorldModelPolicy, PointCloudScaler, WindowedWorldModelPolicy
from .expert_policy import PCDExpertPolicy
from .pcd_wrapper import PointCloudWrapper

register(
    id='swm/OGBCubePCD-v0',
    entry_point='envs.cube_pcd_env:CubePCDEnv',
)

# RealMan RM75-B + EG2-4C2 instead of UR5e + 2F-85 (envs/cube_rm75_env.py): pixel and point-cloud variants
register(
    id='swm/OGBCubeRM75-v0',
    entry_point='envs.cube_rm75_env:RM75CubeEnv',
)
register(
    id='swm/OGBCubeRM75PCD-v0',
    entry_point='envs.cube_rm75_env:RM75CubePCDEnv',
)

__all__ = ['PointCloudWrapper', 'PCDExpertPolicy', 'PCDWorldModelPolicy', 'PointCloudScaler', 'WindowedWorldModelPolicy']
