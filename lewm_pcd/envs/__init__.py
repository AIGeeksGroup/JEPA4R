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

# OGBench's own scene / puzzle-3x3 (the ogbench package, not stable-worldmodel's ports) for the world-model data:
# envs/ogb_scene_env.py, envs/ogb_puzzle_env.py; point clouds envs/ogb_pcd.py
register(
    id='swm/OGBenchScene-v0',
    entry_point='envs.ogb_scene_env:OGBSceneEnv',
)
register(
    id='swm/OGBenchPuzzle3x3-v0',
    entry_point='envs.ogb_puzzle_env:OGBPuzzleEnv',
)
register(
    id='swm/OGBenchPuzzle4x4-v0',
    entry_point='envs.ogb_puzzle_env:OGBPuzzleEnv',
    kwargs=dict(env_type='4x4'),
)
register(
    id='swm/OGBenchPuzzle4x5-v0',
    entry_point='envs.ogb_puzzle_env:OGBPuzzleEnv',
    kwargs=dict(env_type='4x5'),
)

__all__ = ['PointCloudWrapper', 'PCDExpertPolicy', 'PCDWorldModelPolicy', 'PointCloudScaler', 'WindowedWorldModelPolicy']
