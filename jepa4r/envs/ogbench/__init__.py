"""OGBench manipulation: Cube (1-4 cubes), Scene, Puzzle 3x3 / 4x4 / 4x5 [park2025ogbench].

Wrapper that renders rgb + depth + segmentation from the camera behind OGBench's pixel observations, exposes the robot
state (Cube) and button lights (Puzzle), runs the five official evaluation tasks (env_cfg.eval) and reports success,
and provides the expert control used by the data mixtures.

Modules (one each because their module-level constants -- CAMERA, CROP_BOX, NUM_POINTS, make_env, point_cloud, ... --
differ between Scene and Puzzle):
  cube_pcd_env      swm/OGBCube-v0 + extra fixed cameras (UR5e Cube-Single; the RM75-B twin builds on it)
  expert_policy     stable-worldmodel's plan-oracle ExpertPolicy accepting swm/OGBCubePCD-v0
  ogb_scene_env     OGBench scene-v0 (task mode / data collection), the scene three-part collection plan
  ogb_scene_expert  scripted per-task scene expert (CortexBench-style scene demos)
  ogb_puzzle_env    OGBench puzzle-{3x3,4x4,4x5}-v0 and OGBench's noisy collection policy
  ogb_pcd           their single-depth-camera point clouds (subclasses of jepa4r.data.pointcloud.PointCloudWrapper)
Importing this package registers the gymnasium ids the collection configs name; the simulator modules load on first
use.
"""

# gymnasium registrations
ENV_IDS = {
    "swm/OGBCubePCD-v0": ("jepa4r.envs.ogbench.cube_pcd_env:CubePCDEnv", {}),
    "swm/OGBenchScene-v0": ("jepa4r.envs.ogbench.ogb_scene_env:OGBSceneEnv", {}),
    "swm/OGBenchPuzzle3x3-v0": ("jepa4r.envs.ogbench.ogb_puzzle_env:OGBPuzzleEnv", {}),
    "swm/OGBenchPuzzle4x4-v0": ("jepa4r.envs.ogbench.ogb_puzzle_env:OGBPuzzleEnv", {"env_type": "4x4"}),
    "swm/OGBenchPuzzle4x5-v0": ("jepa4r.envs.ogbench.ogb_puzzle_env:OGBPuzzleEnv", {"env_type": "4x5"}),
}


def register_envs():
    try:
        from gymnasium.envs.registration import register, registry
    except ModuleNotFoundError:  # gymnasium is needed to run the envs, not to import this package
        return
    for env_id, (entry, kwargs) in ENV_IDS.items():
        if env_id not in registry:
            register(id=env_id, entry_point=entry, kwargs=kwargs)


register_envs()
