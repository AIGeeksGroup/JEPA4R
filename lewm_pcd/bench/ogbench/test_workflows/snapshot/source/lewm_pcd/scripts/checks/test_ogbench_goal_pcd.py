import gymnasium as gym
import mujoco
import numpy as np

import envs  # noqa: F401
from envs import PointCloudWrapper


def test_native_reset_returns_current_and_official_goal_point_clouds():
    base_env = gym.make(
        "swm/OGBCubePCD-v0",
        ob_type="states",
        env_type="single",
        max_episode_steps=2,
        terminate_at_goal=False,
        visualize_info=False,
    )
    env = PointCloudWrapper(
        base_env,
        num_points=1200,
        cameras=("front_pixels",),
        crop_box=((0.10, -0.50, -0.02), (0.75, 0.50, 0.60)),
        voxel_size=0.005,
        cube_fraction=0.25,
    )

    try:
        _, info = env.reset(
            seed=10001,
            options={"task_id": 1, "render_goal": True},
        )
        model = env.unwrapped._model
        camera_names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
            for camera_id in range(model.ncam)
        }
        front_camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "front_pixels"
        )
    finally:
        env.close()

    assert env.cameras == ("front_pixels",)
    assert front_camera_id >= 0
    assert model.cam_bodyid[front_camera_id] == 0
    assert not any(name.startswith("pcd_") for name in camera_names if name)
    assert info["pcd_xyz"].shape == (1200, 3)
    assert info["pcd_rgb"].shape == (1200, 3)
    assert info["goal_pcd_xyz"].shape == (1200, 3)
    assert info["goal_pcd_rgb"].shape == (1200, 3)
    assert info["goal_pcd_xyz"].dtype == np.float32
    assert info["goal_pcd_rgb"].dtype == np.uint8
    assert np.isfinite(info["goal_pcd_xyz"]).all()
    assert np.count_nonzero(np.any(info["goal_pcd_xyz"] != 0, axis=1)) == 1200
