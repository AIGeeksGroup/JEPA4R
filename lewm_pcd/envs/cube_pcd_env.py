"""``swm/OGBCube-v0`` plus four extra fixed cameras around the workspace.

The extra cameras exist only so that :class:`envs.PointCloudWrapper` can render
depth from several viewpoints and fuse them into one point cloud that covers
the robot and the cubes from all sides. Cameras are non-physical MuJoCo objects:
``qpos``/``qvel``, the dynamics, the env RNG and the expert policy are exactly
those of ``swm/OGBCube-v0``, so trajectories are identical for identical seeds.

Camera layout (all aimed at the centre of ``ManipSpaceEnv._workspace_bounds``):

    front_pixels      (original) +x side, the camera the official pixel dataset used
    pcd_front_left    azimuth  45 deg (+x,+y)  \
    pcd_back_left     azimuth 135 deg (-x,+y)   } distance 1.0 m, elevation 40 deg,
    pcd_back_right    azimuth 225 deg (-x,-y)   } diagonal so no camera looks straight
    pcd_front_right   azimuth 315 deg (+x,-y)  /  along the upper arm
    pcd_top           1.1 m straight above, image "up" = -x
"""

from __future__ import annotations

import numpy as np
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

# centre of ManipSpaceEnv._workspace_bounds ([0.25, -0.35, 0.02] .. [0.6, 0.35, 0.35]),
# lowered towards the table so the cubes stay central in every view
WORKSPACE_CENTER = np.array([0.425, 0.0, 0.08])
RING_CAMERAS = {  # azimuth in deg, measured from +x towards +y
    'pcd_front_left': 45.0,
    'pcd_back_left': 135.0,
    'pcd_back_right': 225.0,
    'pcd_front_right': 315.0,
}


def look_at_xyaxes(pos, target):
    """MuJoCo ``xyaxes`` (image right, image up) for a camera at ``pos`` looking at ``target``."""
    forward = np.asarray(target, float) - np.asarray(pos, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return tuple(float(v) for v in np.concatenate([right, up]))


class CubePCDEnv(CubeEnv):
    def __init__(
        self,
        *args,
        pcd_cam_distance: float = 1.0,
        pcd_cam_elevation: float = 40.0,
        pcd_top_height: float = 1.1,
        **kwargs,
    ):
        self._pcd_cam_distance = pcd_cam_distance
        self._pcd_cam_elevation = pcd_cam_elevation
        self._pcd_top_height = pcd_top_height
        super().__init__(*args, **kwargs)

    def add_objects(self, arena_mjcf):
        super().add_objects(arena_mjcf)  # cubes, targets, original cameras

        T = WORKSPACE_CENTER
        d = self._pcd_cam_distance
        el = np.deg2rad(self._pcd_cam_elevation)
        extra = {}
        for name, az_deg in RING_CAMERAS.items():
            az = np.deg2rad(az_deg)
            pos = T + d * np.array(
                [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
            )
            extra[name] = {
                'pos': tuple(float(v) for v in pos),
                'xyaxes': look_at_xyaxes(pos, T),
            }
        top = T + np.array([0.0, 0.0, self._pcd_top_height])
        extra['pcd_top'] = {
            'pos': tuple(float(v) for v in top),
            'xyaxes': (0.0, 1.0, 0.0, -1.0, 0.0, 0.0),  # right = +y, up = -x
        }

        for name, kw in extra.items():
            arena_mjcf.worldbody.add('camera', name=name, **kw)
        self.cameras.update(extra)
