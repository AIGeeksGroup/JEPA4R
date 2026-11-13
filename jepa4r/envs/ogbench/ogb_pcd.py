# Moved from cortexbench@9331494:lewm_pcd/envs/ogb_pcd.py (verbatim except: PointCloudWrapper imported from
# jepa4r.data.pointcloud; lines wrapped at 120 columns)
"""The repo's single-depth-camera point cloud (envs/pcd_wrapper.py) for OGBench's own puzzle-3x3 and scene envs
(envs/ogb_puzzle_env.py, envs/ogb_scene_env.py). What differs from the cube's PointCloudWrapper and is overridden here:

* the depth camera's renders come from the env's front_pixels ``Camera`` (``render_flat``: colour, depth and
  segmentation without multisampling, so an edge pixel cannot decode to an unrelated geom id; the 2D frame keeps the
  multisampled renderer);
* foreground and object geoms are the env's (``Camera``): puzzle's objects are the 9 button caps (25 % of the points,
  the cube fraction of the base class), scene's are 5 objects (cube, two button caps, drawer, sliding pane) that split
  the 25 % evenly, without the window glass; the sampling is the env module's ``point_cloud``;
* the private RNG is seeded with the reset seed + 7919, so a point cloud is a function of the episode seed and the row;
* every row also gets the visible pixel counts the probes use: ``cap_visible_px`` (puzzle: the 9 caps) and
  ``visible_px`` (scene: the cube and the two caps), in both datasets (no 3D-only prefix).
"""

from __future__ import annotations

import numpy as np

from . import ogb_puzzle_env, ogb_scene_env
from jepa4r.data.pointcloud import PointCloudWrapper

SEED_OFFSET = 7_919


class _OGBPointCloudWrapper(PointCloudWrapper):
    def __init__(self, env, crop_box, **kwargs):
        super().__init__(env, cameras=('front_pixels',), crop_box=crop_box, **kwargs)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed + SEED_OFFSET)
        info.update(self._point_cloud())
        return obs, info


class PuzzlePointCloudWrapper(_OGBPointCloudWrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env, crop_box=ogb_puzzle_env.CROP_BOX, **kwargs)

    def _point_cloud(self) -> dict:
        u = self.env.unwrapped
        rgb, depth, seg, cap_px, R, t = u.cam.render_flat(u._u._data)
        xyz, col = ogb_puzzle_env.point_cloud(u.cam, rgb, depth, seg, R, t, self._rng)
        return {'pcd_xyz': xyz, 'pcd_rgb': col, 'cap_visible_px': cap_px}


class ScenePointCloudWrapper(_OGBPointCloudWrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env, crop_box=ogb_scene_env.CROP_BOX, **kwargs)

    def _point_cloud(self) -> dict:
        u = self.env.unwrapped
        rgb, depth, seg, R, t = u.cam.render_flat(u._u._data)
        xyz, col = ogb_scene_env.point_cloud(u.cam, rgb, depth, seg, R, t, self._rng)
        return {'pcd_xyz': xyz, 'pcd_rgb': col, 'visible_px': u.cam.visible_px(seg)}
