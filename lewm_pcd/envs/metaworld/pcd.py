"""The repo's single-depth-camera point cloud (envs/pcd_wrapper.py: crop, 5 mm voxel de-duplication, stratified draw
with a guaranteed object share, shuffle) for the MetaWorld envs. Two things differ and are overridden here:

* the depth camera is the free camera reproducing CortexBench's top_cap2, so its pose is read from the renderer
  (MWTaskEnv.camera_pose) instead of data.cam_xpos / cam_xmat;
* foreground = every geom whose body is not welded to the world (arm links, gripper, the task's objects), plus the
  task's ``fixtures``: bodies welded to the world that a reset places anew and that carry the task state (assembly's
  peg); the rules of the CortexBench 3D demos (bench/cortexbench/pcd_fork.py). Table, bins, walls and the robot
  pedestal are background. The object share (``cube_fraction``) is drawn from the foreground outside the robot:
  the cube in bin-picking, the nut and the peg in assembly.
* depth, segmentation and the points' colours come from a renderer without multisampling (MWTaskEnv.render(multisample=
  False)), as the CortexBench 3D demos: with multisampling an edge pixel between two background geoms can decode to a
  foreground id (a stray point; bin-picking data collected before this had about 3% such points, the other tasks next
  to none).
"""

from __future__ import annotations

import mujoco
import numpy as np

from ..pcd_wrapper import PointCloudWrapper

# workspace around the bins / the peg; drops the robot's shoulder / head behind it (y < 0.3) and the floor below the table
DEFAULT_CROP_BOX = ((-0.50, 0.30, -0.02), (0.50, 1.00, 0.60))
GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)  # a plain int: comparing an array with the pybind enum is ~1000x slower


class MWPointCloudWrapper(PointCloudWrapper):
    def __init__(self, env, cameras=('top_cap2',), crop_box=DEFAULT_CROP_BOX, fixtures=(), **kwargs):
        super().__init__(env, cameras=cameras, crop_box=crop_box, **kwargs)
        self.fixtures = set(fixtures)

    def _setup(self, model: mujoco.MjModel) -> None:
        robot_root = model.body_rootid[model.body('hand').id]
        fg = np.zeros(model.ngeom, dtype=bool)
        obj = np.zeros(model.ngeom, dtype=bool)
        for g in range(model.ngeom):
            body = model.geom_bodyid[g]
            root = model.body_rootid[body]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root) or ''
            if model.body_weldid[body] == 0 and name not in self.fixtures:  # static: table, bins, walls, pedestal
                continue
            if model.body_mocapid[root] >= 0:  # the mocap body driving the hand
                continue
            fg[g] = True
            obj[g] = root != robot_root  # bin-picking: the cube (its visible box is on the child body 'objA')
        self._geom_fg, self._geom_cube = fg, obj
        self._model_ref = model
        self._grids = {}

    def _point_cloud(self) -> dict:
        u = self.env.unwrapped
        if self._model_ref is not u._model:
            self._setup(u._model)
        rgb = u.render(multisample=False)
        depth = u.render(depth=True, multisample=False)
        seg2 = u.render(segmentation=True, multisample=False)
        seg = np.where(seg2[..., 1] == GEOM, seg2[..., 0], -1).reshape(-1)  # sites render too
        t, R, fovy = u.camera_pose()

        valid = seg >= 0
        if self.foreground_only:
            valid &= self._geom_fg[np.where(valid, seg, 0)]
        idx = np.flatnonzero(valid)
        if idx.size:
            xn, yn = self._grid(*depth.shape, fovy)
            z = depth.reshape(-1)[idx]
            p_cam = np.stack([xn[idx] * z, yn[idx] * z, -z], axis=1)
            xyz, col = self._sample(p_cam @ R.T + t, rgb.reshape(-1, 3)[idx], self._geom_cube[seg[idx]])
        else:
            xyz = np.zeros((self.num_points, 3), dtype=np.float32)
            col = np.zeros((self.num_points, 3), dtype=np.uint8)
        return {'pcd_xyz': xyz, 'pcd_rgb': col}
