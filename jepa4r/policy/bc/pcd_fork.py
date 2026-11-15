"""Point clouds from CortexBench's MetaWorld env (metaworld fork 4e896a5, mujoco-py 2.0): the top_cap2 camera's depth
image back-projected into the world frame, with MuJoCo's segmentation telling which geom each pixel belongs to.

    no_multisampling(sim)                    # once, before the sim renders anything
    xyz, geom = render_xyz_geom(sim, 256)    # (H, W, 3) world xyz (m); (H, W) geom id, -1 = not a geom
    pcm = MWPointClouds(sim, 'bin-picking'); pcm.seed(s); xyz, rgb = pcm()   # (1200, 3) float32, (1200, 3) uint8

Segmentation needs multisampling off: with it, an edge pixel's colour is the average of two ids' colours and decodes
to a third, unrelated geom. MuJoCo reads offsamples once, when the sim's render context is created, and a second
context in the same process breaks the first one's rendering (mujoco-py 2.0 / EGL), so a process renders either the
multisampled RGB CortexBench shows the policy or, with no_multisampling, depth + segmentation. The physics does not
depend on rendering, so the two passes replay the same trajectory (checked: qpos / qvel bitwise equal).
"""

import importlib.util
from pathlib import Path

import numpy as np

# the depth back-projection and the point sampling are the Adroit point clouds' (envs/adroit/surface_geometry.py,
# loaded from its file like there: this module also runs in interpreters without the jepa4r package's dependencies)
_spec = importlib.util.spec_from_file_location(
    "_mw_surface_geometry", Path(__file__).resolve().parents[2] / "envs" / "adroit" / "surface_geometry.py"
)
_geometry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geometry)
_depth_to_xyz = _geometry._depth_to_xyz

CAM = "top_cap2"
GEOM = 5  # mjOBJ_GEOM
# MWPointClouds: the settings of envs.metaworld.MWPointCloudWrapper, which makes JEPA4R's training data
CROP_BOX = ((-0.50, 0.30, -0.02), (0.50, 1.00, 0.60))
# fixtures: bodies welded to the world that a reset places anew each episode (found by resetting with two seeds and
# comparing body poses). They carry the task's state (assembly's target peg, the button's box, the drawer's case), so
# they are foreground; fixtures that never move (table, walls, bins, the hammer's nail box, the robot's base) are not.
FIXTURE_ROOTS = {"assembly": {"peg"}, "button-press-topdown": {"box"}, "drawer-open": {"drawer"}}


def no_multisampling(sim):
    assert sim._render_context_offscreen is None, "call before the first render"
    sim.model.vis.quality.offsamples = 0


def render_xyz_geom(sim, size=256, cam=CAM, rgb=False):
    cid = sim.model.camera_name2id(cam)
    img, depth = sim.render(width=size, height=size, depth=True, camera_name=cam, device_id=0)
    seg = sim.render(width=size, height=size, camera_name=cam, segmentation=True, device_id=0)
    # OpenGL rows are bottom-up; CortexBench flips its frames the same way
    img, depth, seg = img[::-1], depth[::-1], seg[::-1]
    xyz = _depth_to_xyz(sim, depth, size, cid)
    geom = np.where(seg[..., 0] == GEOM, seg[..., 1], -1)
    return (xyz, geom, img) if rgb else (xyz, geom)


class MWPointClouds:
    """JEPA4R's point cloud of the current frame, made the way jepa4r.envs.metaworld.MWPointCloudWrapper (on
    jepa4r.data.pointcloud.PointCloudWrapper) made its training data: one depth camera (top_cap2, 224 x 224)
    back-projected; foreground = geoms of bodies not welded to the world (arm, gripper, objects; not the mocap) plus
    this task's re-placed fixtures; crop box; 5 mm voxel de-duplication; a stratified draw of num_points with
    obj_fraction from the task objects (foreground outside the robot; in bin-picking the cube, as in training), the rest
    from the robot; shuffled. The draw uses a private RNG seeded per episode (seed()), as the wrapper's is seeded by the
    reset seed."""

    def __init__(self, sim, task, num_points=1200, size=224, voxel_size=0.005, obj_fraction=0.25, crop_box=CROP_BOX):
        m = sim.model
        self.sim, self.size, self.cid = sim, size, m.camera_name2id(CAM)
        self.num_points, self.voxel_size, self.obj_fraction = num_points, voxel_size, obj_fraction
        self.crop = np.asarray(crop_box, dtype=np.float64)
        fixtures = FIXTURE_ROOTS.get(task, set())
        robot_root = m.body_rootid[m.body_name2id("hand")]
        self.fg = np.zeros(m.ngeom, dtype=bool)
        self.obj = np.zeros(m.ngeom, dtype=bool)
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if b == 0:
                continue
            root = m.body_rootid[b]
            if m.body_mocapid[root] >= 0:
                continue
            if m.body_weldid[b] == 0 and m.body_id2name(root) not in fixtures:
                continue
            self.fg[g] = True
            self.obj[g] = root != robot_root
        self.rng = np.random.default_rng()

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        xyz, geom, rgb = render_xyz_geom(self.sim, self.size, rgb=True)
        seg = geom.reshape(-1)
        valid = seg >= 0
        valid &= self.fg[np.where(valid, seg, 0)]
        idx = np.flatnonzero(valid)
        return self._sample(xyz.reshape(-1, 3)[idx], rgb.reshape(-1, 3)[idx], self.obj[seg[idx]])

    _sample = _geometry.SurfaceSampler._sample  # jepa4r.data.pointcloud.PointCloudWrapper._sample, unchanged
