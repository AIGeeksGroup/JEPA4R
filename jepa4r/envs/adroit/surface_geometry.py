# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/cortexbench/surface_geometry.py (verbatim except: lines wrapped at 120
# columns)
"""Depth back-projection and the unchanged shared visible-surface sampling primitive."""
import numpy as np

CAM = "vil_camera"
GEOM = 5

def _depth_to_xyz(sim, depth, size, cid):
    m, d = sim.model, sim.data
    ext = m.stat.extent
    near, far = m.vis.map.znear * ext, m.vis.map.zfar * ext
    z = near / (1.0 - depth.astype(np.float64) * (1.0 - near / far))  # metric distance along the view axis
    f = 0.5 * size / np.tan(np.deg2rad(m.cam_fovy[cid]) / 2)
    v, u = np.mgrid[0:size, 0:size].astype(np.float64)
    c = (size - 1) / 2.0  # pixel centres, as envs/pcd_wrapper.py: (u + 0.5 - W / 2) / f
    pts_cam = np.stack([(u - c) * z / f, -(v - c) * z / f, -z], axis=-1)  # MuJoCo camera: x right, y up, looks along -z
    R, p = d.cam_xmat[cid].reshape(3, 3), d.cam_xpos[cid]
    return pts_cam @ R.T + p

def render_xyz_geom(sim, size=256, cam=CAM, rgb=False):
    cid = sim.model.camera_name2id(cam)
    img, depth = sim.render(width=size, height=size, depth=True, camera_name=cam, device_id=0)
    seg = sim.render(width=size, height=size, camera_name=cam, segmentation=True, device_id=0)
    # OpenGL rows are bottom-up; CortexBench flips its frames the same way
    img, depth, seg = img[::-1], depth[::-1], seg[::-1]
    xyz = _depth_to_xyz(sim, depth, size, cid)
    geom = np.where(seg[..., 0] == GEOM, seg[..., 1], -1)
    return (xyz, geom, img) if rgb else (xyz, geom)


class SurfaceSampler:
    def _sample(self, pts, cols, is_obj):  # envs/pcd_wrapper.PointCloudWrapper._sample, unchanged
            n = self.num_points
            lo, hi = self.crop
            m = np.all((pts >= lo) & (pts <= hi), axis=1)
            pts, cols, is_obj = pts[m], cols[m], is_obj[m]
            if self.voxel_size > 0 and len(pts):
                q = np.floor(pts / self.voxel_size).astype(np.int32)
                _, keep = np.unique(q, axis=0, return_index=True)
                pts, cols, is_obj = pts[keep], cols[keep], is_obj[keep]
            if len(pts) == 0:
                return np.zeros((n, 3), dtype=np.float32), np.zeros((n, 3), dtype=np.uint8)
            obj_idx, rob_idx = np.flatnonzero(is_obj), np.flatnonzero(~is_obj)
            n_obj = min(int(round(self.obj_fraction * n)), obj_idx.size)
            if rob_idx.size == 0:
                sel = self.rng.choice(obj_idx, n, replace=obj_idx.size < n)
            else:
                n_rob = n - n_obj
                sel = np.concatenate([self.rng.choice(obj_idx, n_obj, replace=False),
                                      self.rng.choice(rob_idx, n_rob, replace=rob_idx.size < n_rob)])
            sel = self.rng.permutation(sel)
            return pts[sel].astype(np.float32), cols[sel].astype(np.uint8)


def depth_to_world(
    depth: np.ndarray, camera_position: np.ndarray, camera_rotation: np.ndarray, fovy: float
) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2 or depth.shape[0] != depth.shape[1]:
        raise ValueError(f"depth must be square HxW, got {depth.shape}")
    size = depth.shape[0]
    v, u = np.mgrid[0:size, 0:size].astype(np.float64)
    center = (size - 1.0) / 2.0
    focal = 0.5 * size / np.tan(np.deg2rad(float(fovy)) / 2.0)
    camera_points = np.stack(
        [(u - center) * depth / focal, -(v - center) * depth / focal, -depth], axis=-1
    )
    return camera_points @ np.asarray(camera_rotation, dtype=np.float64).T + np.asarray(
        camera_position, dtype=np.float64
    )
