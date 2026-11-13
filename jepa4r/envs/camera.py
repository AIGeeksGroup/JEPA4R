"""Camera geometry shared by every suite: depth image -> world-frame points (Sec. 4.1, step 1).

The back-projection is the one of the point-cloud wrapper (cortexbench envs/pcd_wrapper.py: ``PointCloudWrapper._grid``
for the pixel rays of a MuJoCo camera with vertical field of view ``fovy``, then ``p_cam = [x z, y z, -z]`` and
``p_world = p_cam R^T + t`` with R = data.cam_xmat, t = data.cam_xpos)."""
import numpy as np


def depth_to_world(depth, K, T_world_cam):
    """[H, W] depth (m) -> [H*W, 3] world-frame points, same pixel order as rgb / seg.
    K: the camera's fovy in degrees; T_world_cam: (R [3, 3], t [3])."""
    from jepa4r.data.pointcloud import PointCloudWrapper

    class _Grids:
        _grids = {}

    depth = np.asarray(depth)
    xn, yn = PointCloudWrapper._grid(_Grids, *depth.shape, float(K))
    R, t = np.asarray(T_world_cam[0]).reshape(3, 3), np.asarray(T_world_cam[1]).reshape(3)
    z = depth.reshape(-1)
    # Moved from cortexbench@9331494:lewm_pcd/envs/pcd_wrapper.py:PointCloudWrapper._point_cloud (lines 166-167; the same
    # formula applied to every pixel instead of the foreground indices only)
    p_cam = np.stack([xn * z, yn * z, -z], axis=1)
    return p_cam @ R.T + t
