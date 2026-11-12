"""Depth image -> 1,200-point xyzrgb cloud (Sec. 4.1). Every place that produces point clouds (data collection,
policy data, online policy evaluation) calls this one function.

Steps (config: env_cfg.pointcloud):
  1. lift the depth image to world-frame points (jepa4r.envs.camera.depth_to_world)
  2. crop to the workspace box                                   crop_box
  3. drop renderer-segmentation classes: table, floor, static fixtures, invisible goal markers;
     keep the task objects (Adroit Relocate also keeps the visible goal marker)   drop / keep_extra
  4. merge points within 5 mm voxels (keeps surfaces near the camera from dominating)   voxel_size
  5. sample N = 1,200 points                                     num_points, sampler
  6. normalise with the fixed affine map (on Cube shared with the real arm)   affine
Output: float32 [1200, 6] = normalised xyz + rgb.
"""


def process(rgb, depth, seg, K, T_world_cam, pcd_cfg, rng=None):
    raise NotImplementedError("TODO(integrate): the existing point-cloud pipeline")


def denormalise(xyz, pcd_cfg):
    """Undo step 6, so that metrics are computed in metres (reported in cm)."""
    raise NotImplementedError("TODO(integrate)")
