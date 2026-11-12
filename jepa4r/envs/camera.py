"""Camera geometry shared by every suite: depth image -> world-frame points (Sec. 4.1, step 1)."""


def depth_to_world(depth, K, T_world_cam):
    """[H, W] depth (m) -> [H*W, 3] world-frame points, same pixel order as rgb / seg."""
    raise NotImplementedError("TODO(integrate): back-projection with intrinsics K and extrinsics T_world_cam")
