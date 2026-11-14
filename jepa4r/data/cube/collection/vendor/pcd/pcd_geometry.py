"""The renders that PointCloudWrapper back-projects for the online point clouds of the current and goal frames
(native_pcd_frame.py): the frame's own antialiased RGB, and depth / segmentation renders without antialiasing."""


class PCDRenderer:
    """``provide`` answers PointCloudWrapper's render calls: the given RGB frame (``rgb``), or a depth / segmentation
    render of front_pixels by the attached renderer (``renderer``, scene of ``u``); the last segmentation is kept in
    ``last_seg``."""

    def provide(self, camera=None, depth=False, segmentation=False):
        if camera != "front_pixels" or depth and segmentation:
            raise ValueError("Unexpected main render request")
        if not depth and not segmentation:
            return self.rgb.copy()
        r = self.renderer
        if depth:
            r.disable_segmentation_rendering()
            r.enable_depth_rendering()
        else:
            r.disable_depth_rendering()
            r.enable_segmentation_rendering()
        r.update_scene(self.u._data, camera="front_pixels")
        value = r.render().copy()
        if segmentation:
            self.last_seg = value
        return value
