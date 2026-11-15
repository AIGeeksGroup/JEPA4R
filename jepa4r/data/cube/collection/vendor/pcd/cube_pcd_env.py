"""OGBench Cube with a hook for official-goal point-cloud capture.

Point clouds come only from the environment's original fixed eye-to-hand
``front_pixels`` camera. No auxiliary point-cloud cameras are added.
"""

from __future__ import annotations

from stable_worldmodel.envs.ogbench.cube_env import CubeEnv


class CubePCDEnv(CubeEnv):
    def render(self, camera="front_pixels", *args, **kwargs):
        """Render and let a point-cloud wrapper capture the official goal state.

        ``CubeEnv.initialize_episode`` calls ``self.render()`` while the
        simulator is temporarily at the official goal state. The callback is
        consumed exactly once there, before the environment restores its real
        initial state. A guard prevents the callback's own RGB/depth renders
        from recursively triggering another capture.
        """
        image = super().render(camera=camera, *args, **kwargs)
        callback = getattr(self, "_pcd_goal_capture_callback", None)
        if (
            callback is not None
            and getattr(self, "_render_goal", False)
            and not getattr(self, "_pcd_goal_capture_active", False)
        ):
            self._pcd_goal_capture_active = True
            try:
                self._pcd_goal_capture = callback()
            finally:
                self._pcd_goal_capture_active = False
                self._pcd_goal_capture_callback = None
        return image
