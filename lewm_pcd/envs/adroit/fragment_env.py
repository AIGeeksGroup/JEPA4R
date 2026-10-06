"""Gymnasium registration bridge for the independent Pen fragment extension."""
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from .fragment import PenFragment


class PenFragmentEnv(PenFragment, gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 100}

    def __init__(self, width=256, height=256, camera='vil_camera', render_mode='rgb_array', offsamples=None):
        # No native import during module import/registration.
        from mj_envs.envs.hand_manipulation_suite.pen_v0 import PenEnvV0
        native = PenEnvV0()
        PenFragment.__init__(self, native, width, height, camera)
        self.sim, self.render_mode = native.sim, render_mode
        if native.action_space.shape != (24,) or native.observation_space.shape != (45,):
            raise ValueError('native Pen dimensions must be action24/observation45')
        if offsamples is not None:
            if int(offsamples) != 0:
                raise ValueError('only the independent PCD pass may request offsamples=0')
            if self.sim.render_contexts or self.sim._render_context_offscreen is not None:
                raise RuntimeError('HOLD: offsamples=0 requires a fresh native render context')
            self.sim.model.vis.quality.offsamples = 0
        self.action_space = Box(-1., 1., (24,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, (45,), np.float64)

    def render(self):
        return self.sim.render(width=self.width, height=self.height, camera_name=self.camera, device_id=0)[::-1].copy()

    def camera_pose(self, camera=None):
        cid = self.sim.model.camera_name2id(camera or self.camera)
        return dict(position=self.sim.data.cam_xpos[cid].copy(), rotation=self.sim.data.cam_xmat[cid].reshape(3, 3).copy(),
                    fov=np.float64(self.sim.model.cam_fovy[cid]))

    def close(self):
        close = getattr(self.native_env, 'close', None)
        if close:
            close()
