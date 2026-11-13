# Moved from OGBench-cube@4fbb0c6:lewm_pcd/envs/adroit/fragment_env.py (verbatim)
"""Gymnasium registration bridge for the final Relocate fragment extension."""
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from .fragment import NativeFragment, RelocateFragment


class NativeFragmentEnv(NativeFragment, gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 100}

    def _make_native(self):
        raise NotImplementedError('select the final RelocateFragmentEnv')

    def __init__(self, width=256, height=256, camera='vil_camera', render_mode='rgb_array', offsamples=None):
        # No native import during module import/registration.
        native = self._make_native()
        NativeFragment.__init__(self, native, width, height, camera)
        self.sim, self.render_mode = native.sim, render_mode
        if native.action_space.shape != (self.action_dim,) or native.observation_space.shape != (self.observation_dim,):
            raise ValueError('native %s dimensions must be action%d/observation%d' % (self.task, self.action_dim, self.observation_dim))
        if offsamples is not None:
            if int(offsamples) != 0:
                raise ValueError('only the independent PCD pass may request offsamples=0')
            if self.sim.render_contexts or self.sim._render_context_offscreen is not None:
                raise RuntimeError('HOLD: offsamples=0 requires a fresh native render context')
            self.sim.model.vis.quality.offsamples = 0
        self.action_space = Box(-1., 1., (self.action_dim,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, (self.observation_dim,), np.float64)

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


class RelocateFragmentEnv(RelocateFragment, NativeFragmentEnv):
    def _make_native(self):
        from mj_envs.envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0
        return RelocateEnvV0()
