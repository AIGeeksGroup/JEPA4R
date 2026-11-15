"""The opposite camera (the fixed world camera turned half a turn about the vertical axis), set on render copies
only; the simulation itself is not changed."""

import mujoco
import numpy as np

from .native import NativeBridge


class OppositeCameraBridge(NativeBridge):
    def _render_state(self, kind):
        renderer, m, d = super()._render_state(kind)
        cid = m.camera("vil_camera").id
        original = self.env.model
        if int(original.cam_bodyid[cid]) != 0:
            raise ValueError("this half-turn requires the original fixed world camera")
        # Always derive from the native pose: repeated capture must not alternate views.
        m.cam_pos[cid] = np.diag([-1.0, -1.0, 1.0]) @ original.cam_pos[cid]
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, np.array([0.0, 0.0, 0.0, 1.0]), original.cam_quat[cid])
        m.cam_quat[cid] = quat
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera="vil_camera")
        return renderer, m, d
