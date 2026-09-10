"""Point-cloud evaluation policy: stable-worldmodel's WorldModelPolicy with the
observation keys assembled the way the point-cloud world model expects.

The World (swm/OGBCubePCD-v0 + PointCloudWrapper) reports ``pcd_xyz`` / ``pcd_rgb``
per step, and the dataset-driven evaluation injects the goal frame's columns as
``goal_pcd_xyz`` / ``goal_pcd_rgb``. Before the parent's preprocessing we join
them into ``pcd`` (the model's obs_key) and ``goal`` (what ``JEPA.get_cost``
reads), and drop ``pixels`` (rendered for videos only; the planner would
otherwise expand it 300x on the GPU). The fixed affine used in training is
applied through the parent's ``process`` mechanism (``PointCloudScaler``), so
planning, re-planning and action buffering are untouched.
"""

from __future__ import annotations

import numpy as np
from stable_worldmodel.policy import WorldModelPolicy


class PointCloudScaler:
    """Same affine as utils.PointCloudPreprocessor, in the sklearn-like
    ``transform(ndarray)`` form the policy's ``process`` dict expects.
    Operates on (..., N, 3 + 3) arrays: xyz -> (xyz - center) * scale, rgb -> /255."""

    def __init__(self, center=(0.425, 0.0, 0.2), scale=4.0, use_rgb=True):
        self.center = np.asarray([float(c) for c in center], dtype=np.float32)
        self.scale = float(scale)
        self.use_rgb = use_rgb

    def transform(self, v):
        v = np.asarray(v, dtype=np.float32)
        xyz = (v[..., :3] - self.center) * self.scale
        if not self.use_rgb:
            return xyz
        return np.concatenate([xyz, v[..., 3:6] / 255.0], axis=-1)


class PCDWorldModelPolicy(WorldModelPolicy):
    def __init__(self, *args, obs_key='pcd', use_rgb=True, **kwargs):
        kwargs.setdefault('history_keys', (obs_key,))
        super().__init__(*args, **kwargs)
        self.obs_key = obs_key
        self.use_rgb = use_rgb  # must match the PointCloudScaler / training preprocessor
        # set by eval.py for frame-window models (jepa_agg): {'pcd_xyz': (E, K, N, 3), 'pcd_rgb': (E, K, N, 3)},
        # the K frames ending at the goal frame; replaces the World's single goal frame (E, 1, N, 3)
        self.goal_window = None
        # set by eval.py for frame-window models: {'pcd_xyz': (E, L, N, 3), 'pcd_rgb': (E, L, N, 3), 'action': (E, L, A)},
        # the L env steps preceding each start (see get_action)
        self.history_seed = None

    def _join(self, xyz, rgb):
        xyz = np.asarray(xyz, np.float32)
        if not self.use_rgb:  # xyz-only models: keep the array 3-wide so the parent's process/reshape round-trips
            return xyz
        return np.concatenate([xyz, np.asarray(rgb, np.float32)], axis=-1)

    def get_action(self, info_dict, **kwargs):
        """WorldModelPolicy.get_action; at the start of the evaluation episode (every env's history buffer still
        empty) the buffer is first filled with the dataset steps preceding each start (``history_seed``), so the
        first plan already sees ``history_len`` real frames action_block steps apart, like a training clip."""
        hb = getattr(self, '_history_buffer', None)
        if self.history_seed is not None and hb is not None and not any(hb.num_strided()):
            for j in range(self.history_seed['action'].shape[1]):  # oldest first, laid out like the live info entries
                step = {c: np.asarray(v[:, j]).reshape(np.shape(info_dict[c])) for c, v in self.history_seed.items()}
                step = self._prepare_info(step)
                hb.append({k: step[k] for k in (*self.history_keys, 'action')})
        return super().get_action(info_dict, **kwargs)

    def _prepare_info(self, info_dict):
        info_dict = dict(info_dict)
        if 'pcd_xyz' in info_dict:
            info_dict[self.obs_key] = self._join(info_dict.pop('pcd_xyz'), info_dict.pop('pcd_rgb'))
        if 'goal_pcd_xyz' in info_dict:
            xyz, rgb = info_dict.pop('goal_pcd_xyz'), info_dict.pop('goal_pcd_rgb')  # (E, 1, N, 3): the goal frame
            if self.goal_window is not None:
                xyz, rgb = self.goal_window['pcd_xyz'], self.goal_window['pcd_rgb']  # (E, K, N, 3)
            info_dict['goal'] = self._join(xyz, rgb)
        info_dict.pop('pixels', None)
        return super()._prepare_info(info_dict)
