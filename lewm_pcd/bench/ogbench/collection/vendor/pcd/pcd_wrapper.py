"""Single depth-camera point-cloud wrapper for stable-worldmodel's OGBench manipulation envs.

Adds two keys to the info dict on every ``reset()`` / ``step()``:

    pcd_xyz : (num_points, 3) float32, world frame (same frame as ``qpos`` and
              ``privileged/*``; the table top is z = 0)
    pcd_rgb : (num_points, 3) uint8

The point cloud comes from exactly one fixed (eye-to-hand) camera, the way a ToF
RGB-D camera delivers it: its depth image, back-projected. No stereo, no fusion
of several cameras (a list of more than one camera is rejected). The default
camera is ``front_pixels``, the camera of the 2D (pixel) dataset, so the 2D and
3D observations of a step come from the same view.

Pipeline per frame
    1. The env's own ``mujoco.Renderer`` produces one RGB, one depth and one
       segmentation image of the camera. Depth is back-projected with the
       camera's fovy and its pose read from ``data.cam_xpos`` /
       ``data.cam_xmat`` at render time.
    2. Background geoms (anything attached to the world body: floor, table) and
       the invisible target markers (mocap bodies) are dropped.
    3. Points outside ``crop_box`` (robot base / shoulder behind the workspace)
       are dropped.
    4. Voxel de-duplication at ``voxel_size``: a surface close to the camera no
       longer gets more points than a distant one.
    5. Stratified random draw: ``round(cube_fraction * num_points)`` points from
       cube geoms (fewer if fewer are visible), the rest from the robot; with
       replacement only if the pool is too small. Output order is shuffled.

The wrapper owns a private RNG seeded from the reset seed. It never touches the
env RNG or numpy's global RNG (the OGBench oracle draws from the latter), so the
trajectories are exactly those a pixel collection with the same seeds produces.
"""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import mujoco
import numpy as np

DEFAULT_CAMERAS = ('front_pixels',)  # the one depth camera: the official pixel camera (eye-to-hand)
# ManipSpaceEnv._workspace_bounds is x 0.25..0.6, y -0.35..0.35, z 0.02..0.35;
# widened so the wrist / forearm over the workspace stay in, the base column does not
DEFAULT_CROP_BOX = ((0.10, -0.50, -0.02), (0.75, 0.50, 0.60))


class PointCloudWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        num_points: int = 1200,
        cameras: Sequence[str] = DEFAULT_CAMERAS,
        foreground_only: bool = True,
        crop_box: Sequence[Sequence[float]] | None = DEFAULT_CROP_BOX,
        voxel_size: float = 0.005,
        cube_fraction: float = 0.25,
        keep_rgb: bool = False,
    ):
        """
        Args:
            env: a ``swm/OGBCubePCD-v0`` env (any wrapper depth). ``swm/OGBCube-v0``
                works too (``front_pixels`` is one of its own cameras).
            num_points: points per frame.
            cameras: exactly one MuJoCo camera name, the depth camera (a
                sequence, as in the configs).
            foreground_only: drop world-body geoms (floor/table) before sampling.
            crop_box: ``((xmin, ymin, zmin), (xmax, ymax, zmax))`` in world frame,
                or ``None`` to keep everything.
            voxel_size: de-duplication voxel edge in metres; ``0`` disables.
            cube_fraction: guaranteed share of the sample taken from cube geoms.
            keep_rgb: debug / visualisation only. Also stores, per camera, the RGB
                render as ``pixels/<cam>`` and the pose as ``cam/<cam>/pos``,
                ``cam/<cam>/xmat``, ``cam/<cam>/fovy``. Off for the real dataset.
        """
        super().__init__(env)
        self.num_points = int(num_points)
        self.cameras = tuple(cameras)
        if len(self.cameras) != 1:  # one depth camera (ToF-style); multi-camera fusion is not allowed
            raise ValueError(f'PointCloudWrapper takes exactly one depth camera, got {list(self.cameras)}')
        self.foreground_only = foreground_only
        self.crop_box = None if crop_box is None else np.asarray(
            [list(crop_box[0]), list(crop_box[1])], dtype=np.float64
        )
        self.voxel_size = float(voxel_size)
        self.cube_fraction = float(cube_fraction)
        self.keep_rgb = keep_rgb
        self._rng = np.random.default_rng()
        self._model_ref = None  # the MjModel the cached camera ids / geom masks belong to
        self._grids = {}

    # ------------------------------------------------------------------ setup
    def _setup(self, model: mujoco.MjModel) -> None:
        self._cams = {}
        for name in self.cameras:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid < 0:
                raise ValueError(f'camera {name!r} not found in model')
            self._cams[name] = (cid, float(model.cam_fovy[cid]))

        fg = np.zeros(model.ngeom, dtype=bool)
        cube = np.zeros(model.ngeom, dtype=bool)
        for g in range(model.ngeom):
            body = model.geom_bodyid[g]
            if body == 0:  # static scene: floor, table, walls
                continue
            root = model.body_rootid[body]
            if model.body_mocapid[root] >= 0:  # target markers (mocap)
                continue
            fg[g] = True
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root) or ''
            cube[g] = 'object' in name
        self._geom_fg = fg
        self._geom_cube = cube
        self._model_ref = model
        self._grids = {}

    def _grid(self, H: int, W: int, fovy: float):
        key = (H, W, fovy)
        if key not in self._grids:
            f = 0.5 * H / np.tan(np.deg2rad(fovy) / 2)
            v, u = np.mgrid[0:H, 0:W]
            self._grids[key] = (
                ((u + 0.5 - W / 2) / f).reshape(-1),
                (-(v + 0.5 - H / 2) / f).reshape(-1),
            )
        return self._grids[key]

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        options = options or {}
        unwrapped = self.env.unwrapped
        unwrapped._pcd_goal_capture = None
        unwrapped._pcd_goal_capture_callback = self._point_cloud
        try:
            obs, info = self.env.reset(seed=seed, options=options)
        finally:
            goal = getattr(unwrapped, '_pcd_goal_capture', None)
            unwrapped._pcd_goal_capture_callback = None
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options.get('render_goal', False):
            if goal is None:
                raise RuntimeError(
                    'render_goal=True requested, but the wrapped environment '
                    'did not expose an official goal point-cloud capture hook'
                )
            info['goal_pcd_xyz'] = goal['pcd_xyz']
            info['goal_pcd_rgb'] = goal['pcd_rgb']
        info.update(self._point_cloud())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.update(self._point_cloud())
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------- core
    def _point_cloud(self, *, render_fn=None, return_trace=False, empty_crop_fallback=False) -> dict:
        """Optional frame-render injection and source-index trace; defaults stay unchanged."""
        u = self.env.unwrapped
        render = u.render if render_fn is None else render_fn
        traced = render_fn is not None or return_trace or empty_crop_fallback
        pixel_idx, geom_idx, camera_trace = [], [], {}
        if self._model_ref is not u._model:  # the env recompiles its model on every reset
            self._setup(u._model)

        pts, cols, cube, extra = [], [], [], {}
        for name, (cid, fovy) in self._cams.items():
            # order matters: env.render() switches the shared renderer's mode per call
            rgb = render(camera=name)
            depth = render(camera=name, depth=True)
            seg = render(camera=name, segmentation=True)[..., 0].reshape(-1)
            if traced and (rgb.dtype != np.uint8 or rgb.shape != (*depth.shape, 3)):
                raise ValueError('Explicit extraction requires aligned real uint8 RGB/depth')

            valid = seg >= 0  # -1 = nothing hit (sky)
            if self.foreground_only:
                valid &= self._geom_fg[np.where(valid, seg, 0)]
            idx = np.flatnonzero(valid)

            R = u._data.cam_xmat[cid].reshape(3, 3)
            t = u._data.cam_xpos[cid]
            if idx.size:
                xn, yn = self._grid(*depth.shape, fovy)
                z = depth.reshape(-1)[idx]
                p_cam = np.stack([xn[idx] * z, yn[idx] * z, -z], axis=1)
                pts.append(p_cam @ R.T + t)
                cols.append(rgb.reshape(-1, 3)[idx])
                cube.append(self._geom_cube[seg[idx]])
                if return_trace:
                    pixel_idx.append(idx); geom_idx.append(seg[idx])
                    H, W = depth.shape; f = 0.5 * H / np.tan(np.deg2rad(fovy) / 2)
                    T = np.eye(4); T[:3, :3] = R @ np.diag([1., -1., -1.]); T[:3, 3] = t
                    camera_trace = {'camera': name, 'height': H, 'width': W,
                                    'camera_K': np.array([[f, 0., (W-1)/2], [0., f, (H-1)/2], [0., 0., 1.]]),
                                    'world_from_cv': T}

            if self.keep_rgb:
                extra[f'pixels/{name}'] = np.ascontiguousarray(rgb)
                extra[f'cam/{name}/pos'] = t.astype(np.float32).copy()
                extra[f'cam/{name}/xmat'] = R.reshape(-1).astype(np.float32).copy()
                extra[f'cam/{name}/fovy'] = np.float32(fovy)

        if pts:
            sampled = self._sample(
                np.concatenate(pts, 0),
                np.concatenate(cols, 0),
                np.concatenate(cube, 0),
                **({'return_indices': True, 'empty_crop_fallback': empty_crop_fallback} if traced else {}),
            )
            xyz, col = sampled[:2]
            if return_trace:
                selected = sampled[2].pop('indices')
                flat = np.concatenate(pixel_idx)[selected]; W = camera_trace['width']
                extra['pcd_trace'] = {**sampled[2], **camera_trace,
                                      'source_pixel_uv': np.column_stack([flat % W, flat // W]).astype(np.int32),
                                      'source_geom_id': np.concatenate(geom_idx)[selected].astype(np.int32)}
        else:
            if traced: raise ValueError('Actual foreground empty; explicit extraction cannot fabricate a zero cloud')
            xyz = np.zeros((self.num_points, 3), dtype=np.float32)
            col = np.zeros((self.num_points, 3), dtype=np.uint8)

        out = {'pcd_xyz': xyz, 'pcd_rgb': col}
        out.update(extra)
        return out

    def _sample(self, pts, cols, is_cube, *, return_indices=False, empty_crop_fallback=False):
        if return_indices or empty_crop_fallback:
            if pts.ndim != 2 or pts.shape[1] != 3 or cols.shape != pts.shape or cols.dtype != np.uint8 or is_cube.shape != (len(pts),) or is_cube.dtype != np.bool_ or not np.isfinite(pts).all():
                raise ValueError('Explicit sampling requires finite XYZ, uint8 RGB and actual cube mask')
        n = self.num_points
        raw_count = len(pts); indices = np.arange(raw_count) if return_indices else None
        cropped_count = raw_count; fallback = False
        if self.crop_box is not None:
            lo, hi = self.crop_box
            m = np.all((pts >= lo) & (pts <= hi), axis=1)
            cropped_count = int(m.sum())
            if empty_crop_fallback and raw_count and not cropped_count:
                fallback = True
            else:
                pts, cols, is_cube = pts[m], cols[m], is_cube[m]
                if return_indices: indices = indices[m]
        if self.voxel_size > 0 and len(pts):
            q = np.floor(pts / self.voxel_size).astype(np.int32)
            _, keep = np.unique(q, axis=0, return_index=True)
            pts, cols, is_cube = pts[keep], cols[keep], is_cube[keep]
            if return_indices: indices = indices[keep]
        if len(pts) == 0:
            if return_indices or empty_crop_fallback: raise ValueError('Actual raw/cropped point pool empty; no zero-cloud output in explicit extraction')
            return (
                np.zeros((n, 3), dtype=np.float32),
                np.zeros((n, 3), dtype=np.uint8),
            )

        cube_idx = np.flatnonzero(is_cube)
        rob_idx = np.flatnonzero(~is_cube)
        n_cube = min(int(round(self.cube_fraction * n)), cube_idx.size)
        if rob_idx.size == 0:
            sel = self._rng.choice(cube_idx, n, replace=cube_idx.size < n)
        else:
            n_rob = n - n_cube
            sel = np.concatenate([
                self._rng.choice(cube_idx, n_cube, replace=False),
                self._rng.choice(rob_idx, n_rob, replace=rob_idx.size < n_rob),
            ])
        sel = self._rng.permutation(sel)
        result = pts[sel].astype(np.float32), cols[sel].astype(np.uint8)
        if return_indices:
            unique = len(np.unique(sel))
            trace = {'indices': indices[sel], 'raw_foreground_pool': raw_count,
                     'foreground_cropped_pool': cropped_count, 'voxel_unique_pool': len(pts),
                     'cube_unique_pool': len(cube_idx), 'other_unique_pool': len(rob_idx),
                     'sampled_cube_points': int(is_cube[sel].sum()),
                     'actual_selected_unique_pool_indices': unique, 'repeated_samples': n-unique,
                     'fallback_used': fallback,
                     'fallback_reason': 'original_crop_empty_use_actual_foreground' if fallback else None}
            return *result, trace
        return result
