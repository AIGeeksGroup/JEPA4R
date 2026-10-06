"""Single-camera legacy Adroit surface PCD; real rendering remains a G3 gate.

Depth is OpenGL normalized depth, inverted with model near/far times extent.
Images are flipped once from OpenGL bottom-up rows. Camera coordinates use
x right, y up, -z forward; cam_xmat/cam_xpos map these to world metres.
Relocate target is rendered as an opaque site surface in the SAME RGB/depth/
segmentation pass, restoring alpha in finally; occlusion may yield zero points.
Pen uses joint segmentation/depth for surface ownership, ordinary RGB for colour,
and includes only visible native target-body geoms without changing their alpha.
"""
import importlib.util
from pathlib import Path
import numpy as np

_spec = importlib.util.spec_from_file_location(
    '_adroit_same_checkout_pcd_fork',
    Path(__file__).resolve().parents[2] / 'bench/cortexbench/pcd_fork.py')
_PCD_FORK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_PCD_FORK)
DEFAULT_CROP_BOX = ((-.6, -.6, -.05), (.6, .6, 1.1))
TASK_SETTINGS = {
    'pen': {'camera': 'vil_camera', 'object_body': 'Object', 'target_site': None, 'crop_box': DEFAULT_CROP_BOX},
    'relocate': {'camera': 'vil_camera', 'object_body': 'Object', 'target_site': 'target', 'crop_box': DEFAULT_CROP_BOX},
}

class AdroitPointClouds:
    _sample = _PCD_FORK.MWPointClouds._sample

    def __init__(self, sim, task, num_points=1200, size=224, camera='vil_camera',
                 voxel_size=.005, obj_fraction=.25, crop_box=DEFAULT_CROP_BOX,
                 foreground_only=True):
        task = task[:-3] if task.endswith('-v0') else task
        if task not in TASK_SETTINGS:
            raise ValueError('unsupported Adroit task: ' + task)
        if num_points <= 0 or size <= 0 or not 0 <= obj_fraction <= 1 or voxel_size < 0:
            raise ValueError('invalid point-cloud sampling settings')
        self.sim, self.task, self.camera, self.size = sim, task, camera, int(size)
        self.num_points, self.voxel_size, self.obj_fraction = int(num_points), float(voxel_size), float(obj_fraction)
        self.crop = np.asarray(crop_box, dtype=np.float64)
        if self.crop.shape != (2,3) or not np.isfinite(self.crop).all() or np.any(self.crop[0] > self.crop[1]):
            raise ValueError('crop_box must be finite ordered (2,3) bounds')
        self.foreground_only = bool(foreground_only)
        m = sim.model
        self._check_render_context()
        oid = m.body_name2id(TASK_SETTINGS[task]['object_body'])
        object_root = m.body_rootid[oid]
        roots = m.body_rootid[m.geom_bodyid]
        self.obj = roots == object_root
        self.fg = (m.geom_bodyid != 0) & (m.body_weldid[m.geom_bodyid] != 0) & (m.body_mocapid[roots] < 0)
        site = TASK_SETTINGS[task]['target_site']
        self.target_site = None if site is None else m.site_name2id(site)
        self.rng = np.random.default_rng()
        self.last_info = {}
        self.last_labels = np.empty(0, dtype=np.int8)  # 0 robot, 1 Object, 2 target.
        self.last_source_indices = np.empty(0, dtype=np.int64)  # Top-down image pixels.
        self.last_rgb = self.last_depth = self.last_seg = None

    def _check_render_context(self):
        m = self.sim.model
        contexts = getattr(self.sim, '_render_contexts', ())
        created = getattr(self.sim, '_render_context_offscreen', None) is not None or bool(contexts)
        if created and m.vis.quality.offsamples != 0:
            raise RuntimeError('HOLD: existing MSAA render context; PCD needs offsamples=0 before first render; use a separate replay process')
        if not created:
            m.vis.quality.offsamples = 0

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        self._check_render_context()
        if self.task == 'pen':
            return self._pen_point_cloud()
        m = self.sim.model
        old_alpha = None
        try:
            if self.target_site is not None:
                old_alpha = m.site_rgba[self.target_site,3].copy()
                m.site_rgba[self.target_site,3] = 1.
            rgb, depth = self.sim.render(width=self.size, height=self.size, depth=True,
                                        camera_name=self.camera, device_id=0)
            seg = self.sim.render(width=self.size, height=self.size, segmentation=True,
                                  camera_name=self.camera, device_id=0)
        finally:
            if old_alpha is not None:
                m.site_rgba[self.target_site,3] = old_alpha
        rgb, depth, seg = rgb[::-1], depth[::-1], seg[::-1]
        cid = m.camera_name2id(self.camera)
        xyz = _PCD_FORK._depth_to_xyz(self.sim, depth, self.size, cid)
        kind, sid = seg[...,0].reshape(-1), seg[...,1].reshape(-1)
        geom = (kind == 5) & (sid >= 0) & (sid < m.ngeom)
        safe = np.where(geom,sid,0)
        target = np.zeros(sid.shape,dtype=bool) if self.target_site is None else (kind == 6) & (sid == self.target_site)
        valid = ((geom & self.fg[safe]) if self.foreground_only else geom) | target
        valid &= np.isfinite(xyz.reshape(-1,3)).all(axis=1) & np.isfinite(depth.reshape(-1)) & (depth.reshape(-1) < 1.)
        self.last_info = {'camera':self.camera, 'target_visible_pixels':int(np.count_nonzero(target & valid)),
                          'foreground_pixels':int(np.count_nonzero(valid)), 'target_surface_pass':self.target_site is not None,
                          'render_validation':'UNVERIFIED_G3', 'depth_units':'world_metres', 'offsamples':0}
        return self._sample(xyz.reshape(-1,3)[valid], rgb.reshape(-1,3)[valid],
                            (self.obj[safe] & geom)[valid])

    def _pen_point_cloud(self):
        if not self.foreground_only:
            raise ValueError('Pen target sampling requires foreground_only=True')
        m = self.sim.model
        # Deferred lookup keeps existing construction/shared-sampler APIs intact.
        target_body = m.body_name2id('target')
        if target_body < 0:
            raise ValueError('Pen model has no target body')
        target_geoms = m.geom_bodyid == target_body
        self.last_rgb = self.last_depth = self.last_seg = None
        self.last_info = {'camera': self.camera, 'depth_units': 'world_metres',
                          'render_validation': 'UNVERIFIED_G3', 'offsamples': 0,
                          'same_segmentation_depth': 1, 'target_alpha_modified': 0}
        # Ordinary translucent RGB depth can belong to a different surface than
        # the ID pass. Keep colour, but derive XYZ and labels from one ID/depth pass.
        rgb = self.sim.render(width=self.size, height=self.size,
                              camera_name=self.camera, device_id=0)
        seg, depth = self.sim.render(width=self.size, height=self.size, depth=True,
                                    segmentation=True, camera_name=self.camera, device_id=0)
        rgb, depth, seg = rgb[::-1], depth[::-1], seg[::-1]
        cid = m.camera_name2id(self.camera)
        xyz = _PCD_FORK._depth_to_xyz(self.sim, depth, self.size, cid)
        near, far = m.vis.map.znear * m.stat.extent, m.vis.map.zfar * m.stat.extent
        self.last_rgb, self.last_seg = rgb, seg
        self.last_depth = near / (1. - depth.astype(np.float64) * (1. - near / far))
        kind, sid = seg[...,0].reshape(-1), seg[...,1].reshape(-1)
        geom = (kind == 5) & (sid >= 0) & (sid < m.ngeom)
        safe = np.where(geom, sid, 0)
        target = geom & target_geoms[safe]
        obj = geom & self.obj[safe] & ~target
        robot = geom & self.fg[safe] & ~obj & ~target
        valid = robot | obj | target
        valid &= np.isfinite(xyz.reshape(-1,3)).all(axis=1) & np.isfinite(depth.reshape(-1)) & (depth.reshape(-1) < 1.)
        labels = np.where(target, 2, np.where(obj, 1, 0)).astype(np.int8)
        self.last_info.update(target_visible_pixels=int(np.count_nonzero(target & valid)),
                              foreground_pixels=int(np.count_nonzero(valid)))
        pixels = np.flatnonzero(valid).astype(np.int64)
        return self._sample_pen(xyz.reshape(-1,3)[valid], rgb.reshape(-1,3)[valid],
                                labels[valid], pixels)

    def _sample_pen(self, pts, cols, labels, source_indices=None):
        """Global crop/voxel, capped visible Object/target, robot-only fill.

        The private caches follow the output shuffle; source indices identify
        top-down image pixels when called by the renderer, or input rows otherwise.
        """
        labels = np.asarray(labels, dtype=np.int8)
        source = (np.arange(len(pts), dtype=np.int64) if source_indices is None
                  else np.asarray(source_indices, dtype=np.int64))
        if labels.shape != (len(pts),) or source.shape != (len(pts),) or not np.isin(labels, [0,1,2]).all():
            raise ValueError('Pen labels/source indices must match input points')
        self.last_labels = np.empty(0, dtype=np.int8)
        self.last_source_indices = np.empty(0, dtype=np.int64)
        classes = (('robot', 0), ('object', 1), ('target', 2))
        def counts(stage):
            self.last_info.update({stage + '_' + name: int(np.count_nonzero(labels == code))
                                   for name, code in classes})
        counts('visible')
        lo, hi = self.crop
        inside = np.all((pts >= lo) & (pts <= hi), axis=1)
        pts, cols, labels, source = pts[inside], cols[inside], labels[inside], source[inside]
        counts('crop')
        if self.voxel_size > 0 and len(pts):
            q = np.floor(pts / self.voxel_size).astype(np.int32)
            _, keep = np.unique(q, axis=0, return_index=True)
            pts, cols, labels, source = pts[keep], cols[keep], labels[keep], source[keep]
        counts('voxel')
        robot, obj, target = (np.flatnonzero(labels == code) for code in (0,1,2))
        n_obj = min(300, int(round(self.obj_fraction * self.num_points)), obj.size)
        n_target = min(100, target.size, self.num_points - n_obj)
        n_robot = self.num_points - n_obj - n_target
        self.last_info.update(sampling_failed=0, required_robot=n_robot, available_robot=int(robot.size),
                              total_sampled=0, sample_unique_total=0, sample_unique_source_pixels=0)
        for name, _ in classes:
            self.last_info.update({ 'sampled_' + name: 0, name + '_unique_sampled': 0,
                                    name + '_duplicate_samples': 0, name + '_unique_source_pixels': 0 })
        if robot.size == 0:
            self.last_info['sampling_failed'] = 1
            raise RuntimeError('Pen PCD has no visible robot points after crop/voxel; cannot fill sample')
        selected = np.concatenate([self.rng.choice(obj, n_obj, replace=False),
                                   self.rng.choice(target, n_target, replace=False),
                                   self.rng.choice(robot, n_robot, replace=robot.size < n_robot)])
        selected = self.rng.permutation(selected)
        self.last_labels, self.last_source_indices = labels[selected], source[selected]
        sampled_pts = pts[selected].astype(np.float32)
        for name, code in classes:
            mask = self.last_labels == code
            picked = self.last_source_indices[mask]
            unique = np.unique(sampled_pts[mask], axis=0).shape[0]
            self.last_info.update({'sampled_' + name: int(picked.size),
                                   name + '_unique_sampled': int(unique),
                                   name + '_duplicate_samples': int(picked.size - unique),
                                   name + '_unique_source_pixels': int(np.unique(picked).size)})
        self.last_info.update(total_sampled=int(selected.size),
                              sample_unique_total=int(np.unique(sampled_pts, axis=0).shape[0]),
                              sample_unique_source_pixels=int(np.unique(self.last_source_indices).size))
        return sampled_pts, cols[selected].astype(np.uint8)

class _WrapperMethods:
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None: self._pcd.seed(seed)
        return obs, dict(info, **self.pcd_info())

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, dict(info, **self.pcd_info())

    def pcd_info(self):
        xyz,rgb = self._pcd()
        return {'pcd_xyz':xyz,'pcd_rgb':rgb}

class AdroitPointCloudWrapper:
    """Construct a Gymnasium Wrapper lazily; legacy BC imports need no Gymnasium."""
    def __new__(cls, env, num_points=1200, cameras=('vil_camera',),
                crop_box=DEFAULT_CROP_BOX, cube_fraction=.25, foreground_only=True,
                voxel_size=.005, keep_rgb=False, size=224):
        import gymnasium as gym
        if tuple(cameras) != ('vil_camera',):
            raise ValueError('Adroit PCD requires exactly vil_camera')
        if keep_rgb:
            raise ValueError('PCD surface RGB differs from ordinary RGB; keep_rgb unsupported')
        wrapper_type = type('_AdroitGymPCDWrapper', (_WrapperMethods,gym.Wrapper), {})
        out = wrapper_type(env)
        native = env.unwrapped
        if native.sim.model.vis.quality.offsamples != 0:
            raise RuntimeError('HOLD: paired RGB/PCD requires validated separate MSAA and non-MSAA rendering; changing RGB offsamples is forbidden')
        out._pcd = AdroitPointClouds(native.sim, native.task, num_points=num_points,
            size=size, camera=cameras[0], voxel_size=voxel_size, obj_fraction=cube_fraction,
            crop_box=crop_box, foreground_only=foreground_only)
        return out
