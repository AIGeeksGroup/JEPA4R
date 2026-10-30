"""Original Triple AA4/NoAA frame capture, with per-episode independent private streams."""
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from common import episode_seed, load_source, object_sha

STATE = {}


def init_worker(workspace, arrays_receipt, camera, pcd):
    root = Path(workspace)
    original = load_source('_pcd_original_rgb_pose', root / 'official_state_2d_cache_v1/render_adapter.py')
    original.init_worker(workspace, arrays_receipt, camera)
    core = load_source('_pcd_original_core', root / 'source/lewm_pcd/envs/pcd_wrapper.py')
    native = load_source('_pcd_original_native', root / 'external/native_pcd_frame.py')
    STATE.update(original=original, base=original.STATE['base'], data=original.STATE['data'],
        arrays=original.STATE['arrays'], camera=original.STATE['camera'], root=root,
        native=native, core_factory=partial(core.PointCloudWrapper, crop_box=pcd['crop_box']))


def _frame(split, episode):
    root = STATE['root']
    return STATE['native'].NativePCDFrame(root / 'source/lewm_pcd/envs/pcd_wrapper.py',
        root / 'external/pcd_geometry.py', episode_seed(split, episode), core_factory=STATE['core_factory'])


def _capture(frame, split, raw_id):
    base, data = STATE['base'], STATE['data']
    tables = STATE['arrays'][split]
    qpos, qvel = tables['qpos'][raw_id], tables['qvel'][raw_id]
    assert qpos.shape == (35,) and qvel.shape == (32,)
    base.set_state(qpos, qvel)
    rgb = base.render(camera='front_pixels')
    cloud = frame.capture(base, data, rgb, 'current')
    assert np.array_equal(data.qpos, qpos) and np.array_equal(data.qvel, qvel)
    assert cloud.shape == (1200, 6) and cloud.dtype == np.float32 and np.isfinite(cloud).all()
    return cloud, rgb


def render_rows(job):
    """Advance the original sampling stream from episode t0 even for a sparse witness."""
    split, requested = job
    requested = np.asarray(requested, np.int64)
    assert requested.ndim == 1 and len(requested) and np.all(np.diff(requested) > 0)
    points, colors, images, witnesses = [], [], [], {}
    for episode in np.unique(requested // 1001):
        ids = requested[requested // 1001 == episode]
        wanted = set(ids.tolist())
        frame = _frame(split, int(episode))
        try:
            for raw_id in range(int(episode) * 1001, int(ids[-1]) + 1):
                cloud, image = _capture(frame, split, raw_id)
                if raw_id in wanted:
                    points.append(cloud[:, :3].copy())
                    colors.append(cloud[:, 3:].astype(np.uint8))
                    images.append(image.copy())
                    witnesses[str(raw_id)] = dict(frame.last_receipts['current'])
        finally:
            frame.close()
    return dict(raw_ids=requested, pcd_xyz=np.stack(points), pcd_rgb=np.stack(colors),
                pixels=np.stack(images), capture_witnesses=witnesses, camera=STATE['camera'])


def render_episode(job):
    started = time.time()
    split, episode = job
    frame = _frame(split, episode)
    points = np.empty((1001, 1200, 3), np.float32)
    colors = np.empty((1001, 1200, 3), np.uint8)
    image_hash, trace_hash, boundary = hashlib.sha256(), hashlib.sha256(), {}
    xyz_witness = {}
    try:
        for row, raw_id in enumerate(range(episode * 1001, (episode + 1) * 1001)):
            cloud, rgb = _capture(frame, split, raw_id)
            points[row], colors[row] = cloud[:, :3], cloud[:, 3:].astype(np.uint8)
            capture = frame.last_receipts['current']
            image_hash.update(memoryview(rgb))
            trace_hash.update(object_sha(capture).encode())
            if row in (0, 999, 1000):
                boundary[str(row)] = capture
                xyz_witness[str(raw_id)] = [STATE['data'].joint(f'object_joint_{i}').qpos[:3].tolist()
                                           for i in range(3)]
        assert frame.counts == {'current': 1001, 'goal': 0}
        return dict(split=split, episode=episode, pcd_xyz=points, pcd_rgb=colors,
            raw_ids=np.arange(episode * 1001, (episode + 1) * 1001, dtype=np.int64),
            sampling_seed=episode_seed(split, episode), capture_boundaries=boundary,
            capture_receipt_stream_sha256=trace_hash.hexdigest(), RGB_sha256=image_hash.hexdigest(),
            PCD_xyz_sha256=hashlib.sha256(memoryview(points)).hexdigest(),
            PCD_rgb_sha256=hashlib.sha256(memoryview(colors)).hexdigest(),
            camera=STATE['camera'], cube_xyz_witness=xyz_witness,
            worker_pid=os.getpid(), render_started=started, render_completed=time.time(),
            render_geometry=dict(qpos_max_abs_error=0., qvel_max_abs_error=0., nq=35, nv=32, nu=7),
            boundary_clouds={key: dict(pcd_xyz=points[int(key)].copy(), pcd_rgb=colors[int(key)].copy())
                             for key in ('0', '999', '1000')})
    finally:
        frame.close()


def missing_state_witness(job):
    """One same-pose/same-private-RNG PCD check; not a dynamics/contact replay proof."""
    import copy
    import mujoco
    split, raw_id = job
    base, data = STATE['base'], STATE['data']
    frame = _frame(split, int(raw_id) // 1001)
    ctrl, pos, quat = data.ctrl.copy(), data.mocap_pos.copy(), data.mocap_quat.copy()
    target, success = base._target_block, base._success
    rng = copy.deepcopy(frame.cores['current']._rng.bit_generator.state)
    try:
        reference, rgb = _capture(frame, split, raw_id)
        before = frame.last_receipts['current']
        data.ctrl[:] = np.arange(len(ctrl)) * 13. - 7.
        data.mocap_pos[:] += [1.1, -.7, .8]
        data.mocap_quat[:] = [1., 0., 0., 0.]
        base._target_block, base._success = (target + 1) % 3, not success
        mujoco.mj_forward(STATE['original'].STATE['model'], data)
        changed_rgb = base.render(camera='front_pixels')
        frame.cores['current']._rng.bit_generator.state = rng
        changed = frame.capture(base, data, changed_rgb, 'current')
        after = frame.last_receipts['current']
        assert np.array_equal(reference, changed) and np.array_equal(rgb, changed_rgb)
        for key in ('source_UV_sha256', 'source_geom_sha256', 'cloud_sha256', 'rng_before_sha256', 'rng_after_sha256'):
            assert before[key] == after[key], key
        return dict(actual_exit_code=0, split=split, raw_id=int(raw_id), static_RGB_PCD_bitexact=True,
            same_sampling_RNG=True, ctrl_and_target_mocap_changed=True, dynamics_or_contact_replay_proven=False)
    finally:
        data.ctrl[:], data.mocap_pos[:], data.mocap_quat[:] = ctrl, pos, quat
        base._target_block, base._success = target, success
        mujoco.mj_forward(STATE['original'].STATE['model'], data)
        frame.close()
