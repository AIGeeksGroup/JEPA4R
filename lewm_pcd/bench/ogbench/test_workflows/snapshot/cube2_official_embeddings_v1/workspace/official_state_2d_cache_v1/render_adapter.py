"""Static official NPZ poses only; no action replay or observation recomputation."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

STATE = {}


def init_worker(workspace, arrays_receipt, camera):
    import atexit
    sys.path.insert(0, str(Path(workspace) / 'source/ogbench'))
    import gymnasium as gym
    import mujoco
    import ogbench  # Register the unchanged official environment.
    env = gym.make('cube-double-v0', render_mode='rgb_array', width=224, height=224,
                   ob_type='states', visualize_info=False, pixel_transparent_arm=False)
    env.reset(seed=1732001)
    base = env.unwrapped
    model, data = base._model, base._data
    assert (model.nq, model.nv, model.nu) == (28, 26, 7)
    source = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'front')
    target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'front_pixels')
    assert source >= 0 and target >= 0
    position, original = model.cam_pos[source].copy(), model.cam_quat[source].copy()
    np.testing.assert_allclose(position, camera['position'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(original, camera['base_quaternion'], atol=1e-12, rtol=0)
    assert float(model.cam_fovy[source]) == 45.
    rotation, actual = np.empty(4), np.empty(4)
    mujoco.mju_axisAngle2Quat(rotation, np.array([1., 0., 0.]), np.deg2rad(5.))
    mujoco.mju_mulQuat(actual, original, rotation)
    np.testing.assert_allclose(actual, camera['actual_quaternion'], atol=1e-12, rtol=0)
    # Match approved collection's front_pixels optical alias of original front.
    model.cam_pos[target], model.cam_quat[target], model.cam_fovy[target] = position, actual, 45.
    model.vis.quality.offsamples = 4  # Before the first lazy renderer is created.
    for i in range(2):
        for gid in base._cube_target_geom_ids_list[i]:
            assert model.geom_rgba[gid, 3] == 0
        for gid in base._cube_geom_ids_list[i]:
            np.testing.assert_array_equal(model.geom_rgba[gid], base._cube_colors[i].astype(np.float32))
    arm_materials = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, i): float(model.mat_rgba[i, 3])
                     for i in range(model.nmat)
                     if 'ur5e' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, i) or '')}
    assert arm_materials and all(alpha == 1. for alpha in arm_materials.values())
    binding = dict(source_name='front', render_name='front_pixels', source_id=int(source), render_id=int(target),
        source_position=position.tolist(), position=model.cam_pos[target].tolist(),
        source_quaternion=original.tolist(), actual_quaternion=model.cam_quat[target].tolist(),
        FOV=45., width=224, height=224, offsamples=int(model.vis.quality.offsamples),
        ob_type='states', visualize_info=False, pixel_transparent_arm=False, pullback=0, crop=None,
        arm_material_alpha=arm_materials)
    receipt = json.loads(Path(arrays_receipt).read_text())
    arrays = {split: {key: np.load(item[key]['path'], mmap_mode='r', allow_pickle=False)
                      for key in ('qpos', 'qvel')} for split, item in receipt['splits'].items()}
    assert all(not table.flags.writeable for item in arrays.values() for table in item.values())
    STATE.update(env=env, base=base, model=model, data=data, arrays=arrays, camera=binding)
    atexit.register(env.close)


def render_rows(job):
    started = time.time()
    split, ids = job
    ids = np.asarray(ids, np.int64)
    tables, base, data = STATE['arrays'][split], STATE['base'], STATE['data']
    images = np.empty((len(ids), 224, 224, 3), np.uint8)
    qpos_error = qvel_error = 0.
    xyz = {}
    for row, raw_id in enumerate(ids):
        qpos, qvel = tables['qpos'][raw_id], tables['qvel'][raw_id]
        base.set_state(qpos, qvel)
        image = base.render(camera='front_pixels')
        assert image.shape == (224, 224, 3) and image.dtype == np.uint8
        images[row] = image
        qpos_error = max(qpos_error, float(np.max(np.abs(data.qpos - qpos))))
        qvel_error = max(qvel_error, float(np.max(np.abs(data.qvel - qvel))))
        if raw_id in (999, 1000, 1001):
            xyz[str(int(raw_id))] = [data.joint(f'object_joint_{i}').qpos[:3].tolist() for i in range(2)]
    assert qpos_error == qvel_error == 0.
    return dict(split=split, raw_ids=ids, pixels=images, camera=STATE['camera'], cube_xyz_witness=xyz,
        worker_pid=os.getpid(), render_started=started, render_completed=time.time(),
        render_geometry=dict(qpos_max_abs_error=qpos_error, qvel_max_abs_error=qvel_error, nq=28, nv=26, nu=7),
        RGB_sha256=hashlib.sha256(memoryview(images)).hexdigest())


def render_episode(job):
    split, episode = job
    return render_rows((split, np.arange(episode * 1001, (episode + 1) * 1001, dtype=np.int64)))


def missing_state_witness(job):
    """One static frame: missing ctrl/target mocap must not alter opaque RGB."""
    import mujoco
    split, raw_id = job
    base, data = STATE['base'], STATE['data']
    reference = render_rows((split, [raw_id]))['pixels'][0]
    ctrl, pos, quat = data.ctrl.copy(), data.mocap_pos.copy(), data.mocap_quat.copy()
    target, success = base._target_block, base._success
    try:
        data.ctrl[:] = np.arange(len(ctrl), dtype=np.float64) * 13. - 7.
        data.mocap_pos[:] += np.array([1.1, -0.7, 0.8])
        data.mocap_quat[:] = np.array([1., 0., 0., 0.])
        base._target_block = (target + 1) % 2
        base._success = not success
        mujoco.mj_forward(STATE['model'], data)
        changed = base.render(camera='front_pixels')
        assert np.array_equal(reference, changed), 'Static RGB depends on missing control/target state'
        assert np.array_equal(data.qpos, STATE['arrays'][split]['qpos'][raw_id])
        assert np.array_equal(data.qvel, STATE['arrays'][split]['qvel'][raw_id])
        return dict(split=split, raw_id=raw_id, actual_exit_code=0, static_RGB_bitexact=True,
                    ctrl_and_target_mocap_changed=True, dynamics_or_contact_replay_proven=False)
    finally:
        data.ctrl[:], data.mocap_pos[:], data.mocap_quat[:] = ctrl, pos, quat
        base._target_block, base._success = target, success
        mujoco.mj_forward(STATE['model'], data)
