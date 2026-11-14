"""Plan or execute an independently measured current Cube camera binding."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()

def build_plan(cube_count, runtime, binding, output, swm_root, collection_seed=0):
    if type(cube_count) is not int or cube_count not in (1, 2, 3, 4):
        raise ValueError('cube_count must be in 1..4')
    if type(collection_seed) is not int or collection_seed < 0:
        raise ValueError('collection_seed must be nonnegative')
    return dict(
        schema='paper_cube_camera_plan_v1',
        cube_count=cube_count,
        protocol='paper_cube_fixed_mix_v1',
        collection_seed=collection_seed,
        reset_seed=1000000 * cube_count + collection_seed,
        runtime=str(Path(runtime).resolve()),
        binding=str(Path(binding).resolve()),
        output=str(Path(output).resolve()),
        swm_root=str(Path(swm_root).resolve()),
        scope='one fixed reset and camera-only geometry probe; no collection or model',
    )

def command(plan, python=None):
    return [
        str(python or sys.executable),
        str(Path(__file__).resolve()),
        '--cube-count',
        str(plan['cube_count']),
        '--runtime',
        plan['runtime'],
        '--binding',
        plan['binding'],
        '--output',
        plan['output'],
        '--swm-root',
        plan['swm_root'],
        '--collection-seed',
        str(plan['collection_seed']),
        '--execute',
    ]

def camera_fields(cube_count, actual):
    """Serialize measured fields; this function does not infer a camera pose."""
    for name, length in (('source_position', 3), ('source_quaternion', 4), ('position', 3), ('quaternion', 4)):
        values = actual[name]
        if len(values) != length or not all((math.isfinite(float(value)) for value in values)):
            raise ValueError('invalid measured camera field: ' + name)
    for name in ('source_quaternion', 'quaternion'):
        if abs(sum((float(value) ** 2 for value in actual[name])) - 1.0) > 1e-06:
            raise ValueError('measured camera quaternion must be normalized')
    if not 0 < float(actual['fov']) < 180 or int(actual['aa_samples']) != 4:
        raise ValueError('actual camera FOV/AA4 protocol differs')
    result = {
        key: actual[key]
        for key in ('source_position', 'source_quaternion', 'position', 'quaternion', 'fov', 'aa_samples')
    }
    result.update(
        source_name='front_pixels' if cube_count == 1 else 'front',
        render_name='front_pixels',
        local_x_up_degrees=0.0 if cube_count == 1 else 5.0,
        pullback_m=0.1 if cube_count == 4 else 0.0,
        width=224,
        height=224,
        pointcloud_crop_box=[[0.1, -0.5, -0.02], [0.75, 0.5, 0.6]]
        if cube_count < 3
        else [[-0.25, -0.5, -0.02], [0.75, 0.5, 0.9]],
    )
    return result

def _verify(plan):
    binding = json.loads(Path(plan['binding']).read_text(encoding='utf8'))
    files = {**binding.get('files', {}), **binding.get('reused_files', {})}
    root, swm = (Path(plan['runtime']), Path(plan['swm_root']))
    required = [
        root / 'code/lewm_pcd/scripts/data/double_collection.py',
        root / 'code/lewm_pcd/scripts/data/triple_collection.py',
        root / 'code/lewm_pcd/envs/cube_pcd_env.py',
        root / 'code/lewm_pcd/envs/pcd_wrapper.py',
        root / 'third_party/ogbench/ogbench/manipspace/envs/cube_env.py',
        root / 'collection_adapters/camera_pullback.py',
        swm / 'stable_worldmodel/envs/ogbench/cube_env.py',
    ]
    for path in required:
        if str(path.resolve()) not in files:
            raise ValueError('camera probe source is not bound: ' + str(path))
    if any((sha(path) != expected for path, expected in files.items())):
        raise ValueError('camera probe source checksum differs')
    return files

def _read_actual_camera(plan):
    """Actual simulation is intentionally confined to this child-only function."""
    import copy
    import importlib.util
    import os
    import pickle
    root, swm = (Path(plan['runtime']), Path(plan['swm_root']))
    sys.path[:0] = [
        str(root / 'code/lewm_pcd'),
        str(root / 'code/lewm_pcd/scripts/data'),
        str(swm),
        str(root / 'third_party/ogbench'),
    ]
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('EGL_PLATFORM', 'surfaceless')
    import mujoco
    import numpy as np
    from double_collection import local_pitch_quaternion
    cube = plan['cube_count']
    common = dict(
        env_type=('single', 'double', 'triple', 'quadruple')[cube - 1],
        ob_type='states',
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=False,
    )
    if cube == 1:
        from envs.cube_pcd_env import CubePCDEnv
        from collect_cube_pcd import _with_seeded_cube_reset
        from collection_protocol import fixed_task_info, reset_options_for_task
        env = CubePCDEnv(**common, mode='data_collection', multiview=False)
        _with_seeded_cube_reset(env)
        options = reset_options_for_task(fixed_task_info(env, 1))
    else:
        if cube == 2:
            from double_collection import make_official_double_env as factory
        else:
            from triple_collection import make_official_triple_env as factory
        env = factory(**common, mode='task', permute_blocks=False)
        options = {'task_id': 1, 'render_goal': False}
    try:
        env.reset(seed=plan['reset_seed'], options=options)
        u = env.unwrapped
        if int(u._num_cubes) != cube or tuple(u.action_space.shape) != (5,):
            raise ValueError('actual Cube/action contract differs')
        model, data = (u._model, u._data)
        source_name = 'front_pixels' if cube == 1 else 'front'
        source = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, source_name)
        render = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'front_pixels')
        if source < 0 or render < 0 or int(model.cam_bodyid[source]) != 0 or (int(model.cam_bodyid[render]) != 0):
            raise ValueError('expected actual world-fixed source/render cameras')
        position, quaternion, fov = (
            model.cam_pos[source].copy(),
            model.cam_quat[source].copy(),
            float(model.cam_fovy[source]),
        )
        before = {key: getattr(data, key).copy() for key in ('qpos', 'qvel', 'ctrl', 'act', 'mocap_pos', 'mocap_quat')}
        rng = pickle.dumps(copy.deepcopy(u.np_random.bit_generator.state))
        model.cam_pos[render] = position
        model.cam_fovy[render] = fov
        model.cam_quat[render] = quaternion if cube == 1 else local_pitch_quaternion(quaternion, 5.0)
        mujoco.mj_camlight(model, data)
        if cube == 4:
            path = root / 'collection_adapters/camera_pullback.py'
            spec = importlib.util.spec_from_file_location('paper_cube_camera_math', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model.cam_pos[render] = module.camera_position(position, data.cam_xmat[render], 0.1)
            mujoco.mj_camlight(model, data)
        if any((not np.array_equal(values, getattr(data, key)) for key, values in before.items())):
            raise ValueError('camera-only probe changed physical state')
        if rng != pickle.dumps(u.np_random.bit_generator.state):
            raise ValueError('camera-only probe changed environment RNG')
        return dict(
            source_position=position.tolist(),
            source_quaternion=quaternion.tolist(),
            position=data.cam_xpos[render].tolist(),
            quaternion=model.cam_quat[render].tolist(),
            fov=float(model.cam_fovy[render]),
            aa_samples=int(model.vis.quality.offsamples),
            physical_state_unchanged=True,
            environment_rng_unchanged=True,
        )
    finally:
        env.close()

def probe_camera(plan):
    """Run only in the explicitly dispatched independent child, then write a new receipt."""
    output = Path(plan['output'])
    if output.exists():
        raise FileExistsError(output)
    expected = build_plan(
        plan['cube_count'], plan['runtime'], plan['binding'], output, plan['swm_root'], plan['collection_seed']
    )
    if plan != expected:
        raise ValueError('camera probe plan differs')
    files = _verify(plan)
    actual = _read_actual_camera(plan)
    if any((sha(path) != expected_sha for path, expected_sha in files.items())):
        raise ValueError('camera probe source changed during execution')
    record = dict(
        schema='paper_cube_camera_binding_v1',
        protocol=plan['protocol'],
        cube_count=plan['cube_count'],
        camera=camera_fields(plan['cube_count'], actual),
        actual_reset_seed=plan['reset_seed'],
        probe_scope=plan['scope'],
        physical_state_unchanged=actual['physical_state_unchanged'],
        environment_rng_unchanged=actual['environment_rng_unchanged'],
        producer_binding=dict(files=files, probe_source_sha256=sha(__file__)),
        actual_exit_code=0,
        collection_executed=False,
        model_loaded=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    return record

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cube-count', type=int, required=True, choices=(1, 2, 3, 4))
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--binding', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--swm-root', required=True)
    parser.add_argument('--collection-seed', type=int, default=0)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = build_plan(args.cube_count, args.runtime, args.binding, args.output, args.swm_root, args.collection_seed)
    print(json.dumps(plan, indent=2), flush=True)
    if args.execute:
        probe_camera(plan)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
