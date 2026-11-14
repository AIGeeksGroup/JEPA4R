"""camera stage of the Cube workflow: the camera of the Cube collection (its pose and field of view), read from the
simulator in a separate process and written to CAMERA.json."""

import argparse
import json
import math
from pathlib import Path
import sys


def build_plan(cube_count, runtime, output, swm_root, collection_seed=0):
    if cube_count not in (1, 2, 3, 4):
        raise ValueError("cube_count must be in 1..4")
    return dict(
        cube_count=cube_count,
        collection_seed=collection_seed,
        reset_seed=1000000 * cube_count + collection_seed,
        runtime=str(Path(runtime).resolve()),
        output=str(Path(output).resolve()),
        swm_root=str(Path(swm_root).resolve()),
    )


def command(plan, python=None):
    return [
        str(python or sys.executable),
        str(Path(__file__).resolve()),
        "--cube-count",
        str(plan["cube_count"]),
        "--runtime",
        plan["runtime"],
        "--output",
        plan["output"],
        "--swm-root",
        plan["swm_root"],
        "--collection-seed",
        str(plan["collection_seed"]),
    ]


def camera_fields(cube_count, actual):
    """The camera record of CAMERA.json from the values read from the simulator."""
    for name, length in (("source_position", 3), ("source_quaternion", 4), ("position", 3), ("quaternion", 4)):
        values = actual[name]
        if len(values) != length or not all((math.isfinite(float(value)) for value in values)):
            raise ValueError("invalid measured camera field: " + name)
    for name in ("source_quaternion", "quaternion"):
        if abs(sum((float(value) ** 2 for value in actual[name])) - 1.0) > 1e-06:
            raise ValueError("measured camera quaternion must be normalized")
    if not 0 < float(actual["fov"]) < 180 or int(actual["aa_samples"]) != 4:
        raise ValueError("actual camera FOV/AA4 protocol differs")
    result = {
        key: actual[key]
        for key in ("source_position", "source_quaternion", "position", "quaternion", "fov", "aa_samples")
    }
    result.update(
        source_name="front_pixels" if cube_count == 1 else "front",
        render_name="front_pixels",
        local_x_up_degrees=0.0 if cube_count == 1 else 5.0,
        pullback_m=0.1 if cube_count == 4 else 0.0,
        width=224,
        height=224,
        pointcloud_crop_box=[[0.1, -0.5, -0.02], [0.75, 0.5, 0.6]]
        if cube_count < 3
        else [[-0.25, -0.5, -0.02], [0.75, 0.5, 0.9]],
    )
    return result


def _read_actual_camera(plan):
    """The camera of the collection env after its first reset, as the collection sets it up."""
    import importlib.util
    import os

    root, swm = (Path(plan["runtime"]), Path(plan["swm_root"]))
    sys.path[:0] = [
        str(root / "code/collection"),
        str(root / "code/collection/scripts/data"),
        str(swm),
        str(root / "third_party/ogbench"),
    ]
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    import mujoco
    from double_collection import local_pitch_quaternion

    cube = plan["cube_count"]
    common = dict(
        env_type=("single", "double", "triple", "quadruple")[cube - 1],
        ob_type="states",
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=False,
    )
    if cube == 1:
        from envs.cube_pcd_env import CubePCDEnv
        from collect_cube_pcd import _with_seeded_cube_reset
        from collection_protocol import fixed_task_info, reset_options_for_task

        env = CubePCDEnv(**common, mode="data_collection", multiview=False)
        _with_seeded_cube_reset(env)
        options = reset_options_for_task(fixed_task_info(env, 1))
    else:
        if cube == 2:
            from double_collection import make_official_double_env as factory
        else:
            from triple_collection import make_official_triple_env as factory
        env = factory(**common, mode="task", permute_blocks=False)
        options = {"task_id": 1, "render_goal": False}
    try:
        env.reset(seed=plan["reset_seed"], options=options)
        u = env.unwrapped
        if int(u._num_cubes) != cube or tuple(u.action_space.shape) != (5,):
            raise ValueError("actual Cube/action contract differs")
        model, data = (u._model, u._data)
        source_name = "front_pixels" if cube == 1 else "front"
        source = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, source_name)
        render = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_pixels")
        if source < 0 or render < 0 or int(model.cam_bodyid[source]) != 0 or (int(model.cam_bodyid[render]) != 0):
            raise ValueError("expected actual world-fixed source/render cameras")
        position, quaternion, fov = (
            model.cam_pos[source].copy(),
            model.cam_quat[source].copy(),
            float(model.cam_fovy[source]),
        )
        model.cam_pos[render] = position
        model.cam_fovy[render] = fov
        model.cam_quat[render] = quaternion if cube == 1 else local_pitch_quaternion(quaternion, 5.0)
        mujoco.mj_camlight(model, data)
        if cube == 4:
            path = root / "collection_adapters/camera_pullback.py"
            spec = importlib.util.spec_from_file_location("paper_cube_camera_math", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model.cam_pos[render] = module.camera_position(position, data.cam_xmat[render], 0.1)
            mujoco.mj_camlight(model, data)
        return dict(
            source_position=position.tolist(),
            source_quaternion=quaternion.tolist(),
            position=data.cam_xpos[render].tolist(),
            quaternion=model.cam_quat[render].tolist(),
            fov=float(model.cam_fovy[render]),
            aa_samples=int(model.vis.quality.offsamples),
        )
    finally:
        env.close()


def probe_camera(plan):
    """Read the camera in the simulator and write CAMERA.json."""
    actual = _read_actual_camera(plan)
    record = dict(
        cube_count=plan["cube_count"], reset_seed=plan["reset_seed"], camera=camera_fields(plan["cube_count"], actual)
    )
    output = Path(plan["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-count", type=int, required=True, choices=(1, 2, 3, 4))
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--swm-root", required=True)
    parser.add_argument("--collection-seed", type=int, default=0)
    args = parser.parse_args()
    probe_camera(build_plan(args.cube_count, args.runtime, args.output, args.swm_root, args.collection_seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
