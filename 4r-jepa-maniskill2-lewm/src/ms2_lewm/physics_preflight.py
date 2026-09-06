"""Renderer-free PickCube physics, exclusively for expert-policy prechecks.

This is not a data-collection environment: it cannot produce images or point
clouds. Physics, robot, controller, reset, reward and success logic are inherited
from the pinned ManiSkill2 0.5.3 implementation. Only renderer/visual setup is
omitted. The original installed files and vendor repositories are not edited.

Exact native configuration retained: PickCubeEnv constructor retains
obj_init_rot_z=True, cube half-size [0.02,0.02,0.02], StationaryManipulationEnv
retains panda and robot_init_qpos_noise=0.02, and BaseEnv retains sim_freq=500,
control_freq=20 with 25 physics steps per action. This factory selects
pd_ee_delta_pose, as in the validated RL baseline. The native reward default
is normalized_dense; selecting it is unchanged. Native friction, restitution,
contact offset, solver iterations, URDF collision materials and controller
parameters are inherited, not copied or retuned.

Render-only exemptions: no SapienRenderer/RenderClient or engine renderer,
no camera instances/lights/background, cube visual geometry/material omitted,
ground render=False, and goal visual sphere omitted while preserving the same
collision-free static goal actor. State initialization/random draws and all
collision/body builders retain native semantics. This precheck cannot certify
image/pointcloud collection, which must use the actual rendered environment.
"""
from collections import OrderedDict
import os

import numpy as np
import sapien.core as sapien

from mani_skill2 import logger
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv
from mani_skill2.sensors.camera import update_camera_cfgs_from_dict
from mani_skill2.utils.common import convert_observation_to_space


class PhysicsOnlyBase(BaseEnv):
    """BaseEnv initialization with graphics resources deliberately absent."""

    preflight_only = True

    def __init__(self, obs_mode=None, reward_mode=None, control_mode=None,
                 render_mode=None, sim_freq=500, control_freq=20,
                 renderer="sapien", renderer_kwargs=None, shader_dir="ibl",
                 render_config=None, enable_shadow=False, camera_cfgs=None,
                 render_camera_cfgs=None, bg_name=None):
        if obs_mode not in (None, "state", "state_dict", "none"):
            raise ValueError("Physics preflight supports only actual simulator state, never visual observations")
        if render_mode is not None or bg_name is not None:
            raise ValueError("Rendering/backgrounds are unavailable in physics preflight")
        # Matches BaseEnv setup; crucially no renderer is created or installed.
        self._engine = sapien.Engine()
        self._engine.set_log_level(os.getenv("MS2_SIM_LOG_LEVEL", "error"))
        self._renderer_type = "physics_preflight"
        self._renderer = None
        self._sim_freq = sim_freq
        self._control_freq = control_freq
        if sim_freq % control_freq != 0:
            logger.warning(f"sim_freq({sim_freq}) is not divisible by control_freq({control_freq}).")
        self._sim_steps_per_control = sim_freq // control_freq
        self._obs_mode = self.SUPPORTED_OBS_MODES[0] if obs_mode is None else obs_mode
        self._reward_mode = self.SUPPORTED_REWARD_MODES[0] if reward_mode is None else reward_mode
        if self._reward_mode not in self.SUPPORTED_REWARD_MODES:
            raise NotImplementedError(self._reward_mode)
        self._control_mode = control_mode
        if control_mode == "*":
            raise NotImplementedError("Multiple controllers are not supported by official BaseEnv")
        self.render_mode = None
        self._viewer = None
        self._configure_agent()
        # Camera configurations are pure data and retain upstream config setup;
        # _setup_cameras below never creates any camera or rendering resource.
        self._configure_cameras()
        self._configure_render_cameras()
        if camera_cfgs is not None:
            update_camera_cfgs_from_dict(self._camera_cfgs, camera_cfgs)
        if render_camera_cfgs is not None:
            update_camera_cfgs_from_dict(self._render_camera_cfgs, render_camera_cfgs)
        self.enable_shadow = enable_shadow
        self.bg_name = None
        self._main_seed = None
        self.set_main_rng(2022)
        obs, _ = self.reset(seed=2022, options=dict(reconfigure=True))
        self.observation_space = convert_observation_to_space(obs)
        self.action_space = self.agent.action_space

    def _setup_cameras(self):
        self._cameras = OrderedDict()
        self._render_cameras = OrderedDict()

    def _setup_lighting(self):
        pass

    def _load_background(self):
        assert self.bg_name is None

    def _add_ground(self, altitude=0.0, render=True):
        # Upstream method uses identical ground collision/material defaults.
        return super()._add_ground(altitude=altitude, render=False)

    def render(self, *args, **kwargs):
        raise RuntimeError("Physics-only preflight cannot render or collect visual data")

    render_human = render
    render_rgb_array = render
    render_cameras = render
    update_render = render
    take_picture = render
    get_images = render
    get_camera_params = render
    _get_obs_images = render
    collect = render
    export_dataset = render
    save_trajectory = render


class PhysicsOnlyPickCube(PickCubeEnv, PhysicsOnlyBase):
    """Official PickCube with only visual actor components excluded."""

    def _build_cube(self, half_size, color=(1, 0, 0), name="cube", static=False,
                    render_material=None):
        # Exact upstream collision and body construction, without visual calls.
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        return builder.build_static(name) if static else builder.build(name)

    def _build_sphere_site(self, radius, color=(0, 1, 0), name="goal_site"):
        # Keep the same static, collision-free actor and pose/state ordering.
        return self._scene.create_actor_builder().build_static(name)


def create_physics_preflight(**kwargs):
    """Construct a raw state-only env; caller may add official TimeLimit(200)."""
    kwargs.setdefault("obs_mode", "state")
    kwargs.setdefault("control_mode", "pd_ee_delta_pose")
    return PhysicsOnlyPickCube(**kwargs)


def _diagnose():
    import argparse
    import hashlib
    import inspect
    import json
    from pathlib import Path
    import time
    from unittest.mock import patch

    import pynvml

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' for the isolated CPU-only diagnostic")
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

    def own_gpu_bytes():
        return sum(p.usedGpuMemory for p in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu)
                   if p.pid == os.getpid())

    started = time.monotonic()
    samples = [own_gpu_bytes()]
    with (patch.object(sapien, "SapienRenderer", side_effect=AssertionError("Forbidden renderer construction")) as renderer,
          patch.object(sapien, "RenderClient", side_effect=AssertionError("Forbidden render client construction")) as client):
        env = create_physics_preflight()
        samples.append(own_gpu_bytes())
        obs, _ = env.reset(seed=20260906)
        before = env.get_state().copy()
        result = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        after = env.get_state().copy()
        samples.append(own_gpu_bytes())
        source_methods = ["_get_default_scene_config", "_setup_scene", "_load_agent", "reset",
                          "step", "step_action", "_initialize_actors", "_initialize_agent",
                          "_initialize_task", "evaluate", "compute_dense_reward"]
        inherited = {name: getattr(PhysicsOnlyPickCube, name) is getattr(PickCubeEnv, name)
                     for name in source_methods}
        summary = dict(kind="physics-only expert preflight; no images, no point clouds, no dataset",
                       pid=os.getpid(), renderer_calls=renderer.call_count, render_client_calls=client.call_count,
                       own_gpu_bytes_samples=samples, observation_shape=list(obs.shape),
                       action_shape=list(env.action_space.shape), state_shape=list(before.shape),
                       state_l1_change=float(np.abs(after-before).sum()), reward=float(result[1]),
                       success=bool(result[4]["success"]), finite_state=bool(np.isfinite(after).all()),
                       inherited_physics_methods=inherited,
                       actor_names=[actor.name for actor in env.get_actors()],
                       cube_mass=float(env.obj.mass), cube_collision_shapes=len(env.obj.get_collision_shapes()),
                       robot_dof=int(env.agent.robot.dof), sim_freq=env.sim_freq, control_freq=env.control_freq,
                       elapsed_seconds=time.monotonic()-started,
                       upstream_base_source_sha256=hashlib.sha256(Path(inspect.getfile(BaseEnv)).read_bytes()).hexdigest())
        native_signature = inspect.signature(BaseEnv.__init__)
        local_signature = inspect.signature(PhysicsOnlyBase.__init__)
        summary["native_base_constructor_defaults_match"] = {
            key: local_signature.parameters[key].default == parameter.default
            for key, parameter in native_signature.parameters.items() if key != "self"}
        scene_config = env._get_default_scene_config()
        summary["native_configuration"] = dict(
            robot=env.robot_uid, robot_init_qpos_noise=env.robot_init_qpos_noise,
            obj_init_rot_z=env.obj_init_rot_z, cube_half_size=env.cube_half_size.tolist(),
            control_mode=env.control_mode, reward_mode=env.reward_mode,
            sim_steps_per_control=env._sim_steps_per_control,
            default_dynamic_friction=scene_config.default_dynamic_friction,
            default_static_friction=scene_config.default_static_friction,
            default_restitution=scene_config.default_restitution,
            contact_offset=scene_config.contact_offset,
            enable_pcm=scene_config.enable_pcm,
            solver_iterations=scene_config.solver_iterations,
            solver_velocity_iterations=scene_config.solver_velocity_iterations)
        rejected = {}
        for name in ("render", "render_human", "render_rgb_array", "render_cameras",
                     "update_render", "get_images", "get_camera_params", "take_picture",
                     "_get_obs_images", "collect", "export_dataset", "save_trajectory"):
            try:
                getattr(env, name)()
            except RuntimeError:
                rejected[name] = True
            else:
                raise AssertionError(f"Graphics/data export unexpectedly accepted: {name}")
        for mode in ("image", "pointcloud"):
            try:
                create_physics_preflight(obs_mode=mode)
            except ValueError:
                rejected[f"obs_mode={mode}"] = True
            else:
                raise AssertionError(f"Visual observation unexpectedly accepted: {mode}")
        summary["forbidden_graphics_and_data_interfaces_rejected"] = rejected
        assert all(summary["native_base_constructor_defaults_match"].values())
        samples.append(own_gpu_bytes())
        env.close()
    assert renderer.call_count == client.call_count == 0
    assert max(samples) == 0 and all(inherited.values()) and summary["finite_state"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    _diagnose()
