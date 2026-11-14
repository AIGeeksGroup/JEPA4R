"""The online evaluation of a Cube policy: OGBench's own evaluate() on the official CubeEnv, five tasks x 50 episodes
(seed 970000000 + 100000 x task + episode), the frozen encoder applied online to the current and the goal view at every
policy call (RGB frame, or the point cloud of the same camera for JEPA4R)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np

from ._source import PCD_ROOT, VENDOR, install_impls

ROOT = VENDOR
LATENT_DIM = 192
OnlineEncoder = None  # set by the caller (workflow_downstream.py) before use
CUBE_NAMES = {1: "single", 2: "double", 3: "triple", 4: "quadruple"}
GROUPS = {}


def group_contract(group: str) -> dict:
    cfg = GROUPS[group]
    cubes = cfg.get("cube_count", 2)
    result = {
        "cube_count": cubes,
        "state_dim": cfg.get("state_dim", 19 + 9 * cubes),
        "current_dim": cfg.get("current_dim", 211),
        "goal_dim": cfg.get("goal_dim", 192 + 3 * cubes),
        "max_episode_steps": cfg.get("max_episode_steps", 500),
        "mode": cfg["mode"],
    }
    if cfg.get("pcd_crop_box") is not None:
        result["pcd_crop_box"] = cfg["pcd_crop_box"]
    return result


def camera_for_group(group):
    if group not in GROUPS:
        raise ValueError(group)
    return dict(GROUPS[group]["camera"], **{"geometry_offsamples": 0} if GROUPS[group].get("mode") == "pcd" else {})


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def seed_for(task: int, episode_index: int) -> int:
    if task not in range(1, 6) or episode_index not in range(50):
        raise ValueError("task must be 1..5 and episode_index must be 0..49")
    return 970000000 + task * 100000 + episode_index


def validate_latent(value: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(value)
    if value.shape != (LATENT_DIM,) or value.dtype != np.float32 or (not np.isfinite(value).all()):
        raise ValueError(
            f"{label} latent must be finite float32 shape ({LATENT_DIM},), got {value.shape} {value.dtype}"
        )
    return value


def goal_xyz(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state)
    if (
        state.ndim != 1
        or state.size < 28
        or (state.size - 19) % 9
        or (state.dtype != np.float32)
        or (not np.isfinite(state).all())
    ):
        raise ValueError("goal state must be finite float32 with the official 19+9N layout")
    cubes = (state.size - 19) // 9
    return np.asarray(
        [state[index] for cube in range(cubes) for index in range(19 + 9 * cube, 22 + 9 * cube)], dtype=np.float32
    )


def make_policy_inputs(
    current_latent: np.ndarray, goal_latent: np.ndarray, current_state: np.ndarray, goal_state: np.ndarray
):
    current_latent = validate_latent(current_latent, "current")
    goal_latent = validate_latent(goal_latent, "goal")
    current_state = np.asarray(current_state, dtype=np.float32)
    if (
        current_state.ndim != 1
        or current_state.size < 28
        or (current_state.size - 19) % 9
        or (not np.isfinite(current_state).all())
    ):
        raise ValueError("current state must use the official 19+9N layout")
    return (
        np.concatenate([current_latent, current_state[:19]]).astype(np.float32),
        np.concatenate([goal_latent, goal_xyz(np.asarray(goal_state, dtype=np.float32))]).astype(np.float32),
    )


def configure_camera(base, group: str) -> dict:
    """Set the render camera of the run (the source camera turned up by local_x_up_degrees and pulled back by
    pullback_m along its view axis, with its field of view; 4x antialiasing) and return its values."""
    import mujoco

    model = base._model
    camera = camera_for_group(group)
    source = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera["source_name"])
    target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera["render_name"])
    if source < 0 or target < 0:
        raise RuntimeError("the source / render cameras are missing")
    position = model.cam_pos[source].copy()
    original = model.cam_quat[source].copy()
    if (
        not np.allclose(position, camera.get("source_position", camera["position"]), atol=1e-12, rtol=0)
        or float(model.cam_fovy[source]) != camera["fov"]
    ):
        raise RuntimeError("the source camera's pose / field of view differ from the run's camera")
    rotation = np.empty(4)
    actual = np.empty(4)
    mujoco.mju_axisAngle2Quat(rotation, np.array([1.0, 0.0, 0.0]), np.deg2rad(camera["local_x_up_degrees"]))
    mujoco.mju_mulQuat(actual, original, rotation)
    w, x, y, z = actual
    local_z = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    model.cam_pos[target] = position + camera.get("pullback_m", 0.0) * local_z
    if not np.allclose(model.cam_pos[target], camera["position"], atol=1e-12, rtol=0):
        raise RuntimeError("the configured camera position differs from its rotation / pullback")
    model.cam_quat[target] = actual
    model.cam_fovy[target] = camera["fov"]
    model.vis.quality.offsamples = 4
    result = {**camera, "actual_quaternion": actual.tolist(), "source_quaternion": original.tolist()}
    result.update(camera_for_group(group))
    return result


def load_agent(group: str):
    """The trained GCIQL agent of the run (its last checkpoint), on the CPU."""
    from . import trainer

    return trainer.load_policy(json.loads(Path(GROUPS[group]["training"]).read_text()))


def make_env(group: str, seed: int):
    import gymnasium as gym

    contract = group_contract(group)
    install_impls()
    import ogbench

    if contract["mode"] == "pcd":
        from ogbench.manipspace.envs.cube_env import CubeEnv as OfficialCubeEnv

        shim_package = types.ModuleType("stable_worldmodel.envs.ogbench")
        shim_package.__path__ = []
        shim_parent = types.ModuleType("stable_worldmodel.envs.ogbench.cube_env")
        shim_parent.CubeEnv = OfficialCubeEnv
        sys.modules["stable_worldmodel.envs.ogbench"] = shim_package
        sys.modules["stable_worldmodel.envs.ogbench.cube_env"] = shim_parent
        path = PCD_ROOT / "cube_pcd_env.py"
        spec = importlib.util.spec_from_file_location("_sr_cube_pcd", path)
        pcd_env_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pcd_env_module)
        env_type = CUBE_NAMES[contract["cube_count"]]
        base = pcd_env_module.CubePCDEnv(
            env_type=env_type,
            render_mode="rgb_array",
            width=224,
            height=224,
            ob_type="states",
            visualize_info=False,
            pixel_transparent_arm=False,
        )
        env = gym.wrappers.TimeLimit(base, max_episode_steps=contract["max_episode_steps"])
    else:
        play_env = "cube-" + CUBE_NAMES[contract["cube_count"]] + "-play-v0"
        env = ogbench.make_env_and_datasets(
            play_env,
            env_only=True,
            render_mode="rgb_array",
            width=224,
            height=224,
            ob_type="states",
            visualize_info=False,
            pixel_transparent_arm=False,
        )
        base = env.unwrapped
        env = gym.wrappers.TimeLimit(env, max_episode_steps=contract["max_episode_steps"])
    base.reset(seed=0)
    camera = configure_camera(base, group)
    if contract["mode"] == "rgb":
        original_render = base.render

        def render(*args, camera=None, **kwargs):
            if camera is None and getattr(base, "_render_goal", False):
                camera = "front_pixels"
            return original_render(*args, camera=camera, **kwargs)

        base.render = render
    return env, camera


class OnlineVisualEnv:
    """Thin reset/step wrapper: renders the goal scene at reset and the current view (RGB, or its point cloud) on
    demand."""

    def __init__(self, env, group: str, encoder: OnlineEncoder, seed: int):
        self.env, self.group, self.encoder, self.seed = env, group, encoder, seed
        self.goal_raw = None
        self.frame = None

    def capture(self, rgb, phase):
        base = self.env.unwrapped
        return self.frame.capture(base, base._data, rgb, phase)

    def reset(self, *, options=None, **kwargs):
        options = dict(options or {})
        options["render_goal"] = True
        if group_contract(self.group)["mode"] == "pcd":
            sys.path.insert(0, str(PCD_ROOT))
            from native_pcd_frame import NativePCDFrame

            path = PCD_ROOT / "pcd_wrapper.py"
            spec = importlib.util.spec_from_file_location("_sr_pcd_main", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            crop_box = GROUPS[self.group].get("pcd_crop_box")
            core_factory = (
                module.PointCloudWrapper
                if crop_box is None
                else __import__("functools").partial(module.PointCloudWrapper, crop_box=crop_box)
            )
            self.frame = NativePCDFrame(path, PCD_ROOT / "pcd_geometry.py", self.seed, core_factory=core_factory)
            base = self.env.unwrapped
            base._pcd_goal_capture = None
            base._pcd_goal_capture_callback = lambda: self.capture(base.render(camera="front_pixels"), "goal")
        observation, info = self.env.reset(seed=self.seed, options=options)
        if group_contract(self.group)["mode"] == "rgb":
            goal_raw = np.asarray(info["goal_rendered"])
        else:
            goal_raw = self.env.unwrapped._pcd_goal_capture
            if goal_raw is None:
                raise RuntimeError("CubePCDEnv did not capture the goal scene")
        self.goal_raw = np.array(goal_raw, copy=True)
        return observation, info

    def step(self, action):
        return self.env.step(action)

    def current_raw(self):
        rgb = self.env.unwrapped.render(camera="front_pixels")
        return rgb if group_contract(self.group)["mode"] == "rgb" else self.capture(rgb, "current")

    def close(self):
        if self.frame is not None:
            self.frame.close()
        self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


class OnlineGCIQLPolicy:
    """The GCIQL agent on the online latents: at every call the goal view and the current view are encoded."""

    def __init__(self, agent, env: OnlineVisualEnv):
        self.agent, self.env = agent, env

    def sample_actions(self, observations, goals=None, seed=None, temperature=0.0):
        observation = np.asarray(observations, dtype=np.float32)
        goal = np.asarray(goals, dtype=np.float32)
        if observation.shape == (1, group_contract(self.env.group)["state_dim"]):
            observation = observation[0]
        if goal.shape == (1, group_contract(self.env.group)["state_dim"]):
            goal = goal[0]
        goal_latent = self.env.encoder.encode(self.env.goal_raw)
        latent = self.env.encoder.encode(self.env.current_raw())
        current, goal_input = make_policy_inputs(latent, goal_latent, observation, goal)
        return self.agent.sample_actions(observations=current, goals=goal_input, seed=seed, temperature=temperature)


def run_episode(group: str, agent, encoder: OnlineEncoder, task: int, index: int, output: Path):
    import random

    install_impls()
    from utils.evaluation import evaluate

    seed = seed_for(task, index)
    env, _camera = make_env(group, seed)
    policy_env = None
    try:
        random.seed(seed)
        np.random.seed(seed)
        policy_env = OnlineVisualEnv(env, group, encoder, seed)
        stats, trajectories, _ = evaluate(
            OnlineGCIQLPolicy(agent, policy_env),
            policy_env,
            task_id=task,
            config=GROUPS[group]["agent_config"],
            num_eval_episodes=1,
            num_video_episodes=0,
            eval_temperature=0,
            eval_gaussian=None,
        )
        traj = trajectories[0]
        actions = np.asarray(traj["action"], dtype=np.float32)
        rewards = np.asarray(traj["reward"], dtype=np.float32)
        successes = np.asarray([bool(row.get("success", False)) for row in traj["info"]], dtype=bool)
        result = dict(
            model=group, task_id=task, episode_index=index, seed=seed, steps=len(actions), success=int(stats["success"])
        )
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output / "steps.npz", actions=actions, rewards=rewards, successes=successes)
        atomic_json(output / "result.json", result)
        return result
    finally:
        if policy_env is not None:
            policy_env.close()
        else:
            env.close()


def main(group: str, output: Path, tasks=(1, 2, 3, 4, 5), episodes_per_task=50):
    if group not in GROUPS:
        raise ValueError(group)
    output.mkdir(parents=True, exist_ok=False)
    agent = load_agent(group)
    encoder = OnlineEncoder(group)
    results = []
    for task in tasks:
        for index in range(episodes_per_task):
            directory = output / "episodes" / f"task{task}" / f"episode_{index:03d}"
            results.append(run_episode(group, agent, encoder, task, index, directory))
    successes = sum(x["success"] for x in results)
    report = dict(
        model=group,
        tasks=list(tasks),
        episodes_per_task=episodes_per_task,
        episodes=len(results),
        successes=successes,
        success_rate=successes / len(results),
        max_episode_steps=group_contract(group)["max_episode_steps"],
        temperature=0.0,
    )
    atomic_json(output / "report.json", report)
    return report
