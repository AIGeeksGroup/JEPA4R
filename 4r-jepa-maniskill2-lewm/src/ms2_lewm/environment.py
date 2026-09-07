"""ManiSkill2 observation/state bridge for the untouched stable World runner.

The RL observation wrapper remains upstream code.  This adapter restores its
Gymnasium reset tuple and exposes flat arrays for the official dataset writer.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess

import gymnasium as gym
import numpy as np


ENV_ID = "ms2_lewm/PickCubePointCloud-v0"
TASK_ROOT = Path(__file__).resolve().parents[2]


def _pack_info_scalars(info):
    """Preserve boolean values in the pinned upstream pool's numeric columns.

    NumPy bool_ is not an np.number. The official EnvPool otherwise stacks
    these values as Python lists, which World.collect intentionally skips.
    Convert only this scalar type; arrays and all underlying values are kept.
    """
    return {key: bool(value) if isinstance(value, np.bool_) else value
            for key, value in info.items()}


def official_rl_env_config(horizon: int = 200) -> dict:
    """Read the same PointNet PPO config; resolve explicit baseline options."""
    from maniskill2_learn.utils.meta import Config

    cfg = Config.fromfile(
        str(TASK_ROOT / "vendor/ManiSkill2-Learn/configs/mfrl/ppo/maniskill2_pn.py")
    )
    env_cfg = dict(cfg.env_cfg)
    env_cfg.pop("type", None)
    env_cfg.update(
        env_name="PickCube-v0",
        obs_mode="pointcloud",
        control_mode="pd_ee_delta_pose",
        obs_frame="world",
        n_points=1200,
        n_goal_points=-1,
        horizon=horizon,
        # render_rgb_array uses only Color. Do not copy the unused 512x512
        # float4 Position buffer to the CPU; point-cloud cameras keep both.
        render_camera_cfgs={"render_camera": {"texture_names": ("Color",)}},
    )
    return env_cfg


class ManiSkillPointCloudEnv(gym.Env):
    """Expose official RL xyz/rgb points and replayable simulator state."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode="rgb_array",
        horizon=200,
        terminate_at_goal=False,
        env_cfg=None,
        episode_callback=None,
    ):
        from maniskill2_learn.env.env_utils import make_gym_env

        self.render_mode = render_mode
        self.terminate_at_goal = terminate_at_goal
        self._env_type = "single"
        # Original OGBench retreat-sampling box, translated to the ManiSkill
        # cube workspace center; only used by the unchanged oracle reset.
        self._arm_sampling_bounds = np.asarray([[0.25, -0.35, 0.20], [0.6, 0.35, 0.35]]) - np.array([0.425, 0.0, 0.0])
        cfg = official_rl_env_config(horizon)
        if env_cfg is not None:
            cfg.update(env_cfg)
        cfg.pop("type", None)
        # Termination is managed as in official OGB data/evaluation configs.
        cfg["ignore_dones"] = not terminate_at_goal
        cfg["horizon"] = horizon
        self.rl_env = make_gym_env(**cfg)
        self.action_space = self.rl_env.action_space
        self.observation_space = gym.spaces.Dict(
            pointcloud=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(cfg["n_points"], 6), dtype=np.float32
            )
        )
        self._last_obs = None
        self._episode_seed = 0
        self._episode_callback = episode_callback
        self._episode_success = False
        self._steps = 0

    @property
    def simulator(self):
        return self.rl_env.unwrapped

    def _observations(self, observation):
        xyz = np.asarray(observation["xyz"], dtype=np.float32)
        rgb = np.asarray(observation["rgb"], dtype=np.float32)
        if xyz.shape != (1200, 3) or rgb.shape != xyz.shape:
            raise ValueError(f"Expected unchanged RL 1200 xyz/rgb points, got {xyz.shape}, {rgb.shape}")
        self._last_obs = observation
        obs = {"pointcloud": np.concatenate([xyz, rgb], axis=-1)}
        obs["observation"] = np.asarray(observation["state"], dtype=np.float32)
        obs["proprio"] = obs["observation"]
        # Keep all original numeric policy inputs without renormalization.
        for name, value in observation.items():
            value = np.asarray(value)
            if value.dtype.kind not in "OUSV":
                obs[f"rl_{name}"] = value
        return obs

    def _state_info(self):
        env = self.simulator
        cube = np.asarray(env.obj.pose.p)
        tcp = np.asarray(env.tcp.pose.p)
        origin = getattr(self, "_initial_cube_position", cube)
        return {
            "sim_state": np.asarray(env.get_state()).copy(),
            "cube_pos": cube.copy(),
            "seed": np.int64(self._episode_seed),
            # Observable operation coverage; none of these extra columns
            # enters the visual encoder or changes the native success rule.
            "object_grasped": bool(env.agent.check_grasp(env.obj)),
            "object_height_above_table": np.float32(cube[2] - env.cube_half_size[2]),
            "tcp_object_distance": np.float32(np.linalg.norm(tcp - cube)),
            "object_goal_distance": np.float32(np.linalg.norm(cube - env.goal_pos)),
            "object_displacement": np.float32(np.linalg.norm(cube - origin)),
        }

    def oracle_info(self):
        """Provide the original OGB oracle schema at the policy boundary."""
        env = self.simulator
        tcp_matrix = env.tcp.pose.to_transformation_matrix()
        cube_matrix = env.obj.pose.to_transformation_matrix()
        tcp_yaw = np.arctan2(tcp_matrix[1, 0], tcp_matrix[0, 0])
        cube_yaw = np.arctan2(cube_matrix[1, 0], cube_matrix[0, 0])
        closedness = self.gripper_closedness()
        return {
            "privileged/target_task": "cube",
            "privileged/target_block": np.int64(0),
            "privileged/block_0_pos": np.asarray(env.obj.pose.p).copy(),
            "privileged/block_0_yaw": np.asarray([cube_yaw]),
            "privileged/target_block_pos": np.asarray(env.goal_pos).copy(),
            "privileged/target_block_yaw": np.asarray([cube_yaw]),
            "proprio/effector_pos": np.asarray(env.tcp.pose.p).copy(),
            "proprio/effector_yaw": np.asarray([tcp_yaw]),
            "proprio/gripper_opening": np.asarray([closedness]),
            # OGB's dimensionless contact thresholds classify a secure grasp.
            # Panda supplies that semantic signal natively; its pad-force
            # scale is different from the original Robotiq's force sensor.
            "proprio/gripper_contact": np.asarray([float(env.agent.check_grasp(env.obj))]),
        }

    def gripper_closedness(self):
        """Use the same native physical controller bounds in both directions."""
        simulator = self.simulator
        bounds = simulator.agent.controller.controllers["gripper"]._action_space
        lower, upper = float(bounds.low[0]), float(bounds.high[0])
        qpos = float(np.mean(simulator.agent.robot.get_qpos()[-2:]))
        return float(np.clip((upper - qpos) / (upper - lower), 0.0, 1.0))

    def action_adapter_state(self):
        """Read-only controller snapshot, also usable across process workers."""
        simulator = self.simulator
        bounds = simulator.agent.controller.controllers["gripper"]._action_space
        return {
            "rotation": simulator.tcp.pose.to_transformation_matrix()[:3, :3].copy(),
            "qpos": float(np.mean(simulator.agent.robot.get_qpos()[-2:])),
            "lower": float(bounds.low[0]),
            "upper": float(bounds.high[0]),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = int(seed)
        self._episode_success = False
        self._steps = 0
        # ExtendedEnv.reset returns observations only; preserve upstream
        # conversion rather than bypassing the original wrapper.
        obs = self.rl_env.reset(seed=seed, options=options)
        self._initial_cube_position = np.asarray(self.simulator.obj.pose.p).copy()
        info = dict(self.simulator.evaluate())
        info.update(self._state_info())
        return self._observations(obs), _pack_info_scalars(info)

    def step(self, action):
        obs, reward, term, trunc, info = self.rl_env.step(action)
        info = dict(info)
        self._steps += 1
        self._episode_success |= bool(info.get("success", False))
        # stable EverythingToInfoWrapper adds the canonical transition fields
        # from the return tuple; ManiSkill already duplicates reward in info.
        for key in ("reward", "terminated", "truncated", "action", "step_idx", "id"):
            info.pop(key, None)
        info.update(self._state_info())
        if (term or trunc) and self._episode_callback is not None:
            self._episode_callback({
                "seed": self._episode_seed,
                "success": self._episode_success,
                "steps": self._steps,
            })
        return self._observations(obs), float(reward), bool(term), bool(trunc), _pack_info_scalars(info)

    def set_sim_state(self, state):
        """Dataset-evaluation callable; state is the full ManiSkill snapshot."""
        self.simulator.set_state(np.asarray(state))

    def set_goal_cube_position(self, target_pos):
        """Dataset-evaluation callable using PickCube's native success goal."""
        from sapien.core import Pose

        env = self.simulator
        env.goal_pos = np.asarray(target_pos).copy()
        if hasattr(env, "goal_site"):
            env.goal_site.set_pose(Pose(env.goal_pos))

    def set_new_target(self, p_stack=0.0):
        """Resample within PickCube's reset workspace, even while holding a cube.

        Native _initialize_task is a reset hook: its vertical origin is the
        cube's current position, which is on the table during reset. Calling
        it mid-episode would add another 0--0.5 m above an already lifted cube.
        Preserve its XY range, height range, RNG and separation rejection,
        but anchor height to the cube center on the table (z=half-size).
        """
        env = self.simulator
        obj_pos = np.asarray(env.obj.pose.p)
        table_cube_z = float(env.cube_half_size[2])
        for _ in range(100):
            goal_xy = env._episode_rng.uniform(-0.1, 0.1, [2])
            goal_z = env._episode_rng.uniform(0, 0.5) + table_cube_z
            goal_pos = np.hstack([goal_xy, goal_z])
            if np.linalg.norm(goal_pos - obj_pos) > env.min_goal_dist:
                break
        self.set_goal_cube_position(goal_pos)
        return self._last_obs, self.oracle_info()

    def render(self):
        return self.rl_env.render()

    def close(self):
        self.rl_env.close()


def register_environment():
    if ENV_ID not in gym.registry:
        gym.register(
            ENV_ID,
            entry_point="ms2_lewm.environment:ManiSkillPointCloudEnv",
            disable_env_checker=True,
            order_enforce=False,
        )


def ensure_renderer_headroom(num_envs=1, extra_model_mib=0):
    """Fail closed before GPU renderer creation while another task is busy."""
    icd = os.environ.get("VK_ICD_FILENAMES", "")
    if "lvp_icd" in icd and os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return {"mode": "software_only", "icd": icd}
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    devices = []
    for line in result.stdout.strip().splitlines():
        index, utilization, used, free = [int(value.strip()) for value in line.split(",")]
        devices.append({"index": index, "utilization_percent": utilization, "used_mib": used, "free_mib": free})
    required = max(2048, 1024 + 256 * num_envs) + extra_model_mib
    report = {"mode": "gpu", "devices": devices, "required_free_mib": required,
              "passed": bool(devices) and all(device["free_mib"] >= required for device in devices)}
    path = TASK_ROOT / "artifacts/last_renderer_resource_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError(f"GPU renderer deferred to protect existing tasks: need {required} MiB free; snapshot {path}")
    return report


def _make_worker_env(max_episode_steps, terminate_at_goal):
    """Spawn-safe factory using the exact official wrapper construction."""
    from functools import partial
    from stable_worldmodel.world.world import _make_env
    from stable_worldmodel.wrapper import MegaWrapper

    register_environment()
    wrappers = [partial(MegaWrapper, image_shape=(224, 224), pixels_transform=None,
                        goal_transform=None, separate_goal=True, image_resample=None)]
    return _make_env(ENV_ID, max_episode_steps, wrappers, horizon=max_episode_steps,
                     terminate_at_goal=terminate_at_goal)


def make_world(*, num_envs, max_episode_steps, terminate_at_goal=False,
               episode_callback=None, env_backend="serial"):
    # The collector explicitly selects its default process backend. Keep
    # existing evaluation/preflight callers on their original serial pool.
    import stable_worldmodel as swm

    if env_backend not in {"process", "serial"}:
        raise ValueError("env_backend must be process or serial")
    ensure_renderer_headroom(num_envs)
    register_environment()
    if env_backend == "process" and num_envs > 1:
        if episode_callback is not None:
            raise ValueError("Episode callback preflight uses one serial environment")
        from functools import partial
        from ms2_lewm.process_env_pool import ProcessWorld

        factory = partial(_make_worker_env, max_episode_steps, terminate_at_goal)
        return ProcessWorld([factory] * num_envs)
    return swm.World(
        env_name=ENV_ID,
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
        image_shape=(224, 224),
        horizon=max_episode_steps,
        terminate_at_goal=terminate_at_goal,
        episode_callback=episode_callback,
    )


def dataset_eval_callables():
    return [
        {"method": "set_sim_state", "args": {"state": {"value": "sim_state"}}},
        {
            "method": "set_goal_cube_position",
            "args": {"target_pos": {"value": "goal_cube_pos"}},
        },
    ]
