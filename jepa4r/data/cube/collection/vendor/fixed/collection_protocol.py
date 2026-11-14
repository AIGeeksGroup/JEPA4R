"""Small protocol helpers for fixed-task, paired OGBench cube collection."""

from __future__ import annotations


import numpy as np

STRATEGY_EXPERT_LOW_NOISE = "expert_low_noise"
STRATEGY_EXPERT_HIGH_NOISE = "expert_high_noise"
STRATEGY_EXPERT_RANDOM = "expert_random"
STRATEGY_EXPERT_RANDOM_BRANCH100 = "expert_random_branch100"
STRATEGY_EXPERT_RANDOM_BRANCH45 = "expert_random_branch45"
STRATEGY_RANDOM = "random"

STRATEGY_ALIASES = {
    STRATEGY_EXPERT_RANDOM_BRANCH100: STRATEGY_EXPERT_RANDOM,
    STRATEGY_EXPERT_RANDOM_BRANCH45: STRATEGY_EXPERT_RANDOM,
}

STRATEGY_WEIGHTS = {
    STRATEGY_EXPERT_LOW_NOISE: 2,
    STRATEGY_EXPERT_HIGH_NOISE: 4,
    STRATEGY_EXPERT_RANDOM: 2,
    STRATEGY_RANDOM: 2,
}
STRATEGY_NOISE = {
    STRATEGY_EXPERT_LOW_NOISE: 0.05,
    STRATEGY_EXPERT_HIGH_NOISE: 0.2,
    STRATEGY_EXPERT_RANDOM: 0.05,
    STRATEGY_EXPERT_RANDOM_BRANCH100: 0.05,
    STRATEGY_EXPERT_RANDOM_BRANCH45: 0.05,
    STRATEGY_RANDOM: 0.0,
}
MIXED_SWITCH_STEPS = {
    STRATEGY_EXPERT_RANDOM: 100,
    STRATEGY_EXPERT_RANDOM_BRANCH100: 100,
    STRATEGY_EXPERT_RANDOM_BRANCH45: 45,
}
SUPPORTED_STRATEGIES = frozenset((*STRATEGY_WEIGHTS, *STRATEGY_ALIASES))
GOAL_ATOL = 1e-7  # meters, accommodates the float32 info representation only


def mixed_switch_step_for_strategy(strategy: str, mixed_switch_step=None) -> int:
    """Resolve the zero-based expert-to-random boundary for a collection strategy."""
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"unknown collection strategy: {strategy}")
    if mixed_switch_step is None:
        return MIXED_SWITCH_STEPS.get(strategy, 100)
    if isinstance(mixed_switch_step, bool) or not isinstance(mixed_switch_step, int):
        raise TypeError("mixed_switch_step must be an integer")
    if mixed_switch_step <= 0:
        raise ValueError("mixed_switch_step must be positive")
    return mixed_switch_step


def strategy_counts(num_episodes: int) -> dict[str, int]:
    """Return exact per-strategy episode quotas for the requested 2:4:2:2 mix."""
    if isinstance(num_episodes, bool) or not isinstance(num_episodes, int):
        raise TypeError("num_episodes must be an integer")
    total_weight = sum(STRATEGY_WEIGHTS.values())
    if num_episodes <= 0 or num_episodes % total_weight:
        raise ValueError(f"num_episodes must be a positive multiple of {total_weight}")
    return {strategy: num_episodes * weight // total_weight for strategy, weight in STRATEGY_WEIGHTS.items()}


def action_source(strategy: str, step_idx: int, mixed_switch_step=None) -> str:
    """Select expert or random control for one zero-based episode step."""
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"unknown collection strategy: {strategy}")
    if step_idx < 0:
        raise ValueError("step_idx must be non-negative")
    mixed_switch_step = mixed_switch_step_for_strategy(strategy, mixed_switch_step)
    if strategy == STRATEGY_RANDOM:
        return "random"
    if strategy in (STRATEGY_EXPERT_RANDOM, *STRATEGY_ALIASES) and step_idx >= mixed_switch_step:
        return "random"
    return "expert"


def reset_options_for_task(task_info: dict) -> dict:
    """Fix single-cube task positions while resampling start yaw per reset."""
    init_xyzs = np.asarray(task_info["init_xyzs"], dtype=np.float64)
    goal_xyzs = np.asarray(task_info["goal_xyzs"], dtype=np.float64)
    if init_xyzs.shape != (1, 3) or goal_xyzs.shape != (1, 3):
        raise ValueError("fixed cube task_info must contain one 3D start and goal")
    return {
        "variation": ["cube.start_yaw"],
        "variation_values": {
            "cube.start_position": init_xyzs[:, :2],
            "cube.goal_position": goal_xyzs[:, :2],
            "cube.goal_yaw": np.zeros(1, dtype=np.float64),
        },
    }


def fixed_task_info(env, task_id: int) -> dict:
    """The fixed task's definition; also in data_collection mode, which skips set_tasks."""
    if not hasattr(env, "task_infos"):
        env.set_tasks()
    return env.task_infos[task_id - 1]


def seeded_fixed_task_reset(
    env,
    *,
    seed,
    options,
    reset_variation_space,
    base_reset,
    default_variations,
):
    """CubeEnv reset for fixed-task collection; the seed goes to both the variation space and the base reset."""
    if seed is None:
        raise ValueError("fixed-task collection requires an explicit reset seed")
    options = options or {}
    if options.get("state") is not None:
        raise ValueError("state overrides are unsupported for fixed-task collection")
    reset_variation_space(
        env.variation_space,
        seed=int(seed),
        options=options,
        default_variations=default_variations,
    )
    observation, info = base_reset(env, seed=int(seed), options=options)
    info["collection_reset_seed"] = int(seed)
    return observation, info


def validate_fixed_paired_schema(
    episode: dict,
    *,
    expected_action_dim: int,
    expected_rgb_shape: tuple[int, int, int],
    expected_pcd_shape: tuple[int, int],
) -> None:
    """Validate the aligned RGB, point-cloud and action arrays of a fixed-task paired episode."""
    required = ("action", "pixels", "pcd_xyz", "pcd_rgb")
    missing = [key for key in required if key not in episode]
    if missing:
        raise ValueError(f"fixed paired episode missing modality: {missing[0]}")

    frames = len(episode["step_idx"])
    action = np.asarray(episode["action"])
    if action.shape != (frames, expected_action_dim):
        raise ValueError(f"action shape must be ({frames}, {expected_action_dim}), got {action.shape}")

    pixels = np.asarray(episode["pixels"])
    expected_pixels = (frames, *expected_rgb_shape)
    if pixels.shape != expected_pixels:
        raise ValueError(f"pixels shape must be {expected_pixels}, got {pixels.shape}")
    if pixels.dtype != np.uint8:
        raise ValueError(f"pixels dtype must be uint8, got {pixels.dtype}")

    xyz = np.asarray(episode["pcd_xyz"])
    expected_pcd = (frames, *expected_pcd_shape)
    if xyz.shape != expected_pcd:
        raise ValueError(f"pcd_xyz shape must be {expected_pcd}, got {xyz.shape}")
    if xyz.dtype != np.float32:
        raise ValueError(f"pcd_xyz dtype must be float32, got {xyz.dtype}")

    rgb = np.asarray(episode["pcd_rgb"])
    if rgb.shape != expected_pcd:
        raise ValueError(f"pcd_rgb shape must be {expected_pcd}, got {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"pcd_rgb dtype must be uint8, got {rgb.dtype}")


class FixedTaskPlanPolicyMixin:
    """Plan-oracle collection policy that preserves the reset target for 200 steps."""

    def __init__(self, strategy: str, mixed_switch_step=None, **kwargs):
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"unknown collection strategy: {strategy}")
        self.strategy = strategy
        self.mixed_switch_step = mixed_switch_step_for_strategy(strategy, mixed_switch_step)
        self.oracle_episode_seeds = {}
        super().__init__(policy_type="plan_oracle", **kwargs)

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        envs = [getattr(env, "unwrapped", env) for env in self.env.envs]
        actions = np.zeros(self.env.action_space.shape, dtype=np.float32)

        for i, env in enumerate(envs):
            action_space = env.action_space
            info = {key: value[i][0] for key, value in info_dict.items() if not key.startswith("_")}
            if info.get("privileged/target_task") != "cube":
                raise ValueError("fixed-task collection requires a cube target")

            if int(info["step_idx"]) == 0:
                episode_seed = int(self.rng.integers(0, 2**32, dtype=np.uint64))
                if "collection_reset_seed" in info:
                    self.oracle_episode_seeds[int(info["collection_reset_seed"])] = episode_seed
                numpy_state = np.random.get_state()
                try:
                    np.random.seed(episode_seed)
                    self._agents[i] = self._oracle_agents["cube"][i]
                    self._agents[i].reset(None, info)
                finally:
                    np.random.set_state(numpy_state)

            if (
                action_source(self.strategy, int(info["step_idx"]), getattr(self, "mixed_switch_step", None))
                == "random"
            ):
                action = self.rng.uniform(action_space.low, action_space.high)
            else:
                action = self._agents[i].select_action(None, info)

            action = np.asarray(action, dtype=np.float32)
            if action.shape != action_space.shape:
                raise ValueError(f"expected action shape {action_space.shape}, got {action.shape}")
            if not np.isfinite(action).all():
                raise ValueError("collection policy produced a non-finite action")
            actions[i] = np.clip(action, action_space.low, action_space.high)

        return actions
