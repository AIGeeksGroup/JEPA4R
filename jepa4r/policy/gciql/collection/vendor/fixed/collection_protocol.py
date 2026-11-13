"""Small protocol helpers for fixed-task, paired OGBench cube collection."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

STRATEGY_EXPERT_LOW_NOISE = 'expert_low_noise'
STRATEGY_EXPERT_HIGH_NOISE = 'expert_high_noise'
STRATEGY_EXPERT_RANDOM = 'expert_random'
STRATEGY_EXPERT_RANDOM_BRANCH100 = 'expert_random_branch100'
STRATEGY_EXPERT_RANDOM_BRANCH45 = 'expert_random_branch45'
STRATEGY_RANDOM = 'random'

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
        raise ValueError(f'unknown collection strategy: {strategy}')
    if mixed_switch_step is None:
        return MIXED_SWITCH_STEPS.get(strategy, 100)
    if isinstance(mixed_switch_step, bool) or not isinstance(mixed_switch_step, int):
        raise TypeError('mixed_switch_step must be an integer')
    if mixed_switch_step <= 0:
        raise ValueError('mixed_switch_step must be positive')
    return mixed_switch_step


def strategy_counts(num_episodes: int) -> dict[str, int]:
    """Return exact per-strategy episode quotas for the requested 2:4:2:2 mix."""
    if isinstance(num_episodes, bool) or not isinstance(num_episodes, int):
        raise TypeError('num_episodes must be an integer')
    total_weight = sum(STRATEGY_WEIGHTS.values())
    if num_episodes <= 0 or num_episodes % total_weight:
        raise ValueError(
            f'num_episodes must be a positive multiple of {total_weight}'
        )
    return {
        strategy: num_episodes * weight // total_weight
        for strategy, weight in STRATEGY_WEIGHTS.items()
    }


def action_source(strategy: str, step_idx: int, mixed_switch_step=None) -> str:
    """Select expert or random control for one zero-based episode step."""
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f'unknown collection strategy: {strategy}')
    if step_idx < 0:
        raise ValueError('step_idx must be non-negative')
    mixed_switch_step = mixed_switch_step_for_strategy(strategy, mixed_switch_step)
    if strategy == STRATEGY_RANDOM:
        return 'random'
    if strategy in (STRATEGY_EXPERT_RANDOM, *STRATEGY_ALIASES) and step_idx >= mixed_switch_step:
        return 'random'
    return 'expert'


def reset_options_for_task(task_info: dict) -> dict:
    """Fix single-cube task positions while resampling start yaw per reset."""
    init_xyzs = np.asarray(task_info['init_xyzs'], dtype=np.float64)
    goal_xyzs = np.asarray(task_info['goal_xyzs'], dtype=np.float64)
    if init_xyzs.shape != (1, 3) or goal_xyzs.shape != (1, 3):
        raise ValueError('fixed cube task_info must contain one 3D start and goal')
    return {
        'variation': ['cube.start_yaw'],
        'variation_values': {
            'cube.start_position': init_xyzs[:, :2],
            'cube.goal_position': goal_xyzs[:, :2],
            'cube.goal_yaw': np.zeros(1, dtype=np.float64),
        },
    }


def fixed_task_info(env, task_id: int) -> dict:
    """Read pinned definitions also in data_collection mode, which skips set_tasks."""
    if not hasattr(env, 'task_infos'):
        env.set_tasks()
    return env.task_infos[task_id - 1]


def goal_matches(recorded, expected) -> bool:
    return bool(np.allclose(recorded, expected, rtol=0.0, atol=GOAL_ATOL))


def seeded_fixed_task_reset(
    env, *, seed, options, reset_variation_space, base_reset, default_variations,
):
    """Pinned CubeEnv reset path, with both variation and base seeds forwarded."""
    if seed is None:
        raise ValueError('fixed-task collection requires an explicit reset seed')
    options = options or {}
    if options.get('state') is not None:
        raise ValueError('state overrides are unsupported for fixed-task collection')
    reset_variation_space(
        env.variation_space, seed=int(seed), options=options,
        default_variations=default_variations,
    )
    observation, info = base_reset(env, seed=int(seed), options=options)
    info['collection_reset_seed'] = int(seed)
    return observation, info


def validate_fixed_paired_schema(
    episode: dict,
    *,
    expected_action_dim: int,
    expected_rgb_shape: tuple[int, int, int],
    expected_pcd_shape: tuple[int, int],
) -> None:
    """Validate the fixed pilot's aligned RGB, point-cloud, and action arrays."""
    required = ('action', 'pixels', 'pcd_xyz', 'pcd_rgb')
    missing = [key for key in required if key not in episode]
    if missing:
        raise ValueError(f'fixed paired episode missing modality: {missing[0]}')

    frames = len(episode['step_idx'])
    action = np.asarray(episode['action'])
    if action.shape != (frames, expected_action_dim):
        raise ValueError(
            f'action shape must be ({frames}, {expected_action_dim}), got {action.shape}'
        )

    pixels = np.asarray(episode['pixels'])
    expected_pixels = (frames, *expected_rgb_shape)
    if pixels.shape != expected_pixels:
        raise ValueError(f'pixels shape must be {expected_pixels}, got {pixels.shape}')
    if pixels.dtype != np.uint8:
        raise ValueError(f'pixels dtype must be uint8, got {pixels.dtype}')

    xyz = np.asarray(episode['pcd_xyz'])
    expected_pcd = (frames, *expected_pcd_shape)
    if xyz.shape != expected_pcd:
        raise ValueError(f'pcd_xyz shape must be {expected_pcd}, got {xyz.shape}')
    if xyz.dtype != np.float32:
        raise ValueError(f'pcd_xyz dtype must be float32, got {xyz.dtype}')

    rgb = np.asarray(episode['pcd_rgb'])
    if rgb.shape != expected_pcd:
        raise ValueError(f'pcd_rgb shape must be {expected_pcd}, got {rgb.shape}')
    if rgb.dtype != np.uint8:
        raise ValueError(f'pcd_rgb dtype must be uint8, got {rgb.dtype}')


def episode_record(episode: dict, *, expected_actions: int, expected_num_cubes: int = 2) -> dict:
    """Validate a paired episode before either writer receives it."""
    lengths = {key: len(value) for key, value in episode.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f'paired episode column length mismatch: {lengths}')
    frames = len(episode['step_idx'])
    if frames != expected_actions + 1:
        raise ValueError(f'expected horizon {expected_actions}, got {frames - 1}')
    if not np.array_equal(episode['step_idx'], np.arange(frames)):
        raise ValueError('episode step indices are not contiguous from reset')
    actions = np.asarray(episode['action'])
    if not np.isfinite(actions[:-1]).all() or not np.isnan(actions[-1]).all():
        raise ValueError('actions require finite executed rows and a terminal NaN sentinel')
    for key in ('qpos', 'qvel', 'pcd_xyz', 'pcd_rgb'):
        if key in episode and not np.isfinite(episode[key]).all():
            raise ValueError(f'non-finite {key}')
    seeds = np.asarray(episode['collection_reset_seed'])
    if not np.all(seeds == seeds[0]):
        raise ValueError('reset seed changed within an episode')
    double_task = 'collection_goal_xyzs' in episode
    fixed_keys = ('collection_goal_xyzs', 'collection_init_xyzs') if double_task else (
        'privileged/target_block_pos', 'privileged/target_block_yaw'
    )
    for key in fixed_keys:
        values = np.asarray(episode[key])
        if not np.all(values == values[0]):
            raise ValueError(f'fixed target changed within an episode: {key}')
    initial = np.concatenate([
        np.asarray(episode[key][0], dtype='<f8').reshape(-1)
        for key in ('qpos', 'qvel')
    ])
    record = {
        'frames': frames, 'actions': expected_actions,
        'reset_seed': int(seeds[0]),
        'initial_state_sha256': hashlib.sha256(initial.tobytes()).hexdigest(),
        'success_ever': bool(np.any(episode['success'])),
        'terminated': bool(episode['terminated'][-1]),
        'truncated': bool(episode['truncated'][-1]),
        'termination_reason': 'terminated' if episode['terminated'][-1] else 'horizon',
        'target_position': np.asarray(episode['privileged/target_block_pos'][0]).tolist(),
        'target_yaw': np.asarray(episode['privileged/target_block_yaw'][0]).tolist(),
    }
    if double_task:
        from double_collection import whole_success
        if (isinstance(expected_num_cubes, bool) or not isinstance(expected_num_cubes, int)
                or expected_num_cubes not in (2, 3, 4)):
            raise ValueError('expected_num_cubes must be 2, 3 or 4')
        goals = np.asarray(episode['collection_goal_xyzs'])
        label = {2: 'Double', 3: 'Triple', 4: 'Quadruple'}[expected_num_cubes]
        if goals.shape != (frames, expected_num_cubes, 3) or not np.isfinite(goals).all():
            raise ValueError(f'{label} collection requires full (frames,{expected_num_cubes},3) fixed goals')
        initial_xyzs = np.asarray(episode['collection_init_xyzs'])
        if initial_xyzs.shape != goals.shape or not np.isfinite(initial_xyzs).all():
            raise ValueError(f'{label} collection requires full finite initial XYZ')
        blocks = np.stack([episode[f'privileged/block_{i}_pos'] for i in range(expected_num_cubes)], axis=1)
        computed = np.array([whole_success(xyzs, goal) for xyzs, goal in zip(blocks, goals)])
        if not np.array_equal(computed, np.asarray(episode['success'], dtype=bool)):
            quantifier = {2: 'both', 3: 'all three', 4: 'all four'}[expected_num_cubes]
            raise ValueError(f'{label} success differs from {quantifier} fixed cube goals')
        if not np.array_equal(np.asarray(episode['collection_action_source'])[-1:], [2]):
            raise ValueError(f'{label} terminal row must have terminal control source')
        record.update(
            goal_xyzs=goals[0].tolist(), task_id=int(episode['collection_task_id'][0]),
            success_final=bool(computed[-1]),
            reset_actions=np.asarray(episode['collection_reset_actions'][0]).tolist(),
            last_stage=int(episode['collection_stage'][-1]),
            plan_revisions=int(episode['collection_plan_revision'][-1]),
        )
    return record


class CollectionManifest:
    """Fresh per-file provenance and atomic progress updates for the paired run."""

    def __init__(self, paths, config):
        self.paths = [Path(path) for path in paths]
        existing = [str(path) for path in self.paths if path.exists()]
        if existing:
            raise FileExistsError(f'refusing to overwrite manifests: {existing}')
        self.state = {
            'config': config, 'started_at': datetime.now(timezone.utc).isoformat(),
            'episodes': [], 'completed': 0, 'failed': 0,
        }
        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8') as stream:
                json.dump(self.state, stream)
        self.update('collecting', exit_code=None, error=None)

    def add_episode(self, record):
        if any(
            item['reset_seed'] == record['reset_seed']
            or item['initial_state_sha256'] == record['initial_state_sha256']
            for item in self.state['episodes']
        ):
            raise ValueError('duplicate reset seed or full initial state')
        self.state['episodes'].append(record)
        self.state['completed'] = len(self.state['episodes'])
        self.update('collecting', exit_code=None, error=None)

    def update(self, stage, *, exit_code, error, pix_written=None, pcd_written=None):
        self.state.update(
            stage=stage, updated_at=datetime.now(timezone.utc).isoformat(),
            exit_code=exit_code, error=error, failed=int(stage == 'failed'),
        )
        if pix_written is not None:
            self.state['pix_episodes_written'] = pix_written
            self.state['pcd_episodes_written'] = pcd_written
        for path in self.paths:
            with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=path.parent, prefix=path.name + '.',
                suffix='.tmp', delete=False,
            ) as stream:
                json.dump(self.state, stream, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
                temporary = stream.name
            os.replace(temporary, path)


class FixedTaskPlanPolicyMixin:
    """Plan-oracle collection policy that preserves the reset target for 200 steps."""

    def __init__(self, strategy: str, mixed_switch_step=None, **kwargs):
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f'unknown collection strategy: {strategy}')
        self.strategy = strategy
        self.mixed_switch_step = mixed_switch_step_for_strategy(strategy, mixed_switch_step)
        self.oracle_episode_seeds = {}
        super().__init__(policy_type='plan_oracle', **kwargs)

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        envs = [getattr(env, 'unwrapped', env) for env in self.env.envs]
        actions = np.zeros(self.env.action_space.shape, dtype=np.float32)

        for i, env in enumerate(envs):
            action_space = env.action_space
            info = {
                key: value[i][0]
                for key, value in info_dict.items()
                if not key.startswith('_')
            }
            if info.get('privileged/target_task') != 'cube':
                raise ValueError('fixed-task collection requires a cube target')

            if int(info['step_idx']) == 0:
                episode_seed = int(self.rng.integers(0, 2**32, dtype=np.uint64))
                if 'collection_reset_seed' in info:
                    self.oracle_episode_seeds[int(info['collection_reset_seed'])] = episode_seed
                numpy_state = np.random.get_state()
                try:
                    np.random.seed(episode_seed)
                    self._agents[i] = self._oracle_agents['cube'][i]
                    self._agents[i].reset(None, info)
                finally:
                    np.random.set_state(numpy_state)

            if action_source(
                self.strategy, int(info['step_idx']), getattr(self, 'mixed_switch_step', None)
            ) == 'random':
                action = self.rng.uniform(action_space.low, action_space.high)
            else:
                action = self._agents[i].select_action(None, info)

            action = np.asarray(action, dtype=np.float32)
            if action.shape != action_space.shape:
                raise ValueError(
                    f'expected action shape {action_space.shape}, got {action.shape}'
                )
            if not np.isfinite(action).all():
                raise ValueError('collection policy produced a non-finite action')
            actions[i] = np.clip(action, action_space.low, action_space.high)

        return actions
