"""Paired 2D + 3D counterpart of stable-worldmodel's ``scripts/data/collect_cube.py``: one rollout, two datasets.

Same env, same expert (plan oracle), same seeds / episode length / episode count
as the paper's ``ogbench/cube_single_expert`` collection. The only changes:
  * the env is ``swm/OGBCubePCD-v0``: OGBCube-v0 with an official-goal PCD
    capture hook and no auxiliary cameras, i.e. the same dynamics;
  * one eye-to-hand camera, ``front_pixels`` (the official pixel camera), observes
    every step as an RGB-D camera: swm's own ``pixels`` (224x224 RGB) is the 2D
    observation, and ``PointCloudWrapper`` (a pre-wrapper) back-projects the same
    camera's depth into ``pcd_xyz`` / ``pcd_rgb``, the 3D observation (ToF-style
    depth: no stereo, no multi-camera fusion);
  * every episode is written twice: all columns but the point cloud to the 2D
    dataset ``pix_dataset_name`` (the layout of ``collect_cube_pix.py``), all
    columns but ``pixels`` / ``render_time`` to the 3D dataset ``dataset_name``
    (~18 KB / frame, as before). States, actions etc. are those of the same
    episodes, row for row, so the two datasets correspond one to one. Separate
    2D and 3D runs cannot give that: ``CubeEnv.reset`` draws start and goal from
    OS entropy and the oracle's noise comes from the unseeded global ``np.random``.
"""

import logging
import hashlib
import inspect
import os
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import MethodType

os.environ['MUJOCO_GL'] = 'egl'
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.data.formats.hdf5 import HDF5Writer
from collect_cube_pix import GzipHDF5Writer  # the 2D dataset's on-disk layout (this script's directory is on sys.path)
from envs import PCDExpertPolicy, PointCloudWrapper
from collection_protocol import (
    GOAL_ATOL, STRATEGY_NOISE, SUPPORTED_STRATEGIES, CollectionManifest, FixedTaskPlanPolicyMixin,
    episode_record, fixed_task_info, goal_matches, reset_options_for_task, seeded_fixed_task_reset,
    mixed_switch_step_for_strategy, validate_fixed_paired_schema,
)


logger = logging.getLogger(__name__)

PIX_COLS = ('pixels', 'render_time')    # swm's AddPixelsWrapper: 2D dataset only
PCD_COLS = ('pcd_', 'pixels/', 'cam/')  # prefixes of PointCloudWrapper's keys (pixels/, cam/ only with keep_rgb): 3D dataset only
_CUBE_RESET_SHA256 = '9afaa8310271973f60e9625a67ac1ee003b1ab4a1f18dbd87125ad27c5005fa3'


class FixedTaskPlanPolicy(FixedTaskPlanPolicyMixin, PCDExpertPolicy):
    pass


def _seeded_cube_reset(env, *, seed=None, options=None):
    from stable_worldmodel import spaces as swm_spaces
    from stable_worldmodel.envs.ogbench.cube_env import CubeEnv, DEFAULT_VARIATIONS

    return seeded_fixed_task_reset(
        env, seed=seed, options=options,
        reset_variation_space=swm_spaces.reset_variation_space,
        base_reset=lambda raw, **kwargs: super(CubeEnv, raw).reset(**kwargs),
        default_variations=DEFAULT_VARIATIONS,
    )


def _with_seeded_cube_reset(env):
    from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

    raw = env.unwrapped
    if not isinstance(raw, CubeEnv) or raw._env_type != 'single':
        raise ValueError('fixed-task collection requires the single-cube CubeEnv')
    digest = hashlib.sha256(inspect.getsource(CubeEnv.reset).encode()).hexdigest()
    if digest != _CUBE_RESET_SHA256:
        raise RuntimeError(f'upstream CubeEnv.reset changed: {digest}')
    # Override only this collector's instance; outer TimeLimit/PCD wrappers still reset normally.
    raw.reset = MethodType(_seeded_cube_reset, raw)
    return env


class PairedWriter:
    """``World.collect`` writer that stores every episode in two datasets: all columns but the point cloud in the
    2D one, all columns but the pixels in the 3D one -- the same episodes in the same order, row for row."""

    def __init__(
        self, pix_writer, pcd_writer, *, expected_actions=None, on_episode=None,
        fixed_schema=False, expected_action_dim=None, expected_rgb_shape=None,
        expected_pcd_shape=None,
    ):
        self.pix, self.pcd = pix_writer, pcd_writer
        self.expected_actions, self.on_episode = expected_actions, on_episode
        self.fixed_schema = fixed_schema
        self.expected_action_dim = expected_action_dim
        self.expected_rgb_shape = expected_rgb_shape
        self.expected_pcd_shape = expected_pcd_shape
        if fixed_schema and (
            expected_actions is None or expected_action_dim is None
            or expected_rgb_shape is None or expected_pcd_shape is None
        ):
            raise ValueError('fixed schema requires expected action, RGB, and PCD shapes')
        self.pix_episodes_written = self.pcd_episodes_written = 0

    def __enter__(self):
        with ExitStack() as stack:  # if the second file cannot be opened, the first one is closed again
            stack.enter_context(self.pix)
            stack.enter_context(self.pcd)
            self._files = stack.pop_all()
        return self

    def __exit__(self, *exc):
        return self._files.__exit__(*exc)

    def write_episodes(self, episodes):
        for ep in episodes:
            if self.fixed_schema:
                validate_fixed_paired_schema(
                    ep,
                    expected_action_dim=self.expected_action_dim,
                    expected_rgb_shape=self.expected_rgb_shape,
                    expected_pcd_shape=self.expected_pcd_shape,
                )
            record = episode_record(ep, expected_actions=self.expected_actions) if self.expected_actions is not None else None
            self.pix.write_episode({k: v for k, v in ep.items() if not k.startswith(PCD_COLS)})
            self.pix_episodes_written += 1
            self.pcd.write_episode({k: v for k, v in ep.items() if k not in PIX_COLS})
            self.pcd_episodes_written += 1
            if self.on_episode is not None:
                self.on_episode(record)


@hydra.main(version_base=None, config_path='./config', config_name='ogb_pcd')
def run(cfg: DictConfig):
    """Run the paired data collection: pixels (2D) and point clouds (3D) of the same rollouts."""

    fixed_task_id = cfg.get('fixed_task_id')
    fixed_task = fixed_task_id is not None
    strategy = cfg.get('strategy', 'expert_low_noise')
    noise = cfg.get('noise')
    split = cfg.get('split', 'pilot')
    if fixed_task:
        if isinstance(fixed_task_id, bool) or not isinstance(fixed_task_id, int) or not 1 <= fixed_task_id <= 5:
            raise ValueError('fixed_task_id must be an integer from 1 through 5')
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f'unknown collection strategy: {strategy}')
        noise = STRATEGY_NOISE[strategy] if noise is None else float(noise)
        if not np.isfinite(noise) or noise < 0 or (strategy == 'random' and noise != 0):
            raise ValueError('noise must be finite/non-negative, and zero for random strategy')
        if strategy == 'expert_random_branch45' and noise != 0.05:
            raise ValueError('expert_random_branch45 requires noise=0.05')
        mixed_switch_step = mixed_switch_step_for_strategy(
            strategy, cfg.get('mixed_switch_step')
        )
        if split not in ('pilot', 'train', 'validation'):
            raise ValueError('split must be pilot, train, or validation')
        if cfg.get('options') is not None:
            raise ValueError('fixed-task mode constructs its own reset options')
        if list(cfg.pcd.cameras) != ['front_pixels']:
            raise ValueError('fixed-task paired collection requires only front_pixels')

    world = swm.World(
        cfg.get('env_id', 'swm/OGBCubePCD-v0'),  # RM-75 configs set env_id (config/ogb_rm75_pcd.yaml); default = the UR5e env
        **cfg.world,
        env_type='single',
        ob_type='states',
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=False,
        mode='data_collection',
        pre_wrappers=[
            *([_with_seeded_cube_reset] if fixed_task else []),
            partial(PointCloudWrapper, **OmegaConf.to_object(cfg.pcd)),
        ],
    )  # add_pixels (default): the 2D dataset's 'pixels', rendered by swm after the point cloud, from the same state

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None
    rng = np.random.default_rng(cfg.seed)
    world_seed = rng.integers(0, 1_000_000).item()
    if cfg.get('world_seed') is not None:  # set by the sharded launcher
        world_seed = int(cfg.world_seed)
    task_info = None
    if fixed_task:
        task_info = fixed_task_info(world.envs.envs[0].unwrapped, fixed_task_id)
        options = reset_options_for_task(task_info)
        policy = FixedTaskPlanPolicy(
            strategy=strategy, action_noise=noise,
            mixed_switch_step=mixed_switch_step,
            noise_smoothing=cfg.get('noise_smoothing', 0.5), seed=int(cfg.seed),
        )
    else:
        policy = PCDExpertPolicy(policy_type='plan_oracle')
    world.set_policy(policy)

    datasets = Path(cfg.cache_dir or swm.data.utils.get_cache_dir()) / 'datasets'
    out, out_pix = datasets / cfg.dataset_name, datasets / cfg.pix_dataset_name
    if out == out_pix:
        raise ValueError('paired modalities require distinct output files')
    manifest = None
    on_episode = None
    if fixed_task:
        existing = [str(path) for path in (out, out_pix) if path.exists()]
        if existing:
            raise FileExistsError(f'refusing to overwrite datasets: {existing}')
        manifest = CollectionManifest(
            [Path(str(path) + '.manifest.json') for path in (out_pix, out)],
            {
                'task_id': fixed_task_id, 'task_name': task_info['task_name'],
                'nominal_init_xyzs': task_info['init_xyzs'].tolist(),
                'goal_xyzs': task_info['goal_xyzs'].tolist(),
                'start_position_jitter': 0.0, 'start_yaw': 'seeded_uniform_0_2pi',
                'goal_yaw': 0.0, 'strategy': strategy, 'actual_noise': noise,
                'goal_comparison_atol_m': GOAL_ATOL,
                'noise_smoothing': policy.noise_smoothing, 'split': split,
                'policy_seed': int(cfg.seed), 'world_seed': world_seed,
                'mixed_switch_action_index': mixed_switch_step,
                'mixed_switch_step': mixed_switch_step,
                'requested_episodes': int(cfg.num_traj),
                'horizon_actions': int(cfg.world.max_episode_steps),
                'rgb_dataset': str(out_pix), 'pcd_dataset': str(out),
                'action_alignment': 'action[t] drives observation[t] to observation[t+1]; terminal action is NaN',
                'pcd_processing': OmegaConf.to_container(cfg.pcd, resolve=True),
                'segmentation_assisted_foreground_and_cube_quota': True,
                'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'protocol_sha256': hashlib.sha256(Path(__file__).with_name('collection_protocol.py').read_bytes()).hexdigest(),
                'upstream_cube_reset_sha256': _CUBE_RESET_SHA256,
                'imported_swm_path': swm.__file__,
                'resolved_config': OmegaConf.to_container(cfg, resolve=True),
            },
        )

        def on_episode(record):
            record['oracle_reset_seed'] = policy.oracle_episode_seeds[record['reset_seed']]
            if not goal_matches(record['target_position'], task_info['goal_xyzs'][0]):
                raise ValueError('recorded goal differs from the fixed task definition')
            manifest.add_episode(record)

    writer = PairedWriter(
        GzipHDF5Writer(out_pix, mode='error'), HDF5Writer(out, mode='error'),
        expected_actions=int(cfg.world.max_episode_steps) if fixed_task else None,
        on_episode=on_episode,
        fixed_schema=fixed_task,
        expected_action_dim=(5 if fixed_task else None),
        expected_rgb_shape=((224, 224, 3) if fixed_task else None),
        expected_pcd_shape=(
            (int(cfg.pcd.get('num_points', 1200)), 3) if fixed_task else None
        ),
    )
    try:
        world.collect(writer=writer, episodes=cfg.num_traj, seed=world_seed, options=options)
        if manifest is not None:
            if manifest.state['completed'] != int(cfg.num_traj):
                raise RuntimeError('paired completed episode count differs from requested count')
            manifest.update(
                'complete', exit_code=0, error=None,
                pix_written=writer.pix_episodes_written, pcd_written=writer.pcd_episodes_written,
            )
    except BaseException as error:
        if manifest is not None:
            manifest.update(
                'failed', exit_code=1, error=f'{type(error).__name__}: {error}',
                pix_written=writer.pix_episodes_written, pcd_written=writer.pcd_episodes_written,
            )
        raise
    finally:
        world.close()

    logger.info(f'🎉🎉🎉 Completed paired data collection -> 3D {out} + 2D {out_pix} 🎉🎉🎉')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(name)s | %(message)s',
    )
    run()
