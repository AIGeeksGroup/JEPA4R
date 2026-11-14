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
import os
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import MethodType

os.environ["MUJOCO_GL"] = "egl"
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.data.formats.hdf5 import HDF5Writer
from collect_cube_pix import GzipHDF5Writer  # the 2D dataset's on-disk layout (this script's directory is on sys.path)
from envs import PCDExpertPolicy, PointCloudWrapper
from collection_protocol import (
    STRATEGY_NOISE,
    SUPPORTED_STRATEGIES,
    FixedTaskPlanPolicyMixin,
    fixed_task_info,
    reset_options_for_task,
    seeded_fixed_task_reset,
    mixed_switch_step_for_strategy,
    validate_fixed_paired_schema,
)


logger = logging.getLogger(__name__)

PIX_COLS = ("pixels", "render_time")  # swm's AddPixelsWrapper: 2D dataset only
# prefixes of PointCloudWrapper's keys (pixels/, cam/ only with keep_rgb): 3D dataset only
PCD_COLS = ("pcd_", "pixels/", "cam/")


class FixedTaskPlanPolicy(FixedTaskPlanPolicyMixin, PCDExpertPolicy):
    pass


def _seeded_cube_reset(env, *, seed=None, options=None):
    from stable_worldmodel import spaces as swm_spaces
    from stable_worldmodel.envs.ogbench.cube_env import CubeEnv, DEFAULT_VARIATIONS

    return seeded_fixed_task_reset(
        env,
        seed=seed,
        options=options,
        reset_variation_space=swm_spaces.reset_variation_space,
        base_reset=lambda raw, **kwargs: super(CubeEnv, raw).reset(**kwargs),
        default_variations=DEFAULT_VARIATIONS,
    )


def _with_seeded_cube_reset(env):
    from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

    raw = env.unwrapped
    if not isinstance(raw, CubeEnv) or raw._env_type != "single":
        raise ValueError("fixed-task collection requires the single-cube CubeEnv")
    # Override only this collector's instance; outer TimeLimit/PCD wrappers still reset normally.
    raw.reset = MethodType(_seeded_cube_reset, raw)
    return env


class PairedWriter:
    """``World.collect`` writer that stores every episode in two datasets: all columns but the point cloud in the
    2D one, all columns but the pixels in the 3D one -- the same episodes in the same order, row for row."""

    def __init__(
        self,
        pix_writer,
        pcd_writer,
        *,
        fixed_schema=False,
        expected_action_dim=None,
        expected_rgb_shape=None,
        expected_pcd_shape=None,
    ):
        self.pix, self.pcd = pix_writer, pcd_writer
        self.fixed_schema = fixed_schema
        self.expected_action_dim = expected_action_dim
        self.expected_rgb_shape = expected_rgb_shape
        self.expected_pcd_shape = expected_pcd_shape
        if fixed_schema and (expected_action_dim is None or expected_rgb_shape is None or expected_pcd_shape is None):
            raise ValueError("fixed schema requires expected action, RGB and PCD shapes")
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
            self.pix.write_episode({k: v for k, v in ep.items() if not k.startswith(PCD_COLS)})
            self.pix_episodes_written += 1
            self.pcd.write_episode({k: v for k, v in ep.items() if k not in PIX_COLS})
            self.pcd_episodes_written += 1


@hydra.main(version_base=None, config_path="./config", config_name="ogb_pcd")
def run(cfg: DictConfig):
    """Run the paired data collection: pixels (2D) and point clouds (3D) of the same rollouts."""

    fixed_task_id = cfg.get("fixed_task_id")
    fixed_task = fixed_task_id is not None
    env_type = cfg.get("env_type", "single")
    if env_type not in ("single", "double", "triple", "quadruple"):
        raise ValueError("collection env_type must be single, double, triple, or quadruple")
    double_task = env_type == "double"
    triple_task = env_type == "triple"
    quadruple_task = env_type == "quadruple"
    official_fixed_task = double_task or triple_task or quadruple_task
    if double_task and (not fixed_task or int(cfg.world.max_episode_steps) != 500):
        raise ValueError("Double collection requires a fixed task and 500 actions")
    if (triple_task or quadruple_task) and (not fixed_task or int(cfg.world.max_episode_steps) != 1000):
        raise ValueError("Triple/Quadruple collection requires a fixed task and 1000 actions")
    strategy = cfg.get("strategy", "expert_low_noise")
    noise = cfg.get("noise")
    split = cfg.get("split", "pilot")
    if fixed_task:
        if isinstance(fixed_task_id, bool) or not isinstance(fixed_task_id, int) or not 1 <= fixed_task_id <= 5:
            raise ValueError("fixed_task_id must be an integer from 1 through 5")
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"unknown collection strategy: {strategy}")
        noise = STRATEGY_NOISE[strategy] if noise is None else float(noise)
        if not np.isfinite(noise) or noise < 0 or (strategy == "random" and noise != 0):
            raise ValueError("noise must be finite/non-negative, and zero for random strategy")
        if strategy == "expert_random_branch45" and noise != 0.05:
            raise ValueError("expert_random_branch45 requires noise=0.05")
        mixed_switch_step = mixed_switch_step_for_strategy(strategy, cfg.get("mixed_switch_step"))
        if split not in ("pilot", "train", "validation"):
            raise ValueError("split must be pilot, train, or validation")
        if cfg.get("options") is not None:
            raise ValueError("fixed-task mode constructs its own reset options")
        if list(cfg.pcd.cameras) != ["front_pixels"]:
            raise ValueError("fixed-task paired collection requires only front_pixels")

    if official_fixed_task:
        if quadruple_task:
            from quadruple_collection import (
                QUADRUPLE_ENV_ID as collection_env_id,
                QuadrupleFixedPlanPolicyMixin as CollectionPlanPolicyMixin,
                make_quadruple_wrapper as make_collection_wrapper,
                register_quadruple_environment as register_collection_environment,
            )
        elif triple_task:
            from triple_collection import (
                TRIPLE_ENV_ID as collection_env_id,
                TripleFixedPlanPolicyMixin as CollectionPlanPolicyMixin,
                make_triple_wrapper as make_collection_wrapper,
                register_triple_environment as register_collection_environment,
            )
        else:
            from double_collection import (
                DOUBLE_ENV_ID as collection_env_id,
                DoubleFixedPlanPolicyMixin as CollectionPlanPolicyMixin,
                make_double_wrapper as make_collection_wrapper,
                register_double_environment as register_collection_environment,
            )
        register_collection_environment()

        class OfficialFixedPlanPolicy(CollectionPlanPolicyMixin, FixedTaskPlanPolicy):
            pass

        pre_wrappers = [
            partial(
                make_collection_wrapper,
                task_id=fixed_task_id,
                strategy=strategy,
                mixed_switch_step=mixed_switch_step,
                capture=OmegaConf.to_object(cfg.pcd_capture),
            )
        ]
    else:
        pre_wrappers = [
            *([_with_seeded_cube_reset] if fixed_task else []),
            partial(PointCloudWrapper, **OmegaConf.to_object(cfg.pcd)),
        ]

    world = swm.World(
        collection_env_id if official_fixed_task else cfg.get("env_id", "swm/OGBCubePCD-v0"),
        **cfg.world,
        env_type=env_type,
        ob_type="states",
        **({"permute_blocks": False, "add_pixels": False} if official_fixed_task else {"multiview": False}),
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=False,
        mode="task" if official_fixed_task else "data_collection",
        pre_wrappers=pre_wrappers,
    )  # add_pixels (default): the 2D dataset's 'pixels', rendered by swm after the point cloud, from the same state

    options = cfg.get("options")
    options = OmegaConf.to_object(options) if options is not None else None
    rng = np.random.default_rng(cfg.seed)
    world_seed = rng.integers(0, 1_000_000).item()
    if cfg.get("world_seed") is not None:  # set by the sharded launcher
        world_seed = int(cfg.world_seed)
    task_info = None
    if fixed_task:
        task_info = fixed_task_info(world.envs.envs[0].unwrapped, fixed_task_id)
        options = {"task_id": fixed_task_id} if official_fixed_task else reset_options_for_task(task_info)
        policy = (OfficialFixedPlanPolicy if official_fixed_task else FixedTaskPlanPolicy)(
            strategy=strategy,
            action_noise=noise,
            mixed_switch_step=mixed_switch_step,
            noise_smoothing=cfg.get("noise_smoothing", 0.5),
            seed=int(cfg.seed),
        )
    else:
        policy = PCDExpertPolicy(policy_type="plan_oracle")
    world.set_policy(policy)

    datasets = Path(cfg.cache_dir or swm.data.utils.get_cache_dir()) / "datasets"
    out, out_pix = datasets / cfg.dataset_name, datasets / cfg.pix_dataset_name
    if out == out_pix:
        raise ValueError("paired modalities require distinct output files")
    if fixed_task:
        existing = [str(path) for path in (out, out_pix) if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite datasets: {existing}")

    writer = PairedWriter(
        GzipHDF5Writer(out_pix, mode="error"),
        HDF5Writer(out, mode="error"),
        fixed_schema=fixed_task,
        expected_action_dim=(5 if fixed_task else None),
        expected_rgb_shape=((224, 224, 3) if fixed_task else None),
        expected_pcd_shape=((int(cfg.pcd.get("num_points", 1200)), 3) if fixed_task else None),
    )
    try:
        world.collect(writer=writer, episodes=cfg.num_traj, seed=world_seed, options=options)
    finally:
        world.close()

    logger.info(f"🎉🎉🎉 Completed paired data collection -> 3D {out} + 2D {out_pix} 🎉🎉🎉")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    run()
