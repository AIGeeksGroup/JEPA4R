"""Paired 2D + 3D counterpart of stable-worldmodel's ``scripts/data/collect_cube.py``: one rollout, two datasets.

Same env, same expert (plan oracle), same seeds / episode length / episode count
as the paper's ``ogbench/cube_single_expert`` collection. The only changes:
  * the env is ``swm/OGBCubePCD-v0``: OGBCube-v0 plus extra (non-physical) cameras
    that are not used by default, i.e. the same dynamics;
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

os.environ['MUJOCO_GL'] = 'egl'
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.data.formats.hdf5 import HDF5Writer
from collect_cube_pix import GzipHDF5Writer  # the 2D dataset's on-disk layout (this script's directory is on sys.path)
from envs import PCDExpertPolicy, PointCloudWrapper


logger = logging.getLogger(__name__)

PIX_COLS = ('pixels', 'render_time')    # swm's AddPixelsWrapper: 2D dataset only
PCD_COLS = ('pcd_', 'pixels/', 'cam/')  # prefixes of PointCloudWrapper's keys (pixels/, cam/ only with keep_rgb): 3D dataset only


class PairedWriter:
    """``World.collect`` writer that stores every episode in two datasets: all columns but the point cloud in the
    2D one, all columns but the pixels in the 3D one -- the same episodes in the same order, row for row."""

    def __init__(self, pix_writer, pcd_writer):
        self.pix, self.pcd = pix_writer, pcd_writer

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
            self.pix.write_episode({k: v for k, v in ep.items() if not k.startswith(PCD_COLS)})
            self.pcd.write_episode({k: v for k, v in ep.items() if k not in PIX_COLS})


@hydra.main(version_base=None, config_path='./config', config_name='ogb_pcd')
def run(cfg: DictConfig):
    """Run the paired data collection: pixels (2D) and point clouds (3D) of the same rollouts."""

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
        pre_wrappers=[partial(PointCloudWrapper, **OmegaConf.to_object(cfg.pcd))],
    )  # add_pixels (default): the 2D dataset's 'pixels', rendered by swm after the point cloud, from the same state

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None
    rng = np.random.default_rng(cfg.seed)
    world_seed = rng.integers(0, 1_000_000).item()
    if cfg.get('world_seed') is not None:  # set by the sharded launcher
        world_seed = int(cfg.world_seed)
    world.set_policy(PCDExpertPolicy(policy_type='plan_oracle'))

    datasets = Path(cfg.cache_dir or swm.data.utils.get_cache_dir()) / 'datasets'
    out, out_pix = datasets / cfg.dataset_name, datasets / cfg.pix_dataset_name
    world.collect(
        # never append to an existing file (a rerun would silently duplicate episodes)
        writer=PairedWriter(GzipHDF5Writer(out_pix, mode='error'), HDF5Writer(out, mode='error')),
        episodes=cfg.num_traj,
        seed=world_seed,
        options=options,
    )

    logger.info(f'🎉🎉🎉 Completed paired data collection -> 3D {out} + 2D {out_pix} 🎉🎉🎉')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(name)s | %(message)s',
    )
    run()
