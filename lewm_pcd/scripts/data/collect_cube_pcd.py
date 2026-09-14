"""Point-cloud counterpart of stable-worldmodel's ``scripts/data/collect_cube.py``.

Same env, same expert (plan oracle), same seeds / episode length / episode count
as the paper's ``ogbench/cube_single_expert`` collection. The only changes:
  * the env is ``swm/OGBCubePCD-v0``: OGBCube-v0 plus 4 extra (non-physical) cameras;
  * ``PointCloudWrapper`` is inserted as a pre-wrapper and adds ``pcd_xyz`` /
    ``pcd_rgb`` to every step's info (-> two extra dataset columns);
  * no 224x224 images are rendered or stored (``add_pixels=False``,
    ``ob_type='states'``), which is what keeps the dataset at ~18 KB / frame.
"""

import logging
import os
from functools import partial
from pathlib import Path

os.environ['MUJOCO_GL'] = 'egl'
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.data.formats.hdf5 import HDF5Writer
from envs import PCDExpertPolicy, PointCloudWrapper


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./config', config_name='ogb_pcd')
def run(cfg: DictConfig):
    """Run data collection with point-cloud observations."""

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
        add_pixels=False,
        pre_wrappers=[partial(PointCloudWrapper, **OmegaConf.to_object(cfg.pcd))],
    )

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None
    rng = np.random.default_rng(cfg.seed)
    world_seed = rng.integers(0, 1_000_000).item()
    if cfg.get('world_seed') is not None:  # set by the sharded launcher
        world_seed = int(cfg.world_seed)
    world.set_policy(PCDExpertPolicy(policy_type='plan_oracle'))

    out = (
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / cfg.dataset_name
    )
    world.collect(
        writer=HDF5Writer(out, mode='error'),  # never append to an existing file (a rerun would silently duplicate episodes)
        episodes=cfg.num_traj,
        seed=world_seed,
        options=options,
    )

    logger.info(f'🎉🎉🎉 Completed point-cloud data collection -> {out} 🎉🎉🎉')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(name)s | %(message)s',
    )
    run()
