"""Paired 2D + 3D collection of a MetaWorld task (envs/metaworld) with the three-part plan: one rollout, two datasets.
The task is the config (scripts/data/config/mw_<task>.yaml): env id, the point-cloud wrapper (``pcd._target_``) and
the collection policy (``policy._target_``).

The paired design of ``collect_cube_pcd.py`` (its ``PairedWriter``: every episode goes to both datasets, row for
row), with the MetaWorld env and its collection policy instead of the cube and its expert:
  2D  ``pix_dataset_name``  ``pixels`` = 224x224 RGB of the reproduced top_cap2 camera (``GzipHDF5Writer``, the
                            layout of ``collect_cube_pix.py``)
  3D  ``dataset_name``      ``pcd_xyz`` / ``pcd_rgb`` = the same camera's depth, back-projected
                            (``envs.metaworld.MWPointCloudWrapper``); ``GzipHDF5Writer`` without gzip, i.e. 20-row
                            chunks already, so no ``rechunk_h5.py`` pass is needed
Every row also has qpos / qvel / mocap (``set_state`` in the planning evaluation), hand / object positions, the
place target and the episode's plan: sample_type (0 backbone, 1 branch, 2 random), rand_action (this row's action is
random), noise_std, branch_t, branch_len.

Seeds: episode i of a run is reset with world_seed + i; the policy's noise and random actions come from
``default_rng(world_seed)``. ``collect_cube_pcd_sharded.py --script collect_mw.py --no-merge`` gives every shard its
own world_seed and names only its 3D file; ``pix_dataset_name: null`` then names the 2D file after it (the 3D name
without ``_pcd``). ``scripts/data/run_mw_collection.sh`` runs the whole collection (train shards, test set, merge).

    STABLEWM_HOME=... PYTHONPATH=. python scripts/data/collect_mw.py [--config-name mw_binpick] num_traj=150 \
        world_seed=2000000 dataset_name=metaworld/mw_binpick_test_pcd.h5
"""

import logging
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import envs.metaworld  # noqa: F401  registers the MetaWorld env ids
import stable_worldmodel as swm
from collect_cube_pcd import PairedWriter  # this script's directory is on sys.path
from collect_cube_pix import GzipHDF5Writer

logger = logging.getLogger(__name__)


def pix_name(pcd_name: str) -> str:
    """The 2D dataset named after the 3D one: ``metaworld/x_pcd.h5`` -> ``metaworld/x.h5``,
    ``.../x_pcd.shard007.h5`` -> ``.../x.shard007.h5``."""
    p = Path(pcd_name)
    if '_pcd' not in p.name:
        raise ValueError(f"cannot name the 2D dataset after {pcd_name!r} (no '_pcd' in it): set pix_dataset_name")
    return str(p.with_name(p.name.replace('_pcd', '', 1)))


@hydra.main(version_base=None, config_path='./config', config_name='mw_binpick')
def run(cfg: DictConfig):
    """Run the paired data collection: pixels (2D) and point clouds (3D) of the same rollouts."""
    rng = np.random.default_rng(cfg.seed)
    world_seed = rng.integers(0, 1_000_000).item()
    if cfg.get('world_seed') is not None:  # set by the sharded launcher
        world_seed = int(cfg.world_seed)

    world = swm.World(
        cfg.env_id,
        **cfg.world,
        pre_wrappers=[hydra.utils.instantiate(cfg.pcd, _partial_=True)],
        **OmegaConf.to_object(cfg.env),  # the env's collection plan (episode length, sample-type mix, ...)
    )  # add_pixels (default): the 2D dataset's 'pixels', rendered by swm after the point cloud, from the same state
    world.set_policy(hydra.utils.instantiate(cfg.policy, seed=world_seed))

    datasets = Path(cfg.cache_dir or swm.data.utils.get_cache_dir()) / 'datasets'
    out = datasets / cfg.dataset_name
    out_pix = datasets / (cfg.pix_dataset_name or pix_name(cfg.dataset_name))
    for p in (out, out_pix):
        p.parent.mkdir(parents=True, exist_ok=True)
    world.collect(
        # never append to an existing file (a rerun would silently duplicate episodes)
        writer=PairedWriter(GzipHDF5Writer(out_pix, mode='error'), GzipHDF5Writer(out, mode='error', gzip_cols=())),
        episodes=cfg.num_traj,
        seed=world_seed,
    )
    logger.info(f'Completed paired data collection (world seed {world_seed}) -> 3D {out} + 2D {out_pix}')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(name)s | %(message)s',
    )
    run()
