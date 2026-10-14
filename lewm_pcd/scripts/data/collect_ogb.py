"""Paired 2D + 3D collection of OGBench's own scene and puzzle-3x3 (envs/ogb_scene_env.py, envs/ogb_puzzle_env.py) for
the LeWM world-model data: collect_cube_pcd.py's design (stable-worldmodel's World, the point-cloud pre-wrapper,
PairedWriter: every episode goes to the 2D and to the 3D dataset, row for row) with the env, point-cloud wrapper and
policy of the config (scripts/data/config/ogb_scene_pcd.yaml, ogb_puzzle_pcd.yaml), plus what the OGBench recipes need:
  * explicit episode seeds: shard ``shard`` of ``num_shards`` collects episodes i = shard, shard + num_shards, ... at
    seed seed_base + i, World.collect once per episode with one env (OGBench's oracles draw from numpy's global RNG,
    which an env pool would share);
  * OGBench's health check (``health_check: true``, scene): an episode whose cube left the view is collected again at
    seed + 10^7 k, k = 1, 2, ... (OGBench's generate_manipspace.py drops and recollects such episodes);
  * no row after the last action: World records the state after the last step too (with a NaN action); the datasets
    hold the state before every action (200 rows per scene episode, 1001 per puzzle episode).
scripts/data/run_scene_collection.sh / run_puzzle_collection.sh run the shards and merge them. One shard (lewm_mw python):
    STABLEWM_HOME=... PYTHONPATH=.:third_party/stable-worldmodel python scripts/data/collect_ogb.py \
        --config-name ogb_scene_pcd env.task=drawer seed_base=3000000 episodes=2500 shard=0 num_shards=6 \
        dataset_name=ogbench/shards/scene_wm10000_pcd.drawer.shard000.h5
"""

import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import envs  # noqa: F401  registers swm/OGBenchScene-v0, swm/OGBenchPuzzle3x3-v0
import stable_worldmodel as swm
from collect_cube_pcd import PairedWriter  # this script's directory is on sys.path
from collect_cube_pix import GzipHDF5Writer
from envs.ogb_scene_env import healthy

logger = logging.getLogger(__name__)
RETRY_STRIDE = 10_000_000
MAX_TRIES = 50


class _Episodes:
    """World.collect writer that keeps the collected episodes in memory."""

    def __enter__(self):
        self.eps = []
        return self

    def __exit__(self, *exc):
        return False

    def write_episodes(self, episodes):
        self.eps.extend(episodes)


def pix_name(pcd_name: str) -> str:  # collect_mw.pix_name (collect_mw imports envs.metaworld, which needs metaworld)
    p = Path(pcd_name)
    if '_pcd' not in p.name:
        raise ValueError(f"cannot name the 2D dataset after {pcd_name!r} (no '_pcd' in it): set pix_dataset_name")
    return str(p.with_name(p.name.replace('_pcd', '', 1)))


@hydra.main(version_base=None, config_path='./config', config_name='ogb_scene_pcd')
def run(cfg: DictConfig):
    """Collect this shard's episodes, paired: pixels (2D) and point clouds (3D) of the same rollouts."""
    world = swm.World(
        cfg.env_id,
        **cfg.world,
        pre_wrappers=[hydra.utils.instantiate(cfg.pcd, _partial_=True)],
        **OmegaConf.to_object(cfg.env),
    )  # add_pixels (default): the 2D dataset's 'pixels', the env's render() of the same state
    world.set_policy(hydra.utils.instantiate(cfg.policy))

    datasets = Path(cfg.cache_dir or swm.data.utils.get_cache_dir()) / 'datasets'
    out = datasets / cfg.dataset_name
    out_pix = datasets / (cfg.pix_dataset_name or pix_name(cfg.dataset_name))
    for p in (out, out_pix):
        p.parent.mkdir(parents=True, exist_ok=True)
    eps = range(cfg.shard, cfg.episodes, cfg.num_shards)
    t0, retries = time.time(), 0
    # never append to an existing file (a rerun would silently duplicate episodes)
    with PairedWriter(GzipHDF5Writer(out_pix, mode='error'), GzipHDF5Writer(out, mode='error', gzip_cols=())) as w:
        for n, i in enumerate(eps):
            for k in range(MAX_TRIES):
                seed = int(cfg.seed_base + i + k * RETRY_STRIDE)
                buf = _Episodes()
                world.collect(writer=buf, episodes=1, seed=seed, progress=False)
                ep = {col: vals[:-1] for col, vals in buf.eps[0].items()}  # the state after the last action is no row
                if not cfg.health_check or healthy(np.asarray(ep['cube_pos'], np.float64)):
                    break
                retries += 1
                logger.info(f'shard {cfg.shard}: episode {i} seed {seed} unhealthy (cube out of view), retrying')
            else:
                sys.exit(f'episode {i}: {MAX_TRIES} unhealthy tries')
            w.write_episodes([ep])
            logger.info(f'shard {cfg.shard}: episode {i} ({n + 1}/{len(eps)}) seed {seed}, '
                        f'{(time.time() - t0) / (n + 1):.1f} s/episode')
    logger.info(f'shard {cfg.shard}: DONE {len(eps)} episodes, {retries} unhealthy retried, {time.time() - t0:.0f} s '
                f'-> 3D {out} + 2D {out_pix}')
    os._exit(0)  # skips the EGL renderers' destructors (a harmless EGLError at interpreter exit)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
    run()
