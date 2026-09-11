"""Pixel (paper-layout) collection for the OGBench cube: the settings of stable-worldmodel's
``scripts/data/collect_cube.py`` as used for the paper's ``ogbench/cube_single_expert`` dataset --
``swm/OGBCube-v0``, single env type, front camera 224x224, expert plan oracle, 10 envs x 200 steps,
10000 episodes, seed 3072 -- written as HDF5 in the layout the released ``config/train/data/ogb.yaml`` and
``config/eval/cube.yaml`` read (``pixels`` uint8 (224,224,3), ``observation`` = states, proprio_*, qpos, qvel,
...; run ``scripts/data/add_eval_columns.py`` after merging, as for the point-cloud dataset).

Only difference to the official script: on-disk encoding. ``pixels`` is stored in gzip-compressed 5-frame chunks
(303 GB raw -> ~42 GB, lossless, transparent to every h5py reader, measured 7.3x on real frames) and the other
columns in 20-row chunks (HEAD's 1-row chunks blow up the reader's chunk index). Trajectories are identical to
the point-cloud dataset's (same World seeds; bit-exact check in scripts/checks/check_equivalence.py).
"""

import logging
import os
from pathlib import Path

os.environ['MUJOCO_GL'] = 'egl'

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.data.formats.hdf5 import HDF5Writer, _is_string_col
from stable_worldmodel.envs.ogbench import ExpertPolicy

logger = logging.getLogger(__name__)


class GzipHDF5Writer(HDF5Writer):
    """HDF5Writer whose datasets use multi-row chunks and gzip for the listed columns (schema creation only;
    episode writing, append validation and bookkeeping are inherited unchanged)."""

    def __init__(self, path, *, mode='append', chunk_rows=20, gzip_cols=('pixels',), gzip_chunk_rows=5, level=4):
        super().__init__(path, mode=mode)
        self.chunk_rows, self.gzip_cols, self.gzip_chunk_rows, self.level = chunk_rows, set(gzip_cols), gzip_chunk_rows, level

    def _init_schema(self, sample_ep: dict) -> None:
        for col, vals in sample_ep.items():
            if _is_string_col(vals[0]):
                self._f.create_dataset(col, shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(), chunks=(self.chunk_rows,))
                continue
            sample = np.asarray(vals[0])
            if col in self.gzip_cols:
                kw = dict(chunks=(self.gzip_chunk_rows, *sample.shape), compression='gzip', compression_opts=self.level, shuffle=True)
            else:
                kw = dict(chunks=(self.chunk_rows, *sample.shape))
            self._f.create_dataset(col, shape=(0, *sample.shape), maxshape=(None, *sample.shape), dtype=sample.dtype, **kw)
        self._f.create_dataset('ep_len', shape=(0,), maxshape=(None,), dtype=np.int32)
        self._f.create_dataset('ep_offset', shape=(0,), maxshape=(None,), dtype=np.int64)


@hydra.main(version_base=None, config_path='./config', config_name='ogb_pix')
def run(cfg: DictConfig):
    """Run the official pixel data collection (HDF5 output)."""
    world = swm.World(
        'swm/OGBCube-v0',
        **cfg.world,
        env_type='single',
        ob_type='states',
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=False,
        mode='data_collection',
    )

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None
    rng = np.random.default_rng(cfg.seed)
    world_seed = rng.integers(0, 1_000_000).item()
    if cfg.get('world_seed') is not None:  # set by the sharded launcher
        world_seed = int(cfg.world_seed)
    world.set_policy(ExpertPolicy(policy_type='plan_oracle'))

    out = Path(cfg.cache_dir or swm.data.utils.get_cache_dir()) / 'datasets' / cfg.dataset_name
    world.collect(
        writer=GzipHDF5Writer(out, mode='error'),  # never append to an existing file
        episodes=cfg.num_traj,
        seed=world_seed,
        options=options,
    )
    logger.info(f'🎉🎉🎉 Completed pixel data collection -> {out} 🎉🎉🎉')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
    run()
