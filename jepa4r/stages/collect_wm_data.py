"""Stage ``collect_wm_data`` -- paired image / point-cloud trajectories for world-model training.

Paper: Sec. 4.1 (one depth camera -> 1,200-point xyzrgb clouds), Sec. 5.2 and supp. sec:supp-data
(mixtures, episode counts and lengths -> ``configs/env/*.yaml: wm_data``).

in : cfg.env (simulator, camera, pointcloud, wm_data)
out: data/wm/<env>/ in the jepa4r.data.storage layout. Images and point clouds record the SAME simulator
     rollouts row by row, so states, actions, splits and starts coincide for the 2D and 3D models.
     wm_data.recipe == "mixture"       -> episodes drawn from wm_data.mixture (jepa4r.data.behaviors)
     wm_data.recipe == "ogbench_noisy" -> OGBench's noisy recipe and official scale (Puzzle)
uses: jepa4r.envs.make_env, jepa4r.data.behaviors, jepa4r.data.pointcloud.process, jepa4r.data.storage

Integrated collection code (cortexbench@9331494:lewm_pcd/scripts/data/): one stable-worldmodel ``World.collect`` per
shard, each shard in its own process (``python -m jepa4r.stages.collect_wm_data <shard.json>``), as the original
launchers ran them (OGBench's oracles draw from numpy's global RNG):
  collect_paired  collect_cube_pcd.py   OGBench Cube with the plan oracle (UR5e reconstruction data, RM75-B twin)
  collect_mw      collect_mw.py         MetaWorld three-part plan (backbone / branch / random)
  collect_ogb     collect_ogb.py        OGBench scene (three-part plan, health check) and puzzle (noisy recipe)
``run`` then merges the shards in the original order (merge_h5_shards.py), checks that the 2D and 3D files hold the
same rollouts (check_paired) and flattens the columns (add_eval_columns.py), as run_*_collection.sh /
run_rm75_pipeline.sh did. Shard seeds: collect_cube_pcd_sharded.py's rule (``shard_seeds``). The OGBench-cube branch's
collectors: ``wm_data.collector: c4_cube`` (Cube1-4: its collection protocol, jepa4r.policy.gciql.collect_cube),
``adroit`` / ``rh56`` (Adroit Relocate / the RH56 twin: its Relocate harness, jepa4r.policy.bc.relocate_stage).
"""
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from jepa4r.data.storage import GzipHDF5Writer, PairedWriter, add_eval_columns, check_paired, merge_shards

logger = logging.getLogger(__name__)

# Moved from cortexbench@9331494:lewm_pcd/scripts/data/collect_cube_pcd.py:run (lines 67-103; verbatim except: no @hydra.main, renamed collect_paired, output paths from cfg.out / cfg.out_pix, imports)
def collect_paired(cfg: DictConfig):
    from functools import partial

    from stable_worldmodel.data.formats.hdf5 import HDF5Writer

    import stable_worldmodel as swm
    from jepa4r.data.pointcloud import PointCloudWrapper
    from jepa4r.envs.ogbench.expert_policy import PCDExpertPolicy
    import jepa4r.envs.ogbench  # noqa: F401  registers swm/OGBCubePCD-v0
    import jepa4r.envs.twins.rm75b  # noqa: F401  registers swm/OGBCubeRM75PCD-v0

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

    out, out_pix = Path(cfg.out), Path(cfg.out_pix)
    world.collect(
        # never append to an existing file (a rerun would silently duplicate episodes)
        writer=PairedWriter(GzipHDF5Writer(out_pix, mode='error'), HDF5Writer(out, mode='error')),
        episodes=cfg.num_traj,
        seed=world_seed,
        options=options,
    )

    logger.info(f'🎉🎉🎉 Completed paired data collection -> 3D {out} + 2D {out_pix} 🎉🎉🎉')


# Moved from cortexbench@9331494:lewm_pcd/scripts/data/collect_mw.py:pix_name, run (lines 42-78; verbatim except: no @hydra.main, renamed collect_mw, output paths from cfg.out (/ cfg.out_pix), imports)
def pix_name(pcd_name: str) -> str:
    """The 2D dataset named after the 3D one: ``metaworld/x_pcd.h5`` -> ``metaworld/x.h5``,
    ``.../x_pcd.shard007.h5`` -> ``.../x.shard007.h5``."""
    p = Path(pcd_name)
    if '_pcd' not in p.name:
        raise ValueError(f"cannot name the 2D dataset after {pcd_name!r} (no '_pcd' in it): set pix_dataset_name")
    return str(p.with_name(p.name.replace('_pcd', '', 1)))


def collect_mw(cfg: DictConfig):
    import hydra
    import stable_worldmodel as swm
    import jepa4r.envs.metaworld  # noqa: F401  registers the MetaWorld env ids

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

    out = Path(cfg.out)
    out_pix = Path(cfg.get('out_pix') or pix_name(cfg.out))
    for p in (out, out_pix):
        p.parent.mkdir(parents=True, exist_ok=True)
    world.collect(
        # never append to an existing file (a rerun would silently duplicate episodes)
        writer=PairedWriter(GzipHDF5Writer(out_pix, mode='error'), GzipHDF5Writer(out, mode='error', gzip_cols=())),
        episodes=cfg.num_traj,
        seed=world_seed,
    )
    logger.info(f'Completed paired data collection (world seed {world_seed}) -> 3D {out} + 2D {out_pix}')


# Moved from cortexbench@9331494:lewm_pcd/scripts/data/collect_ogb.py:RETRY_STRIDE, MAX_TRIES, _Episodes (lines 36-53; verbatim)
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




# Moved from cortexbench@9331494:lewm_pcd/scripts/data/collect_ogb.py:run (lines 61-98; verbatim except: no @hydra.main, renamed collect_ogb, output paths from cfg.out (/ cfg.out_pix), imports; pix_name is collect_mw's (the source copies it))
def collect_ogb(cfg: DictConfig):
    import hydra
    import stable_worldmodel as swm
    import jepa4r.envs.ogbench  # noqa: F401  registers swm/OGBenchScene-v0, swm/OGBenchPuzzle*-v0
    from jepa4r.envs.ogbench.ogb_scene_env import healthy

    """Collect this shard's episodes, paired: pixels (2D) and point clouds (3D) of the same rollouts."""
    world = swm.World(
        cfg.env_id,
        **cfg.world,
        pre_wrappers=[hydra.utils.instantiate(cfg.pcd, _partial_=True)],
        **OmegaConf.to_object(cfg.env),
    )  # add_pixels (default): the 2D dataset's 'pixels', the env's render() of the same state
    world.set_policy(hydra.utils.instantiate(cfg.policy))

    out = Path(cfg.out)
    out_pix = Path(cfg.get('out_pix') or pix_name(cfg.out))
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


def shard_seeds(num_traj, shards, seed):
    """Episode counts and World seeds of the shards (collect_cube_pcd_sharded.py)."""
    class args:  # noqa: N801  the launcher's argparse namespace
        pass
    args.num_traj, args.shards, args.seed = num_traj, shards, seed
    # Moved from cortexbench@9331494:lewm_pcd/scripts/data/collect_cube_pcd_sharded.py:main (shard seeds) (lines 74-83; the shard-seed lines of main(), as a function; sys.exit kept)
    base, rem = divmod(args.num_traj, args.shards)
    counts = [base + (1 if i < rem else 0) for i in range(args.shards)]
    rng = np.random.default_rng(args.seed)
    for attempt in range(100):  # each World consumes seeds [s, s + n + num_envs); require disjoint ranges
        seeds = [rng.integers(0, 1_000_000).item() for _ in range(args.shards)]
        spans = sorted((s, s + n + 64) for s, n in zip(seeds, counts))
        if all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1)):
            break
    else:
        sys.exit('could not draw non-overlapping shard seed ranges')
    return counts, seeds


# Moved from cortexbench@9331494:lewm_pcd/scripts/checks/check_equivalence.py (lines 19-46, 'pcd' mode; adapted: the three
# pinned random sources, the env, the point-cloud wrapper arguments, the number of envs / episodes, the World seed and the
# output path are arguments instead of constants; the 'paper' pixel mode is not needed)
def collect_pinned(cfg: DictConfig):
    """Collection with all three random sources of OGBench Cube fixed (GATES.md D4): the variation space (cube start /
    goal) re-seeded at every reset from one process-wide counter starting at ``variation_seed_start``, each env's RNG
    ``default_rng(env_rng_base + i)``, and numpy's global RNG (the plan oracle's noise) seeded with ``numpy_seed``."""
    import itertools
    from functools import partial

    import stable_worldmodel as swm
    import stable_worldmodel.spaces as swm_spaces
    import jepa4r.envs.ogbench  # noqa: F401  registers swm/OGBCubePCD-v0
    import jepa4r.envs.twins.rm75b  # noqa: F401  registers swm/OGBCubeRM75PCD-v0
    from jepa4r.data.pointcloud import PointCloudWrapper
    from jepa4r.envs.ogbench.expert_policy import PCDExpertPolicy

    pin = cfg.pinned_rng
    np.random.seed(int(pin.numpy_seed))

    # upstream swm CubeEnv.reset() re-seeds its variation space from OS entropy on every
    # reset (cube start/goal position come from there); give it a deterministic sequence
    # instead so both configs draw identical episodes. Test-only patch.
    _orig_reset_vs, _ctr = swm_spaces.reset_variation_space, itertools.count(int(pin.variation_seed_start))
    def _det_reset_vs(variation_space, seed=None, options=None, default_variations=None):
        return _orig_reset_vs(variation_space, seed=next(_ctr), options=options, default_variations=default_variations)
    swm_spaces.reset_variation_space = _det_reset_vs
    common = dict(num_envs=int(cfg.world.num_envs), max_episode_steps=int(cfg.world.max_episode_steps),
                  image_shape=list(cfg.world.image_shape), env_type='single',
                  multiview=False, width=224, height=224, visualize_info=False,
                  terminate_at_goal=False, mode='data_collection')
    world = swm.World(str(cfg.env_id), ob_type='states', add_pixels=False,
                      pre_wrappers=[partial(PointCloudWrapper, **OmegaConf.to_object(cfg.pcd))], **common)
    world.set_policy(PCDExpertPolicy(policy_type='plan_oracle'))
    # upstream swm CubeEnv.reset() drops the `seed` argument, so the env RNG would be
    # entropy-seeded; pin it explicitly (same in both configs) to allow a bit-exact test.
    for i, e in enumerate(world.envs.envs):
        e.unwrapped.np_random = np.random.default_rng(int(pin.env_rng_base) + i)
    world.collect(str(cfg.out), episodes=int(cfg.num_traj), seed=int(cfg.world_seed), format='hdf5', progress=False)


# --------------------------------------------------------------------------- shard plans (the run_*_collection.sh scripts)
COLLECTORS = {'paired': collect_paired, 'mw': collect_mw, 'ogb': collect_ogb, 'pinned': collect_pinned}


def _base(wm):
    """The keys of the original per-shard Hydra config that do not change between shards."""
    keys = ('env_id', 'world', 'pcd', 'policy', 'health_check', 'options')
    cfg = {k: OmegaConf.to_container(wm[k], resolve=True) if OmegaConf.is_config(wm[k]) else wm[k] for k in keys if k in wm}
    kw = wm.get('env_kwargs')
    cfg['env'] = OmegaConf.to_container(kw, resolve=True) if OmegaConf.is_config(kw) else dict(kw or {})
    return cfg


def plan_shards(wm, data_dir):
    """-> (shard specs, merges [(final file, [shard files])], final files). Mirrors run_mw_collection.sh (40 sharded
    train shards + one 150-episode test run), run_scene_collection.sh (per task 6 shards, seeds (k+1)*10^6 + i, merged
    task by task), run_puzzle_collection.sh (8 train shards at 10^6 + i, 2 test shards at 2*10^6 + i) and the paired
    plan-oracle Cube runs (collect_cube_pcd_sharded.py)."""
    data_dir = Path(data_dir)
    shard_dir = data_dir / 'shards'
    collector, base = str(wm.collector), _base(wm)
    specs, merges = [], []
    for part in wm.sets:  # train / test sets of this env, in the order of the original script
        name = str(part.name)
        pcd_final, pix_final = data_dir / f'{name}_pcd.h5', data_dir / f'{name}.h5'
        pcd_parts, pix_parts = [], []
        if collector in ('paired', 'mw') and part.get('world_seed') is not None:  # one run with a fixed World seed
            cfg = dict(base, num_traj=int(part.num_episodes), world_seed=int(part.world_seed), seed=int(wm.seed),
                       out=str(pcd_final), out_pix=str(pix_final))
            specs.append((collector, cfg))
            continue
        if collector in ('paired', 'mw'):
            set_seed = int(part.get('seed', wm.seed))  # the launcher's --seed (a test set may have its own)
            counts, seeds = shard_seeds(int(part.num_episodes), int(part.shards), set_seed)
            for i, (sd, n) in enumerate(zip(seeds, counts)):
                pcd_s, pix_s = shard_dir / f'{name}_pcd.shard{i:03d}.h5', shard_dir / f'{name}.shard{i:03d}.h5'
                specs.append((collector, dict(base, num_traj=n, world_seed=sd, seed=set_seed, out=str(pcd_s), out_pix=str(pix_s))))
                pcd_parts.append(pcd_s)
                pix_parts.append(pix_s)
        else:  # ogb: explicit episode seeds, shard s collects i = s, s + K, ...
            tasks = list(part.get('tasks') or [None])
            for k, task in enumerate(tasks):
                seed_base = int(part.seed_base) if task is None else (k + 1) * int(part.seed_base_step)
                for s in range(int(part.shards)):
                    tag = f'{name}_pcd.{task}.shard{s:03d}.h5' if task else f'{name}_pcd.shard{s:03d}.h5'
                    pcd_s = shard_dir / tag
                    pix_s = shard_dir / tag.replace('_pcd', '', 1)
                    env = dict(base['env'], task=task) if task else base['env']
                    specs.append(('ogb', dict(base, env=env, seed_base=seed_base, episodes=int(part.num_episodes), shard=s,
                                              num_shards=int(part.shards), out=str(pcd_s), out_pix=str(pix_s))))
                    pcd_parts.append(pcd_s)
                    pix_parts.append(pix_s)
        merges += [(pcd_final, pcd_parts), (pix_final, pix_parts)]  # the same shard order for both
    return specs, merges


def run_shards(specs, max_parallel, threads, log_dir):
    """One process per shard (the original launchers: subprocess / background jobs), max_parallel at a time."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    py = os.environ.get('JEPA4R_PY_COLLECT', sys.executable)  # the collection interpreter (MuJoCo 3 / swm stack)
    procs, failed = [], []
    for i, (collector, cfg) in enumerate(specs):
        while sum(p.poll() is None for p, _ in procs) >= max_parallel:
            time.sleep(5)
        spec = log_dir / f'shard_{i:03d}.json'
        Path(cfg['out']).parent.mkdir(parents=True, exist_ok=True)
        spec.write_text(json.dumps({'collector': collector, 'cfg': cfg}, indent=1))
        log = open(log_dir / f'shard_{i:03d}.log', 'w')
        procs.append((subprocess.Popen([py, '-m', 'jepa4r.stages.collect_wm_data', str(spec)], env=env, stdout=log,
                                       stderr=subprocess.STDOUT), log))
    for i, (p, log) in enumerate(procs):
        if p.wait() != 0:
            failed.append(i)
        log.close()
    if failed:
        raise RuntimeError(f'collection shards failed: {failed} (logs in {log_dir})')


def run(cfg, out_dir):
    wm = cfg.env.wm_data
    if wm.collector in ('adroit', 'rh56'):  # the OGBench-cube branch's Relocate harness: collect + data audit / publish
        from jepa4r.policy.bc import relocate_stage
        return {'harness_status': relocate_stage(cfg, 'collect_wm_data', out_dir)['status']}
    if wm.collector == 'c4_cube':
        from jepa4r.policy.gciql import collect_cube
        return collect_cube(cfg, out_dir)
    out_dir = Path(out_dir)
    specs, merges = plan_shards(wm, out_dir)
    finals = [f for f, _ in merges]
    for f in finals:
        if f.exists():
            raise SystemExit(f'refusing to run: {f} exists (HDF5 output must be fresh)')
    run_shards(specs, int(wm.get('max_parallel', len(specs))), int(wm.get('threads', 2)), out_dir / 'logs')
    for final, parts in merges:
        if parts:
            merge_shards(final, parts, delete=True)
    if (out_dir / 'shards').is_dir():
        (out_dir / 'shards').rmdir()
    for part in wm.sets:
        check_paired(out_dir / f'{part.name}_pcd.h5', out_dir / f'{part.name}.h5')
    for f in finals:
        add_eval_columns(f)
    return {'files': [f.name for f in finals]}


if __name__ == '__main__':  # one shard: python -m jepa4r.stages.collect_wm_data <shard.json>
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
    spec = json.loads(Path(sys.argv[1]).read_text())
    COLLECTORS[spec['collector']](OmegaConf.create(spec['cfg']))
