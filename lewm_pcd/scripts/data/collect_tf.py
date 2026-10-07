"""Paired 2D + 3D collection of CortexBench's TriFinger tasks (envs/trifinger: CortexBench's own MoveCubeEnv /
CubeReachEnv) with the three-part plan: collect_mw.py with envs.trifinger's env and policy instead of a MetaWorld env
and policy, the rest reused unchanged (stable-worldmodel's World, PairedWriter and GzipHDF5Writer):
  2D  <out without _pcd>  'pixels' = camera60's frame as CortexBench's rollouts see it, resized to 224 x 224
  3D  <out>               'pcd_xyz' / 'pcd_rgb' = camera60's depth, back-projected (envs.trifinger.TriFingerPCDWrapper:
                          1200 points, 5 mm voxels, 25 % on the cube)
every row in both: qpos / qvel (set_state), fingertips, the fingertip plan, cube pose, task (subtask 0 move / 1 reach)
and goal, the plan (sample_type, rand_action, noise_std, branch_t, branch_len), task_success, action.
Episodes: the task's official length (move 10 steps, reach 5). Episode i is reset with world_seed + i; the policy's
noise and random actions come from default_rng(world_seed). In the lewm_tf python (bench/cortexbench/README.md):
    STABLEWM_HOME=... PYTHONPATH=.:third_party/stable-worldmodel $PY scripts/data/collect_tf.py --task move \
        --num-traj 30 --world-seed 0 --out trifinger/tf_small_pcd.h5
"""
import argparse
import logging
from pathlib import Path

import stable_worldmodel as swm
from collect_cube_pcd import PairedWriter  # this script's directory is on sys.path
from collect_cube_pix import GzipHDF5Writer

from envs.trifinger import EPISODE_STEPS, TriFingerPCDWrapper, TriFingerThreePartPolicy  # registers swm/TriFinger-v0

logger = logging.getLogger(__name__)


def pix_name(pcd_name: str) -> str:  # collect_mw.pix_name (collect_mw imports envs.metaworld, which needs metaworld)
    p = Path(pcd_name)
    if '_pcd' not in p.name:
        raise ValueError(f"cannot name the 2D dataset after {pcd_name!r} (no '_pcd' in it)")
    return str(p.with_name(p.name.replace('_pcd', '', 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=('move', 'reach'), required=True)
    ap.add_argument('--num-traj', type=int, required=True)
    ap.add_argument('--world-seed', type=int, required=True)
    ap.add_argument('--out', required=True, help='3D dataset name under $STABLEWM_HOME/datasets (with _pcd)')
    ap.add_argument('--num-envs', type=int, default=5)
    ap.add_argument('--mix', type=float, nargs=3, default=(0.5, 0.3, 0.2))
    args = ap.parse_args()

    world = swm.World(
        'swm/TriFinger-v0',
        num_envs=args.num_envs,
        max_episode_steps=EPISODE_STEPS[args.task],
        image_shape=(224, 224),
        pre_wrappers=[TriFingerPCDWrapper],
        task=args.task,
        mix=tuple(args.mix),
    )  # add_pixels (default): the 2D dataset's 'pixels', the env's camera60 frame of the same row
    world.set_policy(TriFingerThreePartPolicy(seed=args.world_seed))

    datasets = Path(swm.data.utils.get_cache_dir()) / 'datasets'
    out, out_pix = datasets / args.out, datasets / pix_name(args.out)
    for p in (out, out_pix):
        p.parent.mkdir(parents=True, exist_ok=True)
    world.collect(
        writer=PairedWriter(GzipHDF5Writer(out_pix, mode='error'), GzipHDF5Writer(out, mode='error', gzip_cols=())),
        episodes=args.num_traj,
        seed=args.world_seed,
    )
    logger.info(f'Completed paired data collection ({args.task}, world seed {args.world_seed}) -> 3D {out} + 2D {out_pix}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
    main()
