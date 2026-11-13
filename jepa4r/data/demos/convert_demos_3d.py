# Moved from cortexbench@9331494:lewm_pcd/bench/cortexbench/convert_demos_3d.py (verbatim except: imports / paths
# adapted to the package; lines wrapped at 120 columns)
"""The official MetaWorld expert demos as point clouds: every demo replayed exactly (fresh env, seed 12345 + i, the
recorded actions; README.md, "3D demos"), and at every step the 3D LeWM's point cloud of the frame
(pcd_fork.MWPointClouds, the settings of its training data), instead of the 256 x 256 RGB frame.

Written in the official pickle format, so that CortexBench's train loop reads it unchanged: 'images' holds the point
clouds, (500, 1200, 6) float32 = xyz in metres (world frame) + rgb 0..255; actions, rewards, observations and env_infos
are the official ones, except env_infos['gripper_proprio'] (the policy's proprioception), which is taken from the
replay (identical for the exactly replayed demos, see the printed check) so that it matches the point clouds.
-> $STABLEWM_HOME/datasets/cortexbench/metaworld-expert-v1.0-pcd/<task>-v2-goal-observable.pickle,
   $STABLEWM_HOME/logs/cortexbench/convert_<task>.json

    source env.sh && $PY convert_demos_3d.py <task>
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

from jepa4r.policy.bc.pcd_fork import MWPointClouds, no_multisampling

HOME = Path(os.environ['JEPA4R_DEMOS_HOME'])  # data/policy/<env>
SRC = HOME / 'metaworld-expert-v1.0'
DST = HOME / 'metaworld-expert-v1.0-pcd'
LOG = HOME / 'logs'
BASE_SEED = 12345
task = sys.argv[1]
name = f'{task}-v2-goal-observable'

with open(SRC / f'{name}.pickle', 'rb') as f:
    demos = pickle.load(f)
    os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
t0 = time.time()
stats = []
for i, demo in enumerate(demos):
    del demo['images']
    e = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[name]()
    e._freeze_rand_vec = False  # as CortexBench's env_constructor
    no_multisampling(e.sim)
    e.seed(BASE_SEED + i)
    np.random.seed(BASE_SEED + i)
    obs = [e.reset()]
    pcm = MWPointClouds(e.sim, task)
    pcm.seed(BASE_SEED + i)
    T = len(demo['actions'])
    pcd = np.zeros((T, pcm.num_points, 6), dtype=np.float32)
    for t in range(T):
        # the frame before action t (the first render directly after reset, as the official collection)
        xyz, rgb = pcm()
        pcd[t, :, :3], pcd[t, :, 3:] = xyz, rgb
        o, *_ = e.step(demo['actions'][t])
        obs.append(o)
    e.close()
    obs = np.asarray(obs)
    gp = demo['env_infos']['gripper_proprio']
    err_now, err_next = float(np.abs(gp - obs[:T, :4]).max()), float(np.abs(gp - obs[1:T + 1, :4]).max())
    demo['env_infos']['gripper_proprio'] = (obs[:T, :4] if err_now <= err_next else obs[1:T + 1, :4]).astype(gp.dtype)
    demo['images'] = pcd
    stats.append(dict(demo=i, seed=BASE_SEED + i, proprio_err_same_step=err_now, proprio_err_next_step=err_next,
                      state_err_vs_official=float(np.abs(obs[1:T, :18] - demo['observations'][1:T, :18]).max())))
    print(f'{task} demo {i:2d}: proprio vs official: same step {err_now:.1e}, next step {err_next:.1e}; '
          f'state vs official {stats[-1]["state_err_vs_official"]:.1e}; {time.time() - t0:.0f} s', flush=True)

DST.mkdir(parents=True, exist_ok=True)
tmp = DST / f'.{name}.pickle.tmp'
with open(tmp, 'wb') as f:
    pickle.dump(demos, f, protocol=4)
    f.flush()
    os.fsync(f.fileno())
    os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
os.replace(tmp, DST / f'{name}.pickle')
LOG.mkdir(parents=True, exist_ok=True)
(LOG / f'convert_{task}.json').write_text(
    json.dumps(dict(task=task, minutes=round((time.time() - t0) / 60, 1), demos=stats), indent=1)
)
print(f'{task}: wrote {DST / name}.pickle in {time.time() - t0:.0f} s', flush=True)
