"""RM75 CortexBench demos for a task, made the way CortexBench's official ones are: 25 episodes of metaworld's scripted
expert for the task on the RM75 (EXPERTS: bin-picking, drawer-open and assembly need RM75 versions,
envs/metaworld/rm75/expert.py; button-press-topdown runs metaworld's script unchanged), each action recorded as the
script outputs it (p * (target - hand), not clipped; the env clips it when executing, as CortexBench's metaworld does),
in rm75_env.RM75CBEnv, in the official demo format ('observations' (500, 39), 'actions' (500, 4), 'rewards',
'env_infos' (metaworld's info per step + 'gripper_proprio' = obs[:4]), 'images' = the frame before each action).
Seeds 12345 + i as the official demos; an episode not successful at its last step is skipped and the next seed taken
(printed). The 2D pickle ('images' = (500, 256, 256, 3) uint8) and the 3D one ('images' = (500, 1200, 6) float32 point
clouds) come from the same rollouts.
-> $STABLEWM_HOME/datasets/cortexbench/metaworld-rm75-expert{,-pcd}/rm75-<task>-v2-goal-observable.pickle

    source env.sh && $MW_PY make_rm75_demos.py <task>     (MW_PY: the MuJoCo 3 python, MuJoCo 3 + metaworld 3.0)
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np

from jepa4r.policy.bc.rm75_env import HORIZON, RM75CBEnv  # first: it puts the repository on sys.path
from jepa4r.envs.twins.metaworld_rm75 import RM75AssemblyPolicy, RM75BinPickingPolicy, RM75DrawerOpenPolicy
from metaworld.policies import SawyerButtonPressTopdownV3Policy

EXPERTS = {
    "bin-picking": RM75BinPickingPolicy,
    "button-press-topdown": SawyerButtonPressTopdownV3Policy,
    "drawer-open": RM75DrawerOpenPolicy,
    "assembly": RM75AssemblyPolicy,
}
TASK = sys.argv[1]
ENV_NAME, N, BASE_SEED = f"rm75-{TASK}-v2-goal-observable", 25, 12345
DATA = Path(os.environ["JEPA4R_DEMOS_HOME"])  # data/policy/<env>
OUT2D, OUT3D = DATA / "metaworld-rm75-expert", DATA / "metaworld-rm75-expert-pcd"

OUT2D.mkdir(parents=True, exist_ok=True)
OUT3D.mkdir(parents=True, exist_ok=True)
env = RM75CBEnv(TASK)
demos2d, demos3d, seed, skipped = [], [], BASE_SEED, []
while len(demos2d) < N:
    pol = EXPERTS[TASK]()
    env.seed(seed)
    o = env.reset()
    obs_seq, A, R, img_seq, P, infos = [], [], [], [], [], {}
    for t in range(HORIZON):
        img_seq.append(env.rgb())
        P.append(env.pcd())
        a = np.asarray(pol.get_action(o), dtype=np.float64)
        obs_seq.append(o.copy())
        o, r, done, info = env.step(a)
        A.append(a)
        R.append(r)
        for k, v in info.items():
            infos.setdefault(k, []).append(v)
        infos.setdefault("gripper_proprio", []).append(obs_seq[-1][:4].copy())
    if infos["success"][-1] < 1:
        skipped.append(seed)
        seed += 1
        continue
    base = {
        "observations": np.array(obs_seq),
        "actions": np.array(A),
        "rewards": np.array(R),
        "env_infos": {k: np.array(v) for k, v in infos.items()},
    }
    demos2d.append(dict(base, images=np.array(img_seq, dtype=np.uint8)))
    demos3d.append(dict(base, images=np.array(P, dtype=np.float32)))
    print(
        f"demo {len(demos2d):2d}/{N}: seed {seed}, first success at step {int(np.argmax(infos['success'])) + 1}",
        flush=True,
    )
    seed += 1
for d, demos in ((OUT2D, demos2d), (OUT3D, demos3d)):
    with open(d / f"{ENV_NAME}.pickle", "wb") as f:
        pickle.dump(demos, f)
print("skipped seeds (not successful at step 500):", skipped)
env.close()
