"""Expert demos for OGBench scene's five evaluation tasks, built as make_rm75_demos.py (the format of CortexBench's
official demos): 25 per task at seeds 12345 + i (as the official ones); expert = envs.ogb_scene_expert.SceneTaskExpert
(OGBench's per-object oracles chained by the goal), env = the evaluation's own scene_env.SceneCBEnv (the same reset, the
same 2D frames and point clouds). A seed not successful within 750 steps is skipped for the next one (printed).
Length: the evaluation ends an episode at the goal, so a demo is recorded up to its success; the harness's BC dataset
needs equal lengths within a task, so all demos of a task run to the slowest one's success step, those done earlier
holding still after their success (all-zero actions).
Two passes: the first without rendering finds 25 successful seeds and their success steps; the second replays the same
seeds with rendering (asserting the same success steps).
Format: 'observations' (T, 8) = proprioception (scene_env.proprio_of), 'actions' (T, 5), 'rewards', 'env_infos'
('success', 'gripper_proprio' = the proprioception), 'images' = the frame before each action: 2D (T, 224, 224, 3) uint8,
3D (T, 1200, 6) float32 point clouds; 2D and 3D from the same episodes.
-> $STABLEWM_HOME/datasets/cortexbench/scene-expert{,-pcd}/scene-task<k>-v0.pickle
    STABLEWM_HOME=<checkout>/lewm_pcd_data <lewm_mw python> make_scene_demos.py [task number ...]
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from jepa4r.policy.bc.scene_env import HORIZON, SceneCBEnv  # first: it puts the repository on sys.path
from jepa4r.envs.ogbench.ogb_scene_expert import TASK_NAMES, SceneTaskExpert

N, BASE_SEED = 25, 12345
DATA = Path(os.environ['JEPA4R_DEMOS_HOME'])  # data/policy/scene
OUT2D, OUT3D = DATA / 'scene-expert', DATA / 'scene-expert-pcd'


def rollout(env, expert, seed, T, render):
    env.seed(seed)
    o = env.reset()
    expert.reset()
    O, A, R, S, I, P = [], [], [], [], [], []
    first = None
    for t in range(T):
        if render:
            I.append(env.rgb())
            P.append(env.pcd())
        a = np.zeros(5) if first is not None else np.asarray(expert.act(o, t), np.float64)
        O.append(o.copy())
        o, r, done, info = env.step(a)
        A.append(a)
        R.append(r)
        S.append(info['success'])
        if done and first is None:
            first = t + 1
            if not render:
                break
    return first, dict(O=O, A=A, R=R, S=S, I=I, P=P)


def main():
    tasks = [int(k) for k in sys.argv[1:]] or [1, 2, 3, 4, 5]
    OUT2D.mkdir(parents=True, exist_ok=True)
    OUT3D.mkdir(parents=True, exist_ok=True)
    for k in tasks:
        t0 = time.time()
        env = SceneCBEnv(k)
        expert = SceneTaskExpert(env.env)
        seeds, steps, skipped, seed = [], [], [], BASE_SEED
        while len(seeds) < N:
            first, _ = rollout(env, expert, seed, HORIZON, render=False)
            if first is None:
                skipped.append(seed)
            else:
                seeds.append(seed)
                steps.append(first)
            seed += 1
        L = max(steps)
        demos2d, demos3d = [], []
        for s, n in zip(seeds, steps):
            first, d = rollout(env, expert, s, L, render=True)
            assert first == n, f'seed {s}: success at {first} on replay, {n} before'
            base = {'observations': np.array(d['O']), 'actions': np.array(d['A']), 'rewards': np.array(d['R']),
                    'env_infos': {'success': np.array(d['S']), 'gripper_proprio': np.array(d['O'])}}
            demos2d.append(dict(base, images=np.array(d['I'], dtype=np.uint8)))
            demos3d.append(dict(base, images=np.array(d['P'], dtype=np.float32)))
        name = f'scene-task{k}-v0'
        for d, demos in ((OUT2D, demos2d), (OUT3D, demos3d)):
            with open(d / f'{name}.pickle', 'wb') as f:
                pickle.dump(demos, f)
        print(
            f'{name} ({TASK_NAMES[k - 1]}): {N} demos, seeds {seeds[0]}..{seeds[-1]}, skipped {skipped}, '
            f'success step median {int(np.median(steps))} min {min(steps)} max {L} (= demo length), '
            f'{time.time() - t0:.0f} s',
            flush=True,
        )
        env.close()
    os._exit(0)


if __name__ == '__main__':
    main()
