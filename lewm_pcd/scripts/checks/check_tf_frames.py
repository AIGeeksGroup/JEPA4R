"""envs/trifinger/env.py's skipped camera renders change nothing observed. Runs episodes of a task with the
three-part policy (seeded) in TriFingerEnv + TriFingerPCDWrapper and saves every row's frame, observation vector,
info arrays and point cloud; run once with skip_unobserved=1 and once with 0 (two processes: the skip is a module-level
patch), then compare the two files (identical bit for bit). Prints the time per step.
    PYTHONPATH=.:third_party/stable-worldmodel $PY scripts/checks/check_tf_frames.py run move 1 <out.npz>
    ... run move 0 <out0.npz>; ... compare <out.npz> <out0.npz>"""
import sys
import time

import numpy as np


def run(task, skip, out, episodes=6):
    from envs.trifinger import EPISODE_STEPS, TriFingerEnv, TriFingerPCDWrapper, TriFingerThreePartPolicy
    env = TriFingerEnv(task=task, skip_unobserved=bool(int(skip)))
    w = TriFingerPCDWrapper(env)
    pol = TriFingerThreePartPolicy(seed=0)
    pol.env = type('E', (), {'envs': [w], 'num_envs': 1})()
    pol._experts = [None]
    rows = {'frame': [], 'obs': [], 'qpos': [], 'ft_des': [], 'pcd': []}
    n, t0 = 0, time.time()
    for ep in range(episodes):
        obs, info = w.reset(seed=100 + ep)
        for k in range(EPISODE_STEPS[task] + 1):
            rows['frame'].append(env.render()); rows['obs'].append(obs); rows['qpos'].append(info['qpos'])
            rows['ft_des'].append(info['ft_des']); rows['pcd'].append(np.c_[info['pcd_xyz'], info['pcd_rgb']])
            if k == EPISODE_STEPS[task]:
                break
            batch = {key: np.asarray([[v]]) for key, v in {**info, 'step_idx': k}.items() if not isinstance(v, dict)}
            obs, _, _, _, info = w.step(pol.get_action(batch)[0])
            n += 1
    print(f'{task} skip_unobserved={skip}: {(time.time() - t0) / n:.3f} s per step ({n} steps, {episodes} resets)')
    np.savez_compressed(out, **{k: np.stack(v) for k, v in rows.items()})


def compare(a, b):
    A, B = np.load(a), np.load(b)
    for k in A.files:
        same = A[k].shape == B[k].shape and np.array_equal(A[k], B[k])
        print(f'{k}: {A[k].shape} identical={same}' + ('' if same else f' max |diff| {np.abs(A[k].astype(np.float64) - B[k]).max()}'))


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run(*sys.argv[2:5])
    else:
        compare(*sys.argv[2:4])
