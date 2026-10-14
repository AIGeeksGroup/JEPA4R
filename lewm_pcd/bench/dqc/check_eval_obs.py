"""Check: the observation computed in the evaluation (lewm_obs.LeWMObs on the eval env, OGBench's task mode) and the
training set's observation (extract_feats.py: a data_collection-mode env re-rendering a stored state) agree for the same
simulator state.
    python check_eval_obs.py --ckpt-2d <2D checkpoint dir> --ckpt-3d <3D checkpoint dir>
"""
import argparse

import gymnasium
import numpy as np
import ogbench.manipspace  # noqa: F401

import extract_feats as ef  # this script's directory is on sys.path
from lewm_obs import Encoders, LeWMObs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-2d', required=True)
    ap.add_argument('--ckpt-3d', required=True)
    ap.add_argument('--epoch', type=int, default=10)
    a = ap.parse_args()
    env = gymnasium.make('puzzle-3x3-v0', use_oracle_rep=True, width=224, height=224)
    obs2, obs3 = LeWMObs(env, 'lewm2d', a.ckpt_2d, a.epoch), LeWMObs(env, 'lewm3d', a.ckpt_3d, a.epoch)
    ef._init_worker()
    enc = Encoders(a.ckpt_2d, a.ckpt_3d, a.epoch)
    cos = lambda x, y: float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    rng = np.random.default_rng(0)
    u = env.unwrapped
    for task in (1, 3, 5):
        env.reset(options=dict(task_id=task, render_goal=False))
        for step in range(60):
            env.step(rng.uniform(-1, 1, 5).astype(np.float32))
            if step % 20 != 19:
                continue
            f2, f3 = obs2(), obs3()
            qpos, qvel, bst = u._data.qpos.copy(), u._data.qvel.copy(), u._cur_button_states.copy()
            obs2.cam.r2d.update_scene(u._data, camera='front_pixels')  # the eval env's own 2D frame
            pix_eval = obs2.cam.r2d.render().copy()
            _, pix, xyz, col, _ = ef._render_chunk((0, 9, qpos[None].astype(np.float32), qvel[None].astype(np.float32), bst[None]))
            z2, z3 = enc.z2d(pix)[0], enc.z3d(xyz, col)[0]
            print(f'task {task} step {step + 1}: identical pixels {(pix[0] == pix_eval).all(-1).mean():.5f}; '
                  f'lights {np.array_equal(f2[192:], bst.astype(np.float32))}; '
                  f'2D cosine {cos(f2[:192], z2):.5f}, 3D cosine {cos(f3[:192], z3):.4f}', flush=True)


if __name__ == '__main__':
    main()
