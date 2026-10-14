"""Lock probe of OGBench scene: can the two locks' states (button 0 locks the drawer, button 1 the window) be read from a
frozen LeWM latent of one frame?
* Frames: the five evaluation tasks, 10 expert episodes each (the CortexBench demos' expert and seeds 12345..12354,
  bench/cortexbench/scene_env.py), a state every 10 steps; every state rendered four times with only the locks changed
  ([0, 0], [0, 1], [1, 0], [1, 1]; 0 = locked): the 2D frame and the 3D point cloud (the point sampling fixed per state,
  so the four renders draw the same points). Within a state the four frames differ only in the colour of the button
  caps and handles (red locked, white unlocked).
* Encoders: CortexBench's configs (bench/cortexbench/conf/model/<name>.yaml), the latent CortexBench feeds the policy.
* Probe: a logistic regression per lock on the standardised latent; train / test split by state (80 / 20 %), the four
  lock combinations of a state on the same side; chance 50 %. Also: how far the latent moves when only the locks change,
  against the distance between neighbouring states.
In the lewm_mw python (MuJoCo 3 + OGBench, and the LeWM loaders):
    STABLEWM_HOME=... PYTHONPATH=.:third_party/stable-worldmodel python scripts/checks/probe_scene_locks.py \
        --models lewm_scene_small_pix,lewm_scene_small_pcd
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

L = Path(__file__).resolve().parents[2]  # lewm_pcd
sys.path.insert(0, str(L / 'bench' / 'cortexbench'))  # scene_env, lewm_cortex
os.environ.setdefault('MUJOCO_GL', 'egl')


def frames():
    from scene_env import SceneCBEnv
    from envs.ogb_scene_expert import SceneTaskExpert
    imgs, pcds, locks, sids, tasks = [], [], [], [], []
    sid = 0
    for k in range(1, 6):
        env = SceneCBEnv(k)
        expert = SceneTaskExpert(env.env)
        states = []
        for seed in range(12345, 12355):
            env.seed(seed)
            o = env.reset()
            expert.reset()
            for t in range(750):
                if t % 10 == 0:
                    states.append((env.u._data.qpos.copy(), env.u._data.qvel.copy()))
                o, _, done, _ = env.step(np.asarray(expert.act(o, t), np.float64))
                if done:
                    break
        for qpos, qvel in states:
            for lock in ((0, 0), (0, 1), (1, 0), (1, 1)):
                env.u.set_state(qpos, qvel, np.array(lock))
                imgs.append(env.rgb())
                env.pcw._rng = np.random.default_rng(sid)
                pcds.append(env.pcd())
                locks.append(lock)
                sids.append(sid)
                tasks.append(k)
            sid += 1
        print(f'task {k}: {len(states)} states', flush=True)
        env.close()
    return (np.array(imgs, np.uint8), np.array(pcds, np.float32), np.array(locks), np.array(sids), np.array(tasks))


def encode(name, data, is_pcd):
    import hydra
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    model, dim, transforms, _ = hydra.utils.call(OmegaConf.load(L / 'bench' / 'cortexbench' / 'conf' / 'model' / f'{name}.yaml'))
    model = model.eval().cuda()
    X = []
    with torch.no_grad():
        for i in range(0, len(data), 64):
            if is_pcd:
                xb = torch.stack([transforms(p) for p in data[i:i + 64]])
            else:
                xb = torch.stack([transforms(Image.fromarray(a)) for a in data[i:i + 64]])
            X.append(model(xb.cuda()).float().cpu().numpy())
    return np.concatenate(X), dim


def main():
    from sklearn.linear_model import LogisticRegression
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='lewm_scene_pix,lewm_scene_pcd', help='CortexBench encoder configs (conf/model)')
    a = ap.parse_args()
    imgs, pcds, lock, sid, task = frames()
    ids = np.unique(sid)
    test_ids = np.random.default_rng(0).choice(ids, size=len(ids) // 5, replace=False)
    te = np.isin(sid, test_ids)
    tr = ~te
    print(f'{len(ids)} states (train {len(ids) - len(test_ids)}, test {len(test_ids)}), {len(sid)} frames; chance 50 %')
    for name in a.models.split(','):
        is_pcd = name.endswith('_pcd')
        X, dim = encode(name, pcds if is_pcd else imgs, is_pcd)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xs = (X - mu) / sd
        out = []
        for j, lname in enumerate(('drawer lock', 'window lock')):
            clf = LogisticRegression(C=1.0, max_iter=5000).fit(Xs[tr], lock[tr, j])
            pred = clf.predict(Xs[te])
            per = ' '.join(f'{(pred[task[te] == k] == lock[te, j][task[te] == k]).mean() * 100:.0f}' for k in range(1, 6))
            out.append(f'{lname} test accuracy {(pred == lock[te, j]).mean() * 100:.1f} % (train '
                       f'{clf.score(Xs[tr], lock[tr, j]) * 100:.1f} %; tasks 1-5: {per})')
        # how far the latent moves when only the locks change, against neighbouring states
        same = np.mean([np.linalg.norm(X[sid == s][0] - X[sid == s][3]) for s in ids])
        diff = np.mean([np.linalg.norm(X[sid == ids[i]][0] - X[sid == ids[i + 1]][0]) for i in range(len(ids) - 1)])
        print(f'{name} ({dim}-d): ' + '; '.join(out) + f'. latent change with the locks only {same:.3f}, between '
              f'neighbouring states {diff:.3f}', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
