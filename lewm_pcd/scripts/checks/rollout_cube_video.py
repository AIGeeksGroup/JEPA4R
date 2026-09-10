"""Roll out ONE OGBench cube_single episode with stable-worldmodel's own
collection pipeline and dump it as a video.

Same settings as the paper-era `scripts/data/collect_cube.py` (commit 5223ba5):
  swm/OGBCube-v0, env_type=single, ob_type=pixels, multiview=False, 224x224,
  visualize_info=False, terminate_at_goal=False, mode=data_collection,
  ExpertPolicy(policy_type='plan_oracle'), 200 steps / episode, seed 3072.

Only difference: episodes=1 and the built-in 'video' writer instead of hdf5.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

HERE = Path(__file__).resolve().parent
SWM_SRC = '/root/autodl-tmp/test_4d_jepa_v1/lewm_official/stable-worldmodel'
sys.path.insert(0, str(HERE.parents[1] / 'TMP' / 'pydeps'))  # pymunk/pygame/shapely (pusht-only deps pulled in by envs.utils)
sys.path.insert(0, SWM_SRC)

import numpy as np  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.data.format import get_format  # noqa: E402
from stable_worldmodel.envs.ogbench import ExpertPolicy  # noqa: E402

SEED = int(os.environ.get('SEED', 3072))
FPS = 20  # OGBench manipspace control_timestep = 0.05 s -> real-time playback
OUT_DIR = HERE.parents[1] / 'TMP' / 'cube_single_plan_oracle_rollout'

# The OGBench oracles draw their noise from the *global* numpy RNG (not the env
# RNG), so seed it here to make this demo repeatable.
np.random.seed(SEED)

t0 = time.time()
world = swm.World(
    'swm/OGBCube-v0',
    num_envs=1,
    max_episode_steps=200,
    image_shape=[224, 224],
    env_type='single',
    ob_type='pixels',
    multiview=False,
    width=224,
    height=224,
    visualize_info=False,
    terminate_at_goal=False,
    mode='data_collection',
)
world.set_policy(ExpertPolicy(policy_type='plan_oracle'))
print(f'world ready in {time.time() - t0:.1f}s', flush=True)

t0 = time.time()
writer = get_format('video').open_writer(OUT_DIR, fps=FPS, mode='overwrite')
world.collect(writer=writer, episodes=1, seed=SEED)
print(f'rollout + encode in {time.time() - t0:.1f}s', flush=True)

# ---- summary of what was recorded -------------------------------------------
ep_len = int(np.load(OUT_DIR / 'ep_len.npz')['arr_0'][0])
cols = sorted(p.stem for p in OUT_DIR.glob('*.npz') if p.stem not in ('ep_len', 'ep_offset'))
vids = sorted(str(p.relative_to(OUT_DIR)) for p in OUT_DIR.rglob('*.mp4'))
print(f'episode length: {ep_len} steps')
print('video files   :', vids)
print('tabular cols  :', cols)
act = np.load(OUT_DIR / 'action.npz')['arr_0']
print(f'action        : shape={act.shape} min={act.min():.2f} max={act.max():.2f} mean|a|={np.abs(act).mean():.3f}')
for c in ('success', 'privileged/target_task', 'privileged/block_0_pos', 'privileged/target_block_0_pos'):
    f = OUT_DIR / f'{c}.npz'
    if f.exists():
        v = np.load(f, allow_pickle=True)['arr_0']
        print(f'{c:32s}: shape={v.shape} first={v[0]} last={v[-1]}')

# ---- contact sheet (8 evenly spaced frames) for a quick visual check ----------
import imageio.v2 as imageio  # noqa: E402

src = OUT_DIR / 'pixels' / 'ep_0.mp4'
frames = imageio.mimread(src, memtest=False)
idx = np.linspace(0, len(frames) - 1, 8).astype(int)
tiles = [frames[i] for i in idx]
sheet = np.concatenate([np.concatenate(tiles[:4], axis=1), np.concatenate(tiles[4:], axis=1)], axis=0)
imageio.imwrite(HERE.parents[1] / 'TMP' / 'cube_single_plan_oracle_ep0_frames.png', sheet)

final = HERE.parents[1] / 'TMP' / 'cube_single_plan_oracle_ep0.mp4'
final.write_bytes(src.read_bytes())
print('video  ->', final, f'({final.stat().st_size / 1e6:.2f} MB, {len(frames)} frames @ {FPS} fps)')
print('frames ->', HERE.parents[1] / 'TMP' / 'cube_single_plan_oracle_ep0_frames.png', 'indices', idx.tolist())
