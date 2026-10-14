"""GIFs of the OGBench puzzle / scene LeWM data: left the 2D frame (what the 2D model trains on), middle / right the same
row's point cloud (1200 points in their RGB) seen obliquely from above and from straight above.
    puzzle: one episode, steps t0..t1; header: the step and the 9 true lights (red 0, blue 1, laid out 3 x 3 as the buttons)
        python scripts/checks/gif_ogb.py puzzle <2D.h5> <3D.h5> <episode> <t0> <t1> <every> <out.gif>
    scene: per object task three episodes (backbone, branch, random) in one GIF; header: task, episode type, step, whether
        the row's action is random, the two locks (unlocked / LOCKED), drawer and window openings, cube in the drawer
        python scripts/checks/gif_ogb.py scene <2D.h5> <3D.h5> <out dir> [every]
"""
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

VIEWS = {'oblique': (35.0, 35.0), 'top': (0.0, 89.0)}  # azimuth, elevation (deg)
RED, BLUE = (220, 60, 80), (70, 150, 230)
SCENE_TASKS = ('cube', 'button', 'drawer', 'window')
TYPES = ('backbone', 'branch', 'random')
SCENE_VIEW = dict(center=(0.42, -0.02, 0.15), extent=0.55)


def render_pcd(xyz, rgb, view='oblique', size=224, center=(0.425, 0.0, 0.12), extent=0.32, psize=2, bg=255):
    """Orthographic image of a point cloud from a virtual view on front_pixels' side (+x), turned by the azimuth about z
    and looking down by the elevation; points as small squares, near ones over far ones."""
    az, el = np.deg2rad(VIEWS[view][0]), np.deg2rad(VIEWS[view][1])
    fwd = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right) + 1e-9
    up = np.cross(right, fwd)
    p = np.asarray(xyz, float) - np.asarray(center)
    u, v, d = p @ right, p @ up, p @ fwd
    px = ((u / extent + 1) / 2 * (size - 1)).astype(int)
    py = ((1 - (v / extent + 1) / 2) * (size - 1)).astype(int)
    img = np.full((size, size, 3), bg, np.uint8)
    for i in np.argsort(-d):  # far ones first
        x, y = px[i], py[i]
        if 0 <= x < size and 0 <= y < size:
            img[max(0, y - psize // 2): y + psize - psize // 2, max(0, x - psize // 2): x + psize - psize // 2] = rgb[i]
    return img


def puzzle(p2, p3, ep, t0, t1, every, out):
    ep, t0, t1, every = int(ep), int(t0), int(t1), int(every)

    def header(step, states, width):
        img = Image.new('RGB', (width, 34), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((6, 10), f'step {step}', fill=(0, 0, 0))
        d.text((100, 10), 'true lights:', fill=(0, 0, 0))
        for i, s in enumerate(states):
            r, c = divmod(i, 3)
            x0, y0 = 190 + c * 10, 2 + r * 10
            d.rectangle([x0, y0, x0 + 8, y0 + 8], fill=BLUE if s else RED)
        d.text((width - 224 * 2 + 6, 10), '3D point cloud: oblique view | top view', fill=(0, 0, 0))
        return np.asarray(img)

    with h5py.File(p2, 'r') as f2, h5py.File(p3, 'r') as f3:
        off = int(f2['ep_offset'][ep])
        rows = np.arange(off + t0, off + min(t1, int(f2['ep_len'][ep])), every)
        sl = slice(int(rows[0]), int(rows[-1]) + 1)
        pix, bs = f2['pixels'][sl][::every], f2['button_states'][sl][::every]
        xyz, rgb = f3['pcd_xyz'][sl][::every], f3['pcd_rgb'][sl][::every]
    frames = []
    for k in range(len(rows)):
        body = np.concatenate([pix[k], render_pcd(xyz[k], rgb[k], 'oblique'), render_pcd(xyz[k], rgb[k], 'top')], 1)
        frames.append(Image.fromarray(np.concatenate([header(t0 + k * every, bs[k], body.shape[1]), body], 0)))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
    print(f'{out}: {len(frames)} frames')


def scene(p2, p3, out, every=2):
    every = int(every)
    Path(out).mkdir(parents=True, exist_ok=True)

    def header(lines, width):
        img = Image.new('RGB', (width, 30), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((6, 2), lines[0], fill=(0, 0, 0))
        d.text((6, 16), lines[1], fill=(0, 0, 0))
        return np.asarray(img)

    lock = lambda b: 'unlocked' if b else 'LOCKED'  # noqa: E731
    with h5py.File(p2, 'r') as f2, h5py.File(p3, 'r') as f3:
        off, el = f2['ep_offset'][:], f2['ep_len'][:]
        tid, st = f2['task_id'][:][off], f2['sample_type'][:][off]
        for k, task in enumerate(SCENE_TASKS):
            frames = []
            for j, typ in enumerate(TYPES):
                ep = int(np.flatnonzero((tid == k) & (st == j))[0])
                rows = np.arange(off[ep], off[ep] + el[ep], every)
                sl = slice(int(rows[0]), int(rows[-1]) + 1)
                pix, xyz, rgb = f2['pixels'][sl][::every], f3['pcd_xyz'][sl][::every], f3['pcd_rgb'][sl][::every]
                bs, dr, wi = f2['button_states'][sl][::every], f2['drawer_pos'][sl][::every], f2['window_pos'][sl][::every]
                cid, ra = f2['cube_in_drawer'][sl][::every], f2['rand_action'][sl][::every]
                for i in range(len(pix)):
                    body = np.concatenate([pix[i], render_pcd(xyz[i], rgb[i], 'oblique', **SCENE_VIEW),
                                           render_pcd(xyz[i], rgb[i], 'top', **SCENE_VIEW)], 1)
                    l1 = f'task {task} | episode {ep} ({typ}) | step {i * every}{"  RANDOM ACTION" if ra[i] else ""}'
                    l2 = (f'drawer {lock(bs[i][0])} {-dr[i]:.2f} m open | window {lock(bs[i][1])} {wi[i]:.2f} m open | '
                          f'cube {"in drawer" if cid[i] else "on table"}')
                    frames.append(Image.fromarray(np.concatenate([header((l1, l2), body.shape[1]), body], 0)))
            path = Path(out) / f'scene_{task}.gif'
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
            print(f'{path}: {len(frames)} frames', flush=True)


if __name__ == '__main__':
    {'puzzle': puzzle, 'scene': scene}[sys.argv[1]](*sys.argv[2:])
