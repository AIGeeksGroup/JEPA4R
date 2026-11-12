"""Side-by-side video: RGB render vs. point cloud, from each camera a keep_rgb pilot recorded."""
import sys
from pathlib import Path
import h5py, numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
H5 = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1] / 'TMP' / 'datasets/ogbench/pilot_cube_single_expert_pcd.h5'
EP = int(sys.argv[2]) if len(sys.argv) > 2 else 0
FPS = 20
ORDER = ['front_pixels', 'pcd_front_left', 'pcd_back_left', 'pcd_back_right', 'pcd_front_right', 'pcd_top']
LABEL = {'front_pixels': 'front (official cam)', 'pcd_front_left': 'front-left', 'pcd_back_left': 'back-left', 'pcd_back_right': 'back-right', 'pcd_front_right': 'front-right', 'pcd_top': 'top'}
WS = np.array([[0.25, -0.35], [0.6, 0.35]])  # workspace x/y bounds, drawn at z=0

with h5py.File(H5, 'r') as f:
    off, L = int(f['ep_offset'][EP]), int(f['ep_len'][EP]); sl = slice(off, off + L)
    xyz, rgb = f['pcd_xyz'][sl], f['pcd_rgb'][sl]
    ORDER = [c for c in ORDER if f'pixels/{c}' in f]  # the cameras the pilot recorded: the one depth camera
    imgs = {c: f[f'pixels/{c}'][sl] for c in ORDER}
    cam = {c: (f[f'cam/{c}/pos'][sl], f[f'cam/{c}/xmat'][sl].reshape(L, 3, 3), float(f[f'cam/{c}/fovy'][off])) for c in ORDER}
    success = f['success'][sl]
H, W = imgs[ORDER[0]].shape[1:3]

def project(p, pos, R, fovy):
    fpx = 0.5 * H / np.tan(np.deg2rad(fovy) / 2)
    pc = (p - pos) @ R                     # R^T (p - t)
    zc = -pc[:, 2]; ok = zc > 1e-3
    u = fpx * pc[:, 0] / np.maximum(zc, 1e-6) + W / 2 - 0.5
    v = -fpx * pc[:, 1] / np.maximum(zc, 1e-6) + H / 2 - 0.5
    return u, v, zc, ok

def pcd_panel(points, colors, pos, R, fovy):
    im = Image.new('RGB', (W, H), (30, 30, 30)); d = ImageDraw.Draw(im)
    corners = np.array([[WS[0,0], WS[0,1], 0], [WS[1,0], WS[0,1], 0], [WS[1,0], WS[1,1], 0], [WS[0,0], WS[1,1], 0]])
    u, v, _, ok = project(corners, pos, R, fovy)
    if ok.all():
        d.polygon(list(zip(u.tolist(), v.tolist())), outline=(85, 85, 85))
    canvas = np.array(im)
    u, v, zc, ok = project(points, pos, R, fovy)
    for i in np.argsort(-zc):              # far first, near last
        if not ok[i]: continue
        ui, vi = int(round(u[i])), int(round(v[i]))
        if 0 <= ui < W - 1 and 0 <= vi < H - 1:
            canvas[vi:vi + 2, ui:ui + 2] = colors[i]
    return canvas

def label(arr, text):
    im = Image.fromarray(arr); ImageDraw.Draw(im).text((4, 3), text, fill=(255, 255, 255)); return np.array(im)

frames = []
for t in range(L):
    top = [label(np.ascontiguousarray(imgs[c][t]), f'RGB {LABEL[c]}') for c in ORDER]
    bot = [label(pcd_panel(xyz[t], rgb[t], cam[c][0][t], cam[c][1][t], cam[c][2]), f'PCD {LABEL[c]}') for c in ORDER]
    fr = np.vstack([np.hstack(top), np.hstack(bot)])
    fr = label(fr, f't={t:3d}  success={int(success[t])}') if False else fr
    frames.append(fr)

out = HERE.parents[1] / 'TMP' / f'pcd_multiview_ep{EP}.mp4'
with imageio.get_writer(out, fps=FPS, codec='libx264', quality=8) as w:
    for fr in frames: w.append_data(fr)
sheet = np.vstack([frames[i] for i in (0, 50, 100, 150)])
imageio.imwrite(HERE.parents[1] / 'TMP' / f'pcd_multiview_ep{EP}_sheet.png', sheet)
print('video ->', out, f'{out.stat().st_size/1e6:.2f} MB, {L} frames @ {FPS} fps, {frames[0].shape[1]}x{frames[0].shape[0]}')
print('sheet ->', HERE.parents[1] / 'TMP' / f'pcd_multiview_ep{EP}_sheet.png')
