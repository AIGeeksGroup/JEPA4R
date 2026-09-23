"""Videos of reconstructed trajectories, and the object's path seen from above.

For every episode: real point cloud | what each given reconstruction run decodes (drawn as points), from an oblique and
a top view -> <out>/ep<E>_<mode>.mp4, plus <out>/ep<E>_<mode>_object_path.png: top view of the object's path --
simulator, read from the real cloud, read from each run's decode (the object is found by its colour, measured on the
real clouds of that episode).

  --mode recon      every frame is encoded and decoded on its own (what the latent of a real frame holds)
  --mode openloop   the world model sees the first --context model frames, the rest is predicted from its own
                    predictions with the recorded actions (``recon.rollout``) and decoded

Several runs are shown side by side (e.g. stage A and stage B, 3D and 4D); they must reconstruct the same target
dataset. Other trajectories: --input / --target, as in eval_frames.py (a pixel checkpoint needs both).

    python scripts/recon/render_episodes.py --recon rm75_1cam_page4d_frozen rm75_1cam_page4d_lora --episodes 3,126 --out <dir>
    python scripts/recon/render_episodes.py --recon rm75_1cam_page4d_lora --episodes 126 --mode openloop
"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party' / 'stable-worldmodel')]

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import stable_worldmodel as swm
import torch

from recon import metrics as M
from recon import render as R
from recon.model import open_run
from recon.rollout import action_blocks, open_loop, training_setup

VIEWS = {'oblique': dict(az_deg=35.0, el_deg=30.0, dist=1.25), 'top': dict(az_deg=180.0, el_deg=89.0, dist=1.35)}   # top: +x up, +y to the left, as in the path figure
COLOURS = ['#2a78d6', '#1baf7a', '#a05cd6', '#d6a02a']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recon', required=True, nargs='+', help='one or more <run>[/weights_epoch_N.pt] under checkpoints/recon/')
    ap.add_argument('--episodes', required=True, help='comma separated episode indices')
    ap.add_argument('--mode', choices=['recon', 'openloop'], default='recon')
    ap.add_argument('--context', type=int, help='openloop: given model frames (default: the world model\'s history size)')
    ap.add_argument('--stride', type=int, help='env steps between video frames (recon: 1; openloop: always the frameskip)')
    ap.add_argument('--input', help='trajectories the encoder reads (default: the dataset each run was trained on)')
    ap.add_argument('--target', help='point clouds of the same rows')
    ap.add_argument('--object', default='privileged_block_0_pos', help='position column of the object whose path is drawn')
    ap.add_argument('--size', type=int, default=320, help='pixels per panel')
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--chunk', type=int, default=64, help='frames decoded at a time (GPU memory; the result does not depend on it)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', help='default: <run dir>/render (one run); required for several')
    args = ap.parse_args()
    assert args.out or len(args.recon) == 1, 'several runs: say where to write with --out'
    torch.manual_seed(0)

    runs = [open_run(name, args.input, args.target, args.device) for name in args.recon]
    cfg0 = runs[0][2]
    assert all(c.data.target == cfg0.data.target for _, _, c, _ in runs), 'the runs reconstruct different target datasets'
    center, scale = cfg0.data.target_preprocessor.center, cfg0.data.target_preprocessor.scale
    setups = [training_setup(c.ckpt) for _, _, c, _ in runs] if args.mode == 'openloop' else None
    if setups:
        assert len({s[0] for s in setups}) == 1, 'the world models use different frameskips'
    stride = setups[0][0] if setups else (args.stride or 1)
    out_dir = Path(args.out) if args.out else Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), 'recon', args.recon[0].split('/')[0], 'render')
    out_dir.mkdir(parents=True, exist_ok=True)
    cams = {v: R.make_cam(size=args.size, device=args.device, **kw) for v, kw in VIEWS.items()}
    names = [n.split('/')[0] for n in args.recon]

    for ep in [int(e) for e in args.episodes.split(',')]:
        decoded = []                                                       # per run: its point cloud in metres, per frame
        with torch.no_grad():
            for i, (wm, decoder, cfg, episodes) in enumerate(runs):
                data = episodes.load(ep, stride)
                latents = torch.cat([wm(o[None].to(args.device))[0] for o in data[wm.obs_key].split(64)])                  # (K, D)
                if args.mode == 'openloop':
                    frameskip, history, mean, std = setups[i]
                    blocks = action_blocks(episodes.column('action', ep), frameskip, mean, std)[:latents.size(0) - 1].to(args.device)
                    latents = open_loop(wm.world_model, latents[None], blocks[None], args.context or history, history)[0]
                decoded.append(torch.cat([M.to_metres(decoder.to_points(decoder(z), cfg.num_points).float(), center, scale) for z in latents.split(args.chunk)]))
                if i == 0:
                    real = M.to_metres(data['target'].to(args.device).float(), center, scale)                               # (K, N, C)
        K = real.size(0)

        given = (args.context or setups[0][1]) if setups else 0
        header = np.concatenate([np.pad(R.label(args.size, t), ((0, 0), (4, 4), (0, 0)), constant_values=0.16) for t in ['real', *names]], 1)
        video = []
        for k in range(K):
            rows = [[R.render_points(real[k], cam), *(R.render_points(pts[k], cam) for pts in decoded)] for cam in cams.values()]
            state = 'given' if k < given else ('predicted' if setups else 'reconstructed')
            body = R.tile(rows)
            foot = R.label(body.shape[1], f'episode {ep}   env step {k * stride}   {state}')
            frame = np.concatenate([np.pad(header, ((0, 0), (0, body.shape[1] - header.shape[1]), (0, 0)), constant_values=0.16), body, foot], 0)
            frame = np.pad(frame, ((0, -frame.shape[0] % 16), (0, -frame.shape[1] % 16), (0, 0)), constant_values=0.16)   # codec wants multiples of 16
            video.append((frame * 255).astype(np.uint8))
        path = out_dir / f'ep{ep}_{args.mode}.mp4'
        imageio.mimsave(path, video, fps=args.fps if stride == 1 else max(2, args.fps // stride), quality=8)
        print(f'[render] {path}  ({K} frames)')

        try:
            sim = torch.from_numpy(runs[0][3].column(args.object, ep, stride)[:K]).float().to(args.device)
        except KeyError:
            print(f'[render] no column {args.object}: object path skipped'); continue
        signature, purity = M.colour_signature(real, sim, 0.025)
        if signature is None or purity > 0.01:
            print(f'[render] the object cannot be told apart by colour in this data: object path skipped'); continue
        fig, ax = plt.subplots(figsize=(5.4, 5.2))
        tracks = [('simulator', '#0b0b0b', sim), ('real cloud', '#eb6834', M.object_centroid(real, signature))]
        tracks += [(n, COLOURS[i % len(COLOURS)], M.object_centroid(decoded[i], signature)) for i, n in enumerate(names)]
        for name, colour, xyz in tracks:
            xy = xyz[:, :2].cpu().numpy()
            ax.plot(xy[:, 1], xy[:, 0], '-', lw=1.6, color=colour, label=name, alpha=0.9)
            ax.plot(xy[0, 1], xy[0, 0], 'o', color=colour, ms=5)
        ax.set_aspect('equal'); ax.invert_xaxis(); ax.grid(alpha=0.3); ax.set_xlabel('y (m)'); ax.set_ylabel('x (m)'); ax.legend(frameon=False, fontsize=8)
        ax.set_title(f'episode {ep}: object path, top view ({args.mode}' + (f', {given} frames given)' if setups else ')'), fontsize=10)
        fig.tight_layout(); fig.savefig(out_dir / f'ep{ep}_{args.mode}_object_path.png', dpi=140, bbox_inches='tight'); plt.close(fig)
        print(f'[render] {out_dir / f"ep{ep}_{args.mode}_object_path.png"}')


if __name__ == '__main__':
    main()
