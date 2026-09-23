"""Open-loop evaluation of a world model through a trained reconstruction run.

Per episode the model is given the first --context model frames (default: its own history size, 3 = env steps
0, 5, 10 -- the clip it was trained on and the window the 4D model aggregates); every later latent1 is predicted from
earlier PREDICTIONS with the recorded action blocks (``recon.rollout``), then decoded and compared with the real frame.

Reported per model frame, averaged over the episodes (distances in cm):
  true     decode of the real frame's latent1 -- the decoder's own floor
  pred     decode of the open-loop prediction -- floor + accumulated dynamics error
  hold     the last given frame held for the rest of the episode
  wrong    the same rollout driven by ANOTHER episode's actions -- does the prediction follow the actions?
for: the whole cloud (nn), the regions around the position columns (object / effector: real points of the region ->
decoded cloud), the object's position read out by colour (3D, horizontal and vertical part), and in latent space the
relative distance and cosine between predicted and real latent1.

    python scripts/recon/eval_rollout.py --recon rm75_1cam_page4d_lora --episodes 0:64
    python scripts/recon/eval_rollout.py --recon <run> --input <pixels>.h5 --target <point clouds of the same rows>.h5
"""
import argparse, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party' / 'stable-worldmodel')]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import stable_worldmodel as swm
import torch

from recon import metrics as M
from recon.model import open_run
from recon.rollout import action_blocks, open_loop, training_setup

KINDS = {'true': '#eb6834', 'pred': '#2a78d6', 'wrong': '#1baf7a', 'hold': '#8a8984'}


def parse_episodes(spec, n):
    out = []
    for part in spec.split(','):
        a, _, b = part.partition(':')
        out += list(range(int(a), int(b))) if b else [int(a)]
    assert out and max(out) < n, f'episodes {spec} outside 0..{n - 1}'
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recon', required=True, help='<run> or <run>/weights_epoch_N.pt under checkpoints/recon/')
    ap.add_argument('--input', help='trajectories the encoder reads (default: the dataset the run was trained on)')
    ap.add_argument('--target', help='point clouds of the same rows (default: as trained; = --input for point-cloud checkpoints)')
    ap.add_argument('--episodes', default='0:64', help='a:b ranges and single indices, comma separated')
    ap.add_argument('--context', type=int, help='given model frames (default: the world model\'s history size)')
    ap.add_argument('--regions', nargs='*', default=['object=privileged_block_0_pos:3', 'effector=proprio_effector_pos:6'])
    ap.add_argument('--batch', type=int, default=8, help='episodes per batch')
    ap.add_argument('--chunk', type=int, default=64, help='latents decoded at a time (GPU memory; the result does not depend on it)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', help='default: <run dir>/eval_rollout[_<target name>]')
    args = ap.parse_args()
    torch.manual_seed(0)

    wm, decoder, cfg, episodes = open_run(args.recon, args.input, args.target, args.device)
    frameskip, history, a_mean, a_std = training_setup(cfg.ckpt)
    context = args.context or history
    eps = parse_episodes(args.episodes, len(episodes))
    center, scale = cfg.data.target_preprocessor.center, cfg.data.target_preprocessor.scale
    regions, radius = {}, {}
    for spec in args.regions:  # name=position column:radius in cm
        name, _, rest = spec.partition('=')
        col, _, cm = rest.partition(':')
        try:
            episodes.column(col, eps[0]); regions[name], radius[name] = col, float(cm or 3) / 100
        except KeyError:
            print(f'[eval_rollout] no column {col}: {name} errors skipped')
    obj = next(iter(regions), None)

    def decode(latents):  # (E, K, D) -> (E, K, N, C) metres; --chunk at a time: the decoder's dense map is 224 x 224
        n = cfg.num_points
        pts = torch.cat([decoder.to_points(decoder(z), n).float() for z in latents.flatten(0, 1).split(args.chunk)])
        return M.to_metres(pts, center, scale).view(*latents.shape[:2], *pts.shape[1:])

    res, pos, gts = {}, {k: [] for k in ('sim', 'real', *KINDS)}, []
    with torch.no_grad():
        for i in range(0, len(eps), args.batch):
            chunk = eps[i:i + args.batch]
            loaded = [episodes.load(e, frameskip) for e in chunk]
            K = min(x['target'].size(0) for x in loaded)
            obs = torch.stack([x[wm.obs_key][:K] for x in loaded]).to(args.device)
            gt = M.to_metres(torch.stack([x['target'][:K] for x in loaded]).to(args.device).float(), center, scale)       # (E, K, N, C)
            blocks = torch.stack([action_blocks(episodes.column('action', e), frameskip, a_mean, a_std)[:K - 1] for e in chunk]).to(args.device)
            true = wm(obs)                                                                                         # (E, K, D)
            lat = {'true': true, 'pred': open_loop(wm.world_model, true, blocks, context, history),
                   'wrong': open_loop(wm.world_model, true, blocks.roll(1, 0), context, history),
                   'hold': true[:, context - 1:context].expand_as(true)}
            region_pos = {n: torch.from_numpy(np.stack([episodes.column(c, e, frameskip)[:K] for e in chunk])).float().to(args.device) for n, c in regions.items()}
            flat = lambda t: t.flatten(0, 1)
            for kind, z in lat.items():
                dec = decode(z)
                spurious, missing, _ = M.nn_both_sides(flat(dec), flat(gt))
                out = {'nn': 50 * (spurious + missing)}
                for n, p in region_pos.items():
                    out[f'{n}_missing'] = 100 * M.region_error(flat(dec), flat(gt), flat(p), radius[n])[0]
                out['latent_rel'] = flat((z - true).norm(dim=-1) / true.norm(dim=-1))
                out['latent_cos'] = flat(torch.nn.functional.cosine_similarity(z, true, dim=-1))
                for k, v in out.items():
                    res.setdefault(f'{k}/{kind}', []).append(v.view(len(chunk), K).cpu())
                pos[kind].append(dec.cpu())
            gts.append(gt.cpu())
            if obj:
                pos['sim'].append(region_pos[obj].cpu())
            print(f'[eval_rollout] episodes {i + len(chunk)}/{len(eps)}', flush=True)

    res = {k: torch.cat(v).numpy() for k, v in res.items()}                                                        # (E, K)
    K = next(iter(res.values())).shape[1]
    steps = np.arange(K) * frameskip
    purity, track = None, {}
    if obj:  # the object's position, read out by its colour in the real clouds of these episodes
        gt_all, sim = torch.cat(gts), torch.cat(pos['sim'])
        signature, purity = M.colour_signature(gt_all.flatten(0, 1), sim.flatten(0, 1), radius[obj] * 0.85)
        read = lambda clouds: M.object_centroid(clouds.flatten(0, 1), signature).view(*clouds.shape[:2], 3)
        track = {'sim': sim, 'real': read(gt_all), **{k: read(torch.cat(pos[k])) for k in KINDS}}
        for k in KINDS:
            d = track[k] - track['real']
            res[f'object_pos/{k}'] = (d.norm(dim=-1) * 100).numpy()
            res[f'object_pos_xy/{k}'] = (d[..., :2].norm(dim=-1) * 100).numpy()
            res[f'object_pos_z/{k}'] = (d[..., 2].abs() * 100).numpy()

    out_dir = Path(args.out) if args.out else Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), 'recon', args.recon.split('/')[0],
                                                     'eval_rollout' + (f'_{Path(cfg.data.target).stem}' if args.input or args.target else ''))
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = {k: np.nanmean(v, 0).tolist() for k, v in res.items()}
    meta = dict(recon=args.recon, world_model=cfg.ckpt, decoder=cfg.decoder._target_, input=cfg.data.input, target=cfg.data.target,
                episodes=args.episodes, context_frames=context, frameskip=frameskip, env_steps=steps.tolist(), regions=args.regions,
                object_colour_elsewhere_pct=None if purity is None else float(purity) * 100)
    json.dump({'meta': meta, 'mean_per_frame': curves}, open(out_dir / 'rollout.json', 'w'), indent=1)
    with open(out_dir / 'rollout.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'env_step', *res])
        w.writerows([e, int(steps[k]), *(f'{res[m][j, k]:.4f}' for m in res)] for j, e in enumerate(eps) for k in range(K))

    # curves: what deviates, and how fast
    panels = [m for m in ('nn', f'{obj}_missing' if obj else None, 'object_pos' if obj else None, 'latent_rel') if m and f'{m}/pred' in res]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.8))
    for ax, m in zip(np.atleast_1d(axes), panels):
        ax.axvspan(-frameskip / 2, steps[context - 1] + frameskip / 2, color='#efeee9', lw=0)
        for kind, colour in KINDS.items():
            k0 = 0 if kind == 'true' else context - 1
            ax.plot(steps[k0:], np.nanmean(res[f'{m}/{kind}'], 0)[k0:], color=colour, lw=2, label=kind, ls='--' if kind == 'hold' else '-')
        ax.set_title(m + ('' if m.startswith('latent') else '  (cm)')); ax.set_xlabel('env step'); ax.set_ylim(bottom=0); ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(frameon=False)
    fig.suptitle(f'{args.recon}: open loop from {context} given frames, {len(eps)} episodes of {Path(cfg.data.target).stem}', fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / 'rollout.png', dpi=130); plt.close(fig)

    if obj:  # top view of the object's path: simulator, read from the real cloud, read from the open-loop prediction
        show = list(range(min(8, len(eps))))
        fig, axes = plt.subplots(2, 4, figsize=(15, 7.6)) if len(show) > 4 else plt.subplots(1, len(show), figsize=(3.8 * len(show), 3.9))
        for ax, j in zip(np.atleast_1d(axes).ravel(), show):
            for k, colour, name in (('sim', '#0b0b0b', 'simulator'), ('real', KINDS['true'], 'real cloud'), ('pred', KINDS['pred'], 'open loop')):
                xy = track[k][j, :, :2].numpy()
                ax.plot(xy[:, 1], xy[:, 0], '-o', ms=2.5, lw=1.3, color=colour, label=name)
            ax.set_title(f'episode {eps[j]}', fontsize=9); ax.set_aspect('equal'); ax.invert_xaxis(); ax.grid(alpha=0.3); ax.set_xlabel('y (m)'); ax.set_ylabel('x (m)')
        np.atleast_1d(axes).ravel()[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f'{args.recon}: object path, top view', fontsize=10)
        fig.tight_layout(); fig.savefig(out_dir / 'object_paths.png', dpi=130, bbox_inches='tight'); plt.close(fig)

    print(f'[eval_rollout] {args.recon}: {len(eps)} episodes, {context} given frames -> {out_dir}')
    if purity is not None and purity > 0.01:
        print(f'[eval_rollout] WARNING the object colour also covers {float(purity) * 100:.1f} % of the rest of the scene: object_pos is unreliable here')
    at = sorted({context - 1, context, K // 2, K - 1})
    print('  env step'.ljust(30) + ''.join(f'{steps[k]:>9d}' for k in at))
    for m in res:
        print(f'  {m:28s}' + ''.join(f'{np.nanmean(res[m][:, k]):9.3f}' for k in at))


if __name__ == '__main__':
    main()
