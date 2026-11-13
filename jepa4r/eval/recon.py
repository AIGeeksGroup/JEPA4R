"""Reconstruction evaluation (stage eval_recon): Tab. 3, Tab. 4a, the open-loop figure, Fig. 1a.

per frame  decode encoder latents of cfg.decoder.eval.per_frame.num_frames validation frames; also decode the
           latent of another frame (shuffled control)
open loop  for cfg.decoder.eval.open_loop.num_episodes episodes: encode the context frames at env steps 0/5/10,
           roll the predictor forward with the episode's own actions (imagination), with another episode's
           actions (control), and decode encoder latents of the true frames (true latents); decode every step
Metrics: jepa4r/metrics/pointcloud.py, in cm in the world frame. Output keys: jepa4r/stages/eval_recon.py.

Integrated code (lewm-pcd-recon-page4d@d3c2d07): scripts/recon/eval_frames.py and eval_rollout.py, moved as the functions
``eval_frames(argv)`` / ``eval_rollout(argv)`` (their command lines unchanged). ``evaluate`` runs both on the stage-B
decoder of outputs/train_decoder/<env>/<decoder>/<tag>/ and reads the table entries off their outputs:
  eval_frames/metrics.json   errors.<name>.{mean, median, p90} over the frames (nn, outliers_pct, object_pos, nn_shuffled)
  eval_rollout/rollout.json  mean_per_frame['<metric>/<kind>'] per model frame (kind: true, pred, wrong, hold),
                             meta.env_steps
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import stable_pretraining as spt  # noqa: E402
import torch  # noqa: E402

from jepa4r.decoder.imagine import action_blocks, open_loop, training_setup  # noqa: E402
from jepa4r.decoder.model import open_run  # noqa: E402
from jepa4r.metrics import pointcloud as M  # noqa: E402

try:  # stable-worldmodel (third_party/stable-worldmodel) is needed to run, not to import this module
    import stable_worldmodel as swm
except ImportError:  # pragma: no cover
    swm = None


# Moved from lewm-pcd-recon-page4d@d3c2d07:lewm_pcd/scripts/recon/eval_frames.py:main (lines 36-129; verbatim except: main() -> eval_frames(argv), parse_args(argv))
def eval_frames(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recon', required=True, help='<run> or <run>/weights_epoch_N.pt under checkpoints/recon/')
    ap.add_argument('--input', help='dataset the encoder reads (default: the one the run was trained on)')
    ap.add_argument('--target', help='point clouds of the same rows (default: as trained; = --input for point-cloud checkpoints)')
    ap.add_argument('--split', choices=['val', 'all'], help='default: val on the training datasets, all otherwise')
    ap.add_argument('--frames', type=int, default=4096)
    ap.add_argument('--regions', nargs='*', default=['object=privileged_block_0_pos:3', 'effector=proprio_effector_pos:6'],
                    help='name=position column:radius in cm; the first one is the object whose position is read out')
    ap.add_argument('--outlier', type=float, default=3.0, help='outlier threshold, cm')
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', help='default: <run dir>/eval_frames[_<target name>]')
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)

    wm, decoder, cfg, episodes = open_run(args.recon, args.input, args.target, args.device)
    other_data = args.input is not None or args.target is not None
    split = args.split or ('all' if other_data else 'val')
    index = np.arange(len(episodes.frames))
    if split == 'val':  # the frames train_recon.py held out
        g = torch.Generator().manual_seed(cfg.seed)
        index = np.asarray(spt.data.random_split(episodes.frames, [cfg.train_split, 1 - cfg.train_split], generator=g)[1].indices)
    index = np.sort(np.random.default_rng(args.seed).choice(index, min(args.frames, len(index)), replace=False))

    regions, radius = {}, {}
    for spec in args.regions:  # position columns that exist in this dataset
        name, _, rest = spec.partition('=')
        col, _, cm = rest.partition(':')
        try:
            regions[name], radius[name] = torch.from_numpy(episodes.column(col)[index]).float(), float(cm or 3) / 100
        except KeyError:
            print(f'[eval_frames] no column {col}: {name} errors skipped')
    obj = next(iter(regions), None)

    center, scale = cfg.data.target_preprocessor.center, cfg.data.target_preprocessor.scale
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(episodes.frames, index.tolist()), batch_size=args.batch, num_workers=6)
    rows, clouds = [], []
    with torch.no_grad():
        for b, batch in enumerate(loader):
            latent = wm(batch[wm.obs_key].to(args.device))[:, 0]
            gt = M.to_metres(batch['target'][:, 0].to(args.device).float(), center, scale)
            sl = slice(b * args.batch, b * args.batch + gt.size(0))
            for tag, z in (('', latent), ('_shuffled', latent.roll(1, 0))):
                pred = M.to_metres(decoder.to_points(decoder(z), gt.size(1)).float(), center, scale)
                spurious, missing, d_pg = M.nn_both_sides(pred, gt)
                out = {'nn': 50 * (spurious + missing), 'nn_spurious': 100 * spurious, 'nn_missing': 100 * missing,
                       'outliers_pct': 100 * (d_pg > args.outlier / 100).float().mean(-1), 'sliced_w': 100 * M.sliced_w(pred, gt)}
                if gt.size(-1) > 3 and pred.size(-1) > 3:
                    out['colour'] = M.colour_error(pred, gt)
                for name, pos in regions.items():
                    err, count = M.region_error(pred, gt, pos[sl].to(args.device), radius[name])
                    out[f'{name}_missing'], out[f'{name}_count'] = 100 * err, count
                clouds.append((tag, pred.cpu(), gt.cpu() if not tag else None))
                rows.append((tag, {k: v.cpu() for k, v in out.items()}))

    # the object's position, read out by its colour in the real clouds of these very frames
    gt_all = torch.cat([g for _, _, g in clouds if g is not None])
    res = {}
    for tag in ('', '_shuffled'):
        for k in rows[0][1]:
            res[k + tag] = torch.cat([r_[k] for t, r_ in rows if t == tag])
    purity = None
    if obj is not None:
        signature, purity = M.colour_signature(gt_all, regions[obj], radius[obj] * 0.85)
        c_real = M.object_centroid(gt_all, signature)
        for tag in ('', '_shuffled'):
            c_dec = M.object_centroid(torch.cat([p for t, p, _ in clouds if t == tag]), signature)
            res['object_pos' + tag] = (c_dec - c_real).norm(dim=-1) * 100
            res['object_pos_sim' + tag] = (c_dec - regions[obj]).norm(dim=-1) * 100
        res['object_pos_real_sim'] = (c_real - regions[obj]).norm(dim=-1) * 100

    stat = lambda v: {'mean': float(np.nanmean(v)), 'median': float(np.nanmedian(v)), 'p90': float(np.nanpercentile(v, 90)),
                      'nan_pct': float(np.isnan(v).mean() * 100)}
    summary = {k: stat(v.numpy()) for k, v in res.items()}
    out_dir = Path(args.out) if args.out else Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), 'recon', args.recon.split('/')[0],
                                                     'eval_frames' + (f'_{Path(cfg.data.target).stem}' if other_data else ''))
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(recon=args.recon, world_model=cfg.ckpt, decoder=cfg.decoder._target_, input=cfg.data.input, target=cfg.data.target, split=split,
                frames=len(index), outlier_cm=args.outlier, regions=args.regions,
                object_colour=None if purity is None else [round(float(c), 3) for c in signature],
                object_colour_elsewhere_pct=None if purity is None else float(purity) * 100)
    json.dump({'meta': meta, 'errors': summary}, open(out_dir / 'metrics.json', 'w'), indent=2)
    with open(out_dir / 'frames.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', *res])
        w.writerows([int(i), *(f'{float(res[k][j]):.4f}' for k in res)] for j, i in enumerate(index))

    print(f'[eval_frames] {args.recon}: {len(index)} {split} frames of {cfg.data.target} -> {out_dir}')
    if purity is not None and purity > 0.01:
        print(f'[eval_frames] WARNING the object colour also covers {float(purity) * 100:.1f} % of the rest of the scene: object_pos is unreliable here')
    for k, v in summary.items():
        print(f'  {k:28s} mean {v["mean"]:8.3f}  median {v["median"]:8.3f}  p90 {v["p90"]:8.3f}' + (f'  (no readout in {v["nan_pct"]:.1f} %)' if v['nan_pct'] else ''))


# Moved from lewm-pcd-recon-page4d@d3c2d07:lewm_pcd/scripts/recon/eval_rollout.py:KINDS, parse_episodes, main (lines 36-173; verbatim except: main() -> eval_rollout(argv), parse_args(argv))
KINDS = {'true': '#eb6834', 'pred': '#2a78d6', 'wrong': '#1baf7a', 'hold': '#8a8984'}


def parse_episodes(spec, n):
    out = []
    for part in spec.split(','):
        a, _, b = part.partition(':')
        out += list(range(int(a), int(b))) if b else [int(a)]
    assert out and max(out) < n, f'episodes {spec} outside 0..{n - 1}'
    return out


def eval_rollout(argv=None):
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
    args = ap.parse_args(argv)
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


def evaluate(cfg, out_dir) -> dict:
    from jepa4r.config import stage_dir

    out_dir, e = Path(out_dir), cfg.decoder.eval
    run = stage_dir('train_decoder', cfg.env.name, None, cfg.decoder.name, str(cfg.tag)) / 'stage_b' / 'checkpoints'
    regions = [f'object={e.cube_position.column}:{float(e.cube_position.radius_cm):g}',
               f'effector={e.effector_region.column}:{float(e.effector_region.radius_cm):g}']
    eval_frames(['--recon', str(run), '--split', str(e.per_frame.split), '--frames', str(int(e.per_frame.num_frames)),
                 '--outlier', str(float(e.per_frame.outlier_cm)), '--seed', str(int(e.per_frame.seed)),
                 '--out', str(out_dir / 'eval_frames'), '--regions', *regions])
    ol = e.open_loop
    context = len(ol.context_env_steps)
    eval_rollout(['--recon', str(run), '--episodes', f'0:{int(ol.num_episodes)}', '--context', str(context),
                  '--batch', str(int(ol.batch)), '--out', str(out_dir / 'eval_rollout'), '--regions', *regions])
    frames = json.loads((out_dir / 'eval_frames' / 'metrics.json').read_text())['errors']
    rollout = json.loads((out_dir / 'eval_rollout' / 'rollout.json').read_text())
    steps, curve = rollout['meta']['env_steps'], rollout['mean_per_frame']
    metrics = {'nn_mean': frames['nn']['mean'], 'nn_median': frames['nn']['median'], 'nn_p90': frames['nn']['p90'],
               'outlier_pct': frames['outliers_pct']['mean'], 'cube_err': frames['object_pos']['mean'],
               'shuffled_nn': frames['nn_shuffled']['mean']}
    for name, key in (('imag_nn', 'nn/pred'), ('true_nn', 'nn/true'), ('other_actions_nn', 'nn/wrong'),
                      ('imag_cube', 'object_pos/pred')):
        for s in ol.report_env_steps:
            metrics[f'{name}@{int(s)}'] = curve[key][steps.index(int(s))]
    predicted = [k for k, s in enumerate(steps) if int(ol.context_env_steps[-1]) < s <= int(ol.horizon_env_steps)]
    metrics['imag_nn_max'] = max(curve['nn/pred'][k] for k in predicted)
    np.savez(out_dir / 'curves.npz', env_step=np.asarray(steps), imag_nn=np.asarray(curve['nn/pred']),
             true_nn=np.asarray(curve['nn/true']), imag_cube=np.asarray(curve['object_pos/pred']),
             true_cube=np.asarray(curve['object_pos/true']))
    return metrics
