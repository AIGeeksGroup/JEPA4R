"""Single-frame reconstruction errors of a trained reconstruction run: encode a real frame with the frozen world
model, decode its latent1, compare with the real point cloud of that frame. All distances in cm.

  nn                two-sided mean nearest-neighbour distance (the training readout ``nn_cm``)
  nn_spurious       decoded -> nearest real point: geometry that is not there
  nn_missing        real -> nearest decoded point: geometry that was not decoded
  outliers_pct      share of decoded points farther than --outlier cm from every real point
  sliced_w          root sliced-Wasserstein distance: shape AND how many points each part got
  colour            L1 between the rgb of matched pairs (clouds with colour)
  <region>_missing  nn_missing restricted to the real points around a position column of the dataset (--regions)
  <region>_count    decoded / real number of points in that region (1 = right amount)
  object_pos        distance between the object's centroid read from the decoded cloud and from the real cloud
                    (both by the object's colour, measured on the real clouds; like-for-like surface centroids)
  object_pos_sim    the same decoded centroid against the simulator's object centre (includes the readout's own bias:
                    see object_pos_real_sim, the real cloud's centroid against the simulator)
  *_shuffled        the same error with the latent of ANOTHER frame: what a decoder that ignores its input would score

    python scripts/recon/eval_frames.py --recon cube_pcd_ppt_point
    python scripts/recon/eval_frames.py --recon <run> --input <pixels>.h5 --target <point clouds of the same rows>.h5 --split all
Frames: the run's own validation split (train_recon.py's seed) or, with --split all / other datasets, any frame."""
import argparse, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party' / 'stable-worldmodel')]

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch

from recon import metrics as M
from recon.model import open_run


def main():
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
    args = ap.parse_args()
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


if __name__ == '__main__':
    main()
