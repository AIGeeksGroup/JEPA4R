"""Probes of OGBench puzzle-3x3: does a frozen encoder's representation hold the lights (the 9 button states) and the
gripper position (effector xyz)?
* Encoders: LeWM 2D (ViT-tiny) and 3D (DexArt PointNet) at the given checkpoints / epoch, and the same architectures
  randomly initialised (control: what random features carry). Loading and preprocessing as CortexBench's
  (bench/cortexbench/lewm_cortex.py): 2D the 224 frame normalised with ImageNet's mean / std; 3D (xyz - center) * scale
  with rgb / 255.
* Features: the projector output (192-d, what LeWM's predictor and CortexBench use), and the encoder CLS (before the
  projector, 192-d).
* Data: probe training on the training set's frames (every 10th, ~20 k frames; the encoder trained on these episodes);
  test on the separate test episodes (every 2nd frame, ~10 k frames; never seen by the encoder).
* Probes: linear (a logistic regression per button; ridge regression for xyz, regularisation by cross-validation) and a
  two-layer MLP (256 hidden, buttons and xyz together, 10 % of the probe training set held out to pick the epoch).
  Features standardised on the probe training set.
* Metrics (test set): buttons: mean accuracy over the 9, the share of frames with all 9 right, and the accuracy split by
  whether the button cap is visible or completely hidden by the arm (a hidden cap cannot be read from one frame);
  gripper: mean R2 over xyz and the error (cm). Chance: per button, the test set's more frequent state.
    python scripts/checks/probe_puzzle.py --ckpt-2d $STABLEWM_HOME/checkpoints/puzzle_3x3_small_pix \\
        --ckpt-3d $STABLEWM_HOME/checkpoints/puzzle_3x3_small_pcd_dexart --train-set ogbench/puzzle_3x3_noisy \\
        --test-set ogbench/puzzle_3x3_noisy_small_test --out probe_results
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
L = HERE.parents[1]  # lewm_pcd
sys.path.insert(0, str(L / 'bench' / 'cortexbench'))
from hydra.utils import instantiate  # noqa: E402
from lewm_cortex import IMAGENET, PcdTransform, ViTCLS, _dexart_pointnet_class  # noqa: E402

D = Path(os.environ.get('STABLEWM_HOME', '.')) / 'datasets'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
CK_DIR = {}  # '2d' / '3d' -> checkpoint directory (--ckpt-2d / --ckpt-3d)


class Emb(nn.Module):
    """Encoder + projector, returning both outputs."""

    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder, self.projector = encoder, projector

    def forward(self, x):
        e = self.encoder(x)
        return e, self.projector(e)


def load_2d(epoch, random_init):
    """As lewm_cortex.build_lewm_embedding, returning the CLS too; also accepts ckpt_vit.ViTCLSCkpt (the ViTCLS with
    gradient checkpointing the small run's 2D training used: the same parameters)."""
    ckdir = Path(CK_DIR['2d'])
    cfg = json.loads((ckdir / 'config.json').read_text())
    assert cfg['encoder']['_target_'] in ('encoders.ViTCLS', 'encoders.vit.ViTCLS', 'ckpt_vit.ViTCLSCkpt'), cfg['encoder']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        enc = ViTCLS(**{k: v for k, v in cfg['encoder'].items() if k != '_target_'})
        proj = instantiate(cfg['projector'])
    if not random_init:
        sd = torch.load(ckdir / f'weights_epoch_{epoch}.pt', map_location='cpu')
        enc.load_state_dict({k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')})
        proj.load_state_dict({k[len('projector.'):]: v for k, v in sd.items() if k.startswith('projector.')})
    return Emb(enc, proj).eval()


def load_3d(epoch, random_init):
    ckdir = Path(CK_DIR['3d'])
    cfg = json.loads((ckdir / 'config.json').read_text())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        enc = _dexart_pointnet_class()(**{k: v for k, v in cfg['encoder'].items() if k != '_target_'})
        proj = instantiate(cfg['projector'])
    if not random_init:
        sd = torch.load(ckdir / f'weights_epoch_{epoch}.pt', map_location='cpu')
        enc.load_state_dict({k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')})
        proj.load_state_dict({k[len('projector.'):]: v for k, v in sd.items() if k.startswith('projector.')})
    return Emb(enc, proj).eval()


def rows_of(path, every):
    with h5py.File(path, 'r') as f:
        n = int(f['ep_len'][:].sum())
    return np.arange(0, n, every)


def labels(path, rows):
    with h5py.File(path, 'r') as f:
        return {k: f[k][:][rows] for k in ('button_states', 'effector_pos', 'cap_visible_px', 'ep_idx')}


@torch.no_grad()
def encode(model, path, rows, kind, bs=None):
    bs = bs or (256 if kind == '2d' else 64)  # 3D: DexArt PointNet keeps 4320-d per point, 64 clouds a batch ~1.3 GB
    model = model.to(DEV)
    mean = torch.tensor(IMAGENET[0], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET[1], device=DEV).view(1, 3, 1, 1)
    tf = PcdTransform(center=(0.425, 0.0, 0.2), scale=4.0, use_rgb=True)  # = config/train/data/ogb_pcd.yaml's preprocessor
    E, Z = [], []
    with h5py.File(path, 'r') as f:
        for i in range(0, len(rows), bs):
            r = rows[i: i + bs]
            if kind == '2d':
                x = torch.from_numpy(f['pixels'][r]).to(DEV).permute(0, 3, 1, 2).float() / 255  # r ascending: only the chunks used are read
                x = (x - mean) / std
            else:
                xyz = f['pcd_xyz'][r]
                rgb = f['pcd_rgb'][r]
                x = torch.stack([tf(np.concatenate([a, b], 1)) for a, b in zip(xyz, rgb)]).to(DEV)
            e, z = model(x)
            E.append(e.float().cpu())
            Z.append(z.float().cpu())
    model.cpu()
    return torch.cat(E).numpy(), torch.cat(Z).numpy()


def mlp_probe(Xtr, Btr, Ptr, Xte, seed=0, epochs=100):
    torch.manual_seed(seed)
    n = len(Xtr)
    perm = np.random.default_rng(seed).permutation(n)
    va, tr = perm[: n // 10], perm[n // 10:]
    pm, ps = Ptr[tr].mean(0), Ptr[tr].std(0) + 1e-8
    X = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    B = torch.tensor(Btr, dtype=torch.float32, device=DEV)
    P = torch.tensor((Ptr - pm) / ps, dtype=torch.float32, device=DEV)
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 12)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    bce, best, best_state = nn.BCEWithLogitsLoss(), float('inf'), None
    tr_t = torch.tensor(tr, device=DEV)
    for ep in range(epochs):
        net.train()
        for idx in tr_t[torch.randperm(len(tr_t), device=DEV)].split(512):
            out = net(X[idx])
            loss = bce(out[:, :9], B[idx]) + ((out[:, 9:] - P[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            out = net(X[va])
            vl = (bce(out[:, :9], B[va]) + ((out[:, 9:] - P[va]) ** 2).mean()).item()
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    with torch.no_grad():
        out = net(torch.tensor(Xte, dtype=torch.float32, device=DEV)).cpu().numpy()
    return (out[:, :9] > 0).astype(int), out[:, 9:] * ps + pm


def linear_probe(Xtr, Btr, Ptr, Xte):
    pred_b = np.zeros((len(Xte), 9), int)
    for j in range(9):
        if len(np.unique(Btr[:, j])) < 2:
            pred_b[:, j] = Btr[0, j]
            continue
        clf = LogisticRegression(C=1.0, max_iter=3000)
        clf.fit(Xtr, Btr[:, j])
        pred_b[:, j] = clf.predict(Xte)
    reg = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xtr, Ptr)
    return pred_b, reg.predict(Xte)


def metrics(pred_b, pred_p, te):
    yb, yp, vis = te['button_states'].astype(int), te['effector_pos'], te['cap_visible_px'] > 0
    correct = pred_b == yb
    return {
        'button_acc': float(correct.mean()),
        'all9_acc': float(correct.all(1).mean()),
        'button_acc_visible': float(correct[vis].mean()),
        'button_acc_hidden': float(correct[~vis].mean()) if (~vis).any() else float('nan'),
        'all9_acc_all_visible': float(correct[vis.all(1)].all(1).mean()),  # all 9 right, among frames with all 9 caps visible
        'gripper_r2': float(np.mean([r2_score(yp[:, k], pred_p[:, k]) for k in range(3)])),
        'gripper_err_cm': float(np.linalg.norm(pred_p - yp, axis=1).mean() * 100),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epoch', type=int, default=10)
    ap.add_argument('--out', default='probe_results')
    ap.add_argument('--kinds', default='2d,3d')
    ap.add_argument('--every-train', type=int, default=10)
    ap.add_argument('--every-test', type=int, default=2)
    ap.add_argument('--ckpt-2d', required=True, help='checkpoint directory of the 2D LeWM')
    ap.add_argument('--ckpt-3d', required=True, help='checkpoint directory of the 3D LeWM')
    ap.add_argument('--train-set', default='ogbench/puzzle_3x3_noisy', help='the 2D probe training set under datasets/ (3D: + _pcd)')
    ap.add_argument('--test-set', default='ogbench/puzzle_3x3_noisy_small_test', help='the 2D test set under datasets/ (3D: + _pcd)')
    ap.add_argument('--skip-random', action='store_true', help='no randomly initialised control')
    ap.add_argument('--linear-only', action='store_true', help='linear probes only')
    a = ap.parse_args()
    CK_DIR.update({'2d': a.ckpt_2d, '3d': a.ckpt_3d})
    t0 = time.time()
    p2_tr, p3_tr = D / f'{a.train_set}.h5', D / f'{a.train_set}_pcd.h5'
    p2_te, p3_te = D / f'{a.test_set}.h5', D / f'{a.test_set}_pcd.h5'
    r_tr, r_te = rows_of(p2_tr, a.every_train), rows_of(p2_te, a.every_test)
    tr, te = labels(p2_tr, r_tr), labels(p2_te, r_te)
    yb_te = te['button_states'].astype(int)
    chance = {'button_acc': float(np.mean([max(yb_te[:, j].mean(), 1 - yb_te[:, j].mean()) for j in range(9)])),
              'hidden_share': float((te['cap_visible_px'] == 0).mean()),
              'rows_with_a_hidden_cap': float((te['cap_visible_px'] == 0).any(1).mean())}
    print(f'probe train rows {len(r_tr)}, test rows {len(r_te)}; chance {chance}', flush=True)
    results = {'chance': chance, 'n_train': int(len(r_tr)), 'n_test': int(len(r_te)), 'epoch': a.epoch,
               'ckpt': dict(CK_DIR), 'rows': []}
    for kind, loader, ptr, pte in [('2d', load_2d, p2_tr, p2_te), ('3d', load_3d, p3_tr, p3_te)]:
        if kind not in a.kinds.split(','):
            continue
        for init in (('trained',) if a.skip_random else ('trained', 'random')):
            model = loader(a.epoch, init == 'random')
            Etr, Ztr = encode(model, ptr, r_tr, kind)
            Ete, Zte = encode(model, pte, r_te, kind)
            for feat, Xtr, Xte in (('projector', Ztr, Zte), ('encoder CLS', Etr, Ete)):
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
                Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd
                for probe in (('linear',) if a.linear_only else ('linear', 'mlp')):
                    f = linear_probe if probe == 'linear' else mlp_probe
                    pb, pp = f(Xtr_s, tr['button_states'].astype(int), tr['effector_pos'], Xte_s)
                    m = metrics(pb, pp, te)
                    row = {'encoder': f'LeWM {kind.upper()}', 'weights': init, 'feature': feat, 'probe': probe, **m}
                    results['rows'].append(row)
                    print(json.dumps(row), f'({time.time() - t0:.0f}s)', flush=True)
    out = Path(a.out)
    (out.with_suffix('.json')).write_text(json.dumps(results, indent=1))
    lines = ['| encoder | weights | feature | probe | button accuracy | all 9 right | visible buttons | hidden buttons | gripper R2 | gripper error (cm) |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in results['rows']:
        lines.append(f"| {r['encoder']} | {r['weights']} | {r['feature']} | "
                     f"{r['probe']} | {100 * r['button_acc']:.1f} | {100 * r['all9_acc']:.1f} | "
                     f"{100 * r['button_acc_visible']:.1f} | {100 * r['button_acc_hidden']:.1f} | {r['gripper_r2']:.3f} | "
                     f"{r['gripper_err_cm']:.2f} |")
    lines.append(f"\nchance: button accuracy {100 * chance['button_acc']:.1f} (each button's more frequent state); button caps "
                 f"completely hidden by the arm: {100 * chance['hidden_share']:.1f} % of the test caps, frames with a hidden cap "
                 f"{100 * chance['rows_with_a_hidden_cap']:.1f} %")
    (out.with_suffix('.md')).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
