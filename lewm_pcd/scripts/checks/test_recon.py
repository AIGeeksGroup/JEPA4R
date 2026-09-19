"""Checks for the reconstruction module (CPU, tiny world models, synthetic datasets in a temp STABLEWM_HOME).

  1. losses      Chamfer / sliced Wasserstein vanish on identical sets; the sliced term moves points to an
                 under-populated region where Chamfer alone leaves them; unequal set sizes; the mixture likelihood
                 equals torch.distributions, chunked == unchunked.
  2. decoders    output shapes with and without colour, loss, to_points, gradients.
  3. checkpoints a 2D, a 3D and a 4D world model saved the way train.py saves them -> load_world_model gives
                 latent1 (B, T, 192), stays in eval under .train(), takes no gradient; input_pipeline returns the
                 columns / preprocessor of that run.
                 recon.rollout.open_loop reproduces each model's own planner rollout (JEPA.rollout, AggJEPA.rollout).
  4. training    train_recon.py end to end: 3D + point, 4D + gaussian, 2D + point on a row-aligned pair;
                 a pair that is not the same rows is refused.
  5. evaluation  the object readout finds an object of ANY colour from the data alone (NaN without colour);
                 eval_frames.py, eval_rollout.py, render_episodes.py run on those runs, 2D pair included.

    python scripts/checks/test_recon.py
"""
import os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TMP = Path(tempfile.mkdtemp(prefix='test_recon_'))
os.environ.update(STABLEWM_HOME=str(TMP), SPT_CACHE_DIR=str(TMP / 'spt'), HYDRA_FULL_ERROR='1')
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party' / 'stable-worldmodel')]

import h5py, numpy as np, torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from stable_worldmodel.wm.utils import load_pretrained, save_pretrained

from recon.decoders import GaussianDecoder, PointDecoder
from recon.losses import chamfer, gaussian_nll, nn_distance, quat_to_rot, sample_gaussians, sliced_wasserstein
from recon import metrics as M
from recon.model import input_pipeline, load_world_model
from recon.rollout import open_loop

torch.manual_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------------------------- 1. losses
x = torch.randn(2, 300, 6)
assert float(chamfer(x, x)[0]) < 1e-10 and float(sliced_wasserstein(x, x)) < 1e-10
assert torch.isfinite(sliced_wasserstein(torch.randn(2, 250, 3), torch.randn(2, 300, 3)))


def blobs(n_left, n_right):
    c = torch.cat([torch.full((n_left,), -1.0), torch.full((n_right,), 1.0)])
    return torch.stack([c, torch.zeros_like(c), torch.zeros_like(c)], -1)[None] + 0.05 * torch.randn(1, n_left + n_right, 3)


target = blobs(300, 300)
for use_sw in (False, True):  # 570 / 30 decoded points for a 300 / 300 target, optimised directly
    p = blobs(570, 30).requires_grad_(True)
    opt = torch.optim.Adam([p], lr=0.02)
    for _ in range(300):
        loss = chamfer(p, target)[0] + (sliced_wasserstein(p, target, 50) if use_sw else 0.0)
        opt.zero_grad(); loss.backward(); opt.step()
    right = int((p[0, :, 0] > 0).sum())
    print(f'1. allocation  300/300 target, 570/30 start -> right blob after {"Chamfer + sliced W." if use_sw else "Chamfer alone      "}: {right}')
    assert (abs(right - 300) <= 15) if use_sw else (right <= 60)

B, N, K = 3, 40, 5
g = dict(mu=torch.randn(B, K, 3), log_s=torch.randn(B, K, 3) * 0.3 - 1, quat=torch.nn.functional.normalize(torch.randn(B, K, 4), dim=-1),
         weight_logit=torch.randn(B, K), rgb=torch.rand(B, K, 3))
pts = torch.cat([torch.randn(B, N, 3), torch.rand(B, N, 3)], -1)
R = quat_to_rot(g['quat'])
cov = R @ torch.diag_embed(g['log_s'].exp() ** 2) @ R.transpose(-1, -2)
D = torch.distributions
ref = D.MixtureSameFamily(D.Categorical(logits=g['weight_logit']), D.MultivariateNormal(g['mu'], cov))
ref_nll = -ref.log_prob(pts[..., :3].transpose(0, 1)).mean()
nll = gaussian_nll(g, pts, chunk=B)[1]['nll']
assert torch.allclose(nll, ref_nll, atol=1e-4), (nll, ref_nll)
gg = {k: v.clone().requires_grad_(True) for k, v in g.items()}
l1 = gaussian_nll(gg, pts, chunk=1)[0]; l1.backward()
assert torch.allclose(l1, gaussian_nll(g, pts, chunk=B)[0], atol=1e-5) and all(v.grad is not None for v in gg.values())
assert sample_gaussians(g, 77).shape == (B, 77, 6) and sample_gaussians({**g, 'rgb': None}, 77).shape == (B, 77, 3)
print(f'1. losses      mixture NLL {float(nll):.4f} == torch.distributions {float(ref_nll):.4f}; chunked == unchunked')

# ---------------------------------------------------------------------------------------------- 2. decoders
small = dict(dim=32, hidden=64, depth=2, query_dim=16)
z = torch.randn(4, 192)
for colour in (True, False):
    tgt = torch.rand(4, 1200, 6 if colour else 3)
    for dec in (PointDecoder(num_points=1200, out_dim=6 if colour else 3, **small), GaussianDecoder(num_gauss=64, color=colour, **small)):
        out = dec(z)
        loss, logs = dec.loss(out, tgt)
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None for p in dec.parameters())
        assert dec.to_points(out, 1200).shape == tgt.shape and nn_distance(dec.to_points(out, 1200), tgt).shape == (4,)
        assert ('col' in logs) == colour
print('2. decoders    point / gaussian, with and without colour: shapes, loss, gradients ok')

# ---------------------------------------------------------------------------------------------- 3. checkpoints
FORMS = {   # name: (config name, overrides) -- the project's own model configs, shrunk
    'wm2d': ('lewm', ['data=ogb', 'model=lewm', 'data.dataset.name=pix.h5']),
    'wm3d': ('lewm', ['data=ogb_pcd', 'model=lewm_pcd', 'model/encoder=dexart', 'model.encoder.mlp_spec=[32,64]', 'data.dataset.name=pcd.h5']),
    'wm4d': ('lewm_agg', ['data=ogb_pcd', 'model/encoder=dexart', 'model.encoder.mlp_spec=[32,64]', 'model.aggregator.depth=1', 'data.dataset.name=pcd.h5']),
}
OBS = {'wm2d': ('pixels', (2, 3, 3, 224, 224)), 'wm3d': ('pcd', (2, 3, 1200, 6)), 'wm4d': ('pcd', (2, 3, 1200, 6))}
for name, (config_name, overrides) in FORMS.items():
    with initialize_config_dir(str(ROOT / 'config' / 'train'), version_base=None):
        cfg = compose(config_name, overrides + ['model.predictor.depth=1', 'model.action_encoder.input_dim=25', f'subdir={name}'])
    save_pretrained(instantiate(cfg.model), run_name=name, config=cfg.model, filename='weights_epoch_1.pt')
    OmegaConf.save(cfg, TMP / 'checkpoints' / name / 'config.yaml')

    wm = load_world_model(f'{name}/weights_epoch_1.pt')
    key, shape = OBS[name]
    assert wm.obs_key == key
    wm.train()
    assert not any(m.training for m in wm.modules()), 'frozen world model left eval mode'
    before = {k: v.clone() for k, v in wm.state_dict().items()}
    latent = wm(torch.rand(*shape))
    assert latent.shape == (2, 3, 192) and not latent.requires_grad
    assert all(torch.equal(v, before[k]) for k, v in wm.state_dict().items()), 'a forward changed the frozen model (BatchNorm statistics?)'
    assert not any(p.requires_grad for p in wm.parameters())
    keys, pre = input_pipeline(OmegaConf.create({'ckpt': f'{name}/weights_epoch_1.pt'}))
    assert keys == (['pixels'] if key == 'pixels' else ['pcd_xyz', 'pcd_rgb']), keys
    # open loop == the model's own planner rollout: 3 given frames, 4 more action blocks
    model, H, T = wm.world_model, 3, 4
    obs, blocks = torch.rand(1, H, *shape[2:]), torch.randn(1, H - 1 + T, 25)
    true = torch.cat([wm(obs), torch.zeros(1, T, 192)], 1)                     # later real latents are never read
    mine = open_loop(model, true, blocks, context=H, history=3)[:, H:]
    info = {key: obs[:, None]}
    with torch.no_grad():
        if hasattr(model, 'aggregate'):
            info['action_history'] = blocks[:, None, :H - 1]
            theirs = model.rollout(info, blocks[:, None, H - 1:])['predicted_emb'][:, 0, H:]
        else:
            theirs = model.rollout(info, blocks[:, None], history_size=3)['predicted_emb'][:, 0, H:]
    assert mine.shape == theirs.shape == (1, T, 192) and torch.allclose(mine, theirs, atol=1e-5), (mine - theirs).abs().max()
    print(f'3. checkpoint  {name}: {type(wm.world_model).__name__}, obs_key {key}, latent {tuple(latent.shape)}, input columns {keys}, {type(pre).__name__}')

# ---------------------------------------------------------------------------------------------- 4. training
rng = np.random.default_rng(0)
E, T = 3, 26                                                                   # 6 model frames at frameskip 5
rows = E * T
common = dict(ep_len=np.full(E, T, np.int32), ep_offset=np.arange(E, dtype=np.int64) * T, qpos=rng.normal(size=(rows, 21)),
              action=rng.normal(size=(rows, 5)).astype(np.float32), privileged_block_0_pos=rng.random((rows, 3)) * 0.2 + [0.3, -0.1, 0.02])
clouds = dict(pcd_xyz=(rng.random((rows, 1200, 3)) * 0.4 + [0.2, -0.2, 0.0]).astype(np.float32),
              pcd_rgb=rng.integers(0, 255, (rows, 1200, 3), dtype=np.uint8))
files = {'pcd': {**common, **clouds}, 'pix': {**common, 'pixels': rng.integers(0, 255, (rows, 224, 224, 3), dtype=np.uint8)},
         'pcd_other': {**common, **clouds, 'qpos': rng.normal(size=(rows, 21))}}   # same layout, different episodes
(TMP / 'datasets').mkdir(exist_ok=True)
for name, cols in files.items():
    with h5py.File(TMP / 'datasets' / f'{name}.h5', 'w') as f:
        for k, v in cols.items():
            f.create_dataset(k, data=v)


def train(ckpt, run, *overrides):
    cmd = [sys.executable, str(ROOT / 'train_recon.py'), f'ckpt={ckpt}/weights_epoch_1.pt', f'subdir={run}', f'output_model_name={run}',
           'trainer.accelerator=cpu', 'trainer.precision=32', 'trainer.max_epochs=1', '+trainer.limit_train_batches=2', '+trainer.limit_val_batches=1',
           'loader.batch_size=4', 'num_workers=0', 'loader.persistent_workers=False', 'loader.prefetch_factor=null',
           'decoder.dim=32', 'decoder.hidden=64', *overrides]
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path[:2] + [os.environ.get('PYTHONPATH', '')])}
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


RUNS = [('wm3d', 'r3d_point', ['data.input=pcd.h5']),
        ('wm4d', 'r4d_gaussian', ['data.input=pcd.h5', 'decoder=gaussian', 'decoder.num_gauss=64']),
        ('wm2d', 'r2d_point', ['data=ogb_pix_pcd', 'data.input=pix.h5', 'data.target=pcd.h5'])]
for ckpt, run, overrides in RUNS:
    r = train(ckpt, run, *overrides)
    out = TMP / 'checkpoints' / 'recon' / run
    assert r.returncode == 0 and (out / 'weights_epoch_1.pt').exists() and (out / 'config.yaml').exists(), (r.stdout + r.stderr)[-3000:]
    dec = load_pretrained(f'recon/{run}/weights_epoch_1.pt')   # config.json names the decoder class: loads like any checkpoint
    assert not any(k.startswith('world_model') for k in dec.state_dict()), 'the frozen world model leaked into the decoder checkpoint'
    print(f'4. training    {ckpt} -> {run}: ok, reloads as {type(dec).__name__} ({sum(p.numel() for p in dec.parameters())} parameters)')

r = train('wm2d', 'r2d_refused', 'data=ogb_pix_pcd', 'data.input=pix.h5', 'data.target=pcd_other.h5')
assert r.returncode != 0 and 'not the same rows' in r.stdout + r.stderr, (r.stdout + r.stderr)[-3000:]
print('4. training    a point-cloud dataset of other episodes is refused')

# ---------------------------------------------------------------------------------------------- 5. evaluation
for colour in ([0.9, 0.1, 0.1], [0.1, 0.2, 0.9]):                              # a red object, then a blue one: no code knows either
    centre = torch.tensor([[0.4, 0.1, 0.05], [0.5, -0.2, 0.05]])
    scene = torch.cat([torch.rand(2, 900, 3) * 0.6, torch.full((2, 900, 3), 0.5) + 0.05 * torch.randn(2, 900, 3)], -1)
    thing = torch.cat([centre[:, None] + 0.01 * torch.randn(2, 300, 3), torch.tensor(colour).expand(2, 300, 3) + 0.02 * torch.randn(2, 300, 3)], -1)
    cloud = torch.cat([scene, thing], 1)
    signature, elsewhere = M.colour_signature(cloud, centre)
    assert elsewhere < 0.01 and (M.object_centroid(cloud, signature) - centre).norm(dim=-1).max() < 0.005
    assert M.colour_signature(cloud[..., :3], centre) == (None, None) and torch.isnan(M.object_centroid(cloud[..., :3], None)).all()
    err, count = M.region_error(cloud, cloud, centre, 0.03)
    assert err.max() < 1e-6 and torch.allclose(count, torch.ones(2))
print('5. readout     object found by a colour measured on the data (red, blue); NaN without colour; region error ok')


def script(name, *args):
    cmd = [sys.executable, str(ROOT / 'scripts' / 'recon' / name), '--device', 'cpu', *args]
    r = subprocess.run(cmd, env={**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path[:2])}, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]


RECON = TMP / 'checkpoints' / 'recon'
for run in ('r3d_point', 'r4d_gaussian', 'r2d_point'):
    script('eval_frames.py', '--recon', run, '--frames', '8', '--batch', '4')
    script('eval_rollout.py', '--recon', run, '--episodes', '0:2', '--batch', '2')
    assert (RECON / run / 'eval_frames' / 'metrics.json').exists() and (RECON / run / 'eval_rollout' / 'rollout.png').exists()
script('render_episodes.py', '--recon', 'r3d_point', 'r4d_gaussian', '--episodes', '1', '--stride', '5', '--size', '64', '--out', str(TMP / 'render'))
script('render_episodes.py', '--recon', 'r2d_point', '--episodes', '0', '--mode', 'openloop', '--size', '64')
assert (TMP / 'render' / 'ep1_recon.mp4').exists() and (RECON / 'r2d_point' / 'render' / 'ep0_openloop.mp4').exists()
print('5. evaluation  eval_frames / eval_rollout / render_episodes ran on the 3D, 4D and 2D runs')
print('ALL OK')
