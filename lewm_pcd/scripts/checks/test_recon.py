"""Checks for the reconstruction module around the decoder (CPU, tiny world models, synthetic datasets in a temp
STABLEWM_HOME). The PAGE-4D decoder itself, training and the evaluation scripts: scripts/checks/check_page4d.py (GPU).

  1. losses      Chamfer / sliced Wasserstein vanish on identical sets; the sliced term moves points to an
                 under-populated region where Chamfer alone leaves them; unequal set sizes.
  2. checkpoints a 2D, a 3D and a 4D world model saved the way train.py saves them -> load_world_model gives
                 latent1 (B, T, 192), stays in eval under .train(), takes no gradient; input_pipeline returns the
                 columns / preprocessor of that run.
                 recon.rollout.open_loop reproduces each model's own planner rollout (JEPA.rollout, AggJEPA.rollout).
  3. data        one point-cloud dataset -> encoder input + target of the same frame; a pixel dataset + the
                 point clouds of the same rows -> a pair; a point-cloud dataset of other episodes is refused.
  4. readout     the object readout finds an object of ANY colour from the data alone (NaN without colour).

    python scripts/checks/test_recon.py
"""
import os, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TMP = Path(tempfile.mkdtemp(prefix='test_recon_'))
os.environ.update(STABLEWM_HOME=str(TMP), SPT_CACHE_DIR=str(TMP / 'spt'), HYDRA_FULL_ERROR='1')
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party' / 'stable-worldmodel')]

import h5py, numpy as np, torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from stable_worldmodel.wm.utils import save_pretrained

from recon.data import build_dataset
from recon.losses import chamfer, sliced_wasserstein
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

# ---------------------------------------------------------------------------------------------- 2. checkpoints
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
    print(f'2. checkpoint  {name}: {type(wm.world_model).__name__}, obs_key {key}, latent {tuple(latent.shape)}, input columns {keys}, {type(pre).__name__}')

# ---------------------------------------------------------------------------------------------- 3. data
rng = np.random.default_rng(0)
E, T = 3, 26
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


def frames(ckpt, *overrides):
    with initialize_config_dir(str(ROOT / 'config' / 'recon'), version_base=None):
        cfg = compose('recon', [f'ckpt={ckpt}/weights_epoch_1.pt', *overrides])
    keys, pre = input_pipeline(cfg)
    return build_dataset(cfg.data, load_world_model(cfg.ckpt).obs_key, keys, pre)


one = frames('wm3d', 'data.input=pcd.h5')[5]
assert one['pcd'].shape == (1, 1200, 6) and one['target'].shape == (1, 1200, 6), {k: getattr(v, 'shape', v) for k, v in one.items()}
two = frames('wm2d', 'data=ogb_pix_pcd', 'data.input=pix.h5', 'data.target=pcd.h5')
assert len(two) == rows and two[5]['pixels'].shape[:2] == (1, 3) and two[5]['target'].shape == (1, 1200, 6)
try:
    frames('wm2d', 'data=ogb_pix_pcd', 'data.input=pix.h5', 'data.target=pcd_other.h5')
    raise RuntimeError('a point-cloud dataset of other episodes was accepted')
except AssertionError as e:
    assert 'not the same rows' in str(e), e
print('3. data        3D: input + target of the same frame; 2D + the point clouds of the same rows: a pair; other episodes: refused')

# ---------------------------------------------------------------------------------------------- 4. readout
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
print('4. readout     object found by a colour measured on the data (red, blue); NaN without colour; region error ok')
print('ALL OK')
