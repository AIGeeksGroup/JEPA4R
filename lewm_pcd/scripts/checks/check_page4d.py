"""Checks of the PAGE-4D decoder with the released weights (GPU; $STABLEWM_HOME/checkpoints/page4d/checkpoint_nomask.pt,
see README), and of training + evaluation end to end on synthetic data.

  1. splice      the decoder's token path, fed with PAGE-4D's own patch tokens of an image, gives PAGE-4D's own point
                 map (the output affine undone): the splice drops nothing but the pixel -> token conv.
  2. modes       frozen: only generator / frame / rgb train; lora: + 72 q / v adapters, and at initialisation its output
                 equals the frozen decoder's; gradients reach the adapters and no pretrained weight; checkpoints hold only
                 what trains and reload; init_from picks up the new parts of a stage-A checkpoint.
  3. end to end  train_recon.py stage A then stage B (2 steps each) on a tiny 3D world model and synthetic point clouds
                 in a temp STABLEWM_HOME; load_pretrained; eval_frames / eval_rollout / render_episodes on stage B.

    python scripts/checks/check_page4d.py
"""
import os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party' / 'stable-worldmodel')]
os.environ.setdefault('STABLEWM_HOME', str(ROOT.parent / 'lewm_pcd_data'))

import h5py, numpy as np, torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from stable_worldmodel.wm.utils import load_pretrained, save_pretrained

from recon.decoders.page4d import ADAPTER_KEYS, PAGE4DDecoder, checkpoint_path, load_page4d

WEIGHTS = checkpoint_path('page4d/checkpoint_nomask.pt')              # the real STABLEWM_HOME, before the temp one of part 3
assert WEIGHTS.is_file(), f'{WEIGHTS} not found: see README (3D reconstruction)'
NEW = ('generator', 'frame', 'rgb')
trained = lambda k: k.split('.')[0] in NEW or any(t in k for t in ADAPTER_KEYS)
torch.manual_seed(0)
dev = 'cuda'

# ---------------------------------------------------------------------------------------------- 1. splice
ref = load_page4d(WEIGHTS).to(dev).eval()
frozen = PAGE4DDecoder(weights=WEIGHTS, train='frozen').to(dev).eval()
img = torch.rand(2, 1, 3, 224, 224, device=dev)
with torch.no_grad():
    want = ref(img)['world_points'][:, 0].float()                            # (2, 224, 224, 3), PAGE-4D's own forward
    a = ref.aggregator
    tokens = a.patch_embed.patch_embed(((img - a._resnet_mean) / a._resnet_std).view(2, 3, 224, 224))
    got, _ = frozen.tokens_to_map(tokens)
    got = ((got[..., :3] - frozen.frame.bias) @ torch.linalg.inv(frozen.frame.weight).T).view(2, 224, 224, 3)
diff = (got - want).abs()
print(f'1. splice      spliced path vs PAGE-4D itself (fp32): max |diff| {float(diff.max()):.2e}, mean {float(diff.mean()):.2e}, output scale {float(want.abs().mean()):.3f}')
assert float(diff.mean()) < 1e-4 * float(want.abs().mean())
del ref; torch.cuda.empty_cache()

# ---------------------------------------------------------------------------------------------- 2. modes
lora = PAGE4DDecoder(weights=WEIGHTS, train='lora').to(dev)
lora.load_state_dict(frozen.state_dict(), strict=False)                     # same new parts; the adapters start at zero
z, tgt = torch.randn(2, 192, device=dev), torch.rand(2, 1200, 6, device=dev)
with torch.no_grad():
    assert torch.allclose(lora.eval()(z), frozen(z), atol=1e-5), 'LoRA at initialisation changes the output'
tr = lambda m: [k for k, p in m.named_parameters() if p.requires_grad]
assert {k.split('.')[0] for k in tr(frozen)} == set(NEW) and all(trained(k) for k in tr(lora)) and len(lora.adapters) == 72
lora.train()
lora.loss(lora(z), tgt)[0].backward()
assert all(p.grad is not None for w in lora.adapters for p in (w.down.weight, w.up_q.weight, w.up_v.weight))
assert all(p.grad is None for p in lora.parameters() if not p.requires_grad)
for m in (frozen, lora):
    sd = m.state_dict()
    assert sd and all(trained(k) for k in sd), [k for k in sd if not trained(k)][:5]
n_adapter = sum(p.numel() for w in lora.adapters for p in w.parameters() if p.requires_grad)
print(f'2. modes       frozen trains {sum(p.numel() for p in frozen.parameters() if p.requires_grad) / 1e6:.2f}M, lora + {len(lora.adapters)} adapters '
      f'({n_adapter / 1e6:.2f}M); LoRA at init == frozen; gradients reach the adapters only; checkpoints hold only what trains')
with tempfile.TemporaryDirectory() as d:
    torch.save(frozen.state_dict(), Path(d) / 'a.pt')
    del lora; torch.cuda.empty_cache()
    b = PAGE4DDecoder(weights=WEIGHTS, train='lora', init_from=str(Path(d) / 'a.pt'))
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in frozen.cpu().state_dict().items()), 'init_from did not pick up the stage-A parts'
print('2. modes       init_from: a lora decoder starts from the stage-A new parts')
del frozen, b; torch.cuda.empty_cache()

# ---------------------------------------------------------------------------------------------- 3. end to end
TMP = Path(tempfile.mkdtemp(prefix='check_page4d_'))
os.environ.update(STABLEWM_HOME=str(TMP), SPT_CACHE_DIR=str(TMP / 'spt'), HYDRA_FULL_ERROR='1')
with initialize_config_dir(str(ROOT / 'config' / 'train'), version_base=None):
    cfg = compose('lewm', ['data=ogb_pcd', 'model=lewm_pcd', 'model/encoder=dexart', 'model.encoder.mlp_spec=[32,64]', 'data.dataset.name=pcd.h5',
                           'model.predictor.depth=1', 'model.action_encoder.input_dim=25', 'subdir=wm3d'])
save_pretrained(instantiate(cfg.model), run_name='wm3d', config=cfg.model, filename='weights_epoch_1.pt')
OmegaConf.save(cfg, TMP / 'checkpoints' / 'wm3d' / 'config.yaml')
rng = np.random.default_rng(0)
E, T = 3, 26                                                                   # 6 model frames at frameskip 5
rows = E * T
(TMP / 'datasets').mkdir(exist_ok=True)
with h5py.File(TMP / 'datasets' / 'pcd.h5', 'w') as f:
    for k, v in dict(ep_len=np.full(E, T, np.int32), ep_offset=np.arange(E, dtype=np.int64) * T, qpos=rng.normal(size=(rows, 21)),
                     action=rng.normal(size=(rows, 5)).astype(np.float32), privileged_block_0_pos=rng.random((rows, 3)) * 0.2 + [0.3, -0.1, 0.02],
                     pcd_xyz=(rng.random((rows, 1200, 3)) * 0.4 + [0.2, -0.2, 0.0]).astype(np.float32),
                     pcd_rgb=rng.integers(0, 255, (rows, 1200, 3), dtype=np.uint8)).items():
        f.create_dataset(k, data=v)
env = {**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path[:2])}


def run(*cmd):
    r = subprocess.run([sys.executable, *map(str, cmd)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    return r.stdout


SMALL = ['ckpt=wm3d/weights_epoch_1.pt', 'data.input=pcd.h5', f'decoder.weights={WEIGHTS}', 'trainer.limit_train_batches=2', 'trainer.limit_val_batches=1',
         'loader.batch_size=2', 'num_workers=0', 'loader.persistent_workers=False', 'loader.prefetch_factor=null']
run(ROOT / 'train_recon.py', *SMALL, 'exp=frozen', 'subdir=ra', 'output_model_name=ra')
out = run(ROOT / 'train_recon.py', *SMALL, 'exp=lora', 'decoder.init_from=recon/ra/weights_epoch_1.pt', 'subdir=rb', 'output_model_name=rb')
assert '[page4d] initialised' in out, out[-2000:]
dec = load_pretrained('recon/rb/weights_epoch_1.pt')                         # config.json names the decoder class
assert dec.train_mode == 'lora' and all(trained(k) for k in dec.state_dict()), 'the checkpoint holds pretrained or world-model weights'
print(f'3. training    stage A -> stage B (init_from A) on a 3D world model: ok, reloads as {type(dec).__name__}')
del dec
RECON = TMP / 'checkpoints' / 'recon'
run(ROOT / 'scripts' / 'recon' / 'eval_frames.py', '--recon', 'rb', '--frames', '8', '--batch', '4')
run(ROOT / 'scripts' / 'recon' / 'eval_rollout.py', '--recon', 'rb', '--episodes', '0:2', '--batch', '2')
run(ROOT / 'scripts' / 'recon' / 'render_episodes.py', '--recon', 'ra', 'rb', '--episodes', '1', '--stride', '5', '--size', '64', '--out', TMP / 'render')
run(ROOT / 'scripts' / 'recon' / 'render_episodes.py', '--recon', 'rb', '--episodes', '0', '--mode', 'openloop', '--size', '64')
assert (RECON / 'rb' / 'eval_frames' / 'metrics.json').exists() and (RECON / 'rb' / 'eval_rollout' / 'rollout.png').exists()
assert (TMP / 'render' / 'ep1_recon.mp4').exists() and (RECON / 'rb' / 'render' / 'ep0_openloop.mp4').exists()
print('3. evaluation  eval_frames / eval_rollout / render_episodes ran on the stage-B run')
print('ALL OK')
