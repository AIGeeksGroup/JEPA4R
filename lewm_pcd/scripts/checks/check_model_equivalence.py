"""Upstream jepa.py vs new jepa.py + encoders.ViTCLS: identical embeddings and predictions."""
import importlib.util, sys, torch
sys.path.insert(0, '/root/autodl-tmp/lewm234d/lewm_pcd')
from jepa import JEPA as NewJEPA
from module import ARPredictor, Embedder, MLP
from encoders import ViTCLS
spec = importlib.util.spec_from_file_location('jepa_old', '/root/autodl-tmp/lewm234d/jepa.py')
jepa_old = importlib.util.module_from_spec(spec); spec.loader.exec_module(jepa_old)

torch.manual_seed(0)
enc = ViTCLS('tiny', patch_size=14, image_size=224, pretrained=False, use_mask_token=False)
pred = ARPredictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192, depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1)
act = Embedder(input_dim=25, emb_dim=192)
proj = MLP(192, 2048, 192, norm_fn=torch.nn.BatchNorm1d)
pp = MLP(192, 2048, 192, norm_fn=torch.nn.BatchNorm1d)
new = NewJEPA(enc, pred, act, proj, pp).eval()          # new code, adapter
old = jepa_old.JEPA(enc.vit, pred, act, proj, pp).eval()  # upstream code, raw ViT, same weights

batch = {'pixels': torch.rand(2, 4, 3, 224, 224), 'action': torch.randn(2, 4, 25)}
with torch.no_grad():
    o, n = old.encode(dict(batch)), new.encode(dict(batch))
    po = old.predict(o['emb'][:, :3], o['act_emb'][:, :3])
    pn = new.predict(n['emb'][:, :3], n['act_emb'][:, :3])
print('emb identical:', torch.equal(o['emb'], n['emb']), '| act_emb identical:', torch.equal(o['act_emb'], n['act_emb']),
      '| pred identical:', torch.equal(po, pn), '| emb shape:', tuple(n['emb'].shape))
ko, kn = set(old.state_dict()), set(new.state_dict())
print('state_dict keys equal up to the encoder.vit. prefix:', {k.replace('encoder.vit.', 'encoder.') for k in kn} == ko)
