"""Unit test for AggJEPA.rollout / get_cost (CPU, smoke checkpoint): the framework-rollout copy must equal a plain
reference loop -- window of the last 3 latent1 -> aggregator -> latent2, + the candidate block applied at the newest
frame -> next latent1 -- for H=3 (with the 2 executed history blocks) and H=1 (first plan), 5 predictions each.
Also: get_cost == MSE(latent2 of the last 3 rollout frames, latent2 of the 3-frame goal window) per candidate."""
import os, sys, torch
ROOT = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/lewm234d/lewm_pcd/TMP'
os.environ['STABLEWM_HOME'] = ROOT
sys.path.insert(0, '/root/autodl-tmp/lewm234d/lewm_pcd')
import stable_worldmodel as swm
import jepa_agg  # noqa: registers the class for loading

torch.manual_seed(0)
model = swm.wm.utils.load_pretrained('lewm_agg_smoke/weights_epoch_1.pt').eval().float()
from omegaconf import OmegaConf
A = int(OmegaConf.load(f'{ROOT}/checkpoints/lewm_agg_smoke/config.yaml').model.action_encoder.input_dim)  # blocked action dim (5 x 5)
N, C, D, W, T, S = 1200, 6, 192, model.window, 5, 4
print(f'window={W} blocked action dim={A} cost_space={model.cost_space}')


def reference(frames, act_hist, cands):
    """frames (H, N, C); act_hist (H-1, A); cands (T, A) -> (H+T, D) latent1 list (context then predictions)."""
    with torch.no_grad():
        emb = model.encode({'pcd': frames[None]})['emb'][0]            # (H, D)
        act_emb = model.action_encoder(torch.cat([act_hist, cands], 0)[None])[0]  # (H-1+T, A_emb)
        H = frames.size(0)
        embs = list(emb.unbind(0))
        for t in range(cands.size(0)):
            latent2 = model.aggregator(torch.stack(embs[-W:], 0)[None])[0]     # last <=W frames
            embs.append(model.predict(latent2[None], act_emb[None, H - 1 + t])[0])  # block t leaves the newest frame
        return torch.stack(embs, 0)


ok = True
for H in (3, 2, 1):
    frames = torch.rand(H, N, C); frames[..., :3] = frames[..., :3] * 2 - 1
    act_hist = torch.randn(H - 1, A)
    cands = torch.randn(S, T, A)
    info = {'pcd': frames[None, None].expand(1, S, H, N, C).clone(), 'action': torch.zeros(1, S, 1, A)}
    if H > 1:
        info['action_history'] = act_hist[None, None].expand(1, S, H - 1, A).clone()
    with torch.no_grad():
        out = model.rollout(dict(info), cands[None])
    pe = out['predicted_emb'][0]                                          # (S, H+T, D)
    ref = torch.stack([reference(frames, act_hist, cands[s]) for s in range(S)], 0)
    err = (pe - ref).abs().max().item()
    good = pe.shape == (S, H + T, D) and err < 1e-4
    ok &= good
    print(f'H={H}: predicted_emb {tuple(pe.shape)} (expect {(S, H + T, D)}), max|rollout - reference| = {err:.2e} -> {"OK" if good else "FAIL"}')

    # get_cost: goal window of W frames
    goal = torch.rand(W, N, C); goal[..., :3] = goal[..., :3] * 2 - 1
    info_c = dict(info); info_c['goal'] = goal[None, None].expand(1, S, W, N, C).clone()
    with torch.no_grad():
        cost = model.get_cost(dict(info_c), cands[None])
        g2 = model.aggregator(model.encode({'pcd': goal[None]})['emb'])[0]       # latent2 of the goal window
        ref_cost = torch.stack([((model.aggregator(ref[s, -W:][None])[0] - g2) ** 2).sum() for s in range(S)])
    errc = (cost[0] - ref_cost).abs().max().item()
    goodc = cost.shape == (1, S) and errc < 1e-3
    ok &= goodc
    print(f'H={H}: cost {tuple(cost.shape)}, max|cost - reference| = {errc:.2e} -> {"OK" if goodc else "FAIL"}')

# repeated get_cost on the same dict (CEM iterations share expanded_infos): cached context emb must give the same cost
with torch.no_grad():
    d = dict(info_c); c1 = model.get_cost(d, cands[None]); c2 = model.get_cost(d, cands[None])
same = torch.allclose(c1, c2); ok &= same
print(f'cached-emb second call identical: {same}')
print('ALL OK' if ok else 'SOME CHECKS FAILED'); sys.exit(0 if ok else 1)
