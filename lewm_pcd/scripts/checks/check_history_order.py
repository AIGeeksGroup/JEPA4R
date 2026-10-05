"""Instrumented 2-env CPU eval of a smoke 4D checkpoint: does the policy deliver the 3-frame history / 3-frame goal
to the model the way training saw them? Records, at every planning call: H (frames given), T (candidate blocks),
history action blocks, goal frames, predicted frames returned by the rollout (expect H + T), whether the newest
history frame is the env's current observation, and (first plan) whether the oldest/middle history frames equal
the scaled dataset frames start-10 / start-5 and the history action blocks equal the dataset's.
Usage:
  python scripts/checks/check_history_order.py <ROOT>          # point clouds: TMP/checkpoints/lewm_agg_smoke, datasets/ogbench/pilot_prod.h5
  python scripts/checks/check_history_order.py <ROOT> --pix    # pixels:       TMP/checkpoints/pix_agg_smoke,  datasets/ogbench/pix_smoke.h5
ROOT is used as STABLEWM_HOME."""
import os, sys, numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # lewm_pcd
ROOT = sys.argv[1]; PIX = '--pix' in sys.argv
os.environ.setdefault('MUJOCO_GL', 'osmesa'); os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
cfg_name, policy_name, ds_name = ('cube_pix_agg.yaml', 'pix_agg_smoke', 'ogbench/pix_smoke') if PIX else ('cube_pcd_agg.yaml', 'lewm_agg_smoke', 'ogbench/pilot_prod')
OBS, GOAL_KEY, OBS_KEYS = ('pixels', 'goal', ('pixels',)) if PIX else ('pcd', 'goal_pcd_xyz', ('pcd_xyz', 'pcd_rgb'))
sys.argv = ['eval.py', f'--config-path={REPO}/config/eval', f'--config-name={cfg_name}', f'policy={policy_name}', 'device=cpu', 'solver.device=cpu',
            f'cache_dir={ROOT}', f'eval.dataset_name={ds_name}', 'eval.num_eval=2', 'eval.eval_budget=50', f'hydra.run.dir={ROOT}/eval_hist']
os.environ['STABLEWM_HOME'] = ROOT
import jepa_agg, envs.eval_policy as ep
from torchvision import tv_tensors
log = []
orig_rollout = jepa_agg.AggJEPA.rollout


def scaled_frame(P, HS, i, j):
    """dataset frame j of env i's history seed -> the tensor the model sees for it (same path as the policy's process/transform)"""
    if PIX:
        return P.transform['pixels'](tv_tensors.Image(np.transpose(HS['pixels'][i, j], (2, 0, 1))))
    return torch.as_tensor(P.process['pcd'].transform(P._join(HS['pcd_xyz'][i, j], HS['pcd_rgb'][i, j])))


def rollout(self, info, action_sequence, history_size=None):
    H = info[self.obs_key].size(2); obs = info[self.obs_key][:, 0]          # (E, H, ...) sample 0
    ah = info.get('action_history')
    rec = {'H': H, 'T': action_sequence.shape[2], 'goal_frames': info['goal'].size(2), 'action_history': None if ah is None else ah.size(2),
           'frames_distinct': [float((obs[:, i] - obs[:, i + 1]).abs().float().mean()) for i in range(H - 1)],
           'newest_eq_current': any(torch.equal(obs[0, -1], ep._LAST_CURRENT[i]) for i in range(ep._LAST_CURRENT.size(0))) if ep._LAST_CURRENT is not None else None}
    P = ep._POLICY
    rec['oldest_eq_seed'] = rec['middle_eq_seed'] = rec['act_hist_eq_dataset'] = None
    if P is not None and P.history_seed is not None and H > 1:
        HS = P.history_seed; E = HS['action'].shape[0]; blk = P.cfg.action_block
        rec['oldest_eq_seed'] = any(torch.equal(obs[0, 0], scaled_frame(P, HS, i, 0)) for i in range(E))
        rec['middle_eq_seed'] = any(torch.equal(obs[0, 1], scaled_frame(P, HS, i, blk)) for i in range(E))
        if ah is not None and ah.size(2) == H - 1:  # action_history == scaled dataset action blocks leaving start-10 and start-5
            def blocks(i):
                a = np.concatenate([HS['action'][i, 1:], HS['last_action'][i][None]], 0)   # a[start-10 .. start-1]
                a = P.process['action'].transform(np.nan_to_num(a, nan=0.0)) if 'action' in P.process else a
                return torch.as_tensor(a.reshape(H - 1, -1), dtype=ah.dtype)
            rec['act_hist_eq_dataset'] = any(torch.allclose(ah[0, 0].cpu(), blocks(i), atol=1e-5) for i in range(E))
    out = orig_rollout(self, info, action_sequence, history_size)
    rec['pred_frames'] = out['predicted_emb'].size(2); log.append(rec)
    return out


jepa_agg.AggJEPA.rollout = rollout
ep._LAST_CURRENT = None; ep._POLICY = None
orig_prep = ep.WindowedWorldModelPolicy._prepare_info
_printed = [0]


def prep(self, info_dict):
    ep._POLICY = self
    if _printed[0] < 2 and GOAL_KEY in info_dict:
        gw = self.goal_window
        print(f'PREP {GOAL_KEY} shape:', np.asarray(info_dict[GOAL_KEY]).shape, '| goal_window shapes:', None if gw is None else {k: v.shape for k, v in gw.items()},
              '| goal_window last frame == World goal frame:', None if gw is None else bool(np.array_equal(gw[GOAL_KEY][:, -1], np.asarray(info_dict[GOAL_KEY])[:, 0])), flush=True)
    out = orig_prep(self, info_dict)
    if _printed[0] < 2 and 'goal' in out:
        print('PREP out goal shape:', tuple(out['goal'].shape), f'| {OBS} shape:', tuple(out[OBS].shape), flush=True); _printed[0] += 1
    ep._LAST_CURRENT = out[OBS][:, -1].clone() if torch.is_tensor(out.get(OBS)) else torch.as_tensor(np.asarray(out[OBS])[:, -1])
    return out


ep.WindowedWorldModelPolicy._prepare_info = prep
import eval as ev
try:
    ev.run()
except SystemExit:
    pass
from collections import Counter
print(f'{len(log)} planning calls (CEM iterations)')
print(Counter((r['H'], r['T'], r['action_history'], r['goal_frames'], r['pred_frames'], r['newest_eq_current'], r['oldest_eq_seed'], r['middle_eq_seed'], r['act_hist_eq_dataset']) for r in log))
print('legend: (H frames, T candidate blocks, action_history blocks, goal frames, predicted frames, newest frame == current obs, oldest == dataset start-10, middle == start-5, action_history == dataset blocks) -> count')
for r in log[:1] + log[-1:]:
    print('example:', r)
