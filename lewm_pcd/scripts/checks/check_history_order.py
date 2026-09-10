"""Instrumented 2-env CPU eval of the smoke 4D checkpoint (config/eval/cube_pcd_agg.yaml, pilot dataset):
at every planning call records H (frames given), T (candidate blocks), the goal's frame count, the number of
predicted frames returned by the rollout (expect H + T), and whether the newest history frame is the env's
current point cloud. Also checks that the last frame of eval.py's goal window equals the World's goal frame.
Usage: python scripts/checks/check_history_order.py <ROOT with checkpoints/lewm_agg_smoke and datasets/ogbench/pilot_prod.h5>"""
import os, sys, numpy as np, torch
ROOT = sys.argv[1]
os.environ.setdefault('MUJOCO_GL', 'osmesa'); os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
sys.argv = ['eval.py', '--config-path=/root/autodl-tmp/lewm234d/lewm_pcd/config/eval', '--config-name=cube_pcd_agg.yaml', 'policy=lewm_agg_smoke', 'device=cpu', 'solver.device=cpu',
            f'cache_dir={ROOT}', 'eval.dataset_name=ogbench/pilot_prod', 'eval.num_eval=2', 'eval.eval_budget=50', f'hydra.run.dir={ROOT}/eval_hist']
os.environ['STABLEWM_HOME'] = ROOT
import jepa_agg, envs.eval_policy as ep
log = []
orig_rollout = jepa_agg.AggJEPA.rollout
def rollout(self, info, action_sequence, history_size=None):
    H = info[self.obs_key].size(2); pcd = info[self.obs_key][:, 0]          # (E, H, N, 6) sample 0
    ah = info.get('action_history')
    rec = {'H': H, 'T': action_sequence.shape[2], 'goal_frames': info['goal'].size(2), 'action_history': None if ah is None else ah.size(2),
           'frames_distinct': [float((pcd[:, i] - pcd[:, i + 1]).abs().mean()) for i in range(H - 1)],
           'newest_eq_current': any(torch.equal(pcd[0, -1], ep._LAST_CURRENT[i]) for i in range(ep._LAST_CURRENT.size(0))) if ep._LAST_CURRENT is not None else None}
    P = ep._POLICY
    rec['oldest_eq_seed'] = rec['middle_eq_seed'] = rec['act_hist_eq_dataset'] = None
    if P is not None and P.history_seed is not None and H > 1:
        HS = P.history_seed; E = HS['action'].shape[0]; blk = P.cfg.action_block
        sc = lambda i, j: torch.as_tensor(P.process['pcd'].transform(P._join(HS['pcd_xyz'][i, j], HS['pcd_rgb'][i, j])))
        # first plan: oldest == dataset start-10, middle == start-5 (scaled), of the same env
        rec['oldest_eq_seed'] = any(torch.equal(pcd[0, 0], sc(i, 0)) for i in range(E))
        rec['middle_eq_seed'] = any(torch.equal(pcd[0, 1], sc(i, blk)) for i in range(E))
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
orig_prep = ep.PCDWorldModelPolicy._prepare_info
_printed = [0]
def prep(self, info_dict):
    ep._POLICY = self
    if _printed[0] < 2 and 'goal_pcd_xyz' in info_dict:
        gw = self.goal_window
        print('PREP goal_pcd_xyz shape:', np.asarray(info_dict['goal_pcd_xyz']).shape, '| goal_window shapes:', None if gw is None else {k: v.shape for k, v in gw.items()},
              '| goal_window last frame == World goal frame:', None if gw is None else bool(np.array_equal(gw['pcd_xyz'][:, -1], np.asarray(info_dict['goal_pcd_xyz'])[:, 0])), flush=True)
    out = orig_prep(self, info_dict)
    if _printed[0] < 2 and 'goal' in out:
        print('PREP out goal shape:', tuple(out['goal'].shape), '| pcd shape:', tuple(out['pcd'].shape), flush=True); _printed[0] += 1
    ep._LAST_CURRENT = out['pcd'][:, -1].clone() if torch.is_tensor(out.get('pcd')) else torch.as_tensor(np.asarray(out['pcd'])[:, -1])
    return out
ep.PCDWorldModelPolicy._prepare_info = prep
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
