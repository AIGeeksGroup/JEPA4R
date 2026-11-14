"""CortexBench evaluation (MetaWorld, Adroit Relocate; reused on Scene): roll out the BC policy at each of the
20 evaluations and report the best (cfg.policy.eval). Episodes / seeds per evaluation: cfg.policy.eval.

During training the 20 evaluations are the official loop's own (eai-vc 76fe35e train_loop.sample_paths every 5
epochs and after the last, inside jepa4r/policy/bc/run_bc*.py); jepa4r.policy.bc reads them back from metrics.jsonl
and aggregates them (best of 20 per run, mean over seeds, mean over tasks). ``evaluate_policy`` runs one such evaluation
again for a saved policy (best_policy.pickle), as GATES.md E1 / E2 do: the run's eai-vc config, the runtime patches of
the runner (the scripts without their last two lines, which start a run), 25 episodes from the run's seed.
Interpreter: JEPA4R_PY_CORTEXBENCH (eai-vc, MuJoCo 2.0 / mujoco-py); JEPA4R_EAI_VC = the eai-vc checkout.
"""
import os
import pickle
import sys
from pathlib import Path

BC = Path(__file__).resolve().parents[2] / 'configs' / 'policy' / 'bc'   # conf/model/<embedding>.yaml


def _apply_runner_patches(script):
    """Execute a runner script without its last two lines (sys.argv = ...; runpy.run_path(...)): its patches only."""
    src = Path(script).read_text().rstrip().splitlines()
    assert src[-1].startswith('runpy.run_path') and src[-2].startswith('sys.argv'), script
    exec(compile('\n'.join(src[:-2]), str(script), 'exec'), {'__file__': str(script), '__name__': 'patches'})


def evaluate_policy(cfg, policy, encoder) -> dict:
    """-> {"success": %, ...} for one evaluation.

    cfg      the run's eai-vc config (a mapping with env_kwargs, eval_num_traj, seed: the run directory's
             .hydra/config.yaml)
    policy   best_policy.pickle of the run (path) or the loaded policy
    encoder  the embedding name: conf/model/<encoder>.yaml under JEPA4R_BC_CONF_HOME (jepa4r.policy.bc.embedding_config)
    """
    import numpy as np
    from omegaconf import OmegaConf

    vi = Path(os.environ['JEPA4R_EAI_VC']) / 'cortexbench' / 'mujoco_vc' / 'visual_imitation'
    for p in (str(vi), str(BC)):
        if p not in sys.path:
            sys.path.insert(0, p)
    run_cfg = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
    run_cfg['env_kwargs']['embedding_name'] = encoder
    conf = Path(os.environ.get('JEPA4R_BC_CONF_HOME', BC)) / 'conf' / 'model' / f'{encoder}.yaml'
    # point-cloud embedding: run_bc3d.py's patches
    if str(OmegaConf.load(conf).get('_target_', '')).endswith('load_lewm_pcd'):
        _apply_runner_patches(BC / 'run_bc3d.py')
    _apply_runner_patches(BC / 'run_bc.py')
    import mujoco_vc.gym_wrapper as gw
    import train_loop
    from mujoco_vc.model_loading import fuse_embeddings_flare

    env = gw.env_constructor(**run_cfg['env_kwargs'], fuse_embeddings=fuse_embeddings_flare)
    if isinstance(policy, (str, Path)):
        with open(policy, 'rb') as f:
            policy = pickle.load(f)
    paths = train_loop.sample_paths(num_traj=run_cfg['eval_num_traj'], env=env, policy=policy, eval_mode=True,
                                    horizon=env.horizon, base_seed=run_cfg['seed'], num_cpu=1)
    success = [int(p['env_infos']['success'][-1]) for p in paths]
    return {'success': 100.0 * float(np.mean(success)), 'per_episode': success,
            'score_mean': float(np.mean([np.sum(p['rewards']) for p in paths]))}
