"""OGBench's official play data (Cube, Puzzle) as training data for the policy learners (supp. sec:supp-policy):
besides the frozen latent, the input carries the robot state (Cube) or the button lights (Puzzle), and the goal
gives the target state of the objects or the lights.

data/policy/<env>/ holds the dataset cfg.env.policy_data.dataset and its -val split as OGBench publishes them:
  Puzzle  <dataset>.npz, <dataset>-val.npz fetched with OGBench's own downloader (ogbench.download_datasets, ogbench
          a977253: files already present are kept). The learner inputs are derived from them when the policy is
          trained: the lights (button_states) and the oracle goal by DQC's loader (puzzle-<b>-play-oraclerep-v0), the
          latents by jepa4r/policy/dqc (cache_latents: every row set back into the simulator, rendered, encoded).
  Cube    the Cube workflow's prepare-data stage (jepa4r/data/cube/official_data.py): the same public files,
          downloaded and checked against the sha256 / byte counts of configs/env/cube/official_sources.json, into
          data/policy/<env>/official/cube<N>/ (OFFICIAL_DATA.json)
Interpreter: JEPA4R_PY_DQC (any python with ogbench installed).
"""
import os
import subprocess
import sys
from pathlib import Path


def prepare_official_play(cfg, out_dir):
    from jepa4r.policy.gciql import is_cube_workflow

    if is_cube_workflow(cfg):
        from jepa4r.policy.gciql import prepare_official

        return prepare_official(cfg, out_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = str(cfg.env.policy_data.dataset)
    code = 'import sys, ogbench; ogbench.download_datasets([sys.argv[1]], sys.argv[2])'
    subprocess.run([os.environ.get('JEPA4R_PY_DQC', sys.executable), '-c', code, name, str(out_dir)], check=True)
    return {'dataset': str(out_dir / f'{name}.npz'), 'val': str(out_dir / f'{name}-val.npz')}
