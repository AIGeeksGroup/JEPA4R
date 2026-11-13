"""Behaviour-cloning demonstrations (supp. sec:supp-policy): 25 per task on MetaWorld and on the five Scene
evaluation tasks, 100 on Adroit Relocate (cfg.env.policy_data.demos_per_task).

Integrated code (moved unchanged apart from paths: data/policy/<env>/ = JEPA4R_DEMOS_HOME):
  unpack_suite.py       cortexbench@9331494:lewm_pcd/bench/cortexbench/unpack_suite.py: the official CortexBench demo zip
                        (downloads/<suite>.zip) -> <suite>/<env id>.pickle (MetaWorld, Adroit)
  convert_demos_3d.py   cortexbench@9331494:.../convert_demos_3d.py: every official MetaWorld demo replayed (fresh env, seed
                        12345 + i, the recorded actions), the point cloud of the frame before each action
                        (jepa4r.policy.bc.pcd_fork, multisampling off) -> metaworld-expert-v1.0-pcd/
  make_scene_demos.py   cortexbench@9331494:.../make_scene_demos.py: 25 scripted-expert demos per Scene task, seeds from
                        12345 (a seed failing within 750 steps is skipped), padded to the slowest success -> scene-expert{,-pcd}/
  prepare_official_assets.py  OGBench-cube@4fbb0c6:scripts/prepare_official_assets.py: the 100 official Relocate demos
                        (relocate-v0.pickle, sha256 pinned in policy/bc/relocate/DEPENDENCIES.json; JEPA4R_ADROIT_DEMOS)
                        -> a light copy (raw labels and non-visual fields kept) + the RH56 task initialisations
  Adroit Relocate / RH56   the Relocate harness's official-demos / rh56_official_demo stage (jepa4r.policy.bc.relocate_stage)
Interpreters: JEPA4R_PY_CORTEXBENCH (eai-vc metaworld fork, MuJoCo 2.0) for MetaWorld, JEPA4R_PY_COLLECT (MuJoCo 3 +
OGBench) for Scene, JEPA4R_PY_RELOCATE_WM for the Relocate light copy (the harness picks its own interpreters).
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUFFIX = '-v2-goal-observable'


def _run(script, args, out_dir, py_var):
    env = dict(os.environ, JEPA4R_DEMOS_HOME=str(out_dir), MUJOCO_GL='egl',
               PYTHONPATH=os.pathsep.join([str(HERE.parents[2]), os.environ.get('PYTHONPATH', '')]))
    subprocess.run([os.environ.get(py_var, sys.executable), str(HERE / script), *map(str, args)], env=env, check=True)


def prepare_demos(cfg, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suite = cfg.env.suite
    if suite == 'metaworld':
        if not (out_dir / 'metaworld-expert-v1.0' / f'{cfg.env.env_id}.pickle').exists():
            _run('unpack_suite.py', ['metaworld-expert-v1.0'], out_dir, 'JEPA4R_PY_CORTEXBENCH')
        _run('convert_demos_3d.py', [str(cfg.env.env_id)[:-len(SUFFIX)]], out_dir, 'JEPA4R_PY_CORTEXBENCH')
        return {'demos': str(out_dir / 'metaworld-expert-v1.0-pcd' / f'{cfg.env.env_id}.pickle')}
    if suite == 'ogbench':  # Scene's five evaluation tasks
        _run('make_scene_demos.py', [1, 2, 3, 4, 5], out_dir, 'JEPA4R_PY_COLLECT')
        return {'demos': [str(out_dir / 'scene-expert-pcd' / f'scene-task{k}-v0.pickle') for k in range(1, 6)]}
    if suite in ('adroit', 'rh56'):  # the OGBench-cube branch's Relocate harness (jepa4r.policy.bc.relocate)
        from jepa4r.policy.bc import ensure_official_assets, relocate_stage

        official = ensure_official_assets()  # light copy of the 100 official demos + RH56 initialisations
        status = relocate_stage(cfg, 'prepare_policy_data', out_dir)
        return {'official_assets': str(official), 'harness_status': status['status']}
    raise ValueError(f'no demonstrations for suite {suite!r}')
