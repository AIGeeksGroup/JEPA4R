"""GCIQL [park2025ogbench] on OGBench Cube (supp. sec:supp-policy, configs/policy/gciql.yaml).

Offline goal-conditioned RL on OGBench's official play data for 1e6 steps. Input: frozen latent and robot state;
goal: target state of the objects. Evaluation: 50 episodes on each of the five official tasks
(jepa4r/eval/ogbench.py).

Integrated code: the OGBench-cube branch's Cube 1-4 workflow ("Current paper Cube workflow.",
OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench), placed along the Integration Map (docs/architecture.md): GCIQL, its
native evaluation and acceptance stages and the controller here; collection and official play data in jepa4r/data/cube,
world-model training in jepa4r/models/cube, PLDM, LeJEPA and the released VISReg / V-JEPA 2.1 weights in
jepa4r/baselines/cube. The controller (workflow.py) runs fifteen stages as child processes (python -m <module>); each
closes with a receipt that the next stage and any resume verify, and every run binds the sha256 of its sources
(INTEGRATED_SOURCES.json and ALGORITHM_SIGNATURES.json here, jepa4r/models/cube/SOURCE_MANIFEST.json,
jepa4r/data/cube/collection/sources.json, downstream/vendor/source_manifest.json,
jepa4r/baselines/cube/pldm/source_manifest.json; a run refuses a pinned file that changed, as the branch's release
did). downstream/vendor/ holds the branch's frozen GCIQL copies (downstream/vendor/source_manifest.json); here they are
wrapped at 120 columns, with LF line ends, without unused imports and with the names downstream/trainer.py sets at run
time declared, otherwise unchanged (the same AST). The stages of this repository run the workflow's stages for
(cfg.env, cfg.method), in the workflow's own order:

  stage                workflow stages                                                  run root (the workflow's
  collect_wm_data      collect, views                                                   output_root)
                                                                                        data/wm/<env>/harness
  prepare_policy_data  prepare-data                                                     data/policy/<env>/harness
  train_wm             collect, views (validated and reused), train                     outputs/train_wm/<env>/
                       (JEPA4R, LeWM, PLDM, LeJEPA)                                     <method>/<tag>/harness
  train_policy         prepare-data (reused), prepare-encoder (VISReg, V-JEPA 2.1),     the same run root
                       camera, bind, encode, cache-audit, gate, reload, gciql, accept,
                       native-sr, native-audit

Shared inputs of every run root of an env: official play data data/policy/<env>/official/cube<N>, the paired collection
data/wm/<env>/collected/<mode>/cube<N> and its training views data/wm/<env>/views/<mode>/cube<N>, the released weights
checkpoints/<visreg|vjepa>/ (prepare-encoder downloads a missing file from the URL pinned in released_models.json).
cfg.env.harness.mode: formal (the paper protocol: 1e6 GCIQL updates, 250 native episodes) or smoke (the workflow's
two-update diagnostic). Interpreters: JEPA4R_PY_CUBE_WM (world model, collection, camera; the paper runs: Python 3.10,
NumPy 2.2.6, MuJoCo 3.3.7) and JEPA4R_PY_CUBE_RL (GCIQL, encoding, native evaluation; JAX 0.7.2, Flax 0.11.2, Torch).
Checkouts: JEPA4R_OGBENCH (OGBench a977253, default third_party/ogbench), JEPA4R_VJEPA2 (V-JEPA 2 204698b, default
third_party/vjepa2); stable-worldmodel is the vendored third_party/stable-worldmodel.
"""
import json
import os
import sys
from pathlib import Path

MODELS = {'jepa4r': 'lewm_pcd', 'lewm': 'lewm_rgb', 'pldm': 'pldm', 'lejepa': 'lejepa', 'visreg': 'visreg',
          'vjepa21': 'vjepa'}   # this repository's methods -> the workflow's models (workflow.MODELS)
STAGES = {
    'collect_wm_data': ('collect', 'views'),
    'prepare_policy_data': ('prepare-data',),
    'train_wm': ('collect', 'views', 'train'),
    'train_policy': ('prepare-data', 'prepare-encoder', 'camera', 'bind', 'encode', 'cache-audit', 'gate', 'reload',
                     'gciql', 'accept', 'native-sr', 'native-audit'),
    'cache_latents': ('prepare-data', 'prepare-encoder', 'camera', 'bind', 'encode', 'cache-audit'),   # features.py
}
SHARED_MODEL = 'lewm_pcd'   # the model named in the method-independent run roots (the stages they run ignore it)


def is_cube_workflow(cfg):
    """An env this workflow runs: Cube 1-4 (configs/env/cube_*.yaml with wm_data.collector c4_cube; not the
    reconstruction env cube_single_recon)."""
    from omegaconf import OmegaConf

    return OmegaConf.select(cfg, 'env.wm_data.collector') == 'c4_cube'


def cube_count(cfg):
    return {'cube_single': 1, 'cube_double': 2, 'cube_triple': 3, 'cube_quadruple': 4}[str(cfg.env.name)]


def run_root(cfg, stage):
    """The workflow output_root of this stage: one per env for the method-independent stages, one per (env, method,
    tag) for train_wm and train_policy."""
    from jepa4r.config import stage_dir

    if stage in ('collect_wm_data', 'prepare_policy_data'):
        return stage_dir(stage, cfg.env.name) / 'harness'
    return stage_dir('train_wm', cfg.env.name, cfg.method.name, None, str(cfg.tag)) / 'harness'


def workflow_config(cfg, stage):
    """The workflow's run config (cube_workflow_v1) of this stage's run root."""
    from jepa4r.config import paths, stage_dir

    n, mode = cube_count(cfg), str(cfg.env.harness.mode)
    if stage in ('collect_wm_data', 'prepare_policy_data'):
        model = SHARED_MODEL
    else:
        if cfg.method.name not in MODELS:
            raise SystemExit(f'the Cube workflow has no {cfg.method.name!r} model (it runs {sorted(MODELS)})')
        model = MODELS[cfg.method.name]
    wm = stage_dir('collect_wm_data', cfg.env.name)
    config = dict(
        schema='cube_workflow_v1',
        cube_count=n,
        model=model,
        mode=mode,
        output_root=str(run_root(cfg, stage).resolve()),
        python=dict(
            world_model=os.path.abspath(os.environ.get('JEPA4R_PY_CUBE_WM') or sys.executable),
            downstream=os.path.abspath(os.environ.get('JEPA4R_PY_CUBE_RL') or sys.executable),
        ),
        official_root=str((stage_dir('prepare_policy_data', cfg.env.name) / 'official' / f'cube{n}').resolve()),
        collection_root=str((wm / 'collected' / mode / f'cube{n}').resolve()),
        data_root=str((wm / 'views' / mode / f'cube{n}').resolve()),
        assets_root=str(paths()['checkpoints'].resolve()),
        workers=int(cfg.env.harness.workers),
        prefetch=int(cfg.env.harness.prefetch),
        collection_seed=0,
    )
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        config['cuda_visible_devices'] = os.environ['CUDA_VISIBLE_DEVICES']
    return config


def cube_stage(cfg, stage, out_dir=None):
    """Run the workflow stages that make up this repository's ``stage`` for (cfg.env, cfg.method); returns the
    workflow's STATUS.json. Completed stages of the run root are verified again and kept (resume)."""
    from jepa4r.policy.gciql import workflow

    config = workflow_config(cfg, stage)
    if os.environ.get('JEPA4R_TMPDIR'):   # the children's TMPDIR (workflow.child_environment), off the system disk
        Path(os.environ['JEPA4R_TMPDIR']).mkdir(parents=True, exist_ok=True)
    plan = workflow.build_plan(config, 'all')
    plan['stages'] = [row for row in plan['stages'] if row['stage'] in STAGES[stage]]
    plan['requested_stage'] = 'all' if stage == 'train_policy' else stage   # PIPELINE_COMPLETE after the last one
    root = Path(config['output_root'])
    status = root / 'STATUS.json'
    resume = status.is_file() and any(
        row['stage'] in json.loads(status.read_text())['completed'] for row in plan['stages']
    )
    print(json.dumps(plan, indent=2), flush=True)
    code = workflow.execute(plan, resume=resume)
    state = json.loads(status.read_text())
    if code != 0:
        raise RuntimeError(f'{cfg.env.name} {stage}: workflow stage failed ({state.get("stage")}): '
                           f'{str(state.get("error"))[-2000:]}\nrecords in {root}')
    return state


def collect_cube(cfg, out_dir):
    """collect_wm_data on Cube 1-4: the workflow's paired RGB / PCD collection and its training views."""
    state = cube_stage(cfg, 'collect_wm_data', out_dir)
    return {'harness_status': state['stage'], 'completed': sorted(state['completed'])}


def prepare_official(cfg, out_dir):
    """prepare_policy_data on Cube 1-4: the byte-pinned public play data (official_data.py, official_sources.json)."""
    state = cube_stage(cfg, 'prepare_policy_data', out_dir)
    official = json.loads((Path(workflow_config(cfg, 'prepare_policy_data')['official_root']) / 'OFFICIAL_DATA.json')
                          .read_text())
    return {'dataset': official['train']['path'], 'val': official['val']['path'], 'harness_status': state['stage']}


def train_world_model(cfg, out_dir):
    """train_wm on Cube 1-4 (JEPA4R, LeWM, PLDM, LeJEPA): the workflow's own trainers (fresh_train.py)."""
    state = cube_stage(cfg, 'train_wm', out_dir)
    return {'harness_status': state['stage'], 'completed': sorted(state['completed'])}


def native_events(root):
    """The native evaluation of a run root, one event per episode (downstream/native/episodes/task*/episode_*/
    result.json): task, episode, success and the evaluated model group."""
    events = []
    for path in sorted((Path(root) / 'downstream' / 'native' / 'episodes').glob('task*/episode_*/result.json')):
        result = json.loads(path.read_text())
        events.append(dict(task=int(result['task_id']), episode_id=int(result['episode_index']),
                           success=bool(result['success']), model=result['group_id']))
    return events


def native_summary(cfg, root):
    """The success of a run root's native evaluation through the A1 raw-event aggregation (jepa4r.eval.aggregation
    .cube_summary): a missing task, a duplicated episode, a wrong denominator or a second model group raises. formal:
    the 5 tasks x 50 episodes of the paper protocol; smoke: the workflow's one diagnostic episode (task 1)."""
    from jepa4r.eval.aggregation import cube_summary

    formal = str(cfg.env.harness.mode) == 'formal'
    return cube_summary(native_events(root), tasks=(1, 2, 3, 4, 5) if formal else (1,),
                        episodes_per_task=50 if formal else 1)


def train(cfg, out_dir) -> dict:
    """train_policy on Cube 1-4: encode the official play data with the frozen encoder, GCIQL (1e6 updates, seed 0),
    then the native evaluation of the final policy (5 tasks x 50 episodes)."""
    cube_stage(cfg, 'train_policy', out_dir)
    summary = native_summary(cfg, run_root(cfg, 'train_policy'))
    success = summary['success_percent']
    return {
        'success': success,
        'success_per_task': {f"task{row['task']}": row['success_percent'] for row in summary['tasks']},
        'evals': [{'step': 1000000 if str(cfg.env.harness.mode) == 'formal' else 2, 'success': success}],
    }


def load_frozen_vjepa21(cfg):
    """V-JEPA 2.1 ViT-L/16 on Cube: the workflow's released-encoder loader (released_encoders.py) on the encoder spec
    its prepare-encoder stage wrote (CUDA, FP16 forward, the mean of the 576 patch tokens, 1024-d); encode() takes the
    raw uint8 RGB [B, 3, 224, 224]."""
    from types import SimpleNamespace

    from jepa4r.baselines.cube.released_encoders import load_encoder

    model = load_encoder(
        run_root(cfg, 'train_policy') / 'encoder' / 'encoder_spec.json', device=str(cfg.get('device', 'cuda'))
    )
    return SimpleNamespace(obs='image', latent_dim=int(model.output_dim), encode=model.encode_uint8, model=model)


def wm_export(cfg):
    """The world-model export that train_wm accepted (the run root's wm/WM_ACCEPTANCE.json): the directory of
    weights_epoch_<n>.pt and config.json from which the workflow's encode and native stages read the frozen encoder."""
    receipt = json.loads((run_root(cfg, 'train_wm') / 'wm' / 'WM_ACCEPTANCE.json').read_text())
    return Path(receipt['checkpoint']['path']).parent


def load_frozen_pldm(cfg):
    """PLDM's Impala encoder on Cube (512-d): the workflow's encoder boundary (workflow_pldm_encoder.py) on the
    accepted export; encode() takes the raw uint8 RGB [B, 224, 224, 3] (resized to 65x65 there, PLDM's input)."""
    from types import SimpleNamespace

    from jepa4r.baselines.cube import workflow_pldm_encoder as adapter

    receipt = json.loads((run_root(cfg, 'train_wm') / 'wm' / 'WM_ACCEPTANCE.json').read_text())
    model = adapter.load_frozen(Path(receipt['checkpoint']['path']), device=str(cfg.get('device', 'cuda')))
    return SimpleNamespace(obs='image', latent_dim=adapter.DIM, model=model,
                           encode=lambda pixels: adapter.encode_uint8(model, pixels))


def cache_latents(cfg, encoder, out_path):
    """features.cache_latents on Cube: the workflow's encode and cache-audit stages for (cfg.env, cfg.method) -- every
    official play row set back into the simulator, rendered and encoded with the run's frozen encoder (the export
    train_wm accepted, or the released weights prepare-encoder bound). The cache stays where the workflow's receipts
    bind it (the run root's downstream/cache), so out_path is not written; ``encoder``, when given, must be that
    encoder.
    -> {split: the feature loader arguments of runtime_recipe.json}"""
    from jepa4r.methods import trained_by_us

    if encoder is not None and trained_by_us(cfg.method) and Path(encoder).resolve() != wm_export(cfg).resolve():
        raise ValueError(f'the Cube workflow encodes with its own accepted export {wm_export(cfg)}, not {encoder}')
    cube_stage(cfg, 'cache_latents')
    recipe = run_root(cfg, 'train_policy') / 'downstream' / 'runtime_recipe.json'
    return json.loads(recipe.read_text())['feature_loader_arguments']


def latent_obs(env, encoder, cfg):
    """features.LatentObs on Cube: the native evaluation's current policy input. ``env`` is the workflow's
    OnlineVisualEnv (downstream/native_core.py: the official CubeEnv, its camera and the run's OnlineEncoder); the
    callable returns [latent of the current render, the first 19 entries of the state] (make_policy_inputs)."""
    import numpy as np

    from jepa4r.policy.gciql.downstream import native_core

    def observation():
        latent = env.encoder.encode(env.current_raw())
        state = np.asarray(env.unwrapped.compute_observation(), dtype=np.float32)
        return native_core.make_policy_inputs(latent, latent, state, state)[0]
    return observation


def evaluate_run(cfg, policy=None, encoder=None):
    """eval/ogbench.evaluate_policy on Cube: the workflow's native evaluation (native-sr, audited by native-audit;
    5 tasks x 50 episodes, online encoding of current and goal views) of the run root ``policy`` (default: this
    config's). Runs the remaining train_policy stages first if the evaluation has not run.
    -> {"success": %, "success_per_task": {task: %}, "episodes": n}"""
    root = Path(policy) if policy else run_root(cfg, 'train_policy')
    if not native_events(root):
        if root.resolve() != run_root(cfg, 'train_policy').resolve():
            raise FileNotFoundError(f'no native evaluation in {root}')
        cube_stage(cfg, 'train_policy')
    summary = native_summary(cfg, root)
    return {'success': summary['success_percent'],
            'success_per_task': {f"task{row['task']}": row['success_percent'] for row in summary['tasks']},
            'episodes': summary['episodes']}
