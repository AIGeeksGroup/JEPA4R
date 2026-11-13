"""Decoupled Q-chunking (DQC) [li2025dqc] on OGBench Puzzle (supp. sec:supp-policy, configs/policy/dqc.yaml).

Offline goal-conditioned RL on the official play data, 1e6 steps. Input: frozen latent and button lights;
goal: target lights. Ten evaluations during training (five tasks x 50 episodes); report the best.

Integrated code (cortexbench@9331494:lewm_pcd/bench/dqc/, moved into this package unchanged apart from paths):
  run_dqc.py       dqc df89825's training loop (the checkout, unchanged, at JEPA4R_DQC or third_party/dqc) with the
                   frozen encoder's latent + the true lights as observation; restartable (resume.pkl); metrics to
                   train.jsonl / eval.jsonl; the agent settings of the run in flags.json
  lewm_obs.py      the frozen encoders as CortexBench loads them (jepa4r/policy/bc/lewm_cortex.py, jepa_baselines.py) and
                   the evaluation-time observations (LeWMObs, BaselineObs)
  extract_feats.py, extract_feats_baselines.py   every row of the official play data set back into the simulator,
                   rendered (front_pixels 224; point sampling seeded per frame with (20261011, split, row)) and encoded
                   (epoch 10) -> one latent per row
  check_eval_obs.py, summarize_dqc.py   the original consistency check and table
``train``: the latents of the official training set (``cache_latents``), one run_dqc.py run with the board's agent
settings (``agent_flags``), then the train_policy metrics (``aggregate``): success = the best of the ten evaluations
during training (steps 100k, 200k, ..., 1M). run_dqc.py, unchanged, also evaluates the untrained agent at step 0: that
evaluation draws from numpy's global RNG, which also drives the batch sampling, so it stays in the run (dropping it
would change the training) and is listed in ``evals`` but not counted. In all 18 Puzzle runs of the paper it is 0.0,
so the best of ten equals the best of the eleven records (GATES.md 3.4).
Interpreter: JEPA4R_PY_DQC (JAX 0.6.2 / flax 0.10.7, MuJoCo 3 + OGBench, torch for the encoders).
"""
import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOADERS = ('vjepa21', 'visreg', 'lejepa', 'pldm')   # lewm_obs.BaselineEncoder.LOADERS (run_dqc.py --obs <name>)


def obs_kind(method_cfg):
    """run_dqc.py --obs of an encoder row: lewm3d / lewm2d (LeWM recipe on point clouds / images) or the baseline."""
    if method_cfg.kind == 'jepa_wm':
        return 'lewm3d' if method_cfg.obs == 'pointcloud' else 'lewm2d'
    if method_cfg.name in LOADERS:
        return str(method_cfg.name)
    raise ValueError(f'no DQC input for method {method_cfg.name!r}')


def board(env_cfg):
    """puzzle-3x3-v0 -> 3x3 (extract_feats.py --env-type)."""
    return str(env_cfg.env_id).split('-')[1]


def task_env(env_cfg):
    """DQC's env name: the play dataset with OGBench's oracle goal representation (the goal = the lights' target state),
    puzzle-<b>-play-v0 -> puzzle-<b>-play-oraclerep-v0 (run_puzzle_baselines.sh:92)."""
    head, _, version = str(env_cfg.policy_data.dataset).rpartition('-')
    return f'{head}-oraclerep-{version}'


def agent_flags(cfg):
    """run_dqc.py --agent.* : cfg.policy.hparams.agent, then the board's entry of hparams.per_env (glob on env.name);
    every other setting is dqc's agents/dqc.py default."""
    from omegaconf import OmegaConf

    hp = dict(OmegaConf.to_container(cfg.policy.hparams.agent, resolve=True))
    for pattern, patch in (cfg.policy.hparams.get('per_env') or {}).items():
        if fnmatch.fnmatchcase(cfg.env.name, str(pattern)):
            hp.update(OmegaConf.to_container(patch, resolve=True))
    return [f'--agent.{k}={v}' for k, v in hp.items()]


def _env():
    from jepa4r.config import paths

    # STABLEWM_HOME: read by the scripts only for defaults that every call here overrides (--data, --dataset_path)
    return dict(os.environ, MUJOCO_GL=os.environ.get('MUJOCO_GL', 'egl'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                STABLEWM_HOME=os.environ.get('STABLEWM_HOME') or str(paths()['data']),
                PYTHONPATH=os.pathsep.join([str(HERE.parents[2]), os.environ.get('PYTHONPATH', '')]))


def cache_latents(cfg, encoder, out_path):
    """The latents of the official training set through the frozen encoder read from ``encoder`` (a checkpoint
    directory, or the released weights file of VISReg / V-JEPA 2.1): extract_feats.py (LeWM recipe) or
    extract_feats_baselines.py (the baselines), restartable -> {kind: path of train_z<kind>.npy}."""
    from jepa4r.config import stage_dir

    kind, out_path = obs_kind(cfg.method), Path(out_path)
    epoch = str(int(cfg.method.get('policy_epoch', 10)))
    if kind in LOADERS:
        script, args, z = 'extract_feats_baselines.py', ['--encoder', f'{kind}={encoder}'], kind
    else:
        script, args, z = 'extract_feats.py', [f'--ckpt-{kind[-2:]}', str(encoder)], kind[-2:]
    done = out_path / (f'train_done_{kind}.npy' if kind in LOADERS else 'train_done.npy')
    if not done.exists() or not _all_done(done):
        cmd = [os.environ.get('JEPA4R_PY_DQC', sys.executable), str(HERE / script), *args, '--epoch', epoch,
               '--env-type', board(cfg.env), '--data', str(stage_dir('prepare_policy_data', cfg.env.name)),
               '--out', str(out_path), '--split', 'train', '--workers', str(int(cfg.policy.latents.workers))]
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / 'extract.log', 'a') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE, env=_env(), check=True)
    return {kind: str(out_path / f'train_z{z}.npy')}


def _all_done(path):
    import numpy as np

    return bool(np.load(path).all())


def latent_obs(env, encoder, cfg):
    """The evaluation-time observation of run_dqc.py for this encoder (lewm_obs.LeWMObs / BaselineObs): call it to get
    [latent, true lights] of the env's current state. Needs the DQC interpreter (JEPA4R_PY_DQC)."""
    if str(HERE) not in sys.path:
        sys.path.insert(1, str(HERE))
    import lewm_obs

    kind, epoch = obs_kind(cfg.method), int(cfg.method.get('policy_epoch', 10))
    if kind in LOADERS:
        return lewm_obs.BaselineObs(env, kind, str(encoder), epoch)
    return lewm_obs.LeWMObs(env, kind, str(encoder), epoch, lights=True)


def run_command(cfg, encoder, feats_dir, run_dir):
    """The run_dqc.py command of one run (as run_puzzle_pipeline.sh / run_puzzle_baselines.sh gave it)."""
    from jepa4r.config import stage_dir

    p = cfg.policy
    data = stage_dir('prepare_policy_data', cfg.env.name) / f'{cfg.env.policy_data.dataset}.npz'
    return [os.environ.get('JEPA4R_PY_DQC', sys.executable), str(HERE / 'run_dqc.py'), '--obs', obs_kind(cfg.method),
            '--ckpt', str(encoder), '--lewm_epoch', str(int(cfg.method.get('policy_epoch', 10))),
            '--feats_dir', str(feats_dir), '--run_dir', str(run_dir), *agent_flags(cfg),
            f'--env_name={task_env(cfg.env)}', f'--dataset_path={data}', f'--offline_steps={int(p.train_steps)}',
            f'--eval_interval={int(p.eval.interval)}', f'--eval_episodes={int(p.eval.episodes)}',
            f'--save_every={int(p.save_every)}', f'--seed={int(p.seed)}']


def aggregate(records, num_evals=10, interval=100000):
    """eval.jsonl records -> the train_policy metrics: success = the highest overall success (%) among the evaluations
    during training (steps interval, ..., num_evals x interval); success_per_task at that evaluation; every record
    (incl. step 0) in evals."""
    during = [r for r in records if 0 < int(r['step']) <= num_evals * interval]
    if len(during) != num_evals:
        raise RuntimeError(f'{len(during)} of {num_evals} evaluations during training found')
    best = max(during, key=lambda r: r['overall_success'])
    per_task = {k[:-len('_success')]: 100 * float(v) for k, v in best.items()
                if k.endswith('_success') and k != 'overall_success'}
    return {'success': 100 * float(best['overall_success']), 'success_per_task': per_task,
            'evals': [{'step': int(r['step']), 'success': 100 * float(r['overall_success'])} for r in records]}


def train(cfg, out_dir) -> dict:
    from jepa4r.policy.frozen import checkpoint

    out_dir = Path(out_dir)
    encoder = checkpoint(cfg)
    feats_dir, run_dir = out_dir / 'feats', out_dir / 'run'
    cache_latents(cfg, encoder, feats_dir)
    (run_dir / '.tmp').mkdir(parents=True, exist_ok=True)
    env = dict(_env(), TMPDIR=str(run_dir / '.tmp'), XLA_PYTHON_CLIENT_PREALLOCATE='false')
    if not (run_dir / 'DONE').exists():   # run_dqc.py resumes from resume.pkl
        with open(out_dir / 'run.log', 'a') as log:
            subprocess.run(run_command(cfg, encoder, feats_dir, run_dir), stdout=log, stderr=subprocess.STDOUT,
                           cwd=HERE, env=env, check=True)
    records = [json.loads(line) for line in (run_dir / 'eval.jsonl').read_text().splitlines() if line.strip()]
    return aggregate(records, int(cfg.policy.eval.num_evals), int(cfg.policy.eval.interval))


def evaluate_run(cfg, run_dir, encoder, episodes=None):
    """Restore a run_dqc.py run (its flags.json + resume.pkl) as run_dqc.py builds and restores it, and evaluate it with
    run_dqc.py's own ``evaluate`` on OGBench's five tasks (episodes per task: ``episodes`` or cfg.env.eval
    .episodes_per_task) -> {'success': %, 'success_per_task': {task: %}}. The observation is ``latent_obs`` of the frozen
    encoder read from ``encoder``. Needs the DQC interpreter (JEPA4R_PY_DQC) and the run's latents (flags.json feats_dir)."""
    import pickle
    import random

    if str(HERE) not in sys.path:
        sys.path.insert(1, str(HERE))
    import run_dqc  # noqa: F401  puts the dqc checkout first on sys.path, as for a run

    import flax
    import ml_collections
    import numpy as np
    from agents import agents
    from envs.env_utils import make_ogbench_env_and_datasets
    from utils.datasets import CGCDataset, Dataset

    run_dir = Path(run_dir)
    fl = json.loads((run_dir / 'flags.json').read_text())
    config = ml_collections.ConfigDict(fl['agent'])
    env, train_dataset, _ = make_ogbench_env_and_datasets(fl['env_name'], dataset_path=fl['dataset_path'],
                                                          compact_dataset=True, add_info=True)
    eval_env = make_ogbench_env_and_datasets(fl['env_name'], dataset_path=fl['dataset_path'], compact_dataset=True,
                                             env_only=True, width=224, height=224)
    kind = fl['obs'].split('_')[0]
    z = np.load(Path(fl['feats_dir']) / f"{fl['feats_split']}_z{kind if kind in LOADERS else kind[-2:]}.npy", mmap_mode='r')
    train_dataset['observations'] = np.concatenate([z.astype(np.float32, copy=False),
                                                    train_dataset['button_states'].astype(np.float32)], axis=1)
    del z
    train_dataset = Dataset.create(**train_dataset)
    eps = 1e-5
    train_dataset = train_dataset.copy(add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + eps, 1 - eps)))
    random.seed(fl['seed'])
    np.random.seed(fl['seed'])
    train_dataset = CGCDataset(Dataset.create(**train_dataset), config)
    example_batch = train_dataset.sample(1)
    agent = agents[config['agent_name']].create(fl['seed'], example_batch, config)
    with open(run_dir / 'resume.pkl', 'rb') as f:
        agent = flax.serialization.from_state_dict(agent, pickle.load(f)['agent'])
    del train_dataset
    obs_fn = latent_obs(eval_env, encoder, cfg)
    n = int(episodes or cfg.env.eval.episodes_per_task)
    task_infos = env.unwrapped.task_infos
    per_task = {}
    for task_id in range(1, len(task_infos) + 1):
        info = run_dqc.evaluate(agent, eval_env, task_id, n, example_batch['actions'].shape[-1], obs_fn)
        per_task[task_infos[task_id - 1]['task_name']] = 100 * float(info['success'])
    return {'success': sum(per_task.values()) / len(per_task), 'success_per_task': per_task}
