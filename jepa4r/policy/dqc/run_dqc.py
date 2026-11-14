"""DQC (Decoupled Q-chunking, github.com/ColinQiyangLi/dqc at df89825, in third_party/dqc unchanged) trained and
evaluated on OGBench puzzle-3x3-play (the official data, 1 M steps). The inputs, everything else equal:
  --obs state  : the original input, the 55-d state vector (the reproduction);
  --obs lewm2d : [the frozen LeWM 2D encoder's 192-d latent, the 9 true lights];
  --obs lewm3d : [the frozen LeWM 3D encoder's 192-d latent, the 9 true lights];
  --obs lewm2d_nolight / lewm3d_nolight : the 192-d latent only (the lights only through the latent);
  --obs vjepa21 / visreg / lejepa / pldm : [a JEPA-family baseline encoder's latent (lewm_obs.BaselineEncoder, as
                                   CortexBench loads it; extract_feats_baselines.py), the true lights]; --ckpt its
                                   checkpoint.
The goal is always the 9 lights' goal state (OGBench's oraclerep); algorithm, hyperparameters, data and evaluation
(OGBench's 5 tasks) are unchanged.
The training loop is dqc's main.py with three changes:
  1. restartable: every --save_every steps the full state (the agent with its optimiser and RNG, numpy / python RNG
     states, the step) goes to disk; on SIGTERM it is saved once more before exiting;
  2. LeWM inputs: the training set's observations are extract_feats.py's latents (+ the lights); in the evaluation, each
     time the policy needs an action, the current frame of the LeWM data's camera (front_pixels, 224, the opaque arm of
     ob_type='states') goes through the same frozen encoder (lewm_obs.LeWMObs, the checkpoint given by --ckpt). The
     original computes an observation every step; here only when the policy uses it (the steps of an action chunk do
     not reach the policy): the same results, less rendering;
  3. no wandb: metrics go to jsonl files.
    python run_dqc.py --obs lewm3d --ckpt $STABLEWM_HOME/checkpoints/puzzle_3x3_pcd_dexart \\
        --feats_dir $STABLEWM_HOME/datasets/ogbench/dqc_feats/puzzle_3x3 --run_dir <dir> [--agent.xxx=...]
"""
import json
import os
import pickle
import random
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('JEPA4R_DQC') or HERE.parents[2] / 'third_party' / 'dqc')  # ColinQiyangLi/dqc @ df89825
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(HERE))  # lewm_obs
DATA = Path(os.environ.get('STABLEWM_HOME', '.')) / 'datasets' / 'ogbench'

import flax
import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_ogbench_env_and_datasets
from utils.datasets import CGCDataset, Dataset
from utils.evaluation import add_to, flatten, supply_rng

from lewm_obs import BaselineEncoder, BaselineObs, LeWMObs

FLAGS = flags.FLAGS
flags.DEFINE_string('obs', 'state', 'state | lewm2d | lewm3d | lewm2d_nolight | lewm3d_nolight | '
                    'vjepa21 | visreg | lejepa | pldm')
flags.DEFINE_string('run_dir', None, 'Output directory (checkpoint, logs).')
flags.DEFINE_string('env_name', 'puzzle-3x3-play-oraclerep-v0', '')
flags.DEFINE_string('dataset_path', str(DATA / 'official' / 'puzzle-3x3-play-v0.npz'), '')
flags.DEFINE_string('feats_dir', None, "extract_feats.py's output directory (LeWM inputs).")
flags.DEFINE_string('feats_split', 'train', 'Prefix of the latent files (val only for smoke tests).')
flags.DEFINE_string('ckpt', None, "Checkpoint directory of the LeWM encoder (LeWM inputs; extract_feats.py's).")
flags.DEFINE_integer('lewm_epoch', 10, 'Epoch of the LeWM weights.')
flags.DEFINE_integer('seed', 100001, '')
flags.DEFINE_integer('offline_steps', 1000000, '')
flags.DEFINE_integer('log_interval', 5000, '')
flags.DEFINE_integer('eval_interval', 100000, '')
flags.DEFINE_integer('eval_episodes', 50, 'Evaluation episodes per task.')
flags.DEFINE_integer('save_every', 25000, 'Steps between restart checkpoints.')
config_flags.DEFINE_config_file('agent', str(REPO / 'agents' / 'dqc.py'), lock_config=False)

_STOP = {'flag': False}


def _on_term(signum, frame):
    _STOP['flag'] = True


def evaluate(agent, env, task_id, num_eval_episodes, action_dim, obs_fn=None):
    """utils/evaluation.py's evaluate (goal_conditioned=True, no video, whole chunks) with obs_fn added: it computes
    the observation whenever the policy needs an action."""
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2 ** 32)))
    stats = defaultdict(list)
    for _ in range(num_eval_episodes):
        if _STOP['flag']:
            return None  # SIGTERM: give up this evaluation round, save and exit; it is redone on restart
        observation, info = env.reset(options=dict(task_id=task_id, render_goal=False))
        goal = info.get('goal')
        done = False
        action_queue = []
        while not done:
            if len(action_queue) == 0:
                ob = obs_fn() if obs_fn is not None else observation
                action = np.array(actor_fn(observations=ob, goals=goal))
                action = np.clip(action, -1, 1).reshape(-1, action_dim)
                action_queue.extend(list(action))
            action = action_queue.pop(0)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        add_to(stats, flatten(info))
    return {k: float(np.mean(v)) for k, v in stats.items()}


def main(_):
    run_dir = Path(FLAGS.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _on_term)
    config = FLAGS.agent
    assert FLAGS.obs in ('state', 'lewm2d', 'lewm3d', 'lewm2d_nolight', 'lewm3d_nolight',
                         *BaselineEncoder.LOADERS), FLAGS.obs
    lewm = FLAGS.obs != 'state'               # a frozen encoder's latent (LeWM or a baseline)
    kind = FLAGS.obs.split('_')[0]            # lewm2d / lewm3d / vjepa21 / lejepa / pldm
    baseline = kind in BaselineEncoder.LOADERS
    with_lights = not FLAGS.obs.endswith('_nolight')
    flag_dict = {
        k: getattr(FLAGS, k)
        for k in FLAGS
        if '.' not in k
        and k
        in (
            'obs',
            'env_name',
            'dataset_path',
            'feats_dir',
            'feats_split',
            'ckpt',
            'lewm_epoch',
            'seed',
            'offline_steps',
            'log_interval',
            'eval_interval',
            'eval_episodes',
            'save_every',
        )
    }
    flag_dict['agent'] = config.to_dict()
    (run_dir / 'flags.json').write_text(json.dumps(flag_dict, indent=1))

    # env and data (as main.py). The LeWM inputs render into a 224 offscreen buffer: the rendering size only, not the
    # simulation.
    env_kwargs = dict(width=224, height=224) if lewm else {}
    env, train_dataset, _ = make_ogbench_env_and_datasets(FLAGS.env_name, dataset_path=FLAGS.dataset_path,
                                                          compact_dataset=True, add_info=True)
    eval_env = make_ogbench_env_and_datasets(FLAGS.env_name, dataset_path=FLAGS.dataset_path, compact_dataset=True,
                                             env_only=True, **env_kwargs)
    if lewm:
        z = np.load(Path(FLAGS.feats_dir) / f'{FLAGS.feats_split}_z{kind if baseline else kind[-2:]}.npy',
                    mmap_mode='r')
        assert len(z) == len(train_dataset['observations']), (z.shape, train_dataset['observations'].shape)
        parts = [z.astype(np.float32, copy=False)] + ([train_dataset['button_states'].astype(np.float32)]
                                                      if with_lights else [])
        train_dataset['observations'] = np.concatenate(parts, axis=1)
        del z, parts  # the latents in memory once (the input; before: also the loaded file and its float32 copy)
    train_dataset = Dataset.create(**train_dataset)
    print(train_dataset.keys(), train_dataset.size, 'obs', train_dataset['observations'].shape, flush=True)
    env.reset()
    eps = 1e-5
    train_dataset = train_dataset.copy(
        add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + eps, 1 - eps))
    )

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    train_dataset = CGCDataset(Dataset.create(**train_dataset), config)
    example_batch = train_dataset.sample(1)
    agent = agents[config['agent_name']].create(FLAGS.seed, example_batch, config)
    metric_rng, _ = jax.random.split(jax.random.PRNGKey(FLAGS.seed))
    action_dim = example_batch['actions'].shape[-1]
    obs_fn = None
    if baseline:
        obs_fn = BaselineObs(eval_env, kind, FLAGS.ckpt, FLAGS.lewm_epoch)
    elif lewm:
        obs_fn = LeWMObs(eval_env, kind, FLAGS.ckpt, FLAGS.lewm_epoch, lights=with_lights)

    # restart
    ckpt = run_dir / 'resume.pkl'
    start = 0
    if ckpt.exists():
        with open(ckpt, 'rb') as f:
            d = pickle.load(f)
        agent = flax.serialization.from_state_dict(agent, d['agent'])
        np.random.set_state(d['np_rng'])
        random.setstate(d['py_rng'])
        metric_rng = jax.numpy.asarray(d['metric_rng'])
        # stopped during an evaluation: this step's update is done, the evaluation is redone
        eval_pending = d.get('eval_pending', False)
        start = d['step'] if eval_pending else d['step'] + 1
        print(f'resumed from step {d["step"]}' + (' (eval pending)' if eval_pending else ''), flush=True)
    else:
        eval_pending = False

    def save(step, pending=False):
        tmp = run_dir / 'resume.pkl.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(dict(agent=jax.device_get(flax.serialization.to_state_dict(agent)), step=step,
                             eval_pending=pending, np_rng=np.random.get_state(), py_rng=random.getstate(),
                             metric_rng=np.asarray(metric_rng)), f)
        os.replace(tmp, ckpt)

    train_log = open(run_dir / 'train.jsonl', 'a')
    eval_log = open(run_dir / 'eval.jsonl', 'a')
    t_last, t_first = time.time(), time.time()
    i = start
    for i in range(start, FLAGS.offline_steps + 1):
        redo_eval = eval_pending and i == start
        if not redo_eval:
            batch = train_dataset.sample(config['batch_size'])
            agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0 and not redo_eval:
            m = {f'training/{k}': float(v) for k, v in jax.device_get(update_info).items()}
            m['time/step_time'] = (time.time() - t_last) / FLAGS.log_interval
            m['time/total_time'] = time.time() - t_first
            t_last = time.time()
            train_log.write(json.dumps(dict(step=i, **m)) + '\n')
            train_log.flush()

        if FLAGS.eval_interval and i % FLAGS.eval_interval == 0:
            t_eval = time.time()
            task_infos = env.unwrapped.task_infos
            res, succ = {}, []
            for task_id in range(1, len(task_infos) + 1):
                info = evaluate(agent, eval_env, task_id, FLAGS.eval_episodes, action_dim, obs_fn)
                if info is None:
                    save(i, pending=True)
                    print(f'SIGTERM during eval: saved at step {i} (eval pending), exiting', flush=True)
                    return
                res[f'{task_infos[task_id - 1]["task_name"]}_success'] = info['success']
                succ.append(info['success'])
            res['overall_success'] = float(np.mean(succ))
            res['eval_time'] = time.time() - t_eval
            eval_log.write(json.dumps(dict(step=i, **res)) + '\n')
            eval_log.flush()
            print(f'step {i} eval {res}', flush=True)
            t_last = time.time()

        if (i % FLAGS.save_every == 0 and i > start) or _STOP['flag']:
            save(i)
            if _STOP['flag']:
                print(f'SIGTERM: saved at step {i}, exiting', flush=True)
                return
    save(i)
    (run_dir / 'DONE').write_text(str(i))
    print('DONE', flush=True)


if __name__ == '__main__':
    app.run(main)
