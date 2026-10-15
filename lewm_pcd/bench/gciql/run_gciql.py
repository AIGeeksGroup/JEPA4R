"""GCIQL (OGBench's goal-conditioned implicit Q-learning: github.com/seohongpark/ogbench at a977253, impls/, in
third_party/ogbench unchanged) trained and evaluated on OGBench puzzle-3x3-play (the official data), the setting of the
DQC runs (bench/dqc/run_dqc.py) with GCIQL in place of DQC. The inputs, everything else equal:
  --obs state  : the original input, the 55-d state vector;
  --obs lewm2d : [the frozen LeWM 2D encoder's 192-d latent, the 9 true lights];
  --obs lewm3d : [the frozen LeWM 3D encoder's 192-d latent, the 9 true lights];
  --obs lewm2d_nolight / lewm3d_nolight : the 192-d latent only.
The goal is the 9 lights' goal state (OGBench's oraclerep, 'puzzle-3x3-play-oraclerep-v0'), as the DQC runs.
Agent: impls/agents/gciql.py with its default config and impls/hyperparameters.sh's puzzle-3x3-play-v0 line
(--agent.alpha=1.0); training loop and evaluation as impls/main.py (seed 0, 1 M steps, evaluation at step 1 and every
100 k steps on the 5 tasks x --eval_episodes=50, eval_temperature 0), with these changes:
  1. oraclerep goals: GCIQLAgent.create sizes its networks with observation-sized goals (ex_goals = ex_observations);
     GCIQLOracleRep.create is that method with this one line replaced by goals of the oracle representation's size (as
     the OGBench authors' later code does, third_party/dqc/agents/dqc.py), and OracleRepGCDataset hands GCDataset the
     oracle representations where it reads goal observations (GCDataset.get_observations, used only for goals here);
  2. LeWM inputs: the training set's observations are extract_feats.py's latents (+ the lights); in the evaluation, each
     step's observation is the current frame of the LeWM data's camera through the same frozen encoder
     (bench/dqc/lewm_obs.LeWMObs, the checkpoint given by --ckpt);
  3. no wandb (jsonl), no validation loss, evaluation on the GPU (main.py's default evaluates a CPU copy of the same agent),
     restartable: every --save_every steps the full state (the agent, numpy / python RNG states, the step) goes to disk,
     on SIGTERM once more before exiting.
    python run_gciql.py --obs lewm3d --ckpt $STABLEWM_HOME/checkpoints/puzzle_3x3_pcd_dexart \\
        --feats_dir $STABLEWM_HOME/datasets/ogbench/dqc_feats/puzzle_3x3 --run_dir <dir>
"""
import copy
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
IMPLS = HERE.parents[1] / 'third_party' / 'ogbench' / 'impls'
sys.path.insert(0, str(IMPLS))  # impls' packages agents and utils come first
sys.path.insert(1, str(HERE.parent / 'dqc'))  # lewm_obs
DATA = Path(os.environ.get('STABLEWM_HOME', '.')) / 'datasets' / 'ogbench'

import flax  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
import ogbench  # noqa: E402
import optax  # noqa: E402
from absl import app, flags  # noqa: E402
from agents.gciql import GCIQLAgent  # noqa: E402
from ml_collections import config_flags  # noqa: E402
from utils.datasets import Dataset, GCDataset  # noqa: E402
from utils.encoders import GCEncoder, encoder_modules  # noqa: E402
from utils.evaluation import add_to, flatten, supply_rng  # noqa: E402
from utils.flax_utils import ModuleDict, TrainState  # noqa: E402
from utils.networks import GCActor, GCDiscreteActor, GCDiscreteCritic, GCValue  # noqa: E402

from lewm_obs import LeWMObs  # noqa: E402

FLAGS = flags.FLAGS
flags.DEFINE_string('obs', 'state', 'state | lewm2d | lewm3d | lewm2d_nolight | lewm3d_nolight')
flags.DEFINE_string('run_dir', None, 'Output directory (checkpoint, logs).')
flags.DEFINE_string('env_name', 'puzzle-3x3-play-oraclerep-v0', '')
flags.DEFINE_string('dataset_path', str(DATA / 'official' / 'puzzle-3x3-play-v0.npz'), '')
flags.DEFINE_string('feats_dir', None, "extract_feats.py's output directory (LeWM inputs).")
flags.DEFINE_string('ckpt', None, "Checkpoint directory of the LeWM encoder (LeWM inputs; extract_feats.py's).")
flags.DEFINE_integer('lewm_epoch', 10, 'Epoch of the LeWM weights.')
flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_integer('train_steps', 1000000, '')
flags.DEFINE_integer('log_interval', 5000, '')
flags.DEFINE_integer('eval_interval', 100000, '')
flags.DEFINE_integer('eval_episodes', 50, 'Evaluation episodes per task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_integer('save_every', 25000, 'Steps between restart checkpoints.')
config_flags.DEFINE_config_file('agent', str(IMPLS / 'agents' / 'gciql.py'), lock_config=False)

_STOP = {'flag': False}


def _on_term(signum, frame):
    _STOP['flag'] = True


class OracleRepGCDataset(GCDataset):
    """GCDataset with goals = the dataset's oracle representations: GCDataset reads value / actor goals through
    get_observations (and nothing else, without frame stacking), so returning the oracle representations there makes
    every goal the 9 lights of the sampled goal state; sampling, rewards and masks are GCDataset's."""

    def __post_init__(self):
        assert self.config['frame_stack'] is None
        super().__post_init__()

    def get_observations(self, idxs):
        return self.dataset['oracle_reps'][idxs]


class GCIQLOracleRep(GCIQLAgent):
    """GCIQLAgent with networks for goals of their own size (see the module docstring); losses, updates and action
    sampling are GCIQLAgent's."""

    @classmethod
    def create(cls, seed, ex_observations, ex_goals, ex_actions, config):
        # GCIQLAgent.create (impls/agents/gciql.py) as is, except `ex_goals = ex_observations` -> the argument ex_goals
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(concat_encoder=encoder_module())
            encoders['critic'] = GCEncoder(concat_encoder=encoder_module())
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        # Define value and actor networks.
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=encoders.get('value'),
        )

        if config['discrete']:
            critic_def = GCDiscreteCritic(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
                action_dim=action_dim,
            )
        else:
            critic_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
            )

        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def evaluate(agent, env, task_id, num_eval_episodes, eval_temperature, obs_fn=None):
    """impls/utils/evaluation.py's evaluate (continuous actions, no video, no Gaussian noise) with obs_fn added: the
    observation the policy gets at each step."""
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    stats = defaultdict(list)
    for _ in range(num_eval_episodes):
        if _STOP['flag']:
            return None  # SIGTERM: give up this evaluation round, save and exit; it is redone on restart
        observation, info = env.reset(options=dict(task_id=task_id, render_goal=False))
        goal = info.get('goal')
        done = False
        while not done:
            ob = obs_fn() if obs_fn is not None else observation
            action = np.array(actor_fn(observations=ob, goals=goal, temperature=eval_temperature))
            action = np.clip(action, -1, 1)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        add_to(stats, flatten(info))
    return {k: float(np.mean(v)) for k, v in stats.items()}


def main(_):
    run_dir = Path(FLAGS.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _on_term)
    config = FLAGS.agent
    assert config['agent_name'] == 'gciql', config['agent_name']
    assert FLAGS.obs in ('state', 'lewm2d', 'lewm3d', 'lewm2d_nolight', 'lewm3d_nolight'), FLAGS.obs
    lewm = FLAGS.obs != 'state'
    kind = FLAGS.obs.split('_')[0]  # lewm2d / lewm3d
    with_lights = not FLAGS.obs.endswith('_nolight')
    flag_dict = {k: getattr(FLAGS, k) for k in FLAGS if '.' not in k and k in (
        'obs', 'env_name', 'dataset_path', 'feats_dir', 'ckpt', 'lewm_epoch', 'seed', 'train_steps', 'log_interval',
        'eval_interval', 'eval_episodes', 'eval_temperature', 'save_every')}
    flag_dict['agent'] = config.to_dict()
    (run_dir / 'flags.json').write_text(json.dumps(flag_dict, indent=1))

    # env and data (as impls' make_env_and_datasets: the compact dataset; add_info keeps the lights for LeWM inputs).
    # The LeWM inputs render into a 224 offscreen buffer: the rendering size only, not the simulation.
    env_kwargs = dict(width=224, height=224) if lewm else {}
    env, train_dataset, _ = ogbench.make_env_and_datasets(FLAGS.env_name, dataset_path=FLAGS.dataset_path,
                                                          compact_dataset=True, add_info=True, **env_kwargs)
    if lewm:
        z = np.load(Path(FLAGS.feats_dir) / f'train_z{kind[-2:]}.npy')
        assert len(z) == len(train_dataset['observations']), (z.shape, train_dataset['observations'].shape)
        parts = [z.astype(np.float32)] + ([train_dataset['button_states'].astype(np.float32)] if with_lights else [])
        train_dataset['observations'] = np.concatenate(parts, axis=1)
    for k in ('qpos', 'qvel', 'button_states'):
        train_dataset.pop(k, None)
    print({k: v.shape for k, v in train_dataset.items()}, flush=True)
    env.reset()

    train_dataset = OracleRepGCDataset(Dataset.create(**train_dataset), config)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    example_batch = train_dataset.sample(1)
    agent = GCIQLOracleRep.create(FLAGS.seed, example_batch['observations'], example_batch['value_goals'],
                                  example_batch['actions'], config)
    obs_fn = LeWMObs(env, kind, FLAGS.ckpt, FLAGS.lewm_epoch, lights=with_lights) if lewm else None

    # restart
    ckpt = run_dir / 'resume.pkl'
    start = 1
    eval_pending = False
    if ckpt.exists():
        with open(ckpt, 'rb') as f:
            d = pickle.load(f)
        agent = flax.serialization.from_state_dict(agent, d['agent'])
        np.random.set_state(d['np_rng'])
        random.setstate(d['py_rng'])
        eval_pending = d.get('eval_pending', False)  # stopped during an evaluation: this step's update is done
        start = d['step'] if eval_pending else d['step'] + 1
        print(f'resumed from step {d["step"]}' + (' (eval pending)' if eval_pending else ''), flush=True)

    def save(step, pending=False):
        tmp = run_dir / 'resume.pkl.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(dict(agent=jax.device_get(flax.serialization.to_state_dict(agent)), step=step,
                             eval_pending=pending, np_rng=np.random.get_state(), py_rng=random.getstate()), f)
        os.replace(tmp, ckpt)

    train_log = open(run_dir / 'train.jsonl', 'a')
    eval_log = open(run_dir / 'eval.jsonl', 'a')
    t_last, t_first = time.time(), time.time()
    i = start
    for i in range(start, FLAGS.train_steps + 1):
        redo_eval = eval_pending and i == start
        if not redo_eval:
            batch = train_dataset.sample(config['batch_size'])
            agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0 and not redo_eval:
            m = {f'training/{k}': float(v) for k, v in jax.device_get(update_info).items()}
            m['time/epoch_time'] = (time.time() - t_last) / FLAGS.log_interval
            m['time/total_time'] = time.time() - t_first
            t_last = time.time()
            train_log.write(json.dumps(dict(step=i, **m)) + '\n')
            train_log.flush()

        if i == 1 or i % FLAGS.eval_interval == 0:
            t_eval = time.time()
            task_infos = env.unwrapped.task_infos
            res, succ = {}, []
            for task_id in range(1, len(task_infos) + 1):
                info = evaluate(agent, env, task_id, FLAGS.eval_episodes, FLAGS.eval_temperature, obs_fn)
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
