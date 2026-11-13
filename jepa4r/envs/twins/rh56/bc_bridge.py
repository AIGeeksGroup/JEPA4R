# Moved from OGBench-cube@4fbb0c6:lewm_pcd/rh56_adapter/bc_bridge.py (verbatim except: imports / paths adapted to the
# package; lines wrapped at 120 columns, no lint exemptions, unused results dropped)
"""Thin RH56 boundary around the unmodified CortexBench frozen-PVR BC loop.

No BC optimizer, loss, feature-fusion or rollout algorithm is implemented here.
The CPU-only contract helpers are importable without the legacy Gym/Torch stack.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

from jepa4r.envs.twins.rh56._runtime import (RUNTIME_HELPERS_SHA256, atomic_json, file_sha256 as _sha,
                                   temporary_bindings, validate_output_location)

ENV_ID = 'rh56-relocate-v0'


def build_run_spec(mode, checkpoint_epoch, train_seed, eval_seed_base, *, bc_recovery=False):
    if mode not in ('smoke', 'profile', 'formal'):
        raise ValueError('unknown BC mode')
    formal = mode == 'formal'
    if bc_recovery and mode != 'profile':
        raise ValueError('RH56 BC recovery is an explicit separate two-epoch profile, not the T3/formal budget')
    if formal and (checkpoint_epoch != 10 or train_seed not in (100, 200, 300)):
        raise ValueError('formal BC requires epoch10 WM and head seed100/200/300')
    if eval_seed_base is None:
        eval_seed_base = 65000000 + int(train_seed) * 1000
    if formal and eval_seed_base != 65000000 + int(train_seed) * 1000:
        raise ValueError('formal evaluation must use its fixed independent seed namespace')
    return dict(
        mode=mode,
        epochs=100 if formal else 2 if bc_recovery else 1,
        num_demos=100 if formal else 1,
        eval_num_traj=25 if formal else 1,
        eval_frequency=5 if formal else 1,
        save_frequency=10 if formal else 1,
        batch_size=256,
        train_seed=int(train_seed),
        eval_seed_base=int(eval_seed_base),
        checkpoint_epoch=int(checkpoint_epoch),
        camera='opposite' if mode != 'smoke' else 'native',
        scope='formal'
        if formal
        else 'explicit two-epoch original-loop recovery case; not T3/formal SR'
        if bc_recovery
        else 'one-minibatch interface profile; not formal SR',
    )


class EvaluationSeedMap:
    """Map only official sampler seed labels, never a real demo restoration seed."""
    def __init__(self, train_seed, physical_base, count):
        self.train_seed, self.physical_base, self.count = int(train_seed), int(physical_base), int(count)

    def __call__(self, requested):
        offset = int(requested) - self.train_seed
        if not 0 <= offset < self.count:
            raise ValueError(f'sampler seed {requested} is outside the frozen evaluation range')
        return self.physical_base + offset


def validate_demo_records(records, expected_count, forbidden_seeds):
    if len(records) != expected_count:
        raise ValueError(f'expected exactly {expected_count} selected demonstrations')
    seeds = [int(r['seed']) for r in records]
    hashes = [str(r['initial_state_sha256']) for r in records]
    paths = [str(r['npz']) for r in records]
    if len(set(seeds)) != len(records) or len(set(hashes)) != len(records) or len(set(paths)) != len(records):
        raise ValueError('selected demonstrations repeat seed, physical initial state, or source path')
    if any(len(h) != 64 for h in hashes) or set(seeds).intersection(forbidden_seeds):
        raise ValueError('invalid demo state hash or demonstration/evaluation seed overlap')
    if any(
        not bool(r.get('success', False)) or r.get('coverage') != 'expert' or float(r.get('noise_std', -1)) != 0.0
        for r in records
    ):
        raise ValueError('BC demonstrations must be successful pure-expert episodes')
    return records


def numeric_flags(value, count):
    raw = np.asarray(value)
    if raw.size != count:
        raise ValueError('flag count does not match actual rollout transitions')
    flat = raw.reshape(-1)
    if raw.dtype.kind == 'O':
        if any(not isinstance(v, (bool, np.bool_, int, np.integer)) or v not in (False, True, 0, 1) for v in flat):
            raise ValueError('legacy flags must contain real scalar booleans, not arbitrary objects')
    elif raw.dtype.kind not in 'biuf' or not np.isin(flat, (0, 1)).all():
        raise ValueError('flags must be booleans or numeric 0/1')
    return flat.astype(np.bool_)


def validate_metric_rows(rows, epochs, eval_frequency):
    losses = [r for r in rows if 'epoch_loss' in r]
    evaluations = [r for r in rows if 'eval/success' in r]
    expected_evals = [e for e in range(epochs) if e > 0 and e % eval_frequency == 0 or e == epochs - 1]
    if len(losses) != epochs or [int(r['_step']) for r in losses] != list(range(1, epochs + 1)):
        raise ValueError('missing or repeated BC epoch-loss records')
    if [int(r['eval/epoch']) for r in evaluations] != expected_evals:
        raise ValueError('evaluation rounds differ from the unchanged official schedule')
    if not all(np.isfinite(float(r['epoch_loss'])) for r in losses):
        raise ValueError('nonfinite BC loss')
    success = [float(r['eval/success']) for r in evaluations]
    if not all(np.isfinite(v) and 0 <= v <= 100 for v in success):
        raise ValueError('invalid online success metric')
    return dict(loss_epochs=len(losses), evaluation_rounds=len(evaluations), evaluation_epochs=expected_evals,
                best_of_20_success_pct=max(success), final_success_pct=success[-1],
                final_loss=float(losses[-1]['epoch_loss']))


def _finite(name, *values):
    if not all(np.isfinite(np.asarray(value)).all() for value in values):
        raise ValueError(f'{name} must contain only finite values')


def demo_directory(output, cache_root=None):
    output = Path(output).resolve()
    if cache_root is None:
        return output / 'demos'
    root = Path(cache_root).resolve()
    destination = (root / output.name).resolve()
    if not destination.is_relative_to(root):
        raise ValueError('demo cache destination escaped its explicit root')
    return destination


def prepare_demo(arrays, route):
    """Convert one genuine T+1 episode to the official pre-action demo layout."""
    if route not in ('rgb', 'pcd'):
        raise ValueError('route must be rgb or pcd')
    observations = np.asarray(arrays['observations'])
    proprio = np.asarray(arrays['proprio'])
    actions = np.asarray(arrays['actions'])
    rewards = np.asarray(arrays['rewards'])
    if observations.shape != (201, 56) or proprio.shape != (201, 13):
        raise ValueError('BC demo requires observations (201,56) and proprio (201,13)')
    if actions.shape != (200, 12) or rewards.shape != (200,):
        raise ValueError('BC demo requires 200 actual actions and rewards')
    _finite('demo', observations, proprio, actions, rewards)
    if np.abs(actions).max() > 1.:
        raise ValueError('executed demo actions exceed [-1,1]')
    robot_qpos = np.concatenate((observations[:, :7], observations[:, 14:20]), axis=1)
    if not np.allclose(proprio, robot_qpos, atol=1e-6, rtol=0):
        raise ValueError('proprio must contain arm7 + active-hand6 qpos, without task GT')
    raw_state = np.asarray(arrays['initial_state_json']).item()
    state = json.loads(raw_state.decode() if isinstance(raw_state, bytes) else str(raw_state))
    if 'seed' not in state:
        raise ValueError('initial state must identify its real reset seed')
    if route == 'rgb':
        pixels = np.asarray(arrays['pixels'])
        if pixels.shape != (201, 256, 256, 3) or pixels.dtype != np.uint8:
            raise ValueError('RGB demo requires uint8 pixels (201,256,256,3)')
        images = pixels[:-1].copy()
    else:
        xyz, rgb = np.asarray(arrays['xyz']), np.asarray(arrays['rgb'])
        if xyz.shape != (201, 1200, 3) or rgb.shape != xyz.shape:
            raise ValueError('PCD demo requires paired (201,1200,3) xyz/rgb')
        _finite('point cloud', xyz, rgb)
        if np.min(rgb) < 0 or np.max(rgb) > 255:
            raise ValueError('PCD RGB must remain in 0..255 before the checkpoint transform')
        images = np.concatenate((xyz[:-1], rgb[:-1]), axis=-1).astype(np.float32)
    return dict(images=images, observations=observations[:-1].copy(),
                actions=actions.copy(), rewards=rewards.copy(),
                env_infos={'proprio': proprio[:-1].copy()}, init_state_dict=state)


class RH56LegacyCore:
    """Old Gym API only; rendering and physics stay in NativeBridge.

    Deliberately has no reset_model: MJRL would call it on the unwrapped env
    and bypass the frozen visual observation wrappers.
    """
    metadata = {'render.modes': ['rgb_array'], 'video.frames_per_second': 100}

    def __init__(self, seed, native_factory=None, route='rgb', seed_map=None):
        if native_factory is None:
            from jepa4r.envs.twins.rh56.native import NativeBridge
            native_factory = NativeBridge
        self.seed_map = seed_map
        physical_seed = int(seed) if seed_map is None else seed_map(seed)
        self.native = native_factory(seed=physical_seed, render=True)
        self.route = route
        self._next_seed = physical_seed
        self._active_physical_seed = physical_seed
        self._last_obs = None
        self.on_state_restored = None
        self.reset_seeds = []
        self.state_restore_seeds = []
        self.spec = SimpleNamespace(id=ENV_ID, max_episode_steps=200)

    @property
    def unwrapped(self):
        return self

    def seed(self, seed=None):
        if seed is not None:
            self._next_seed = int(seed) if self.seed_map is None else self.seed_map(seed)
        return [self._next_seed]

    def reset(self, seed=None, **kwargs):
        self.seed(seed)
        active_seed = self._next_seed
        self._next_seed += 1
        self._active_physical_seed = active_seed
        self._last_obs, _ = self.native.reset(seed=active_seed)
        self.reset_seeds.append(active_seed)
        _finite('reset observation', self._last_obs)
        return self._last_obs.copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (12,):
            raise ValueError(f'RH56 action must be (12,), received {action.shape}')
        _finite('action', action)
        executed = np.clip(action, -1., 1.)
        self._last_obs, reward, terminated, truncated, info = self.native.step(executed)
        info = dict(info)
        info['proprio'] = self.get_proprio()
        info['episode_seed'] = np.int64(self._active_physical_seed)
        info['executed_action'] = executed.copy()
        info['TimeLimit.truncated'] = bool(truncated and not terminated)
        _finite('step', self._last_obs, reward, info['proprio'])
        return self._last_obs.copy(), float(reward), bool(terminated or truncated), info

    def get_obs(self):
        if self._last_obs is None:
            return np.asarray(self.native.env._obs()).copy()
        return self._last_obs.copy()

    def get_proprio(self):
        value = np.asarray(self.native.proprio(), dtype=np.float32)
        if value.shape != (13,):
            raise ValueError('RH56 proprio must be arm7 + active-hand6')
        _finite('proprio', value)
        return value.copy()

    def get_env_infos(self):
        # MJRL prefers this result over step info when nonempty. Return empty
        # so real post-step success/overload is retained, including the last step.
        return {}

    def get_env_state(self):
        return self.native.get_state()

    def set_env_state(self, value):
        if set(value) == {'seed'}:
            # A demonstration is an actual physical reset seed, not a sampler label.
            physical_seed = int(value['seed'])
            self._last_obs, _ = self.native.reset(seed=physical_seed)
            self._active_physical_seed = physical_seed
            self.reset_seeds.append(physical_seed)
            self._next_seed = physical_seed + 1
        else:
            self.native.set_state(value)
            self._last_obs = np.asarray(self.native.env._obs()).copy()
            self._active_physical_seed = int(value['seed'])
        self.state_restore_seeds.append(int(value['seed']))
        if self.on_state_restored is not None:
            self.on_state_restored()

    def get_image(self):
        capture = self.native.capture(modality=self.route)
        if self.route == 'rgb':
            return capture['pixels']
        return np.concatenate((capture['pcd_xyz'], capture['pcd_rgb'].astype(np.float32)), axis=-1)

    def render(self, mode='rgb_array', **kwargs):
        return self.native.capture(modality='rgb')['pixels']

    def close(self):
        if hasattr(self.native, 'close'):
            self.native.close()

    @staticmethod
    def evaluate_success(paths):
        if not paths:
            raise ValueError('success evaluation needs actual paths')
        success = []
        for path in paths:
            info = path['env_infos']
            achieved = np.asarray(info['success'], dtype=bool)
            overload = np.asarray(info.get('overload', False), dtype=bool)
            success.append(bool(achieved[-1]) and not bool(overload.any()))
        return 100. * float(np.mean(success))


def _model_sha(model, parameters_only=False):
    if not parameters_only:
        from jepa4r.policy.bc.relocate.encoder_identity import encoder_digest
        return encoder_digest(model)
    digest = hashlib.sha256()
    items = model.named_parameters()
    for name, tensor in sorted(items):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_demos(args, spec):
    if args.demos_manifest:
        path = Path(args.demos_manifest).resolve()
        records = json.loads(path.read_text(encoding='utf-8'))['selected']
        for record in records:
            source = Path(record['npz'])
            record['npz'] = str(source if source.is_absolute() else path.parent / source)
    else:
        if not args.demo or spec['num_demos'] != 1:
            raise ValueError('formal BC requires a selected-demos manifest; smoke/profile requires one demo')
        with np.load(args.demo, allow_pickle=False) as arrays:
            initial = json.loads(str(np.asarray(arrays['initial_state_json']).item()))
            record = dict(seed=int(initial['seed']), npz=str(Path(args.demo).resolve()),
                          success=bool(np.asarray(arrays['success']).item()), coverage='expert', noise_std=0.,
                          initial_state_sha256=str(np.asarray(arrays['initial_state_sha256']).item()))
        records = [record]
    validate_demo_records(
        records, spec['num_demos'], range(spec['eval_seed_base'], spec['eval_seed_base'] + spec['eval_num_traj'])
    )
    if spec['mode'] == 'formal' and any(not 64000000 <= int(r['seed']) < 65000000 for r in records):
        raise ValueError('formal demonstrations must stay inside the independent 64000000 namespace')
    demos, receipts = [], []
    for record in records:
        path = Path(record['npz']).resolve(); digest = _sha(path)
        if record.get('npz_sha256') and record['npz_sha256'] != digest:
            raise ValueError(f'selected demonstration hash mismatch: {path}')
        with np.load(path, allow_pickle=False) as arrays:
            if not bool(np.asarray(arrays['success']).item()):
                raise ValueError('selected demonstration has unsuccessful producer receipt')
            if str(np.asarray(arrays['initial_state_sha256']).item()) != record['initial_state_sha256']:
                raise ValueError('selected demonstration physical state hash mismatch')
            if spec['camera'] == 'opposite':
                np.testing.assert_allclose(arrays['camera_xpos'], [0., 1.2, 1.2], rtol=0, atol=1e-12)
                matrix = np.asarray(arrays['camera_xmat'])
                if (
                    matrix.shape != (3, 3)
                    or -matrix[1, 2] >= 0
                    or -matrix[2, 2] >= 0
                    or float(arrays['camera_fovy']) != 45.0
                ):
                    raise ValueError('demonstration does not use the frozen opposite camera')
            demo = prepare_demo(arrays, args.route)
        if int(demo['init_state_dict']['seed']) != int(record['seed']):
            raise ValueError('selected record and actual initial-state seed differ')
        from jepa4r.envs.twins.rh56.native import physical_hash
        if physical_hash(demo['init_state_dict']) != record['initial_state_sha256']:
            raise ValueError('actual physical initial state differs from the selected state hash')
        demos.append(demo)
        receipts.append(dict(npz=str(path), seed=int(record['seed']), npz_sha256=digest,
                             initial_state_sha256=record['initial_state_sha256']))
    return demos, receipts


def install_bc_runtime(official, gym_wrapper, *, load_embedding, env_constructor,
                       logger, make_policy, sample_paths, train_rollout):
    """Scope the existing official loop's RH56 boundary and observation-only hooks."""
    return temporary_bindings(
        (gym_wrapper, 'load_pretrained_model', load_embedding),
        (gym_wrapper, 'get_proprioception', lambda env, suite: env.unwrapped.get_proprio()),
        (official, 'load_pretrained_model', load_embedding),
        (official, 'env_constructor', env_constructor),
        (official, 'setup_wandb', lambda config: logger),
        (official, 'BatchNormMLP', make_policy),
        (official, 'sample_paths', sample_paths),
        (official, 'rollout_from_init_states', train_rollout))


def run_bc(args):
    """Adapt inputs/constructors, then call the existing official training loop."""
    out = Path(args.out).resolve()
    baseline = Path(args.baseline).resolve()
    validate_output_location(out, baseline)
    if getattr(args, 'demo_cache_root', None):
        cache = demo_directory(out, args.demo_cache_root)
        validate_output_location(cache, baseline)
    if out.exists():
        raise FileExistsError(out)
    from jepa4r.envs.twins.rh56.input_identity import require_bc_source
    official_source_binding = require_bc_source(args.eai_root)
    out.mkdir(parents=True, exist_ok=False)
    recovery_enabled = getattr(args, 'bc_recovery', False)
    if recovery_enabled:
        from jepa4r.models.relocate.recovery import RecoveryPause
    spec = build_run_spec(args.mode, args.epoch, args.train_seed, args.eval_seed_base, bc_recovery=recovery_enabled)
    state = dict(phase='preflight', started=time.time(), updated=None, route=args.route, mode=args.mode,
                 completed=0, completed_epochs=0, failures=0, exit_code=None, run_spec=spec)

    def save_state():
        state['updated'] = time.time()
        atomic_json(out / 'state.json', state)

    save_state()
    original_cwd = Path.cwd()
    logger = None
    environments = []
    optimizer_hook_handles = []
    bc_controller = None
    try:
        baseline, eai = Path(args.baseline).resolve(), Path(args.eai_root).resolve()
        vi = eai / 'cortexbench/mujoco_vc/visual_imitation'
        for path in (baseline.parent, baseline / 'policy/bc/relocate/cortexbench', eai,
                     eai / 'vc_models/src', eai / 'cortexbench/mujoco_vc/src', vi):
            sys.path.insert(0, str(path))
        import torch
        import gym
        from PIL import Image
        from mjrl.utils.gym_env import GymEnv
        import mujoco_vc.gym_wrapper as gw
        import lewm_cortex
        official = importlib.import_module('train_loop')
        if Path(official.__file__).resolve() != (vi / 'train_loop.py').resolve():
            raise RuntimeError('BC loop resolved outside the explicit official dependency')
        if _sha(official.__file__) != official_source_binding['sha256']:
            raise ValueError('EAI BC source SHA changed before original training call')
        torch.set_num_threads(1)
        demo_paths, demo_receipts = load_demos(args, spec)
        demos = demo_directory(out, args.demo_cache_root)
        demos.mkdir(parents=True, exist_ok=False)
        with (demos / f'{ENV_ID}.pickle').open('xb') as stream:
            pickle.dump(demo_paths, stream, protocol=4)
        del demo_paths
        (out / 'demo_receipts.json').write_text(json.dumps(demo_receipts, indent=2))

        loaded_embeddings = []

        def load_embedding(embedding_name, *unused_args, **unused_kwargs):
            if args.route == 'rgb':
                model, dimension, transform, metadata = lewm_cortex.load_lewm(
                    args.checkpoint, epoch=args.epoch, use_projector=True)
                final_transform = lambda frame: transform(Image.fromarray(np.asarray(frame, dtype=np.uint8))).unsqueeze(
                    0
                )
            else:
                import yaml
                cfg = yaml.safe_load((Path(args.checkpoint) / 'config.yaml').read_text())
                pre = cfg['data']['preprocessor']
                model, dimension, transform, metadata = lewm_cortex.load_lewm_pcd(
                    args.checkpoint, epoch=args.epoch, center=pre['center'], scale=pre['scale'],
                    use_rgb=pre['use_rgb'])
                final_transform = lambda cloud: transform(cloud).unsqueeze(0)
            if dimension != 192:
                raise ValueError(f'expected original latent192, received {dimension}')
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            loaded_embeddings.append((model, _model_sha(model)))
            return model, dimension, final_transform, metadata

        class LegacyEnv(RH56LegacyCore, gym.Env):
            def __init__(self, seed):
                if spec['camera'] == 'opposite':
                    from jepa4r.envs.twins.rh56.camera_preview import OppositeCameraBridge
                    native_factory = OppositeCameraBridge
                else:
                    from jepa4r.envs.twins.rh56.native import NativeBridge
                    native_factory = NativeBridge
                RH56LegacyCore.__init__(self, seed, native_factory=native_factory, route=args.route,
                    seed_map=EvaluationSeedMap(spec['train_seed'], spec['eval_seed_base'], spec['eval_num_traj']))
                self.action_space = gym.spaces.Box(-1., 1., shape=(12,), dtype=np.float32)
                self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(56,), dtype=np.float32)

        class PixelWrapper(gw.MuJoCoPixelObsWrapper):
            def get_image(self):
                return self.unwrapped.get_image()

        def env_constructor(**kwargs):
            core = LegacyEnv(kwargs['seed'])
            pixel = PixelWrapper(core, 256, 256, kwargs.get('camera_name', 'vil_camera'))
            frozen = gw.FrozenEmbeddingWrapper(
                pixel, embedding_name=kwargs['embedding_name'], suite='adroit',
                history_window=3, fuse_embeddings=kwargs['fuse_embeddings'],
                device=args.device, seed=kwargs['seed'], add_proprio=True)
            core.on_state_restored = lambda: frozen.embedding_buffer.clear()
            environment = GymEnv(frozen)
            if environment.observation_dim != 589 or environment.action_dim != 12:
                raise ValueError('BC boundary must be FLARE576 + proprio13 -> action12')
            environments.append(environment)
            return environment

        class JsonLogger:
            def __init__(self):
                self.stream = (out / 'metrics.jsonl').open('x')

            def log(self, data=None, step=None, **kwargs):
                record = {key: float(value) for key, value in (data or {}).items()}
                if not all(np.isfinite(v) for v in record.values()):
                    raise RuntimeError('nonfinite official BC metric')
                if step is not None:
                    record['_step'] = int(step)
                self.stream.write(json.dumps(record) + '\n')
                self.stream.flush()
                if 'epoch_loss' in record:
                    state['completed_epochs'] = int(step)
                    state['latest_loss'] = record['epoch_loss']
                    if not policies or not all(torch.isfinite(p).all() for p in policies[0][0].model.parameters()):
                        raise RuntimeError('nonfinite trained BC head parameter')
                if 'eval/success' in record:
                    state['latest_eval_success_pct'] = record['eval/success']
                save_state()

        logger = JsonLogger()
        policies, online_paths, train_initial_paths = [], [], []
        optimizer_evidence = dict(steps=0, finite_gradient_checks=0)
        optimizers = []
        original_policy = official.BatchNormMLP
        original_sampler = official.sample_paths
        original_train_rollout = official.rollout_from_init_states
        policy_references = []
        evaluation_epochs = [e for e in range(spec['epochs'])
                             if e > 0 and e % spec['eval_frequency'] == 0 or e == spec['epochs'] - 1]

        def make_policy(*policy_args, **policy_kwargs):
            policy = original_policy(*policy_args, **policy_kwargs)
            policies.append((policy, _model_sha(policy.model, parameters_only=True)))
            return policy

        def sample_paths(*sample_args, **sample_kwargs):
            paths = original_sampler(*sample_args, **sample_kwargs)
            if getattr(args, 'policy_evidence', False):
                from jepa4r.policy.bc.relocate.policy_evidence import write_policy_reference
                index = len(policy_references)
                policy_references.append(write_policy_reference(out / 'policy_evidence', f'round{index:03d}',
                    sample_kwargs['policy'], np.asarray(paths[0]['observations']), epoch=evaluation_epochs[index],
                    score=float(np.mean([np.sum(p['rewards']) for p in paths]))))
            online_paths.extend(paths)
            return paths

        def train_rollout(*rollout_args, **rollout_kwargs):
            paths = original_train_rollout(*rollout_args, **rollout_kwargs)
            train_initial_paths.extend(paths)
            return paths

        # Public global hooks observe the original optimizer; neither Adam nor
        # the policy is replaced or modified, and policy pickles stay ordinary.
        from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook
        def is_head_optimizer(opt):
            if not policies:
                return False
            actual = {id(p) for group in opt.param_groups for p in group['params']}
            expected = {id(p) for p in policies[0][0].model.parameters()}
            return actual == expected
        def before_step(opt, unused_args, unused_kwargs):
            if not is_head_optimizer(opt):
                return
            if not isinstance(opt, torch.optim.Adam):
                raise RuntimeError('official BC head optimizer changed from Adam')
            if not optimizers:
                optimizers.append(opt)
            elif optimizers[0] is not opt:
                raise RuntimeError('unexpected second BC head optimizer')
            grads = [p.grad for group in opt.param_groups for p in group['params'] if p.grad is not None]
            if not grads or not torch.isfinite(torch.cat([g.detach().reshape(-1) for g in grads])).all():
                raise RuntimeError('missing or nonfinite actual BC optimizer gradients')
            optimizer_evidence['finite_gradient_checks'] += 1
        def after_step(opt, unused_args, unused_kwargs):
            if is_head_optimizer(opt):
                optimizer_evidence['steps'] += 1
        optimizer_hook_handles.extend((register_optimizer_step_pre_hook(before_step),
                                       register_optimizer_step_post_hook(after_step)))
        config = dict(seed=spec['train_seed'], num_demos=spec['num_demos'], device=args.device, data_dir=str(demos),
                      epochs=spec['epochs'], eval_frequency=spec['eval_frequency'],
                      save_frequency=spec['save_frequency'], eval_num_traj=spec['eval_num_traj'], num_cpu=1,
                      job_name='policy', algorithm='BC', pixel_based=True,
                      env_kwargs=dict(env_name=ENV_ID, suite='adroit', device=args.device,
                                      image_width=256, image_height=256, camera_name='vil_camera',
                                      embedding_name='rh56_' + args.mode + '_' + args.route, pixel_based=True,
                                      render_gpu_id=0, seed=spec['train_seed'], history_window=3,
                                      add_proprio=True, proprio_key='proprio'),
                      bc_kwargs=dict(hidden_sizes='(256, 256, 256)', nonlinearity='relu',
                                     loss_type='MSE', batch_size=256, lr=.001, dropout=0.))
        from contextlib import nullcontext
        bc_context = nullcontext(None)
        if recovery_enabled:
            from jepa4r.policy.bc.relocate.bc_recovery import install_bc_recovery, RecoveryPause, FROZEN_EAI_SHA256
            from jepa4r.policy.bc.relocate.bc_caller_state import (
                save_caller_state,
                load_caller_state,
                epoch_metric_prefix,
            )
            from jepa4r.policy.bc.relocate import bc_caller_state, bc_recovery
            from jepa4r.policy.bc.relocate import encoder_identity
            bc_identity = dict(
                model=dict(kind='original-EAI-BatchNormMLP', input_dim=589, action_dim=12),
                data=dict(demos_sha256=_sha(demos / f'{ENV_ID}.pickle'), demo_receipts=demo_receipts),
                config={key: value for key, value in config.items() if key not in ('data_dir', 'job_name')},
                split='RH56-pure-expert-one-demo-recovery',
                action_dim=12,
                encoder=dict(
                    route=args.route,
                    weights_sha256=_sha(Path(args.checkpoint) / f'weights_epoch_{args.epoch}.pt'),
                    config_sha256=_sha(Path(args.checkpoint) / 'config.yaml'),
                    epoch=args.epoch,
                    loader_sha256=_sha(lewm_cortex.__file__),
                ),
                source=dict(
                    eai_train_loop_sha256=FROZEN_EAI_SHA256,
                    adapter_sha256=_sha(__file__),
                    caller_state_sha256=_sha(bc_caller_state.__file__),
                    recovery_sha256=_sha(bc_recovery.__file__),
                    encoder_identity_sha256=_sha(encoder_identity.__file__),
                ),
            )
            def recovery_event(event, live):
                if event['kind'] != 'resume_loaded':
                    return
                opt = live['optimizer']
                steps = sorted({float(value['step']) for value in opt.state.values() if 'step' in value})
                saved = load_caller_state(
                    args.bc_resume_checkpoint,
                    identity=bc_identity,
                    committed_updates=event['committed_updates'],
                    model_sha256=_model_sha(live['policy'].model, parameters_only=True),
                    adam_steps=steps,
                )
                counts = saved['optimizer_evidence']
                if counts['steps'] != event['committed_updates'] or counts['finite_gradient_checks'] != counts['steps']:
                    raise ValueError('RH56 restored caller observer count differs')
                optimizer_evidence.update(counts)
                if optimizers:
                    raise ValueError('RH56 optimizer ran before resume acceptance')
                optimizers.append(opt)
                if any('evaluation' in record for record in event['epoch_records']):
                    raise ValueError('bounded RH56 recovery unexpectedly contains an earlier evaluation')
                for row in epoch_metric_prefix(event['epoch_records']):
                    logger.log({key: value for key, value in row.items() if key != '_step'}, step=row['_step'])
                state['restored_epochs'] = len(event['epoch_records'])
                state['restored_optimizer_updates'] = event['committed_updates']
                save_state()
            bc_context = install_bc_recovery(
                official,
                bc_identity,
                out / 'recovery',
                enabled=True,
                resume_from=args.bc_resume_checkpoint,
                pause_after_updates=args.bc_pause_after_updates,
                observer=recovery_event,
            )
        def seal_caller():
            if bc_controller is None:
                return None
            if len(policies) != 1 or bc_controller.last_checkpoint is None:
                raise ValueError('RH56 caller has no sealed original head checkpoint')
            return save_caller_state(bc_controller.last_checkpoint, identity=bc_identity,
                state=dict(optimizer_evidence=optimizer_evidence), committed_updates=len(bc_controller.records),
                model_sha256=_model_sha(policies[0][0].model, parameters_only=True))
        (out / 'job_config.json').write_text(json.dumps(config, indent=2))
        state['phase'] = 'official_bc_train_and_online_evaluation'
        save_state()
        os.chdir(out)
        with install_bc_runtime(official, gw, load_embedding=load_embedding,
                env_constructor=env_constructor, logger=logger, make_policy=make_policy,
                sample_paths=sample_paths, train_rollout=train_rollout):
            with bc_context as bc_controller:
                official.bc_pvr_train_loop(config)
        caller_state = seal_caller()
        for handle in optimizer_hook_handles:
            handle.remove()
        os.chdir(original_cwd)
        logger.stream.close()
        state['phase'] = 'reload_and_verify'
        save_state()
        rows = [json.loads(row) for row in (out / 'metrics.jsonl').read_text().splitlines()]
        metrics = validate_metric_rows(rows, spec['epochs'], spec['eval_frequency'])
        expected_episodes = metrics['evaluation_rounds'] * spec['eval_num_traj']
        expected_updates = spec['epochs'] * int(np.ceil(spec['num_demos'] * 200 / spec['batch_size']))
        if (
            len(online_paths) != expected_episodes
            or len(train_initial_paths) != expected_episodes
            or len(policies) != 1
        ):
            raise RuntimeError('actual online/train-initial episode count does not match official schedule')
        if (
            optimizer_evidence['steps'] != expected_updates
            or optimizer_evidence['finite_gradient_checks'] != expected_updates
            or len(optimizers) != 1
        ):
            raise RuntimeError(f'actual original Adam updates differ from expected: {optimizer_evidence}')
        policy, initial_head_sha = policies[0]
        if _model_sha(policy.model, parameters_only=True) == initial_head_sha:
            raise RuntimeError('BC head parameters did not change')
        gradients = [p.grad for p in policy.model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('BC head gradients were missing or nonfinite')
        if not loaded_embeddings or any(_model_sha(model) != digest or
                any(p.requires_grad or p.grad is not None for p in model.parameters())
                for model, digest in loaded_embeddings):
            raise RuntimeError('frozen world-model embedding changed during BC')
        torch.save(optimizers[0].state_dict(), out / 'final_optimizer_state.pt')
        with (out / 'policy/iterations/final_policy.pickle').open('xb') as stream:
            pickle.dump(policy, stream, protocol=4)
        policy_file = out / 'policy/iterations/best_policy.pickle'
        with policy_file.open('rb') as stream:
            reloaded = pickle.load(stream)  # trusted policy produced by this invocation
        with policy_file.open('rb') as stream:
            second_reload = pickle.load(stream)
        feature = np.asarray(online_paths[0]['observations'][0])
        # Original highest-reward selection can save an earlier epoch. Do not
        # incorrectly compare that best policy with the final in-memory head.
        before = second_reload.get_action(feature)[1]['evaluation']
        after = reloaded.get_action(feature)[1]['evaluation']
        _finite('reloaded policy', before, after)
        if feature.shape != (589,) or np.asarray(after).shape != (12,) or not np.array_equal(before, after):
            raise RuntimeError('saved BC policy reload changed its action')
        replay_evidence = {}
        if getattr(args, 'policy_evidence', False):
            import subprocess
            from jepa4r.policy.bc.relocate.policy_evidence import select_best_reference
            selected = select_best_reference(policy_references)
            for role, policy_path, reference in (
                    ('selected', policy_file, selected),
                    ('final', out / 'policy/iterations/final_policy.pickle', policy_references[-1])):
                replay_output = out / ('policy_replay_' + role + '.json')
                command = [sys.executable, '-B', '-m', 'jepa4r.policy.bc.relocate.policy_evidence',
                           '--policy', str(policy_path), '--reference', reference['metadata_path'],
                           '--output', str(replay_output)]
                with (out / ('policy_replay_' + role + '.log')).open('x') as log:
                    actual = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                (out / ('policy_replay_' + role + '.exit')).write_text(str(actual.returncode) + '\n')
                if actual.returncode:
                    raise RuntimeError('independent real-input policy replay failed: ' + role)
                replay_evidence[role] = dict(epoch=reference['epoch'],
                    reference=str(Path(reference['metadata_path']).relative_to(out)),
                    result=json.loads(replay_output.read_text()))
        episode_records = []
        for role, paths in (('heldout_eval', online_paths), ('train_initial', train_initial_paths)):
            for index, path in enumerate(paths):
                T = len(path['actions'])
                _finite('actual online rollout', path['observations'], path['actions'], path['rewards'])
                flags = numeric_flags(path['env_infos']['success'], T)
                overload = numeric_flags(path['env_infos']['overload'], T)
                seeds = np.asarray(path['env_infos']['episode_seed'], dtype=np.int64).reshape(-1)
                executed = np.asarray(path['env_infos']['executed_action'], dtype=np.float32)
                if seeds.shape != (T,) or not np.all(seeds == seeds[0]) or executed.shape != (T, 12):
                    raise RuntimeError('actual rollout seed or executed-action trace changed inside an episode')
                if role == 'heldout_eval':
                    expected_seed = spec['eval_seed_base'] + index % spec['eval_num_traj']
                    if seeds[0] != expected_seed:
                        raise RuntimeError('actual held-out rollout escaped its evaluation seed namespace')
                elif int(seeds[0]) not in {r['seed'] for r in demo_receipts}:
                    raise RuntimeError('training-initial rollout does not restore a selected demonstration')
                if np.abs(executed).max() > 1 or not np.array_equal(
                    executed, np.clip(np.asarray(path['actions'], dtype=np.float32), -1.0, 1.0)
                ):
                    raise RuntimeError('saved actual controls disagree with native execution clipping')
                round_index = index // spec['eval_num_traj']
                record = dict(role=role, round_index=round_index, epoch=metrics['evaluation_epochs'][round_index],
                              seed=int(seeds[0]), steps=T, success=bool(flags[-1] and not overload.any()),
                              overload=bool(overload.any()), return_value=float(np.sum(path['rewards'])))
                episode_records.append(record)
                # Retain complete final-round numeric trajectories. Every earlier
                # evaluation is still recorded in metrics and the episode receipt.
                if role == 'heldout_eval' and round_index == metrics['evaluation_rounds'] - 1:
                    destination = out / f'final_eval_{index % spec["eval_num_traj"]:03d}.npz'
                    np.savez_compressed(destination, observations=path['observations'], actions=executed,
                                        policy_actions=path['actions'], rewards=path['rewards'],
                                        success=flags, overload=overload, episode_seed=seeds)
                    with np.load(destination, allow_pickle=False) as saved:
                        if any(saved[k].dtype.kind == 'O' for k in saved.files):
                            raise RuntimeError('new online artifact contains object/pickle arrays')
        with (out / 'evaluation_episodes.jsonl').open('x') as stream:
            for record in episode_records:
                stream.write(json.dumps(record) + '\n')
        result = dict(
            status='PASS_FORMAL_OFFICIAL_BC_CHAIN' if args.mode == 'formal' else 'PASS_MINIMAL_OFFICIAL_BC_CHAIN',
            route=args.route,
            checkpoint=str(Path(args.checkpoint).resolve()),
            checkpoint_epoch=args.epoch,
            checkpoint_sha256=_sha(Path(args.checkpoint) / f'weights_epoch_{args.epoch}.pt'),
            mode=args.mode,
            run_spec=spec,
            demo_receipts=demo_receipts,
            demo_pickle_cache_directory=str(demos),
            demos_manifest_sha256=_sha(args.demos_manifest) if args.demos_manifest else None,
            demo_seed=demo_receipts[0]['seed'] if len(demo_receipts) == 1 else None,
            eval_seed=spec['eval_seed_base'],
            train_seed=spec['train_seed'],
            eval_seed_base=spec['eval_seed_base'],
            official_loop=str(Path(official.__file__).resolve()),
            official_loop_sha256=_sha(official.__file__),
            adapter_sha256=_sha(__file__),
            runtime_helpers_sha256=RUNTIME_HELPERS_SHA256,
            policy_sha256=_sha(policy_file),
            final_policy_sha256=_sha(out / 'policy/iterations/final_policy.pickle'),
            input_dim=589,
            action_dim=12,
            loss=metrics['final_loss'],
            metrics=metrics,
            optimizer_evidence=optimizer_evidence,
            expected_adam_updates=expected_updates,
            encoder_frozen_and_unchanged=True,
            head_updated=True,
            finite_head_gradients=True,
            policy_reload_max_abs=0.0,
            actual_online_episodes=len(online_paths),
            online_steps=sum(len(p['actions']) for p in online_paths),
            online_success_pct=metrics['final_success_pct'],
            actual_train_initial_episodes=len(train_initial_paths),
            reset_seeds=[env.env.unwrapped.reset_seeds for env in environments],
            state_restore_seeds=[env.env.unwrapped.state_restore_seeds for env in environments],
            source_seed_partition='pure BC demos separate from paired held-out evaluation seeds',
            trusted_native_bundles=[getattr(env.env.unwrapped.native, 'bundle_identity', None) for env in environments],
            policy_selection='original highest-mean-reward best_policy; Best-of-20 SR reported separately',
            policy_evidence=replay_evidence,
            recovery=dict(
                enabled=True,
                checkpoint=str(bc_controller.last_checkpoint),
                caller_state=str(caller_state),
                batch_records=bc_controller.records,
                epoch_records=bc_controller.epoch_records,
            )
            if bc_controller is not None
            else dict(enabled=False),
            scope=spec['scope'],
        )
        (out / 'validation.json').write_text(json.dumps(result, indent=2))
        state.update(phase='complete', completed=1, exit_code=0)
        save_state()
        print(json.dumps(result, indent=2), flush=True)
        return result
    except BaseException as error:
        if recovery_enabled and isinstance(error, RecoveryPause):
            # A failed caller seal remains a genuine error, never an accepted75.
            try:
                caller_state = seal_caller()
            except BaseException:
                state.update(phase='failed_caller_seal', failures=1, exit_code=1, error=traceback.format_exc())
                save_state()
                raise
            state.update(phase='intentional_sealed_pause', completed=0, failures=0, exit_code=75,
                         committed_optimizer_updates=len(bc_controller.records),
                         checkpoint=str(bc_controller.last_checkpoint), caller_state=str(caller_state))
            save_state()
            raise SystemExit(75) from error
        state.update(phase='failed', failures=1, exit_code=1, error=traceback.format_exc())
        save_state()
        raise
    finally:
        for handle in optimizer_hook_handles:
            handle.remove()
        os.chdir(original_cwd)
        if logger is not None and not logger.stream.closed:
            logger.stream.close()
        for env in environments:
            env.env.unwrapped.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=('rgb', 'pcd'), required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--epoch', type=int, default=1)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--demo')
    selection.add_argument('--demos-manifest')
    parser.add_argument('--demo-cache-root', help='Optional temporary input-pickle root; no cache reuse')
    parser.add_argument('--out', required=True)
    parser.add_argument('--baseline', '--baseline-root', dest='baseline',
                        default=str(Path(__file__).resolve().parents[3]),
                        help='Existing LeWM source root; defaults to this checkout')
    parser.add_argument('--eai-root', '--eai-vc-root', dest='eai_root',
                        default=os.environ.get('CORTEXBENCH_EAI_VC_ROOT',
                            str(Path(__file__).resolve().parents[4] / 'third_party/eai-vc')),
                        help='External official eai-vc checkout; CORTEXBENCH_EAI_VC_ROOT also supported')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--mode', choices=('smoke', 'profile', 'formal'), default='smoke')
    parser.add_argument('--train-seed', type=int, default=100)
    parser.add_argument('--eval-seed-base', '--eval-seed', type=int)
    parser.add_argument(
        '--policy-evidence',
        action='store_true',
        help='Retain real policy inputs and verify corresponding selected/final policy outputs in independent '
             'processes',
    )
    parser.add_argument(
        '--bc-recovery', action='store_true', help='Explicit separate two-epoch profile recovery; default off'
    )
    parser.add_argument('--bc-resume-checkpoint', type=Path)
    parser.add_argument('--bc-pause-after-updates', type=int)
    args = parser.parse_args(argv)
    if (args.bc_resume_checkpoint is not None or args.bc_pause_after_updates is not None) and not args.bc_recovery:
        parser.error('BC checkpoint/pause requires --bc-recovery')
    if args.bc_recovery and args.mode != 'profile':
        parser.error('RH56 recovery requires its explicit separate profile mode')
    if args.bc_pause_after_updates is not None and args.bc_pause_after_updates != 1:
        parser.error('bounded RH56 recovery pauses after exactly one update')
    return run_bc(args)


if __name__ == '__main__':
    main()
