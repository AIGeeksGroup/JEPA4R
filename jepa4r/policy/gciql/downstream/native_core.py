# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/downstream/native_core.py (verbatim except: lines wrapped at
# 120 columns, no lint exemptions, unused imports removed, names the caller sets declared)
from __future__ import annotations
import copy
import hashlib
import json
import sys
import time
import importlib.util
import types
import inspect
from pathlib import Path
from typing import Any
import numpy as np
from ._source import VENDOR, IMPLS, OGBENCH_ROOT, PCD_ROOT
ROOT = VENDOR
CASE = VENDOR
OnlineEncoder = None   # set by the caller (downstream/native.py, workflow_downstream.py) before use
EVALUATOR_REFERENCE = None
ENCODING_PROTOCOL = 'fresh_current_and_goal_each_policy_call_2N'
LATENT_DIM = 192
CUBE_NAMES = {1: 'single', 2: 'double', 3: 'triple', 4: 'quadruple'}
CAMERA = {}
GROUPS = {}

def group_contract(group: str) -> dict:
    cfg = GROUPS[group]
    cubes = cfg.get('cube_count', 2)
    result = {
        'cube_count': cubes,
        'state_dim': cfg.get('state_dim', 19 + 9 * cubes),
        'current_dim': cfg.get('current_dim', 211),
        'goal_dim': cfg.get('goal_dim', 192 + 3 * cubes),
        'max_episode_steps': cfg.get('max_episode_steps', 500),
        'mode': cfg['mode'],
    }
    if cfg.get('record_capture_receipts', False):
        result['record_capture_receipts'] = True
    if cfg.get('record_state_trace', False):
        result['record_state_trace'] = True
    if cfg.get('pcd_crop_box') is not None:
        result['pcd_crop_box'] = cfg['pcd_crop_box']
    return result

def camera_for_group(group):
    if group not in GROUPS:
        raise ValueError(group)
    return dict(GROUPS[group]['camera'], **{'geometry_offsamples': 0} if GROUPS[group].get('mode') == 'pcd' else {})

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)

def seed_for(task: int, episode_index: int) -> int:
    if task not in range(1, 6) or episode_index not in range(50):
        raise ValueError('task must be 1..5 and episode_index must be 0..49')
    return 970000000 + task * 100000 + episode_index

def validate_latent(value: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(value)
    if value.shape != (LATENT_DIM,) or value.dtype != np.float32 or (not np.isfinite(value).all()):
        raise ValueError(
            f'{label} latent must be finite float32 shape ({LATENT_DIM},), got {value.shape} {value.dtype}'
        )
    return value

def goal_xyz(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state)
    if (
        state.ndim != 1
        or state.size < 28
        or (state.size - 19) % 9
        or (state.dtype != np.float32)
        or (not np.isfinite(state).all())
    ):
        raise ValueError('goal state must be finite float32 with the official 19+9N layout')
    cubes = (state.size - 19) // 9
    return np.asarray(
        [state[index] for cube in range(cubes) for index in range(19 + 9 * cube, 22 + 9 * cube)], dtype=np.float32
    )

def make_policy_inputs(
    current_latent: np.ndarray, goal_latent: np.ndarray, current_state: np.ndarray, goal_state: np.ndarray
):
    current_latent = validate_latent(current_latent, 'current')
    goal_latent = validate_latent(goal_latent, 'goal')
    current_state = np.asarray(current_state, dtype=np.float32)
    if (
        current_state.ndim != 1
        or current_state.size < 28
        or (current_state.size - 19) % 9
        or (not np.isfinite(current_state).all())
    ):
        raise ValueError('current state must use the official 19+9N layout')
    return (
        np.concatenate([current_latent, current_state[:19]]).astype(np.float32),
        np.concatenate([goal_latent, goal_xyz(np.asarray(goal_state, dtype=np.float32))]).astype(np.float32),
    )

def pcd_components(value: np.ndarray):
    value = np.asarray(value)
    if value.shape != (1200, 6) or value.dtype != np.float32 or (not np.isfinite(value).all()):
        raise ValueError('PCD must be finite float32 XYZRGB shape (1200,6)')
    return (value[:, :3].copy(), np.clip(np.rint(value[:, 3:]), 0, 255).astype(np.uint8))

def configure_camera(base, group: str) -> dict:
    import mujoco
    model = base._model
    camera = camera_for_group(group)
    source = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera['source_name'])
    target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera['render_name'])
    if source < 0 or target < 0:
        raise RuntimeError('bound source/render cameras are missing')
    position = model.cam_pos[source].copy()
    original = model.cam_quat[source].copy()
    if (
        not np.allclose(position, camera.get('source_position', camera['position']), atol=1e-12, rtol=0)
        or float(model.cam_fovy[source]) != camera['fov']
    ):
        raise RuntimeError('bound source camera pose/FOV changed')
    rotation = np.empty(4)
    actual = np.empty(4)
    mujoco.mju_axisAngle2Quat(rotation, np.array([1.0, 0.0, 0.0]), np.deg2rad(camera['local_x_up_degrees']))
    mujoco.mju_mulQuat(actual, original, rotation)
    w, x, y, z = actual
    local_z = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    model.cam_pos[target] = position + camera.get('pullback_m', 0.0) * local_z
    if not np.allclose(model.cam_pos[target], camera['position'], atol=1e-12, rtol=0):
        raise RuntimeError('Configured camera position differs from bound rotation/pullback')
    model.cam_quat[target] = actual
    model.cam_fovy[target] = camera['fov']
    model.vis.quality.offsamples = 4
    result = {**camera, 'actual_quaternion': actual.tolist(), 'source_quaternion': original.tolist()}
    result.update(camera_for_group(group))
    return result

def load_agent(group: str):
    cfg = GROUPS[group]
    contract = group_contract(group)
    training = json.loads(cfg['training'].read_text())
    acceptance = json.loads(cfg['acceptance'].read_text())
    for path_key, hash_key in (
        ('training', 'training_sha256'),
        ('acceptance', 'acceptance_sha256'),
        ('gate', 'gate_sha256'),
    ):
        if hash_key in cfg and sha(cfg[path_key]) != cfg[hash_key]:
            raise RuntimeError(f'{path_key} SHA mismatch')
    if (
        acceptance['actual_exit_code'] != 0
        or acceptance['actual_current_dim'] != contract['current_dim']
        or acceptance['actual_goal_dim'] != contract['goal_dim']
    ):
        raise RuntimeError('training acceptance does not match visual contract')
    if acceptance['final_checkpoint'] != str(cfg['final']):
        raise RuntimeError('final GCIQL checkpoint binding mismatch')
    if sha(cfg['final'] / 'manifest.json') != cfg['manifest_sha256']:
        raise RuntimeError('final GCIQL manifest SHA mismatch')
    sys.path[:0] = [str(CASE), str(ROOT), str(IMPLS), str(OGBENCH_ROOT)]
    import gciql_boundary as boundary
    directory, _sidecar = boundary.restore_saved(training, Path(training['output']), 1000000)
    with np.load(cfg['gate'], allow_pickle=False) as saved:
        example = {key: saved[key][:1].copy() for key in ('observations', 'actions', 'actor_goals')}
    agent = boundary.create_agent(boundary.model.GCIQLAgent, 0, example, boundary.agent_config(training))
    agent = boundary.restore_agent(agent, str(directory), 1000000)
    if boundary.jax.default_backend() != 'cpu' or boundary.fingerprint(agent) != acceptance['state']:
        raise RuntimeError('GCIQL CPU/backend or exact checkpoint state mismatch')
    return agent

def make_env(group: str, seed: int):
    import gymnasium as gym
    contract = group_contract(group)
    sys.path.insert(0, str(OGBENCH_ROOT))
    import ogbench
    if contract['mode'] == 'pcd':
        from ogbench.manipspace.envs.cube_env import CubeEnv as OfficialCubeEnv
        shim_package = types.ModuleType('stable_worldmodel.envs.ogbench')
        shim_package.__path__ = []
        shim_parent = types.ModuleType('stable_worldmodel.envs.ogbench.cube_env')
        shim_parent.CubeEnv = OfficialCubeEnv
        sys.modules['stable_worldmodel.envs.ogbench'] = shim_package
        sys.modules['stable_worldmodel.envs.ogbench.cube_env'] = shim_parent
        path = PCD_ROOT / 'cube_pcd_env.py'
        spec = importlib.util.spec_from_file_location('_sr_cube_pcd', path)
        pcd_env_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pcd_env_module)
        env_type = CUBE_NAMES[contract['cube_count']]
        base = pcd_env_module.CubePCDEnv(
            env_type=env_type,
            render_mode='rgb_array',
            width=224,
            height=224,
            ob_type='states',
            visualize_info=False,
            pixel_transparent_arm=False,
        )
        env = gym.wrappers.TimeLimit(base, max_episode_steps=contract['max_episode_steps'])
    else:
        play_env = 'cube-' + CUBE_NAMES[contract['cube_count']] + '-play-v0'
        env = ogbench.make_env_and_datasets(
            play_env,
            env_only=True,
            render_mode='rgb_array',
            width=224,
            height=224,
            ob_type='states',
            visualize_info=False,
            pixel_transparent_arm=False,
        )
        base = env.unwrapped
        env = gym.wrappers.TimeLimit(env, max_episode_steps=contract['max_episode_steps'])
    base.reset(seed=0)
    camera = configure_camera(base, group)
    if contract['mode'] == 'rgb':
        original_render = base.render

        def render(*args, camera=None, **kwargs):
            if camera is None and getattr(base, '_render_goal', False):
                camera = 'front_pixels'
            return original_render(*args, camera=camera, **kwargs)
        base.render = render
    proof = {
        'callback_calls': 0,
        'goal_capture_calls': 0,
        'current_capture_calls': 0,
        'physics_unchanged': True,
        'env_rng_unchanged': True,
        'global_rng_unchanged': True,
        'captures': [],
        'RGB_offsamples': 4,
    }
    if contract['mode'] == 'pcd':
        proof['geometry_offsamples'] = 0
    return (env, camera, proof)

class OnlineVisualEnv:
    """Thin reset/step wrapper: force the true goal render and retain current visual info."""

    def __init__(self, env, group: str, encoder: OnlineEncoder, seed: int):
        self.env, self.group, self.encoder, self.seed = (env, group, encoder, seed)
        self.goal_raw = None
        self.goal_encoder_calls = 0
        self.goal_raw_sha256 = None
        self.encoder_calls_at_reset = 0
        self.last_info = None
        self.frame = None
        self.capture_proof = None
        self.state_trace = None

    def capture(self, rgb, phase):
        contract = group_contract(self.group)
        base = self.env.unwrapped
        keys = ('qpos', 'qvel', 'ctrl', 'act', 'mocap_pos', 'mocap_quat', 'qacc_warmstart', 'eq_active')
        before = {k: np.asarray(getattr(base._data, k)).copy() for k in keys if hasattr(base._data, k)}
        before_time = float(base._data.time)
        rng = copy.deepcopy(base.np_random.bit_generator.state)
        global_rng = np.random.get_state()
        cloud = self.frame.capture(base, base._data, rgb, phase)
        if before_time != base._data.time or any(
            (not np.array_equal(v, getattr(base._data, k)) for k, v in before.items())
        ):
            raise RuntimeError('PCD capture changed physics')
        if rng != base.np_random.bit_generator.state:
            raise RuntimeError('PCD capture changed environment RNG')
        after_rng = np.random.get_state()
        unchanged_global_rng = (
            global_rng[0] == after_rng[0]
            and np.array_equal(global_rng[1], after_rng[1])
            and (global_rng[2:] == after_rng[2:])
        )
        self.capture_proof['global_rng_unchanged'] &= unchanged_global_rng
        if not unchanged_global_rng:
            raise RuntimeError('PCD capture changed global RNG')
        self.capture_proof['callback_calls'] += 1
        self.capture_proof[f'{phase}_capture_calls'] += 1
        if contract.get('record_capture_receipts', False):
            receipt = dict(self.frame.last_receipts[phase])
            if 'pcd_crop_box' in GROUPS[self.group]:
                receipt['training_crop_box'] = [list(row) for row in GROUPS[self.group]['pcd_crop_box']]
            self.capture_proof['captures'].append(receipt)
        return cloud

    def reset(self, *, options=None, **kwargs):
        options = dict(options or {})
        options['render_goal'] = True
        if group_contract(self.group)['mode'] == 'pcd':
            sys.path.insert(0, str(PCD_ROOT))
            from native_pcd_frame import NativePCDFrame
            path = PCD_ROOT / 'pcd_wrapper.py'
            spec = importlib.util.spec_from_file_location('_sr_pcd_main', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            crop_box = GROUPS[self.group].get('pcd_crop_box')
            core_factory = (
                module.PointCloudWrapper
                if crop_box is None
                else __import__('functools').partial(module.PointCloudWrapper, crop_box=crop_box)
            )
            self.frame = NativePCDFrame(path, PCD_ROOT / 'pcd_geometry.py', self.seed, core_factory=core_factory)
            base = self.env.unwrapped
            base._pcd_goal_capture = None
            base._pcd_goal_capture_callback = lambda: self.capture(base.render(camera='front_pixels'), 'goal')
        observation, info = self.env.reset(seed=self.seed, options=options)
        if int(self.env.unwrapped.np_random_seed) != self.seed:
            raise RuntimeError('environment did not apply the requested episode seed')
        self.last_info = info
        if group_contract(self.group)['mode'] == 'rgb':
            goal_raw = np.asarray(info['goal_rendered'])
        else:
            goal_raw = self.env.unwrapped._pcd_goal_capture
            if goal_raw is None:
                raise RuntimeError('CubePCDEnv did not capture the official goal scene')
        self.goal_raw = np.array(goal_raw, copy=True)
        self.goal_encoder_calls = 0
        self.goal_raw_sha256 = hashlib.sha256(np.ascontiguousarray(goal_raw).tobytes()).hexdigest()
        self.encoder_calls_at_reset = self.encoder.calls
        if group_contract(self.group).get('record_state_trace', False):
            from ..native_trace import NativeTrace
            self.state_trace = NativeTrace()
            self.state_trace.reset(self.env.unwrapped, observation, info['goal'])
        return (observation, info)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.state_trace is not None:
            self.state_trace.step(self.env.unwrapped, action, observation, reward, terminated, truncated, info)
        self.last_info = info
        return (observation, reward, terminated, truncated, info)

    def current_raw(self):
        if group_contract(self.group)['mode'] == 'rgb':
            return self.env.unwrapped.render(camera='front_pixels')
        rgb = self.env.unwrapped.render(camera='front_pixels')
        return self.capture(rgb, 'current')

    def close(self):
        if self.frame is not None:
            self.frame.close()
        self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)

class OnlineGCIQLPolicy:

    def __init__(self, agent, env: OnlineVisualEnv):
        self.agent, self.env = (agent, env)
        self.current_latents = []
        self.current_sha256 = []
        self.input_contracts = []

    def sample_actions(self, observations, goals=None, seed=None, temperature=0.0):
        observation = np.asarray(observations, dtype=np.float32)
        goal = np.asarray(goals, dtype=np.float32)
        if observation.shape == (1, group_contract(self.env.group)['state_dim']):
            observation = observation[0]
        if goal.shape == (1, group_contract(self.env.group)['state_dim']):
            goal = goal[0]
        goal_latent = self.env.encoder.encode(self.env.goal_raw)
        self.env.goal_encoder_calls += 1
        raw = self.env.current_raw()
        latent = self.env.encoder.encode(raw)
        current, goal_input = make_policy_inputs(latent, goal_latent, observation, goal)
        self.current_latents.append(latent)
        self.current_sha256.append(hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest())
        self.input_contracts.append(
            {
                'current_dim': int(current.shape[0]),
                'goal_dim': int(goal_input.shape[0]),
                'current_finite': bool(np.isfinite(current).all()),
                'goal_finite': bool(np.isfinite(goal_input).all()),
            }
        )
        action = self.agent.sample_actions(observations=current, goals=goal_input, seed=seed, temperature=temperature)
        if self.env.state_trace is not None:
            self.env.state_trace.policy(latent, goal_latent, current, goal_input, action)
        return action

def run_episode(group: str, agent, encoder: OnlineEncoder, task: int, index: int, output: Path):
    import random
    from utils.evaluation import evaluate
    ref = json.loads(EVALUATOR_REFERENCE.read_text())
    eval_path = Path(inspect.getfile(evaluate)).resolve()
    expected_eval = ref['source_pins'][str(IMPLS / 'utils/evaluation.py')]
    if eval_path != (IMPLS / 'utils/evaluation.py').resolve() or sha(eval_path) != expected_eval:
        raise RuntimeError('official evaluator source binding changed')
    env, camera, proof = make_env(group, seed_for(task, index))
    seed = seed_for(task, index)
    start_encoder_calls = encoder.calls
    base = env.unwrapped
    contract = group_contract(group)
    policy_env = None
    started = time.time()
    try:
        if (
            base._num_cubes != contract['cube_count']
            or len(base.task_infos) != 5
            or base._permute_blocks is not True
            or (base._use_oracle_rep is not False)
        ):
            raise RuntimeError('official task/environment contract changed')
        random.seed(seed)
        np.random.seed(seed)
        policy_env = OnlineVisualEnv(env, group, encoder, seed)
        policy_env.capture_proof = proof
        policy = OnlineGCIQLPolicy(agent, policy_env)
        train_config = json.loads(GROUPS[group]['training'].read_text())
        stats, trajectories, videos = evaluate(
            policy,
            policy_env,
            task_id=task,
            config=train_config['agent_config'],
            num_eval_episodes=1,
            num_video_episodes=0,
            eval_temperature=0,
            eval_gaussian=None,
        )
        if len(trajectories) != 1 or videos:
            raise RuntimeError('official evaluator returned unexpected episode/video counts')
        traj = trajectories[0]
        actions = np.asarray(traj['action'], dtype=np.float32)
        rewards = np.asarray(traj['reward'], dtype=np.float32)
        successes = np.asarray([bool(row.get('success', False)) for row in traj['info']], dtype=bool)
        contract = group_contract(group)
        if (
            not 1 <= len(actions) <= contract['max_episode_steps']
            or not np.isfinite(actions).all()
            or (not np.isfinite(rewards).all())
        ):
            raise RuntimeError('official evaluator episode output failed finite/length checks')
        if actions.shape != (len(actions), 5) or np.any(np.abs(actions) > 1.000001):
            raise RuntimeError('GCIQL action contract failed')
        if not all(
            (
                x['current_dim'] == contract['current_dim']
                and x['goal_dim'] == contract['goal_dim']
                and x['current_finite']
                and x['goal_finite']
                for x in policy.input_contracts
            )
        ):
            raise RuntimeError('visual policy input contract failed during rollout')
        assert policy_env.goal_encoder_calls == len(actions)
        assert encoder.calls - start_encoder_calls == 2 * len(actions)
        result = {
            'actual_exit_code': 0,
            'group_id': group,
            'task_id': task,
            'episode_index': index,
            'seed': seed,
            'steps': len(actions),
            'success': int(stats['success']),
            'online_encoder_calls': encoder.calls - start_encoder_calls,
            'goal_online_encoder_calls': policy_env.goal_encoder_calls,
            'encoding_protocol': ENCODING_PROTOCOL,
            'current_goal_contract': {
                'current_dim': contract['current_dim'],
                'goal_dim': contract['goal_dim'],
                'synthetic_zero_padding': False,
            },
            'camera': camera,
            'goal_capture_proof': proof,
            'goal_raw_sha256': policy_env.goal_raw_sha256,
            'policy_input_contracts': policy.input_contracts,
            'current_raw_sha256': policy.current_sha256,
            'original_evaluate_reused': True,
            'extra_policy_warmup': 0,
            'evaluator_sha256': expected_eval,
            'temperature': 0.0,
            'gaussian_noise': None,
            'elapsed_seconds': time.time() - started,
        }
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output / 'steps.npz', actions=actions, rewards=rewards, successes=successes)
        if policy_env.state_trace is not None:
            trace = policy_env.state_trace.arrays()
            if not np.array_equal(trace['executed_actions'], actions):
                raise ValueError('Native trace and official evaluator executed actions differ')
            np.savez_compressed(output / 'trajectory.npz', **trace)
            result['state_trace_sha256'] = sha(output / 'trajectory.npz')
            result['state_trace_observations'] = len(actions) + 1
        atomic_json(output / 'result.json', result)
        (output / 'attempt_exit_code').write_text('0\n')
        return result
    except BaseException as exc:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / 'failure.json', {'actual_exit_code': 1, 'error': repr(exc)})
        (output / 'attempt_exit_code').write_text('1\n')
        raise
    finally:
        try:
            if policy_env is not None:
                policy_env.close()
            else:
                env.close()
        except Exception:
            pass

def main(group: str, mode: str, output: Path):
    if group not in GROUPS:
        raise ValueError(group)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(
        output / 'status.json',
        {'stage': 'loading', 'group': group, 'mode': mode, 'completed_episodes': 0, 'started': time.time()},
    )
    encoder = agent = None
    code = 1
    try:
        agent = load_agent(group)
        encoder = OnlineEncoder(group)
        if mode == 'diagnostic':
            tasks = [(1, 0)]
        else:
            tasks = [(task, index) for task in range(1, 6) for index in range(50)]
        results = []
        for task, index in tasks:
            item = run_episode(
                group, agent, encoder, task, index, output / 'episodes' / f'task{task}' / f'episode_{index:03d}'
            )
            results.append(item)
            atomic_json(
                output / 'status.json',
                {
                    'stage': 'evaluating',
                    'group': group,
                    'mode': mode,
                    'completed_episodes': len(results),
                    'successes': sum((x['success'] for x in results)),
                    'updated': time.time(),
                },
            )
        expected = 1 if mode == 'diagnostic' else 250
        if len(results) != expected or len({x['seed'] for x in results}) != expected:
            raise RuntimeError('episode/seed count mismatch')
        report = {
            'actual_exit_code': 0,
            'group_id': group,
            'mode': mode,
            'episodes': expected,
            'successes': sum((x['success'] for x in results)),
            'success_rate': sum((x['success'] for x in results)) / expected,
            'unique_seeds': expected,
            'max_episode_steps': group_contract(group)['max_episode_steps'],
            'temperature': 0.0,
            'gaussian_noise': None,
            'extra_policy_warmup': 0,
            'online_encoder': True,
            'offline_embedding_cache_used_for_policy': False,
            'encoding_protocol': ENCODING_PROTOCOL,
            'current_dim': group_contract(group)['current_dim'],
            'goal_dim': group_contract(group)['goal_dim'],
            'synthetic_zero_padding': False,
            'completed': time.time(),
        }
        atomic_json(output / 'report.json', report)
        atomic_json(output / 'status.json', {'stage': 'complete', **report})
        code = 0
    except BaseException as exc:
        atomic_json(output / 'failure.json', {'actual_exit_code': 1, 'error': repr(exc)})
        atomic_json(
            output / 'status.json',
            {'stage': 'failed', 'group': group, 'mode': mode, 'error': repr(exc), 'updated': time.time()},
        )
        raise
    finally:
        (output / 'actual_exit_code').write_text(f'{code}\n')
        if encoder is not None:
            del encoder
        if agent is not None:
            del agent
