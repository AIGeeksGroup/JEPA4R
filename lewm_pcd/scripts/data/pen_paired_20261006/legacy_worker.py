"""Artifact-only Pen worker. Production env/policy/PCD own all task behavior.

Fresh RGB and PCD processes exchange one episode at a time through NPZ. The
modern parent owns the existing paired writers. Python 3.8 compatible.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace as NS

import numpy as np

EXPERT_SHA = '061dba4907a76b83139ae5a8d0809faa9774dd028d12ce869036b620bd728740'
BACKEND_SHA = 'd4564b9125c66f430bb0b9e92f6dd33973dcf652863e7d3ca4f22a52e3df0bd2'
ATOL, RTOL = 1e-6, 1e-5
RESERVED = {'observation', 'action', 'reward', 'terminated', 'truncated', 'step_idx', 'id'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def value_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     default=json_default).encode('utf-8')).hexdigest()


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False,
                                    default=json_default), encoding='utf-8')
    temporary.replace(path)


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stack(rows):
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def world_row(obs, info, episode_id, step, incoming_action, reward, terminated, truncated):
    assert not RESERVED.intersection(info), 'inner info occupies World reserved keys'
    row = {key: np.asarray(value).copy() for key, value in info.items()}
    assert all(value.dtype.kind in 'biuf' for value in row.values()), 'nonnumeric inner info'
    row.update(observation=np.asarray(obs).copy(), action=np.asarray(incoming_action).copy(),
               reward=np.float64(reward), terminated=np.bool_(terminated), truncated=np.bool_(truncated),
               step_idx=np.int64(step), id=np.int64(episode_id))
    return row


def audit_row(audit, native, complete, image_size):
    row = {key: value.copy() for key, value in complete.items()}
    row.update({'camera_' + key: np.asarray(value).copy() for key, value in audit.camera(native).items()
                if key != 'name'})
    row.update(camera_width=np.int64(image_size), camera_height=np.int64(image_size))
    row['geometry_sha256'] = np.frombuffer(audit.geometry_digest(native.sim.model).encode('ascii'), dtype=np.uint8).copy()
    available, counts = audit.warning_counts(native)
    row.update(warnings_available=np.int64(available), warning_counts=counts.copy())
    return row


def check_value(actual, expected, key, differences):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape, key + ' shape mismatch'
    if a.dtype.kind == 'f' or b.dtype.kind == 'f':
        np.testing.assert_allclose(a, b, atol=ATOL, rtol=RTOL, equal_nan=True, err_msg=key)
        finite = np.isfinite(a) & np.isfinite(b)
        maximum = float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.
        differences[key] = max(differences.get(key, 0.), maximum)
    else:
        np.testing.assert_array_equal(a, b, err_msg=key)


def rgb_episode(env, policy, audit, episode_id, seed, progress=None):
    """T+1 World rows; the production policy is queried exactly once per step."""
    policy.set_env(NS(envs=[env], num_envs=1))
    rng_before = copy.deepcopy(policy.rng.bit_generator.state)
    obs, info = env.reset(seed=seed)
    native = env.native_env
    image_size = int(getattr(env, 'width', 256))
    initial, complete = env.get_env_state(), audit.snapshot(native)
    data_id = int(env.np_random.integers(0, np.iinfo(np.int64).max))  # DataWrapper._gen_id.
    rows, checks = [], []
    incoming, reward, terminated, truncated = np.full(24, np.nan, np.float32), np.nan, False, False
    warmstart_max = 0.
    for step in range(env.episode_len+1):
        before = audit.snapshot(native)
        t0 = time.time()  # Original AddPixelsWrapper clocks only the render call.
        pixels = env.render()
        render_time = np.float64(time.time() - t0)
        warmstart_max = max(warmstart_max, audit.assert_render_state(audit.snapshot(native), before))
        np.testing.assert_array_equal(env.get_obs(), obs, err_msg='RGB render changed native observation')
        assert int(native.sim.model.vis.quality.offsamples) == 8, 'RGB MSAA changed'
        assert pixels.shape == (image_size,image_size,3) and pixels.dtype == np.uint8, 'RGB contract'
        row = world_row(obs, info, data_id, step, incoming, reward, terminated, truncated)
        row['pixels'] = pixels
        row['render_time'] = render_time
        rows.append(row); checks.append(audit_row(audit, native, before, image_size))
        if progress is not None and (step % 25 == 0 or terminated or truncated): progress(step)
        if terminated or truncated: break
        assert step < env.episode_len, 'native horizon was not terminated/truncated'
        batched_info = {key: np.asarray(value)[None,None] for key, value in row.items()}
        incoming = np.asarray(policy.get_action(batched_info)[0]).copy()
        assert incoming.shape == (24,) and np.isfinite(incoming).all() and (np.abs(incoming) <= 1.).all()
        obs, reward, terminated, truncated, info = env.step(incoming.copy())
    arrays = stack(rows)
    arrays['action'] = np.concatenate([arrays['action'][1:], arrays['action'][:1]], axis=0)
    np.testing.assert_array_equal(arrays['action'][:-1], arrays['executed_action'][1:])
    success_percent = None
    if callable(getattr(native, 'evaluate_success', None)):
        success_percent = float(native.evaluate_success([dict(env_infos=dict(goal_achieved=arrays['goal_achieved'][1:]))]))
    metadata = dict(episode_id=int(episode_id), seed=int(seed), rows=len(rows), steps=len(rows)-1,
                    initial_env_state=initial, initial_snapshot=complete,
                    native_initial_sha256=audit.native_digest(initial),
                    initial_native_state_sha=audit.native_digest(initial), image_size=image_size,
                    full_initial_sha256=audit.native_digest(complete),
                    policy_rng_before=rng_before, policy_rng_after=copy.deepcopy(policy.rng.bit_generator.state),
                    policy_rng_before_sha256=value_sha(rng_before), policy_rng_after_sha256=value_sha(policy.rng.bit_generator.state),
                    sample_type=int(info['sample_type']), noise_std=float(info['noise_std']),
                    branch_t=int(info['branch_t']), branch_len=int(info['branch_len']),
                    render_gate='PASS', render_checks=len(rows), render_warmstart_max_abs=warmstart_max,
                    render_state_contract='bitwise except warmstart absolute1e-12, rtol0',
                    gl=audit.gl_info(), goal_rows=int(np.count_nonzero(arrays['goal_achieved'])),
                    terminated=bool(terminated), native_done=bool(terminated), truncated=bool(truncated),
                    terminal_reason='native_done' if terminated else 'horizon',
                    final_goal_achieved=int(arrays['goal_achieved'][-1]),
                    native_episode_success_percent=success_percent,
                    success=None if success_percent is None else bool(success_percent == 100.),
                    success_source='native.evaluate_success' if success_percent is not None else 'UNVERIFIED')
    return arrays, stack(checks), metadata


def pcd_episode(env, backend, audit, rgb, rgb_audit, metadata, progress=None):
    """Replay only stored executed actions; private sampler RNG is episode-seeded."""
    seed = int(metadata['seed'])
    obs, info = env.reset(seed=seed)
    native = env.native_env
    image_size = int(metadata['image_size'])
    assert audit.native_digest(env.get_env_state()) == metadata['native_initial_sha256'], 'native seeded initial state differs'
    assert audit.native_digest(audit.snapshot(native)) == metadata['full_initial_sha256'], 'full seeded initial state differs'
    audit.restore(native, metadata['initial_env_state'], metadata['initial_snapshot'])
    assert audit.native_digest(audit.snapshot(native)) == metadata['full_initial_sha256'], 'restored initial state differs'
    data_id = int(env.np_random.integers(0, np.iinfo(np.int64).max))
    backend.seed(seed)
    sampled, checks, differences = [], [], {}
    incoming, reward, terminated, truncated = np.full(24, np.nan, np.float32), np.nan, False, False
    warmstart_max, state_hash_differing_rows = 0., 0
    for step in range(len(rgb['action'])):
        obs = env.get_obs()
        row = world_row(obs, info, data_id, step, incoming, reward, terminated, truncated)
        for key, value in row.items():
            if key == 'action': continue  # Source actions are outgoing, this temporary row is incoming.
            if key == 'executed_action':
                np.testing.assert_array_equal(value, rgb[key][step], err_msg='actual replay action changed')
            elif key == 'state_sha256':
                state_hash_differing_rows += int(not np.array_equal(value, rgb[key][step]))
            else: check_value(value, rgb[key][step], key, differences)
        before = audit.snapshot(native)
        current_audit = audit_row(audit, native, before, image_size)
        for key, value in current_audit.items(): check_value(value, rgb_audit[key][step], 'audit_' + key, differences)
        xyz, colour = backend()
        warmstart_max = max(warmstart_max, audit.assert_render_state(audit.snapshot(native), before))
        np.testing.assert_array_equal(env.get_obs(), obs, err_msg='PCD render changed native observation')
        assert int(native.sim.model.vis.quality.offsamples) == 0, 'PCD MSAA changed'
        assert xyz.shape == colour.shape == (1200,3) and xyz.dtype == np.float32 and colour.dtype == np.uint8
        assert np.isfinite(xyz).all(), 'nonfinite PCD'
        sampled.append(dict(pcd_xyz=xyz, pcd_rgb=colour))
        current_audit.update({'pcd_' + key: np.asarray(value).copy() for key, value in backend.last_info.items()
                              if isinstance(value, (np.ndarray, np.number, int, float, bool))})
        current_audit.update(pcd_source_labels=backend.last_labels.copy(),
                             pcd_source_indices=backend.last_source_indices.copy())
        checks.append(current_audit)
        if progress is not None and (step % 25 == 0 or step == len(rgb['action'])-1): progress(step)
        if step == len(rgb['action'])-1:
            assert terminated or truncated, 'reference final frame is not done'
            assert np.isnan(rgb['action'][step]).all(), 'missing terminal NaN action'
            break
        assert not (terminated or truncated), 'native episode ended before reference'
        incoming = rgb['action'][step].copy()
        assert np.isfinite(incoming).all() and (np.abs(incoming) <= 1.).all()
        obs, reward, terminated, truncated, info = env.step(incoming)
    payload = {key: value.copy() for key, value in rgb.items() if key not in ('pixels', 'render_time')}
    payload.update(stack(sampled))  # Public numeric values stay bitwise paired with RGB after tolerance checks.
    result = dict(metadata, strict_replay_gate='PASS', render_gate='PASS',
                  replay_atol=ATOL, replay_rtol=RTOL, replay_max_abs=differences,
                  state_sha256_differing_rows=state_hash_differing_rows,
                  render_checks=len(checks), render_warmstart_max_abs=warmstart_max,
                  gl=audit.gl_info(), pcd_contract='visible robot/Object/target;1200;Object<=300;target<=100;robot-only fill')
    return payload, stack(checks), result


def make_runtime(config, phase, policy_seed):
    """Explicit real dependencies only: missing Gymnasium is a failed preflight."""
    os.environ.setdefault('MUJOCO_GL', 'egl'); os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    for name in ('native_root', 'mjrl_root'): sys.path.insert(0, str(Path(config[name]).resolve()))
    code = Path(config['code_root']).resolve()
    for row in config['source_manifest']:
        assert sha(row['path']) == row['sha256'], 'sealed source changed: ' + row['path']
    assert sha(config['expert_path']) == EXPERT_SHA, 'published expert SHA changed'
    assert sha(code/'lewm_pcd/envs/adroit/pcd.py') == BACKEND_SHA, 'sealed target PCD changed'
    audit = load_file(config['audit_helpers'], 'paired_pen_audit')
    image_size = int(config.get('image_size', 256))
    assert image_size in (224,256), 'image_size must be explicitly224 or256'
    audit.SIZE = image_size
    module = load_file(code/'lewm_pcd/envs/adroit/env.py', 'paired_pen_env')
    env = module.AdroitEnv(env_name='pen-v0', width=image_size, height=image_size, episode_len=100,
                          mix=(.5,.3,.2), noise_max=.2, branch_tail_prob=.5, burst_len=(10,30), camera='vil_camera')
    native = env.native_env
    native_path, native_root = str(Path(inspect.getfile(type(native))).resolve()), str(Path(config['native_root']).resolve())
    assert os.path.commonpath([native_path, native_root]) == native_root, 'native Pen imported outside pinned root'
    assert native.action_space.shape == (24,) and native.observation_space.shape == (45,)
    assert int(native.frame_skip) == 5 and np.isclose(native.sim.model.opt.timestep, .002)
    assert int(native.sim.model.vis.quality.offsamples) == 8, 'native RGB default changed'
    assert native.sim._render_context_offscreen is None and not native.sim.render_contexts, 'fresh process/context required'
    policy = backend = None
    if phase == 'rgb':
        pm = load_file(code/'lewm_pcd/envs/adroit/policy.py', 'paired_pen_policy')
        policy = pm.AdroitFeedbackPolicy(config['expert_path'], EXPERT_SHA, env_name='pen-v0', seed=policy_seed)
    else:
        bm = load_file(code/'lewm_pcd/envs/adroit/pcd.py', 'paired_pen_pcd')
        backend = bm.AdroitPointClouds(native.sim, 'pen-v0', size=image_size, num_points=1200, camera='vil_camera',
                                       voxel_size=.005, obj_fraction=.25, crop_box=bm.DEFAULT_CROP_BOX, foreground_only=True)
    source = audit.seal_sources(NS(code_root=code, expert_path=Path(config['expert_path'])), native)
    for key, path in [('worker', Path(__file__)), ('production_env', code/'lewm_pcd/envs/adroit/env.py')]:
        source['files'][key] = dict(path=str(path.resolve()), sha256=sha(path), bytes=path.stat().st_size)
    return env, policy, backend, audit, source


def save_npz(path, arrays):
    assert not path.exists(), 'refusing to overwrite episode payload'
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('xb') as file: np.savez_compressed(file, **arrays)
    temporary.replace(path)


def verify_source_hashes(source, expected):
    required = {'worker','production_env','expert_adapter','pcd_backend','pcd_sampler',
                'policy_pickle','native_pen','script'}
    assert required <= set(expected), 'expected source hashes omit mandatory pinned files'
    for name, digest in expected.items():
        assert name in source['files'] and source['files'][name]['sha256'] == digest, 'source SHA mismatch: ' + name
    return dict(expected)


def run(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    request = json.loads(args.episodes.read_text(encoding='utf-8'))
    episodes = request['episodes']
    assert episodes and len({e['seed'] for e in episodes}) == len(episodes), 'empty/duplicate seeds'
    assert len({e['episode_id'] for e in episodes}) == len(episodes), 'duplicate episode ids'
    out = args.out.resolve()
    if out.exists(): raise FileExistsError('fresh phase staging directory required: ' + str(out))
    out.mkdir(parents=True)
    state = dict(phase=args.phase, stage='preflight', completed=0, failed=0, exit_code=None,
                 started_at=datetime.now(timezone.utc).isoformat(), config_sha256=sha(args.config),
                 episodes_sha256=sha(args.episodes), current_episode=None, step=None)
    def persist():
        state['updated_at'] = datetime.now(timezone.utc).isoformat(); save_json(out/'state.json', state)
    persist(); env = None; exit_code = 1
    try:
        policy_seed = int(request.get('policy_seed', config['policy_seed']))
        env, policy, backend, audit, source = make_runtime(config, args.phase, policy_seed)
        source_hashes = verify_source_hashes(source, config['source_hashes'])
        import torch
        torch.set_num_threads(1)
        if args.phase == 'rgb' and 'policy_rng_state' in request:
            policy.rng.bit_generator.state = copy.deepcopy(request['policy_rng_state'])
        save_json(out/'source_manifest.json', source)
        source_sha = sha(out/'source_manifest.json')
        save_json(out/'environment.json', dict(python=sys.version, executable=sys.executable,
                   versions={name: importlib.metadata.version(name) for name in ('numpy','torch','gym','gymnasium','mujoco-py')},
                   policy_seed=policy_seed, actual_offsamples=int(env.sim.model.vis.quality.offsamples)))
        initial_hashes, native_initial_hashes = set(), set()
        for ep in episodes:
            index, seed = int(ep['episode_id']), int(ep['seed'])
            state.update(stage='episode', current_episode=index, step=0); persist()
            def progress(step): state['step'] = step; persist()
            if args.phase == 'rgb':
                payload, checks, meta = rgb_episode(env, policy, audit, index, seed, progress)
            else:
                assert args.rgb_root is not None, '--rgb-root required for PCD'
                rgb_path = args.rgb_root/('ep_%d_rgb.npz' % index)
                rgb_audit_path = args.rgb_root/('ep_%d_rgb_audit.npz' % index)
                meta_path = args.rgb_root/('ep_%d_rgb.json' % index)
                reference = json.loads(meta_path.read_text(encoding='utf-8'))
                assert reference['seed'] == seed and reference['episode_id'] == index
                assert reference['config_sha256'] == state['config_sha256'] and reference['source_manifest_sha256'] == source_sha
                assert sha(rgb_path) == reference['payload_sha256'] and sha(rgb_audit_path) == reference['audit_sha256']
                with np.load(str(rgb_path), allow_pickle=False) as rgb, np.load(str(rgb_audit_path), allow_pickle=False) as raw:
                    payload, checks, meta = pcd_episode(env, backend, audit, rgb, raw, reference, progress)
                assert sha(rgb_path) == reference['payload_sha256'] and sha(rgb_audit_path) == reference['audit_sha256']
                meta.update(rgb_payload_path=str(rgb_path.resolve()), rgb_payload_sha256=sha(rgb_path),
                            rgb_metadata_path=str(meta_path.resolve()), rgb_metadata_sha256=sha(meta_path))
            assert meta['gl']['gpu_renderer_gate'] == 'PASS_HARDWARE_REPORTED', 'GPU renderer not verified'
            assert meta['full_initial_sha256'] not in initial_hashes, 'duplicate complete initial state'
            assert meta['native_initial_sha256'] not in native_initial_hashes, 'duplicate native initial state'
            initial_hashes.add(meta['full_initial_sha256'])
            native_initial_hashes.add(meta['native_initial_sha256'])
            prefix = 'ep_%d_%s' % (index, args.phase)
            payload_path, audit_path = out/(prefix+'.npz'), out/(prefix+'_audit.npz')
            save_npz(payload_path, payload); save_npz(audit_path, checks)
            meta.update(config_sha256=state['config_sha256'], source_manifest_sha256=source_sha,
                        source_hashes=source_hashes,
                        payload_path=str(payload_path), payload_sha256=sha(payload_path),
                        audit_path=str(audit_path), audit_sha256=sha(audit_path))
            save_json(out/(prefix+'.json'), meta)
            if args.phase == 'rgb': save_json(out/'policy_rng_end.json', meta['policy_rng_after'])
            state['completed'] += 1; persist()
            print(json.dumps(dict(phase=args.phase, episode_id=index, seed=seed, rows=meta['rows'],
                                  sample_type=meta['sample_type'], goal_rows=meta['goal_rows'])), flush=True)
            del payload, checks
        exit_code = 0; state['stage'] = 'complete'
    except BaseException as error:
        state.update(stage='failed', failed=1, error=repr(error))
        (out/'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        traceback.print_exc()
    finally:
        if env is not None:
            try: env.close()
            except Exception as error:
                state.update(stage='failed', error='close: ' + repr(error)); exit_code = 1
        state['exit_code'] = exit_code; persist(); (out/'exit_code').write_text(str(exit_code)+'\n')
    return exit_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('rgb','pcd'), required=True)
    for name in ('config','episodes','out'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--rgb-root', type=Path)
    raise SystemExit(run(parser.parse_args()))
