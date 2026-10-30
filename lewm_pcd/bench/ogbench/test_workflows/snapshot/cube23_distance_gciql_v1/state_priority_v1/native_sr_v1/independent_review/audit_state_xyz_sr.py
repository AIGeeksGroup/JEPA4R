"""Audit committed state/XYZ SR artifacts; no simulator, policy, or JAX imports."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def array_sha(value):
    value = np.asarray(value)
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode() + value.tobytes()).hexdigest()


def episode(folder, config, task, index, diagnostic):
    result = read(folder / 'result.json')
    assert result['actual_exit_code'] == 0 and (folder / 'attempt_exit_code').read_text().strip() == '0'
    assert (result['group_id'], result['task_id'], result['episode']) == (config['group_id'], task, index)
    seed = 970000000 + task * 100000 + index
    assert result['seed'] == seed and result['diagnostic'] is diagnostic
    assert result['environment_closed'] and result['extra_policy_warmup'] == 0 and result['original_evaluate_reused']
    assert set(result['files']) == {'reset.json', 'policy_inputs.json', 'steps.npz'}
    for name, expected in result['files'].items():
        path = folder / name
        assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], path
    reset, inputs = read(folder / 'reset.json'), read(folder / 'policy_inputs.json')
    cubes = config['cube_count']
    current, goal_width = 19 + 9 * cubes, 3 * cubes
    assert reset['seed'] == seed and reset.get('actual_env_seed', seed) == seed
    assert reset['options'] == {'task_id': task, 'render_goal': False}
    assert (reset['task_id'], reset['episode'], reset['cube_count']) == (task, index, cubes)
    full = np.asarray(reset['full_goal'], dtype=reset['full_goal_dtype'])
    assert full.shape == (current,) and np.isfinite(full).all()
    goal = np.concatenate([full[19 + 9 * n:22 + 9 * n] for n in range(cubes)])
    assert goal.shape == (goal_width,) and reset['goal_XYZ'] == goal.tolist()
    assert reset['full_goal_sha256'] == array_sha(full)
    assert reset['goal_XYZ_shape'] == [goal_width] and reset['goal_XYZ_sha256'] == array_sha(goal)
    assert reset['goal_XYZ_offsets'] == [19 + 9 * n for n in range(cubes)]
    assert reset['current_shape'] == [current] and reset['goal_XYZ_is_official_scaled10']
    assert reset['extra_policy_warmup'] == 0 and reset['action_space_seed_explicitly_changed'] is False
    assert reset['legacy_visual_scene_exact_pairing_claimed'] is False
    assert inputs['current_dim'] == current and inputs['goal_dim'] == goal_width
    assert inputs['all_inputs_finite'] and inputs['synthetic_zero_padding'] is False and inputs['encoder_calls'] == 0
    assert inputs['perstep_raw_observations_persisted'] is False
    assert inputs['goal_XYZ_sha256'] == array_sha(goal) and inputs['goal_dtype'] == str(goal.dtype)
    steps = result['steps']
    limit = 4 if diagnostic else config['max_episode_steps']
    assert 1 <= steps <= limit and inputs['policy_calls'] == result['policy_calls'] == steps
    hashes = inputs['observations_sha256']
    assert len(hashes) == steps and hashes[0] == reset['current_sha256']
    assert all(isinstance(h, str) and len(h) == 64 and set(h) <= set('0123456789abcdef') for h in hashes)
    with np.load(folder / 'steps.npz', allow_pickle=False) as raw:
        assert set(raw.files) == {'actions', 'rewards', 'dones', 'successes', 'actor_keys'}
        actions, rewards, dones, successes, keys = (raw[k] for k in ('actions', 'rewards', 'dones', 'successes', 'actor_keys'))
        assert actions.shape == (steps, 5) and np.isfinite(actions).all() and np.all(np.abs(actions) <= 1)
        assert rewards.shape == dones.shape == successes.shape == (steps,) and np.isfinite(rewards).all()
        assert dones.dtype == successes.dtype == np.dtype(bool) and not dones[:-1].any() and dones[-1]
        assert not successes[:-1].any(), 'Native terminate-at-goal must end at the first success'
        assert keys.shape == (steps, 2) and keys.dtype == np.dtype('uint32')
        assert result['success'] in (0, 1) and int(successes[-1]) == result['success']
        assert float(rewards[-1]) == float(result['success'])
        assert steps == limit or result['success'] == 1, 'Failure must reach the original time limit'
    return dict(task=task, index=index, seed=seed, steps=steps, success=result['success'], result_sha256=sha(folder / 'result.json'))


def audit(args):
    config = read(args.config)
    assert sha(args.config) == args.expected_config_sha256
    assert config['group_id'] in ('G01', 'G04', 'G07')
    cubes = {'G01': 1, 'G04': 2, 'G07': 3}[config['group_id']]
    assert config['cube_count'] == cubes and config['current_dim'] == 19 + 9 * cubes and config['goal_dim'] == 3 * cubes
    assert config['dataset_id'] == {1: 'cube-single-play-v0', 2: 'cube-double-play-v0', 3: 'cube-triple-play-v0'}[cubes]
    assert config['max_episode_steps'] == {1: 200, 2: 500, 3: 1000}[cubes] and config['seed_base'] == 970000000
    for path, expected in config['source_pins'].items():
        assert Path(path).suffix not in ('.pkl', '.pt', '.ckpt', '.npy', '.npz', '.h5')
        assert sha(path) == expected, path
    output = args.evaluation_output
    assert (output / 'actual_exit_code').read_text().strip() == '0'
    protocol, binding, report = (read(output / n) for n in ('protocol.json', 'binding.json', 'report.json'))
    mode = protocol['mode']
    assert mode in ('diagnostic', 'full') and report['mode'] == mode
    diagnostic = mode == 'diagnostic'
    assert all(protocol[k] == v for k, v in config.items())
    assert protocol['num_video_episodes'] == protocol['eval_temperature'] == protocol['extra_policy_warmup'] == 0
    assert protocol['eval_gaussian'] is None and protocol['diagnostic_TimeLimit_steps'] == (4 if diagnostic else None)
    assert protocol['original_physics_reset_goal_success_controller_unmodified']
    assert binding['config_path'] == str(args.config) and binding['config_sha256'] == args.expected_config_sha256
    assert binding['adapter_sha256'] == args.expected_adapter_sha256 and binding['backend'] == 'cpu'
    assert binding['original_evaluate_sha256'] == '4d9a2732bc9ef89ce99fbc0f07a76d3f4d035783af3a658251709140ff9de3d5'
    assert binding['actual_exit_code'] == 0 and binding['network_step'] == 1000001 and binding['Adam_count'] == 1000000
    assert binding['synthetic_zero_padding'] is False and binding['current_dim'] == config['current_dim'] and binding['goal_dim'] == config['goal_dim']
    assert binding['new_optimizer_updates'] == binding['encoder_calls'] == binding['WM_dynamics_calls'] == 0 and binding['dataset_loaded'] is False
    accepted = read(config['training_acceptance'])
    assert sha(config['training_acceptance']) == config['training_acceptance_sha256']
    assert accepted['actual_exit_code'] == 0 and accepted['full153_state_exact'] and accepted['state'] == binding['state'] and binding['state']['leaves'] == 153
    assert accepted['network_step'] == binding['network_step'] and accepted['Adam_count'] == binding['Adam_count']
    assert accepted['actual_current_dim'] == config['current_dim'] and accepted['actual_goal_dim'] == config['goal_dim'] and accepted['synthetic_zero_padding'] is False
    manifest_path = Path(config['final_checkpoint']) / 'manifest.json'
    manifest = read(manifest_path)
    assert sha(manifest_path) == config['final_manifest_sha256'] == accepted['final_manifest_sha256']
    assert manifest['counter'] == 1000000 and manifest['agent_sha256'] == binding['checkpoint_sha256']
    assert binding['checkpoint'] == str(Path(config['final_checkpoint']) / 'params_1000000.pkl')
    phase = args.phase_directory
    exit_record, argv = read(phase / 'exit.json'), read(phase / 'actual_argv.json')
    assert exit_record['actual_exit_code'] == 0 and exit_record['residual_owned_members'] == [] and (phase / 'actual_exit_code').read_text().strip() == '0'
    expected_argv = [str(Path(argv['cwd']) / 'env_runtime/bin/python'), '-B', str(args.config.parent.parent / 'evaluate_state_xyz.py'), '--config', str(args.config), '--mode', mode, '--output', str(output)]
    assert argv['argv'] == expected_argv and argv['environment']['CUDA_VISIBLE_DEVICES'] in ('', '-1') and argv['environment']['JAX_PLATFORMS'] == 'cpu'
    assert sha(expected_argv[2]) == args.expected_adapter_sha256
    tasks, count = ([1], 1) if diagnostic else ([1, 2, 3, 4, 5], 50)
    wanted = {output / f'task{task}' / f'episode_{index:03d}' / 'result.json' for task in tasks for index in range(count)}
    assert set(output.glob('task*/episode_*/result.json')) == wanted
    records = [episode(path.parent, config, task, index, diagnostic) for task in tasks for index in range(count) for path in [output / f'task{task}' / f'episode_{index:03d}' / 'result.json']]
    assert len({x['seed'] for x in records}) == len(records) == (1 if diagnostic else 250)
    per_task = {}
    for task in tasks:
        selected = [x for x in records if x['task'] == task]
        summary = read(output / f'task{task}/summary.json')
        expected = dict(actual_exit_code=0, task_id=task, episodes=count, successes=sum(x['success'] for x in selected), policy_steps=sum(x['steps'] for x in selected))
        assert summary == expected
        per_task[str(task)] = dict(episodes=count, successes=expected['successes'], success_rate=expected['successes'] / count)
    successes, total_steps = sum(x['success'] for x in records), sum(x['steps'] for x in records)
    for key, expected in dict(actual_exit_code=0, group_id=config['group_id'], cube_count=cubes, episodes=len(records), successes=successes, success_rate=successes / len(records), policy_steps=total_steps, unique_seeds=len(records), all_attempts_exit0=True, diagnostic_excluded_from_SR=diagnostic, new_training_updates=0).items():
        assert report[key] == expected, key
    return dict(actual_exit_code=0, group_id=config['group_id'], mode=mode, episodes=len(records), unique_seeds=len(records), successes=successes, success_rate=successes / len(records), per_task=per_task, policy_steps=total_steps, checkpoint_sha256=binding['checkpoint_sha256'], state=binding['state'], config_sha256=args.expected_config_sha256, adapter_sha256=args.expected_adapter_sha256, protocol_sha256=sha(output / 'protocol.json'), raw_report_sha256=sha(output / 'report.json'), records=records, no_NN_simulator_or_policy_rehash=True, limits=['Per-step current observation bytes were not stored; counts/hashes are bound to the frozen adapter, not independently reconstructed.', 'Native reset unseeded Box actions prevent strict historical visual scene pairing; no physical/reset change is introduced.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'evaluation-output', 'phase-directory', 'audit-output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-config-sha256', required=True)
    parser.add_argument('--expected-adapter-sha256', required=True)
    args = parser.parse_args()
    args.audit_output.mkdir(parents=True, exist_ok=False)
    code = 1
    try:
        report = audit(args)
        report.update(completed=time.time(), auditor_sha256=sha(__file__))
        (args.audit_output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        code = 0
    except BaseException:
        (args.audit_output / 'failure.json').write_text(json.dumps(dict(actual_exit_code=1, error=traceback.format_exc()), indent=2) + '\n')
        raise
    finally:
        (args.audit_output / 'actual_exit_code').write_text(str(code) + '\n')


if __name__ == '__main__':
    main()
