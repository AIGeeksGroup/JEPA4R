"""Collect all noisy heuristic trajectories and report their operation coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / 'src', ROOT / 'vendor/le-wm',
               ROOT / 'vendor/stable-worldmodel-lewm-compatible', ROOT / 'vendor/ManiSkill2-Learn'):
    sys.path.insert(0, str(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--preflight-episodes', type=int, default=0,
                        help='Optional advisory diagnostic; native success is not a collection gate')
    parser.add_argument('--replay-noise-state', type=Path)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--env-backend', choices=('process', 'serial'), default='process')
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--oracle-mode', choices=('pickcube_hold', 'legacy'), default='pickcube_hold')
    parser.add_argument('--hold-steps', type=int, default=30)
    parser.add_argument('--reserve-gib', type=float, default=10.0)
    parser.add_argument('--storage-safety-factor', type=float, default=1.2)
    args = parser.parse_args()

    import numpy as np
    from omegaconf import OmegaConf
    from ms2_lewm.environment import make_world
    from ms2_lewm.expert import ManiSkillNoisyExpert, capture_numpy_rng, official_noise_config, restore_numpy_rng
    from ms2_lewm.storage import GIB, StorageBudget, collection_writer
    from ms2_lewm.coverage import CoverageWriter

    cfg = OmegaConf.load(ROOT / 'vendor/stable-worldmodel-lewm-compatible/scripts/data/config/ogb.yaml')
    episodes = args.episodes if args.episodes is not None else (2 if args.smoke else int(cfg.num_traj))
    num_envs = args.num_envs if args.num_envs is not None else (1 if args.smoke else int(cfg.world.num_envs))
    horizon = args.horizon if args.horizon is not None else int(cfg.world.max_episode_steps)
    if min(episodes, num_envs, horizon, args.hold_steps) < 1 or args.preflight_episodes < 0:
        parser.error('Counts/horizon must be positive; preflight count must be nonnegative')
    if not args.smoke and any(value is not None for value in (args.episodes, args.num_envs, args.horizon)):
        parser.error('Count/horizon overrides require --smoke; full collection keeps official counts')
    if args.reserve_gib < 0 or args.storage_safety_factor < 1:
        parser.error('Require nonnegative reserve and storage safety factor >=1')
    if args.output.exists():
        raise FileExistsError(f'Dataset exists: {args.output}; choose a new output')
    if args.output.suffix not in ('.lance', '.h5', '.hdf5'):
        parser.error('Output must have .lance, .h5, or .hdf5 suffix')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seed = int(np.random.default_rng(cfg.seed).integers(0, 1_000_000))
    storage_path = args.output.with_suffix(args.output.suffix + '.storage.json')
    coverage_path = args.output.with_suffix(args.output.suffix + '.coverage.json')
    storage_options = dict(expected_episodes=episodes, horizon=horizon,
                           reserve_bytes=int(args.reserve_gib * GIB),
                           safety_margin=args.storage_safety_factor, report_path=storage_path)
    if not args.preflight_only:
        StorageBudget(args.output, **storage_options).check(nominal=True)

    def jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    policy_options = dict(oracle_mode=args.oracle_mode, hold_steps=args.hold_steps)
    initial_policy = ManiSkillNoisyExpert(**policy_options)
    prior = json.loads(args.replay_noise_state.read_text()) if args.replay_noise_state else None
    if prior:
        initial_policy.rng.bit_generator.state = prior['noise_rng_initial_state']
        if 'numpy_rng_initial_state' in prior:
            restore_numpy_rng(prior['numpy_rng_initial_state'])
    noise_state = initial_policy.rng.bit_generator.state
    numpy_state = capture_numpy_rng()
    diagnostic_path = None
    diagnostic_episodes = args.preflight_episodes or (30 if args.preflight_only else 0)
    if diagnostic_episodes:
        episode_records = []
        diagnostic = make_world(num_envs=1, max_episode_steps=int(cfg.world.max_episode_steps),
                                terminate_at_goal=False, episode_callback=episode_records.append)
        try:
            diagnostic.set_policy(initial_policy)
            metrics = diagnostic.evaluate(episodes=diagnostic_episodes, seed=seed)
        finally:
            diagnostic.close()
        diagnostic_path = args.output.with_suffix(args.output.suffix + '.preflight.json')
        diagnostic_report = {
            'advisory_only': True, 'collection_blocked_by_native_success': False,
            'episodes': diagnostic_episodes,
            'successes': sum(bool(row['success']) for row in episode_records),
            'success_definition': 'any native-success transition in the episode',
            'official_final_transition_successes': int(np.asarray(metrics['episode_successes']).sum()),
            'episode_records': episode_records, 'metrics': metrics,
            'noise_config': official_noise_config(), 'oracle_mode': args.oracle_mode,
            'hold_steps': args.hold_steps, 'noise_rng_initial_state': noise_state,
            'numpy_rng_initial_state': numpy_state,
        }
        diagnostic_path.write_text(json.dumps(diagnostic_report, indent=2, default=jsonable) + '\n')
        print('ADVISORY_DIAGNOSTIC ' + json.dumps(diagnostic_report, default=jsonable), flush=True)
        if args.preflight_only:
            return

    provenance = {
        'status': 'starting', 'collector': 'stable_worldmodel.World.collect',
        'stable_commit': 'abdced49809d5eae38e24b27dc7b635c502c4812',
        'official_collection_config': 'scripts/data/config/ogb.yaml',
        'environment': 'ManiSkill2 PickCube-v0', 'oracle_mode': args.oracle_mode,
        'hold_steps': args.hold_steps,
        'policy': 'CubeMarkovOracle with local PickCube hold adaptation; legacy available',
        'noise_implementation': 'upstream ExpertPolicy.get_action unchanged',
        'noise_config': official_noise_config(), 'noise_rng_initial_state': noise_state,
        'numpy_rng_initial_state': numpy_state, 'official_expert_dataset': False,
        'advisory_preflight_report': str(diagnostic_path.resolve()) if diagnostic_path else None,
        'native_success_gate': False,
        'quality_policy': 'retain all successful, failed and intermediate operations; no forced failure ratio',
        'coverage_report': str(coverage_path.resolve()), 'storage_report': str(storage_path.resolve()),
        'smoke': args.smoke, 'episodes': episodes, 'num_envs': num_envs, 'horizon': horizon,
        'env_backend': args.env_backend if num_envs > 1 else 'serial',
        'num_workers': num_envs if args.env_backend == 'process' and num_envs > 1 else 0,
        'worker_start_method': 'spawn' if args.env_backend == 'process' and num_envs > 1 else None,
        'worker_numpy_rng': 'independent stream seeded from each environment reset seed',
        'config_seed': int(cfg.seed), 'world_seed': seed,
        'pointcloud': {'points': 1200, 'frame': 'world', 'channels': 'xyz_rgb',
                       'storage_dtype': 'float16', 'model_input_dtype': 'float32'},
        'format': 'compact Lance with native lossless zstd12/JPEG95' if args.output.suffix == '.lance' else 'direct compressed HDF5',
        'reader': 'unmodified stable_worldmodel.data.load_dataset',
        'removed_exact_aliases': ['rl_xyz', 'rl_rgb', 'rl_state', 'proprio'],
        'intermediate_dataset': None, 'dataset': str(args.output.resolve()),
        'reserve_gib': args.reserve_gib, 'storage_safety_factor': args.storage_safety_factor,
    }
    sidecar = args.output.with_suffix(args.output.suffix + '.json')
    def save_manifest():
        temporary = sidecar.with_suffix(sidecar.suffix + '.tmp')
        temporary.write_text(json.dumps(provenance, indent=2, default=jsonable) + '\n')
        temporary.replace(sidecar)
    save_manifest()
    world = make_world(num_envs=num_envs, max_episode_steps=horizon, terminate_at_goal=False,
                       env_backend=args.env_backend)
    try:
        policy = ManiSkillNoisyExpert(**policy_options)
        policy.rng.bit_generator.state = noise_state
        world.set_policy(policy)
        restore_numpy_rng(numpy_state)
        writer = CoverageWriter(collection_writer(args.output, **storage_options), coverage_path)
        provenance['status'] = 'collecting'
        save_manifest()
        print('FULL_COLLECTION_START ' + json.dumps({
            'episodes': episodes, 'num_envs': num_envs, 'horizon': horizon,
            'oracle_mode': args.oracle_mode, 'native_success_gate': False,
            'output': str(args.output.resolve()),
        }), flush=True)
        world.collect(writer=writer, episodes=episodes, seed=seed, options=None)
        provenance['status'] = 'complete'
        provenance['collected_episode_noise_scales'] = policy.episode_noise_scales
    except BaseException as error:
        provenance['status'] = 'interrupted'
        provenance['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        world.close()
        save_manifest()
    print(json.dumps(provenance, default=jsonable), flush=True)


if __name__ == '__main__':
    main()
