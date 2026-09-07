"""Collect with official stable World.collect; only ManiSkill input bridge differs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for source in (
    ROOT / "src",
    ROOT / "vendor/le-wm",
    ROOT / "vendor/stable-worldmodel-lewm-compatible",
    ROOT / "vendor/ManiSkill2-Learn",
):
    sys.path.insert(0, str(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-episodes", type=int, default=30)
    parser.add_argument("--replay-noise-state", type=Path, help="Replay recorded official PCG64 noise state from a prior report/manifest")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--env-backend", choices=("process", "serial"), default="process",
                        help="Spawn one simulator worker per environment (default); serial is explicit debugging only")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--reserve-gib", type=float, default=10.0,
                        help="Keep this disk reserve; default includes room for CUDA environments/checkpoints")
    args = parser.parse_args()

    import numpy as np
    from omegaconf import OmegaConf
    import stable_worldmodel as swm
    from ms2_lewm.environment import make_world
    from ms2_lewm.expert import ManiSkillNoisyExpert, capture_numpy_rng, official_noise_config, restore_numpy_rng
    from ms2_lewm.storage import GIB, StorageBudget, collection_writer

    cfg = OmegaConf.load(
        ROOT / "vendor/stable-worldmodel-lewm-compatible/scripts/data/config/ogb.yaml"
    )
    episodes = args.episodes or (2 if args.smoke else cfg.num_traj)
    num_envs = args.num_envs or (1 if args.smoke else cfg.world.num_envs)
    horizon = args.horizon or cfg.world.max_episode_steps
    if args.preflight_episodes < 2:
        parser.error("Preflight must evaluate at least 2 episodes before collection")
    if not args.smoke and (args.episodes or args.num_envs or args.horizon):
        parser.error("Recipe overrides require --smoke; full collection keeps official counts/horizon")
    if args.output.exists():
        raise FileExistsError(f"Dataset already exists: {args.output}; choose a new output")
    if args.output.suffix not in (".lance", ".h5", ".hdf5"):
        parser.error("Output must have .lance, .h5, or .hdf5 suffix")
    if args.reserve_gib < 0:
        parser.error("--reserve-gib must be nonnegative")
    rng = np.random.default_rng(cfg.seed)
    seed = rng.integers(0, 1_000_000).item()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    storage_report_path = args.output.with_suffix(args.output.suffix + ".storage.json")
    storage_options = dict(expected_episodes=episodes, horizon=horizon,
                           reserve_bytes=int(args.reserve_gib * GIB), report_path=storage_report_path)
    if not args.preflight_only:
        # Cheap nominal check before renderer creation; actual JPEG/state
        # bytes are included by the writer's per-episode measured check.
        StorageBudget(args.output, **storage_options).check(nominal=True)

    def jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    # User-required quality gate.  This is a small separate evaluation, using
    # the SAME noisy expert and official 200-step cap, before World.collect.
    episode_updates = []
    def on_episode_end(row):
        row = dict(row, episode=len(episode_updates) + 1,
                   noise_scale=float(preflight_policy._xi[0]))
        episode_updates.append(row)
        print("PREFLIGHT_EPISODE " + json.dumps(row), flush=True)
    preflight_world = make_world(
        num_envs=1, max_episode_steps=cfg.world.max_episode_steps,
        terminate_at_goal=True, episode_callback=on_episode_end,
    )
    preflight_policy = ManiSkillNoisyExpert()
    prior = None
    if args.replay_noise_state is not None:
        prior = json.loads(args.replay_noise_state.read_text())
        preflight_policy.rng.bit_generator.state = prior["noise_rng_initial_state"]
    noise_state = preflight_policy.rng.bit_generator.state
    try:
        preflight_world.set_policy(preflight_policy)
        if prior is not None and "numpy_rng_initial_state" in prior:
            restore_numpy_rng(prior["numpy_rng_initial_state"])
        numpy_state = capture_numpy_rng()
        metrics = preflight_world.evaluate(episodes=args.preflight_episodes, seed=seed)
    finally:
        preflight_world.close()
    successes = int(np.asarray(metrics["episode_successes"]).sum())
    report = {
        "episodes": args.preflight_episodes,
        "successes": successes,
        "passed": successes > 0,
        "metrics": metrics,
        "horizon": int(cfg.world.max_episode_steps),
        "world_seed": seed,
        "noise_config": official_noise_config(),
        "noise_rng_initial_state": noise_state,
        "numpy_rng_initial_state": numpy_state,
        "noise_rng_final_state": preflight_policy.rng.bit_generator.state,
        "noise_scale_per_episode": preflight_policy.episode_noise_scales,
        "episode_records": episode_updates,
        "noise_implementation": "upstream ExpertPolicy.get_action (unmodified)",
        "oracle_implementation": "ogbench.manipspace.oracles.markov.cube_markov.CubeMarkovOracle (unmodified)",
        "action_adapter": "OGB .05m/.3rad/delta-closedness canonical5 to ManiSkill .1m/.1rad/target-gripper ee7",
        "env_backend": "serial",
        "num_envs": 1,
        "collection_env_backend": args.env_backend,
        "collection_num_workers": int(num_envs) if args.env_backend == "process" and num_envs > 1 else 0,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".preflight.json")
    report_path.write_text(json.dumps(report, indent=2, default=jsonable) + "\n")
    print(json.dumps(report, indent=2, default=jsonable), flush=True)
    if successes == 0:
        raise RuntimeError(f"ZERO preflight successes: full and smoke collection forbidden. See {report_path}")
    if args.preflight_only:
        return

    world = make_world(num_envs=num_envs, max_episode_steps=horizon, terminate_at_goal=False,
                       env_backend=args.env_backend)
    try:
        policy = ManiSkillNoisyExpert()
        # Replaying the recorded RNG state preserves the upstream seed=None
        # default while making this dataset's noise sequence reproducible.
        policy.rng.bit_generator.state = noise_state
        world.set_policy(policy)
        restore_numpy_rng(numpy_state)
        # Keep the official episode iterator, policy and transition alignment.
        # Only the writer removes aliases and stores pointcloud as float16.
        writer = collection_writer(args.output, **storage_options)
        world.collect(writer=writer, episodes=episodes, seed=seed, options=None)
    finally:
        world.close()

    provenance = {
        "collector": "stable_worldmodel.World.collect",
        "stable_commit": "abdced49809d5eae38e24b27dc7b635c502c4812",
        "official_collection_config": "scripts/data/config/ogb.yaml",
        "environment": "ManiSkill2 PickCube-v0",
        "policy": "original CubeMarkovOracle + original ExpertPolicy noise through ManiSkill state/action adapter",
        "noise_config": official_noise_config(),
        "noise_rng_initial_state": noise_state,
        "numpy_rng_initial_state": numpy_state,
        "preflight_report": str(report_path.resolve()),
        "preflight_successes": successes,
        "official_expert_dataset": False,
        "smoke": args.smoke,
        "episodes": episodes,
        "num_envs": num_envs,
        "env_backend": args.env_backend if num_envs > 1 else "serial",
        "num_workers": int(num_envs) if args.env_backend == "process" and num_envs > 1 else 0,
        "worker_start_method": "spawn" if args.env_backend == "process" and num_envs > 1 else None,
        "worker_numpy_rng": "independent stream, seeded from each environment reset seed",
        "policy_execution": "parent process, unchanged upstream environment-index RNG order",
        "horizon": horizon,
        "config_seed": int(cfg.seed),
        "world_seed": seed,
        "pointcloud": {"points": 1200, "frame": "world", "channels": "xyz_rgb",
                       "storage_dtype": "float16", "model_input_dtype": "float32"},
        "format": "compact streaming Lance/JPEG95" if args.output.suffix == ".lance" else "direct streaming gzip HDF5/JPEG95 round-trip",
        "reader": "unmodified stable_worldmodel.data.load_dataset",
        "removed_exact_aliases": ["rl_xyz", "rl_rgb", "rl_state", "proprio"],
        "intermediate_dataset": None,
        "storage_report": str(storage_report_path.resolve()),
        "dataset": str(args.output.resolve()),
    }
    sidecar = args.output.with_suffix(args.output.suffix + ".json")
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
