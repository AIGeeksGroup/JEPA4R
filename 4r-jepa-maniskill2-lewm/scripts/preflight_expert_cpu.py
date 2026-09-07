"""Check the noisy cube oracle in native physics, without rendering.

This diagnostic cannot collect world-model data: it creates no images or point
clouds. The official World.evaluate/EnvPool/oracle/noise algorithms are reused.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "vendor/le-wm", ROOT / "vendor/stable-worldmodel-lewm-compatible", ROOT / "vendor/ManiSkill2-Learn"):
    sys.path.insert(0, str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/expert_cpu_preflight30.json")
    parser.add_argument("--replay-noise-state", type=Path)
    parser.add_argument("--oracle-mode", choices=("pickcube_hold", "legacy"), default="pickcube_hold")
    parser.add_argument("--hold-steps", type=int, default=30)
    parser.add_argument("--policy-seed", type=int, default=3072)
    parser.add_argument("--terminate-at-goal", action="store_true",
                        help="Stop on native success; default continues for the full collection horizon")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU preflight requires CUDA_VISIBLE_DEVICES='' before Python starts")
    os.nice(19)

    import gymnasium as gym
    import numpy as np
    from omegaconf import OmegaConf
    import pynvml
    import sapien.core as sapien
    from unittest.mock import patch
    import stable_worldmodel as swm
    from stable_worldmodel.world.env_pool import EnvPool
    from stable_worldmodel.wrapper import EverythingToInfoWrapper
    from ms2_lewm.environment import ManiSkillPointCloudEnv
    from ms2_lewm.expert import ManiSkillNoisyExpert, capture_numpy_rng, official_noise_config, restore_numpy_rng
    from ms2_lewm.physics_preflight import create_physics_preflight

    cfg = OmegaConf.load(ROOT / "vendor/stable-worldmodel-lewm-compatible/scripts/data/config/ogb.yaml")
    world_seed = np.random.default_rng(cfg.seed).integers(0, 1_000_000).item()
    horizon = int(cfg.world.max_episode_steps)
    traces, episode_rows, current_rows = [], [], []
    current_phase = [None]

    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    def own_gpu_bytes():
        total = 0
        for handle in handles:
            for getter in (pynvml.nvmlDeviceGetComputeRunningProcesses, pynvml.nvmlDeviceGetGraphicsRunningProcesses):
                try:
                    processes = getter(handle)
                except pynvml.NVMLError_NotSupported:
                    continue
                total += sum(p.usedGpuMemory for p in processes if p.pid == os.getpid())
        return total
    gpu_samples = [own_gpu_bytes()]

    class StateOracleAdapter(ManiSkillPointCloudEnv):
        def __init__(self):
            # Deliberately bypass pointcloud/renderer setup. Actual physics
            # observations are state vectors, never placeholder clouds/images.
            self.rl_env = gym.wrappers.TimeLimit(create_physics_preflight(), horizon)
            self.action_space = self.rl_env.action_space
            self.observation_space = self.rl_env.observation_space
            self.render_mode = None
            self._env_type = "single"
            self._arm_sampling_bounds = np.asarray([[0.25, -0.35, 0.20], [0.6, 0.35, 0.35]]) - np.array([0.425, 0.0, 0.0])
            self._episode_seed = 0
            self._steps = 0
            self.last_ik_success = None
            controller = self.simulator.agent.controller.controllers["arm"]
            original_ik = controller.compute_ik
            def traced_ik(*args, **kwargs):
                result = original_ik(*args, **kwargs)
                self.last_ik_success = result is not None
                return result
            controller.compute_ik = traced_ik

        def reset(self, *, seed=None, options=None):
            gym.Env.reset(self, seed=seed)
            self._episode_seed = int(seed) if seed is not None else self._episode_seed
            self._steps = 0
            obs, info = self.rl_env.reset(seed=seed, options=options)
            self._last_obs = obs
            return obs, {key: value for key, value in info.items() if key not in ("reward", "terminated", "truncated", "action", "step_idx", "id")}

        def step(self, action):
            obs, reward, terminated, truncated, info = self.rl_env.step(action)
            if not args.terminate_at_goal:
                terminated = False
            self._last_obs = obs
            self._steps += 1
            sim = self.simulator
            row = {
                "seed": self._episode_seed, "step": self._steps,
                "phase": current_phase[0], "goal_dist": float(np.linalg.norm(sim.obj.pose.p - sim.goal_pos)),
                "is_obj_placed": bool(info["is_obj_placed"]), "is_robot_static": bool(info["is_robot_static"]),
                "success": bool(info["success"]), "max_abs_arm_qvel": float(np.abs(sim.agent.robot.get_qvel()[:-2]).max()),
                "cube_tcp_dist": float(np.linalg.norm(sim.obj.pose.p - sim.tcp.pose.p)),
                "grasped": bool(sim.agent.check_grasp(sim.obj)), "ik_success": self.last_ik_success,
                "cube_pos": sim.obj.pose.p.tolist(), "tcp_pos": sim.tcp.pose.p.tolist(),
                "goal_pos": sim.goal_pos.tolist(), "noise_scale": float(policy._xi[0]),
            }
            traces.append(row)
            current_rows.append(row)
            if terminated or truncated:
                summary = {
                    "episode": len(episode_rows) + 1, "seed": self._episode_seed,
                    "steps": self._steps, "success": any(x["success"] for x in current_rows),
                    "noise_scale": row["noise_scale"], "min_goal_dist": min(x["goal_dist"] for x in current_rows),
                    "placed_steps": sum(x["is_obj_placed"] for x in current_rows),
                    "static_steps": sum(x["is_robot_static"] for x in current_rows),
                    "grasped_steps": sum(x["grasped"] for x in current_rows),
                    "ik_failures": sum(x["ik_success"] is False for x in current_rows),
                    "phase_counts": dict(Counter(x["phase"] for x in current_rows)),
                }
                episode_rows.append(summary)
                current_rows.clear()
                gpu_samples.append(own_gpu_bytes())
                print("PREFLIGHT_EPISODE " + json.dumps(summary), flush=True)
            return obs, reward, terminated, truncated, {key: value for key, value in info.items() if key not in ("reward", "terminated", "truncated", "action", "step_idx", "id")}

    with (patch.object(sapien, "SapienRenderer", side_effect=AssertionError("GPU renderer forbidden in CPU preflight")) as renderer,
          patch.object(sapien, "RenderClient", side_effect=AssertionError("RenderClient forbidden in CPU preflight")) as client):
        # Same World fields/EnvPool, excluding only the image-adding wrapper.
        # This object is reserved to evaluate; it is never used to collect.
        world = swm.World.__new__(swm.World)
        world.envs = EnvPool([lambda: EverythingToInfoWrapper(StateOracleAdapter())])
        world.policy, world.infos = None, {}
        world.rewards = world.terminateds = world.truncateds = None
        def forbid_collection(*args, **kwargs):
            raise RuntimeError("Physics-only preflight cannot be used for data collection")
        world.collect = forbid_collection
        policy = ManiSkillNoisyExpert(seed=args.policy_seed, oracle_mode=args.oracle_mode, hold_steps=args.hold_steps)
        prior = None
        if args.replay_noise_state:
            prior = json.loads(args.replay_noise_state.read_text())
            policy.rng.bit_generator.state = prior["noise_rng_initial_state"]
        initial_noise = policy.rng.bit_generator.state
        world.set_policy(policy)
        oracle = policy._oracle_agents["cube"][0]
        original_phase = oracle.print_phase
        def phase_log(phase):
            current_phase[0] = phase
            return original_phase(phase)
        oracle.print_phase = phase_log
        if prior is not None and "numpy_rng_initial_state" in prior:
            restore_numpy_rng(prior["numpy_rng_initial_state"])
        initial_numpy_rng = capture_numpy_rng()
        print(json.dumps({"pid": os.getpid(), "kind": "CPU physics-only preflight", "episodes": args.episodes,
                          "horizon": horizon, "world_seed": world_seed, "own_gpu_bytes": gpu_samples[-1]}), flush=True)
        try:
            metrics = world.evaluate(episodes=args.episodes, seed=world_seed)
        finally:
            world.close()
        gpu_samples.append(own_gpu_bytes())
        assert renderer.call_count == client.call_count == 0
        assert max(gpu_samples) == 0

    # Fixed-length collection intentionally continues after native success.
    # World.evaluate's episode_successes records the final transition only;
    # coverage must count success at any transition, including before release.
    successes = sum(row["success"] for row in episode_rows)
    report = {
        "kind": "CPU physics-only expert precheck; no images, no pointclouds, no dataset",
        "episodes": args.episodes, "successes": successes, "passed": successes > 0,
        "success_definition": "any native success transition within each episode",
        "official_final_transition_successes": int(np.asarray(metrics["episode_successes"]).sum()),
        "diagnostic_only": True,
        "world_seed": world_seed, "horizon": horizon, "num_envs": 1,
        "noise_config": official_noise_config(), "noise_rng_initial_state": initial_noise,
        "numpy_rng_initial_state": initial_numpy_rng,
        "noise_rng_final_state": policy.rng.bit_generator.state,
        "oracle": args.oracle_mode, "hold_steps": args.hold_steps,
        "policy_seed": args.policy_seed, "noise": "unmodified stable ExpertPolicy.get_action",
        "terminate_at_goal": args.terminate_at_goal,
        "metrics": metrics, "episode_records": episode_rows,
        "renderer_calls": renderer.call_count, "render_client_calls": client.call_count,
        "own_gpu_bytes_samples": gpu_samples,
    }
    def jsonable(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        raise TypeError(type(value).__name__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=jsonable) + "\n")
    args.output.with_suffix(".trace.json").write_text(json.dumps(traces, indent=2) + "\n")
    print(json.dumps(report, indent=2, default=jsonable), flush=True)


if __name__ == "__main__":
    main()
