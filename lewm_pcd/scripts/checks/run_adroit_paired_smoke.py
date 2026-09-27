"""Run the bounded Pen paired 2D/3D HDF5 smoke and replay checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


HERE = Path(__file__).resolve()
LEWM_ROOT = HERE.parents[2]
CB_ROOT = LEWM_ROOT / "bench" / "cortexbench"
PINNED = Path(os.environ.get("CORTEXBENCH_PINNED_SRC", "/root/autodl-tmp/workspace/users/lewm/cortexbench-official-pinned-src"))
for path in (LEWM_ROOT, CB_ROOT, next(PINNED.glob("mj_envs-*"), PINNED / "mj_envs"), next(PINNED.glob("mjrl-*"), PINNED / "mjrl")):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", "/root/.mujoco/mujoco200")
os.environ.setdefault("MUJOCO_PY_MJKEY_PATH", "/root/.mujoco/mjkey.txt")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ["LD_LIBRARY_PATH"] = f"{Path(os.environ['MUJOCO_PY_MUJOCO_PATH']) / 'bin'}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")

from bench.cortexbench.adroit.smoke import (  # noqa: E402
    AdroitPairedWriter,
    AdroitPCDConfig,
    branch_action,
    draw_episode_plan,
    capture_adroit_frame,
    dump_json,
)


def _pen_env():
    from mj_envs.envs.hand_manipulation_suite.pen_v0 import PenEnvV0

    return PenEnvV0()


def _check_pairing(pixel_path: Path, pcd_path: Path) -> dict:
    with h5py.File(pixel_path, "r") as pix, h5py.File(pcd_path, "r") as pcd:
        for key in ("ep_len", "ep_offset"):
            assert np.array_equal(pix[key][:], pcd[key][:]), key
        for key in ("action", "observation", "qpos", "qvel", "sample_type", "rand_action", "noise_std", "branch_t", "branch_len", "seed", "step_idx"):
            assert np.array_equal(pix[key][:], pcd[key][:]), key
        xyz = pcd["pcd_xyz"][:]
        assert xyz.shape[1:] == (1200, 3)
        assert np.isfinite(xyz).all()
        assert pix["pixels"].shape[1:] == (256, 256, 3)
        return {
            "episodes": int(len(pix["ep_len"])),
            "rows_2d": int(len(pix["action"])),
            "rows_3d": int(len(pcd["action"])),
            "action_equal": True,
            "metadata_equal": True,
            "pcd_finite": True,
        }


def _replay(pixel_path: Path, seed: int, count: int) -> dict:
    with h5py.File(pixel_path, "r") as pix:
        lengths, offsets = pix["ep_len"][:], pix["ep_offset"][:]
        episodes = min(int(count), len(lengths))
        env = _pen_env()
        checked_rows = 0
        try:
            for episode in range(episodes):
                episode_seed = seed + episode
                np.random.seed(episode_seed)
                env.seed(episode_seed)
                obs = np.asarray(env.reset(), dtype=np.float32)
                start, length = int(offsets[episode]), int(lengths[episode])
                for t in range(length):
                    row = start + t
                    np.testing.assert_allclose(obs, pix["observation"][row], rtol=0.0, atol=1e-6)
                    np.testing.assert_allclose(env.data.qpos, pix["qpos"][row], rtol=0.0, atol=1e-7)
                    obs, _, _, _ = env.step(np.asarray(pix["action"][row], dtype=np.float32))
                    checked_rows += 1
        finally:
            env.close()
    return {"episodes": episodes, "rows": checked_rows, "state_replay": True}


def run(args) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pixel_path = args.output_dir / "pen_smoke.h5"
    pcd_path = args.output_dir / "pen_smoke_pcd.h5"
    config = AdroitPCDConfig()
    plan_rng = np.random.default_rng(args.seed)
    action_rng = np.random.default_rng(args.seed + 10_000)
    env = _pen_env()
    type_counts = np.zeros(3, dtype=np.int64)
    rows = 0
    try:
        with AdroitPairedWriter(pixel_path, pcd_path) as writer:
            for episode in range(args.episodes):
                episode_seed = args.seed + episode
                np.random.seed(episode_seed)
                env.seed(episode_seed)
                obs = np.asarray(env.reset(), dtype=np.float32)
                plan = draw_episode_plan(plan_rng, args.episode_len)
                type_counts[plan.sample_type] += 1
                home_qpos = np.asarray(env.data.qpos[:24], dtype=np.float64).copy()
                rows_data = {k: [] for k in (
                    "pixels", "pcd_xyz", "pcd_rgb", "observation", "action", "qpos", "qvel",
                    "sample_type", "rand_action", "noise_std", "branch_t", "branch_len", "seed", "step_idx",
                    "reward", "goal_achieved",
                )}
                for step in range(args.episode_len):
                    rgb, pcd_xyz, pcd_rgb, _ = capture_adroit_frame(env, action_rng, config)
                    state = env.get_env_state()
                    action, is_random = branch_action(plan, step, action_rng, env, home_qpos)
                    next_obs, reward, done, info = env.step(action)
                    rows_data["pixels"].append(rgb)
                    rows_data["pcd_xyz"].append(pcd_xyz)
                    rows_data["pcd_rgb"].append(pcd_rgb)
                    rows_data["observation"].append(obs)
                    rows_data["action"].append(action)
                    rows_data["qpos"].append(np.asarray(state["qpos"], dtype=np.float64))
                    rows_data["qvel"].append(np.asarray(state["qvel"], dtype=np.float64))
                    rows_data["sample_type"].append(plan.sample_type)
                    rows_data["rand_action"].append(int(is_random))
                    rows_data["noise_std"].append(plan.noise_std)
                    rows_data["branch_t"].append(plan.branch_t)
                    rows_data["branch_len"].append(plan.branch_len)
                    rows_data["seed"].append(episode_seed)
                    rows_data["step_idx"].append(step)
                    rows_data["reward"].append(float(reward))
                    rows_data["goal_achieved"].append(int(info.get("goal_achieved", False)))
                    rows += 1
                    obs = np.asarray(next_obs, dtype=np.float32)
                    if done:
                        break
                writer.write_episode({k: np.asarray(v) for k, v in rows_data.items()})
    finally:
        env.close()
    pairing = _check_pairing(pixel_path, pcd_path)
    replay = _replay(pixel_path, args.seed, args.replay_episodes)
    report = {
        "status": "PASS",
        "task": "pen-v0",
        "episodes": args.episodes,
        "rows": rows,
        "episode_len_requested": args.episode_len,
        "sample_type_counts": type_counts.tolist(),
        "sample_type_ratio": (type_counts / max(1, type_counts.sum())).round(4).tolist(),
        "seed_start": args.seed,
        "seed_end": args.seed + args.episodes - 1,
        "pairing": pairing,
        "replay": replay,
        "protocol_note": "bounded plumbing smoke; oracle is not a formal Pen expert controller",
    }
    dump_json(args.output_dir / "smoke_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--episode-len", type=int, default=100)
    parser.add_argument("--replay-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
