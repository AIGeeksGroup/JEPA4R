"""Validate Franka smoke datasets and replay their recorded dynamics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import h5py
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import envs  # noqa: E402,F401

ENV_ID = "swm/OGBCubeFrankaPCD-v0"
REPLAY_ATOL = 1e-10


def validate_episode_layout(
    ep_len: np.ndarray,
    ep_offset: np.ndarray,
    total_rows: int,
    expected_episodes: int | None = None,
) -> None:
    ep_len = np.asarray(ep_len)
    ep_offset = np.asarray(ep_offset)
    assert ep_len.ndim == ep_offset.ndim == 1
    assert len(ep_len) == len(ep_offset)
    if expected_episodes is not None:
        assert len(ep_len) == expected_episodes
    assert np.all(ep_len > 1)
    expected_offsets = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(ep_len[:-1], dtype=np.int64)]
    )
    assert np.array_equal(ep_offset, expected_offsets), "episode offsets are not contiguous"
    assert int(ep_len.sum()) == total_rows


def validate_action_rows(
    actions: np.ndarray,
    ep_len: np.ndarray,
    ep_offset: np.ndarray,
) -> None:
    actions = np.asarray(actions)
    assert actions.ndim == 2 and actions.shape[1] == 5
    terminal = np.asarray(ep_offset) + np.asarray(ep_len) - 1
    terminal_mask = np.zeros(len(actions), dtype=bool)
    terminal_mask[terminal] = True
    assert np.isfinite(actions[~terminal_mask]).all(), "non-terminal actions contain NaN/Inf"
    assert np.isnan(actions[terminal_mask]).all(), "terminal actions must be NaN sentinels"


def point_cloud_frame_stats(xyz: np.ndarray, cube_center: np.ndarray) -> dict[str, int]:
    xyz = np.asarray(xyz)
    cube_center = np.asarray(cube_center)
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    cube_mask = np.all(np.abs(xyz - cube_center[None]) < 0.022, axis=1)
    unique_xyz = np.unique(xyz, axis=0)
    unique_cube = np.all(
        np.abs(unique_xyz - cube_center[None]) < 0.022,
        axis=1,
    )
    return {
        "cube_points": int(cube_mask.sum()),
        "robot_points": int((~cube_mask).sum()),
        "repeated_points": int(len(xyz) - len(unique_xyz)),
        "unique_cube_points": int(unique_cube.sum()),
    }


def _column(handle: h5py.File, nested: str) -> h5py.Dataset:
    if nested in handle:
        return handle[nested]
    flat = nested.replace("/", "_")
    if flat in handle:
        return handle[flat]
    raise KeyError(nested)


def validate_dataset(path: Path, expected_episodes: int = 10) -> dict:
    with h5py.File(path, "r") as handle:
        ep_len = handle["ep_len"][:]
        ep_offset = handle["ep_offset"][:]
        total_rows = int(handle["action"].shape[0])
        validate_episode_layout(ep_len, ep_offset, total_rows, expected_episodes)
        validate_action_rows(handle["action"][:], ep_len, ep_offset)
        state_widths = {
            "observation": 30,
            "target": 30,
            "qpos": 16,
            "qvel": 15,
            "prev_qpos": 16,
            "prev_qvel": 15,
            "control": 8,
        }
        for name, width in state_widths.items():
            values = handle[name]
            expected = (total_rows, width)
            assert values.shape == expected, (
                f"{name} shape {values.shape} does not match Franka expected {expected}"
            )
            assert np.isfinite(values[:]).all(), f"{name} contains NaN/Inf"

        report = {
            "path": str(path),
            "episodes": int(len(ep_len)),
            "rows": total_rows,
            "episode_lengths": ep_len.astype(int).tolist(),
            "action_shape": list(handle["action"].shape),
            "action_dtype": str(handle["action"].dtype),
        }
        if "pcd_xyz" in handle:
            xyz = handle["pcd_xyz"]
            rgb = handle["pcd_rgb"]
            assert xyz.shape == (total_rows, 1200, 3)
            assert rgb.shape == (total_rows, 1200, 3)
            assert xyz.dtype == np.float32
            assert rgb.dtype == np.uint8
            assert np.isfinite(xyz[:]).all()
            cube_pos = _column(handle, "privileged/block_0_pos")
            sample_rows = np.unique(
                np.linspace(0, total_rows - 1, min(32, total_rows), dtype=int)
            )
            frame_stats = [
                point_cloud_frame_stats(xyz[row], cube_pos[row])
                for row in sample_rows
            ]
            for stats in frame_stats:
                if stats["unique_cube_points"] >= 300:
                    assert stats["cube_points"] >= 300
            cube_counts = np.array([s["cube_points"] for s in frame_stats])
            robot_counts = np.array([s["robot_points"] for s in frame_stats])
            report["pcd"] = {
                "shape": list(xyz.shape),
                "xyz_dtype": str(xyz.dtype),
                "rgb_dtype": str(rgb.dtype),
                "sampled_frames": int(len(sample_rows)),
                "cube_points_min": int(cube_counts.min()),
                "cube_points_max": int(cube_counts.max()),
                "cube_fraction_mean": float(cube_counts.mean() / xyz.shape[1]),
                "robot_fraction_mean": float(robot_counts.mean() / xyz.shape[1]),
                "repeated_points": int(sum(s["repeated_points"] for s in frame_stats)),
                "xyz_min": np.min(xyz[sample_rows], axis=(0, 1)).astype(float).tolist(),
                "xyz_max": np.max(xyz[sample_rows], axis=(0, 1)).astype(float).tolist(),
            }
        elif "pixels" in handle:
            pixels = handle["pixels"]
            assert pixels.shape == (total_rows, 224, 224, 3)
            assert pixels.dtype == np.uint8
            sample_rows = np.unique(
                np.linspace(0, total_rows - 1, min(16, total_rows), dtype=int)
            )
            variances = [float(np.var(pixels[row])) for row in sample_rows]
            assert min(variances) > 0.0
            report["pixels"] = {
                "shape": list(pixels.shape),
                "dtype": str(pixels.dtype),
                "sample_variance_min": min(variances),
                "sample_variance_max": max(variances),
            }
        else:
            raise AssertionError("dataset has neither pcd_xyz nor pixels")
    return report


def replay_dataset(
    path: Path,
    collection_seed: int = 3072,
    atol: float = REPLAY_ATOL,
) -> dict:
    world_seed = int(np.random.default_rng(collection_seed).integers(0, 1_000_000))
    episode_reports = []
    overall_max = 0.0
    with h5py.File(path, "r") as handle:
        ep_len = handle["ep_len"][:]
        ep_offset = handle["ep_offset"][:]
        target_pos = _column(handle, "privileged/target_block_pos")
        target_yaw = _column(handle, "privileged/target_block_yaw")
        target_block = _column(handle, "privileged/target_block")
        cube_pos = _column(handle, "privileged/block_0_pos")
        for episode, (offset, length) in enumerate(zip(ep_offset, ep_len)):
            env = gym.make(
                ENV_ID,
                env_type="single",
                ob_type="states",
                mode="data_collection",
                terminate_at_goal=False,
                visualize_info=False,
                width=224,
                height=224,
            )
            try:
                env.reset(seed=world_seed + episode)
                robot = env.unwrapped
                row = int(offset)
                robot.set_state(handle["qpos"][row], handle["qvel"][row])
                robot._data.ctrl[:] = handle["control"][row]
                robot._data.time = float(handle["time"][row, 0])
                robot._prev_qpos = handle["prev_qpos"][row].copy()
                robot._prev_qvel = handle["prev_qvel"][row].copy()
                robot._target_block = int(target_block[row])
                mocap_id = robot._cube_target_mocap_ids[robot._target_block]
                robot._data.mocap_pos[mocap_id] = target_pos[row]
                yaw = float(np.asarray(target_yaw[row]).reshape(-1)[0])
                robot._data.mocap_quat[mocap_id] = [
                    np.cos(yaw / 2),
                    0.0,
                    0.0,
                    np.sin(yaw / 2),
                ]
                robot._reset_next_step = False
                mujoco.mj_forward(robot._model, robot._data)

                episode_max = 0.0
                for step in range(int(length) - 1):
                    expected_row = row + step + 1
                    _, _, _, _, info = env.step(handle["action"][row + step])
                    comparisons = (
                        (robot._data.qpos, handle["qpos"][expected_row]),
                        (robot._data.qvel, handle["qvel"][expected_row]),
                        (info["privileged/block_0_pos"], cube_pos[expected_row]),
                    )
                    step_max = max(
                        float(np.max(np.abs(actual - expected)))
                        for actual, expected in comparisons
                    )
                    episode_max = max(episode_max, step_max)
                overall_max = max(overall_max, episode_max)
                episode_reports.append(
                    {
                        "episode": episode,
                        "steps": int(length) - 1,
                        "max_abs_error": episode_max,
                    }
                )
            finally:
                env.close()
    return {
        "collection_seed": collection_seed,
        "world_seed": world_seed,
        "episodes": episode_reports,
        "checked_steps": int(sum(r["steps"] for r in episode_reports)),
        "max_abs_error": overall_max,
        "atol": atol,
        "passed": overall_max <= atol,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=10)
    parser.add_argument("--collection-seed", type=int, default=3072)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    report = {"dataset": validate_dataset(args.dataset, args.expected_episodes)}
    if not args.validate_only and "pcd" in report["dataset"]:
        report["replay"] = replay_dataset(args.dataset, args.collection_seed)
    report["passed"] = report.get("replay", {"passed": True})["passed"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
