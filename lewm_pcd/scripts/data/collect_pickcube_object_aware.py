#!/usr/bin/env python3
"""Replay ManiSkill PickCube trajectories with object-aware multiview PCD."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np

import mani_skill.envs  # noqa: F401: register ManiSkill environments
from mani_skill.trajectory import utils as trajectory_utils

from maniskill_object_aware_render import MultiViewPickCubeRenderer
from object_aware_sampler import ObjectAwareSamplingError, sample_object_aware_point_cloud
from maniskill_pickcube import (
    align_actions_to_states,
    load_episode_shard,
    save_episode_shard,
    validate_episode_arrays,
)


WORKSPACE_BOUNDS = ((-0.8, 0.5), (-0.6, 0.6), (-0.05, 0.8))
PROTOCOL = "object_aware_v2"


def _first_scene_id(value) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _robot_scene_ids(env) -> set[int]:
    ids: set[int] = set()
    for link in env.unwrapped.agent.robot.links:
        ids.update(int(value) for value in np.asarray(link.per_scene_id).reshape(-1))
    return ids


def _local_table_semantic_ids(
    semantic_ids: np.ndarray,
    points: np.ndarray,
    table_id: int,
    cube_xyz: np.ndarray,
    tcp_xyz: np.ndarray,
    radius: float = 0.20,
) -> np.ndarray:
    """Hide distant table points while preserving all non-table semantics."""
    semantic_ids = np.asarray(semantic_ids).copy()
    table_mask = semantic_ids == table_id
    if not table_mask.any():
        return semantic_ids
    points_xy = np.asarray(points, dtype=np.float32)[table_mask, :2]
    cube_xy = np.asarray(cube_xyz, dtype=np.float32).reshape(3)[:2]
    tcp_xy = np.asarray(tcp_xyz, dtype=np.float32).reshape(3)[:2]
    local = np.minimum(
        np.linalg.norm(points_xy - cube_xy, axis=1),
        np.linalg.norm(points_xy - tcp_xy, axis=1),
    ) <= radius
    table_indices = np.flatnonzero(table_mask)
    semantic_ids[table_indices[~local]] = -1
    return semantic_ids


def _frame_arrays(env, observation, renderer, episode_id: int, frame_idx: int, num_points: int, voxel_size: float):
    merged = renderer.capture(observation)
    cube_id = _first_scene_id(env.unwrapped.cube.per_scene_id)
    table_id = _first_scene_id(env.unwrapped.table_scene.table.per_scene_id)
    robot_ids = _robot_scene_ids(env)
    cube_xyz = env.unwrapped.cube.pose.p[0].detach().cpu().numpy().astype(np.float32)
    tcp_pose = observation["extra"]["tcp_pose"][0].detach().cpu().numpy().astype(np.float32)
    tcp_xyz = tcp_pose[:3]
    semantic_ids = _local_table_semantic_ids(
        merged["semantic_id"], merged["xyz"], table_id, cube_xyz, tcp_xyz
    )
    xyz, colors, groups = sample_object_aware_point_cloud(
        merged["xyz"],
        merged["rgb"],
        semantic_ids,
        tcp_xyz=tcp_xyz,
        cube_id=cube_id,
        robot_ids=robot_ids,
        table_id=table_id,
        seed=3072 + episode_id * 1_000_003 + frame_idx,
        num_points=num_points,
        voxel_size=voxel_size,
        bounds=WORKSPACE_BOUNDS,
        episode=episode_id,
        frame=frame_idx,
    )
    camera = observation["sensor_data"]["base_camera"]
    frame = {
        "rgb": camera["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8),
        "depth": camera["depth"][0].detach().cpu().numpy().astype(np.int16),
        "pcd_xyz": xyz,
        "pcd_rgb": colors,
        "pcd_group": groups,
        "proprio": np.concatenate(
            [
                observation["agent"]["qpos"][0].detach().cpu().numpy(),
                observation["agent"]["qvel"][0].detach().cpu().numpy(),
                tcp_pose,
            ]
        ).astype(np.float32),
        "cube_pose": env.unwrapped.cube.pose.raw_pose[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "goal_pos": observation["extra"]["goal_pos"][0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
    }
    evaluation = env.unwrapped.evaluate()
    frame["success"] = bool(evaluation["success"][0].detach().cpu().item())
    frame["is_grasped"] = bool(observation["extra"]["is_grasped"][0].detach().cpu().item())
    return frame


def replay_episode(env, trajectory, episode_id: int, reset_kwargs: dict, renderer, num_points: int, voxel_size: float):
    actions = np.asarray(trajectory["actions"], dtype=np.float32)
    states = trajectory_utils.dict_to_list_of_dicts(trajectory["env_states"])
    num_frames = len(actions) + 1
    if len(states) != num_frames:
        raise ValueError(f"episode {episode_id}: {len(states)} states for {len(actions)} actions")
    env.reset(**reset_kwargs)
    frame_data = {key: [] for key in ("rgb", "depth", "pcd_xyz", "pcd_rgb", "pcd_group", "proprio", "cube_pose", "goal_pos", "success", "is_grasped")}
    for frame_idx, state in enumerate(states):
        env.unwrapped.set_state_dict(state)
        observation = env.unwrapped.get_obs()
        frame = _frame_arrays(env, observation, renderer, episode_id, frame_idx, num_points, voxel_size)
        for key, value in frame.items():
            frame_data[key].append(value)
    arrays = {key: np.stack(values) for key, values in frame_data.items()}
    arrays["action"] = align_actions_to_states(actions, num_frames)
    arrays["episode_idx"] = np.full(num_frames, episode_id, dtype=np.int32)
    arrays["step_idx"] = np.arange(num_frames, dtype=np.int32)
    arrays["source_success"] = np.concatenate([np.zeros(1, dtype=bool), np.asarray(trajectory["success"], dtype=bool)])
    arrays["source_terminated"] = np.concatenate([np.zeros(1, dtype=bool), np.asarray(trajectory["terminated"], dtype=bool)])
    arrays["source_truncated"] = np.concatenate([np.zeros(1, dtype=bool), np.asarray(trajectory["truncated"], dtype=bool)])
    validate_episode_arrays(arrays, num_frames, num_points)
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-json", required=True, type=Path)
    parser.add_argument("--trajectory-h5", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--num-points", type=int, default=1200)
    parser.add_argument("--extra-resolution", type=int, default=384)
    parser.add_argument("--voxel-size", type=float, default=0.0025)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata = json.loads(args.metadata_json.read_text())
    episodes = metadata["episodes"]
    stop = args.start_episode + args.num_episodes
    if args.start_episode < 0 or stop > len(episodes):
        raise ValueError(f"episode range [{args.start_episode}, {stop}) exceeds {len(episodes)} episodes")
    if metadata["env_info"]["env_id"] != "PickCube-v1":
        raise ValueError(f"expected PickCube-v1, got {metadata['env_info']['env_id']}")

    env_kwargs = dict(metadata["env_info"]["env_kwargs"])
    env_kwargs.update(obs_mode="rgb+depth+position+segmentation", render_mode=None, sim_backend="physx_cpu", render_backend="sapien_cuda")
    env = gym.make(metadata["env_info"]["env_id"], **env_kwargs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_results = []
    renderer = MultiViewPickCubeRenderer(env, extra_resolution=args.extra_resolution)
    try:
        with h5py.File(args.trajectory_h5, "r") as source:
            for episode_pos in range(args.start_episode, stop):
                episode_meta = episodes[episode_pos]
                episode_id = int(episode_meta["episode_id"])
                output_path = args.output_dir / f"episode_{episode_id:04d}.npz"
                if output_path.exists() and not args.overwrite:
                    try:
                        arrays = load_episode_shard(output_path, int(episode_meta["elapsed_steps"]) + 1, args.num_points)
                        if "pcd_group" not in arrays:
                            raise ValueError("uniform shard has no pcd_group")
                        result = {"episode": episode_id, "frames": len(arrays["pcd_xyz"]), "status": "SKIP_VALID", "path": str(output_path)}
                        summary_results.append(result)
                        print(json.dumps(result), flush=True)
                        continue
                    except (OSError, ValueError, KeyError):
                        output_path.unlink()
                started = time.time()
                try:
                    arrays = replay_episode(
                        env,
                        source[f"traj_{episode_id}"],
                        episode_id,
                        episode_meta["reset_kwargs"],
                        renderer,
                        args.num_points,
                        args.voxel_size,
                    )
                except ObjectAwareSamplingError:
                    raise
                save_episode_shard(output_path, arrays)
                result = {
                    "episode": episode_id,
                    "frames": len(arrays["pcd_xyz"]),
                    "status": "WRITTEN",
                    "elapsed_seconds": round(time.time() - started, 3),
                    "cube_min_count": int(np.count_nonzero(arrays["pcd_group"] == 1, axis=1).min()),
                    "cube_mean_count": float(np.count_nonzero(arrays["pcd_group"] == 1, axis=1).mean()),
                    "path": str(output_path),
                }
                summary_results.append(result)
                print(json.dumps(result), flush=True)
    finally:
        env.close()

    summary = {
        "status": "PASS",
        "protocol": PROTOCOL,
        "start_episode": args.start_episode,
        "num_episodes": args.num_episodes,
        "num_points": args.num_points,
        "extra_resolution": args.extra_resolution,
        "voxel_size": args.voxel_size,
        "results": summary_results,
    }
    summary_path = args.output_dir / "last_run_summary.json"
    summary_tmp = summary_path.with_suffix(".json.tmp")
    summary_tmp.write_text(json.dumps(summary, indent=2))
    summary_tmp.replace(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
