#!/usr/bin/env python3
"""Replay official PickCube states and write resumable training-data shards."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np

import mani_skill.envs  # noqa: F401: register ManiSkill environments
from mani_skill.envs.utils.observations.observations import sensor_data_to_pointcloud
from mani_skill.trajectory import utils as trajectory_utils

from maniskill_pickcube import (
    align_actions_to_states,
    load_episode_shard,
    sample_point_cloud,
    save_episode_shard,
    validate_episode_arrays,
)


WORKSPACE_BOUNDS = ((-0.8, 0.5), (-0.6, 0.6), (-0.05, 0.8))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-json", required=True, type=Path)
    parser.add_argument("--trajectory-h5", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--num-points", type=int, default=1200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata = json.loads(args.metadata_json.read_text())
    episodes = metadata["episodes"]
    stop = args.start_episode + args.num_episodes
    if args.start_episode < 0 or stop > len(episodes):
        raise ValueError(
            f"episode range [{args.start_episode}, {stop}) exceeds {len(episodes)} episodes"
        )
    if metadata["env_info"]["env_id"] != "PickCube-v1":
        raise ValueError(f"expected PickCube-v1, got {metadata['env_info']['env_id']}")

    env_kwargs = dict(metadata["env_info"]["env_kwargs"])
    env_kwargs.update(
        obs_mode="rgb+depth+position+segmentation",
        render_mode=None,
        sim_backend="physx_cpu",
        render_backend="sapien_cuda",
    )
    env = gym.make(metadata["env_info"]["env_id"], **env_kwargs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    try:
        with h5py.File(args.trajectory_h5, "r") as source:
            for episode_pos in range(args.start_episode, stop):
                started = time.time()
                episode_meta = episodes[episode_pos]
                episode_id = int(episode_meta["episode_id"])
                trajectory = source[f"traj_{episode_id}"]
                actions = np.asarray(trajectory["actions"], dtype=np.float32)
                num_frames = len(actions) + 1
                output_path = args.output_dir / f"episode_{episode_id:04d}.npz"

                if output_path.exists() and not args.overwrite:
                    try:
                        load_episode_shard(output_path, num_frames, args.num_points)
                        result = {
                            "episode": episode_id,
                            "frames": num_frames,
                            "status": "SKIP_VALID",
                            "path": str(output_path),
                        }
                        results.append(result)
                        print(json.dumps(result), flush=True)
                        continue
                    except (OSError, ValueError, KeyError):
                        output_path.unlink()

                states = trajectory_utils.dict_to_list_of_dicts(
                    trajectory["env_states"]
                )
                if len(states) != num_frames:
                    raise ValueError(
                        f"episode {episode_id}: {len(states)} states for {len(actions)} actions"
                    )

                env.reset(**episode_meta["reset_kwargs"])
                frame_data = {
                    "rgb": [],
                    "depth": [],
                    "pcd_xyz": [],
                    "pcd_rgb": [],
                    "proprio": [],
                    "cube_pose": [],
                    "goal_pos": [],
                    "success": [],
                    "is_grasped": [],
                }
                for frame_idx, state in enumerate(states):
                    env.unwrapped.set_state_dict(state)
                    observation = env.unwrapped.get_obs()
                    camera = next(iter(observation["sensor_data"].values()))
                    frame_data["rgb"].append(
                        camera["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
                    )
                    frame_data["depth"].append(
                        camera["depth"][0].detach().cpu().numpy().astype(np.int16)
                    )

                    pointcloud = sensor_data_to_pointcloud(
                        observation, env.unwrapped._sensors
                    )["pointcloud"]
                    xyz, colors = sample_point_cloud(
                        pointcloud["xyzw"][0].detach().cpu().numpy(),
                        pointcloud["rgb"][0].detach().cpu().numpy(),
                        num_points=args.num_points,
                        seed=3072 + episode_id * 1_000_003 + frame_idx,
                        bounds=WORKSPACE_BOUNDS,
                    )
                    frame_data["pcd_xyz"].append(xyz)
                    frame_data["pcd_rgb"].append(colors)

                    agent = observation["agent"]
                    extra = observation["extra"]
                    proprio = np.concatenate(
                        [
                            agent["qpos"][0].detach().cpu().numpy(),
                            agent["qvel"][0].detach().cpu().numpy(),
                            extra["tcp_pose"][0].detach().cpu().numpy(),
                        ]
                    ).astype(np.float32)
                    frame_data["proprio"].append(proprio)
                    frame_data["cube_pose"].append(
                        env.unwrapped.cube.pose.raw_pose[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    frame_data["goal_pos"].append(
                        extra["goal_pos"][0].detach().cpu().numpy().astype(np.float32)
                    )
                    evaluation = env.unwrapped.evaluate()
                    frame_data["success"].append(
                        bool(evaluation["success"][0].detach().cpu().item())
                    )
                    frame_data["is_grasped"].append(
                        bool(extra["is_grasped"][0].detach().cpu().item())
                    )

                arrays = {key: np.stack(values) for key, values in frame_data.items()}
                arrays["action"] = align_actions_to_states(actions, num_frames)
                arrays["episode_idx"] = np.full(num_frames, episode_id, dtype=np.int32)
                arrays["step_idx"] = np.arange(num_frames, dtype=np.int32)
                arrays["source_success"] = np.concatenate(
                    [
                        np.zeros(1, dtype=bool),
                        np.asarray(trajectory["success"], dtype=bool),
                    ]
                )
                arrays["source_terminated"] = np.concatenate(
                    [
                        np.zeros(1, dtype=bool),
                        np.asarray(trajectory["terminated"], dtype=bool),
                    ]
                )
                arrays["source_truncated"] = np.concatenate(
                    [
                        np.zeros(1, dtype=bool),
                        np.asarray(trajectory["truncated"], dtype=bool),
                    ]
                )

                validate_episode_arrays(arrays, num_frames, args.num_points)
                save_episode_shard(output_path, arrays)
                result = {
                    "episode": episode_id,
                    "frames": num_frames,
                    "status": "WRITTEN",
                    "elapsed_seconds": round(time.time() - started, 3),
                    "pcd_min": arrays["pcd_xyz"].min(axis=(0, 1)).tolist(),
                    "pcd_max": arrays["pcd_xyz"].max(axis=(0, 1)).tolist(),
                    "success_final": bool(arrays["success"][-1]),
                    "source_success_final": bool(arrays["source_success"][-1]),
                    "path": str(output_path),
                }
                results.append(result)
                print(json.dumps(result), flush=True)
    finally:
        env.close()

    summary_path = args.output_dir / "last_run_summary.json"
    summary_tmp = summary_path.with_suffix(".json.tmp")
    summary_tmp.write_text(
        json.dumps(
            {
                "status": "PASS",
                "env_id": metadata["env_info"]["env_id"],
                "start_episode": args.start_episode,
                "num_episodes": args.num_episodes,
                "num_points": args.num_points,
                "workspace_bounds": WORKSPACE_BOUNDS,
                "results": results,
            },
            indent=2,
        )
    )
    summary_tmp.replace(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
