#!/usr/bin/env python3
"""Evaluate a LeWM-PCD checkpoint on held-out ManiSkill PickCube episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

from gymnasium.spaces import Box
import numpy as np


def build_policy_info(
    pcd_xyz: np.ndarray,
    pcd_rgb: np.ndarray,
    goal_xyz: np.ndarray,
    goal_rgb: np.ndarray,
    previous_action: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the batched one-frame contract expected by PCDWorldModelPolicy."""
    return {
        "pcd_xyz": np.asarray(pcd_xyz, np.float32)[None, None],
        "pcd_rgb": np.asarray(pcd_rgb, np.uint8)[None, None],
        "goal_pcd_xyz": np.asarray(goal_xyz, np.float32)[None, None],
        "goal_pcd_rgb": np.asarray(goal_rgb, np.uint8)[None, None],
        "action": np.asarray(previous_action, np.float32)[None, None],
        "terminated": np.zeros(1, dtype=bool),
    }


def clip_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Clip an executed action to the simulator's Box bounds."""
    return np.clip(np.asarray(action, np.float32), low, high).astype(np.float32)


def build_eval_env_kwargs(source: dict, eval_budget: int) -> dict:
    """Copy replay environment settings and honor the requested step budget."""
    if eval_budget <= 0:
        raise ValueError("eval_budget must be positive")
    kwargs = dict(source)
    kwargs["max_episode_steps"] = int(eval_budget)
    kwargs.update(
        obs_mode="rgb+depth+position+segmentation",
        render_mode=None,
        sim_backend="physx_cpu",
        render_backend="sapien_cuda",
    )
    return kwargs


def policy_environment(env):
    """Return the ManiSkill vector env expected by stable-worldmodel policies."""
    vector_env = env.unwrapped
    single_action_space = vector_env.single_action_space
    batch_shape = (vector_env.num_envs, *single_action_space.shape)
    action_space = Box(
        low=np.broadcast_to(single_action_space.low, batch_shape).copy(),
        high=np.broadcast_to(single_action_space.high, batch_shape).copy(),
        dtype=single_action_space.dtype,
    )
    return SimpleNamespace(
        num_envs=vector_env.num_envs,
        single_action_space=single_action_space,
        action_space=action_space,
    )


def _first_scene_id(value) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _robot_scene_ids(env) -> set[int]:
    ids: set[int] = set()
    for link in env.unwrapped.agent.robot.links:
        ids.update(int(item) for item in np.asarray(link.per_scene_id).reshape(-1))
    return ids


def _local_table_semantic_ids(
    semantic_ids: np.ndarray,
    points: np.ndarray,
    table_id: int,
    cube_xyz: np.ndarray,
    tcp_xyz: np.ndarray,
    radius: float = 0.20,
) -> np.ndarray:
    """Keep table points near the task objects for the object-aware protocol."""
    semantic_ids = np.asarray(semantic_ids).copy()
    table_mask = semantic_ids == table_id
    if not table_mask.any():
        return semantic_ids
    table_xy = np.asarray(points, np.float32)[table_mask, :2]
    cube_xy = np.asarray(cube_xyz, np.float32).reshape(3)[:2]
    tcp_xy = np.asarray(tcp_xyz, np.float32).reshape(3)[:2]
    local = np.minimum(
        np.linalg.norm(table_xy - cube_xy, axis=1),
        np.linalg.norm(table_xy - tcp_xy, axis=1),
    ) <= radius
    table_indices = np.flatnonzero(table_mask)
    semantic_ids[table_indices[~local]] = -1
    return semantic_ids


def object_aware_pointcloud(
    env,
    observation,
    renderer,
    episode_id: int,
    frame_idx: int,
    num_points: int,
    voxel_size: float,
    return_groups: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Render and sample the same object-aware PCD contract used for training."""
    from scripts.data.object_aware_sampler import sample_object_aware_point_cloud

    merged = renderer.capture(observation)
    cube_id = _first_scene_id(env.unwrapped.cube.per_scene_id)
    table_id = _first_scene_id(env.unwrapped.table_scene.table.per_scene_id)
    cube_xyz = env.unwrapped.cube.pose.p[0].detach().cpu().numpy().astype(np.float32)
    tcp_xyz = (
        observation["extra"]["tcp_pose"][0].detach().cpu().numpy().astype(np.float32)[:3]
    )
    semantic_ids = _local_table_semantic_ids(
        merged["semantic_id"], merged["xyz"], table_id, cube_xyz, tcp_xyz
    )
    xyz, colors, groups = sample_object_aware_point_cloud(
        merged["xyz"],
        merged["rgb"],
        semantic_ids,
        tcp_xyz=tcp_xyz,
        cube_id=cube_id,
        robot_ids=_robot_scene_ids(env),
        table_id=table_id,
        seed=3072 + episode_id * 1_000_003 + frame_idx,
        num_points=num_points,
        voxel_size=voxel_size,
        bounds=((-0.8, 0.5), (-0.6, 0.6), (-0.05, 0.8)),
        episode=episode_id,
        frame=frame_idx,
        allow_cube_replacement=True,
    )
    return (xyz, colors, groups) if return_groups else (xyz, colors)


def mask_goal_to_cube(
    current_xyz: np.ndarray,
    current_rgb: np.ndarray,
    current_groups: np.ndarray,
    goal_xyz: np.ndarray,
    goal_rgb: np.ndarray,
    goal_groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the goal Cube while preserving the current robot/table context.

    The PCD encoder has no semantic head.  Matching the full final cloud would
    therefore also ask CEM to reproduce the expert's final robot pose.  This
    task-specific target keeps the Cube geometry from the goal frame and the
    non-Cube support points from the current frame.
    """
    current_xyz = np.asarray(current_xyz, np.float32)
    current_rgb = np.asarray(current_rgb, np.uint8)
    current_groups = np.asarray(current_groups)
    goal_xyz = np.asarray(goal_xyz, np.float32)
    goal_rgb = np.asarray(goal_rgb, np.uint8)
    goal_groups = np.asarray(goal_groups)
    current_non_cube = current_groups != 1
    goal_cube = goal_groups == 1
    if current_non_cube.sum() != goal_groups.size - goal_cube.sum():
        raise ValueError("current and goal object-aware group quotas differ")
    masked_xyz = goal_xyz.copy()
    masked_rgb = goal_rgb.copy()
    masked_xyz[~goal_cube] = current_xyz[current_non_cube]
    masked_rgb[~goal_cube] = current_rgb[current_non_cube]
    return masked_xyz, masked_rgb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-json", required=True, type=Path)
    parser.add_argument("--trajectory-h5", required=True, type=Path)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--train-h5", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--start-episode", type=int, default=900)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--num-points", type=int, default=1200)
    parser.add_argument(
        "--sampling-protocol",
        choices=("uniform_v1", "object_aware_v2"),
        default="uniform_v1",
    )
    parser.add_argument("--extra-resolution", type=int, default=320)
    parser.add_argument("--voxel-size", type=float, default=0.0025)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--history-len", type=int, default=1)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument(
        "--var-scale",
        type=float,
        default=1.0,
        help="Initial CEM action-distribution standard deviation in normalized units.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--goal-mask-noncube",
        action="store_true",
        help="Keep current robot/table points in the goal and change only the Cube.",
    )
    parser.add_argument(
        "--final-release-steps",
        type=int,
        default=0,
        help="Diagnostic only: reserve final steps for current-arm hold and gripper open.",
    )
    args = parser.parse_args()

    if args.horizon * args.action_block > args.eval_budget:
        raise ValueError("planning horizon exceeds evaluation budget")
    if args.history_len < 1:
        raise ValueError("history-len must be positive")
    if args.topk > args.num_samples:
        raise ValueError("topk cannot exceed num_samples")
    if args.var_scale <= 0:
        raise ValueError("var-scale must be positive")
    if args.final_release_steps < 0 or args.final_release_steps > args.eval_budget:
        raise ValueError("final-release-steps must be between 0 and eval-budget")

    import gymnasium as gym
    import h5py
    import torch
    from sklearn.preprocessing import StandardScaler

    import mani_skill.envs  # noqa: F401: register environments
    from mani_skill.envs.utils.observations.observations import sensor_data_to_pointcloud
    from mani_skill.trajectory import utils as trajectory_utils
    import stable_worldmodel as swm
    from stable_worldmodel.planning.solver import CEMSolver

    from envs import PCDWorldModelPolicy, PointCloudScaler
    from scripts.data.maniskill_pickcube import sample_point_cloud
    from scripts.data.maniskill_object_aware_render import MultiViewPickCubeRenderer

    metadata = json.loads(args.metadata_json.read_text())
    selected = metadata["episodes"][
        args.start_episode : args.start_episode + args.num_episodes
    ]
    if len(selected) != args.num_episodes:
        raise ValueError(f"only {len(selected)} evaluation episodes are available")

    with h5py.File(args.train_h5, "r") as train_data:
        train_actions = train_data["action"][:]
    train_actions = train_actions[np.isfinite(train_actions).all(axis=1)]
    action_scaler = StandardScaler().fit(train_actions)

    model = swm.wm.utils.load_pretrained(str(args.checkpoint))
    model = model.to("cuda").eval()
    model.requires_grad_(False)

    env_kwargs = build_eval_env_kwargs(metadata["env_info"]["env_kwargs"], args.eval_budget)
    env = gym.make(metadata["env_info"]["env_id"], **env_kwargs)
    renderer = (
        MultiViewPickCubeRenderer(env, extra_resolution=args.extra_resolution)
        if args.sampling_protocol == "object_aware_v2"
        else None
    )
    pcd_scaler = PointCloudScaler(
        center=(-0.15, 0.0, 0.375), scale=1.5, use_rgb=True
    )
    process = {"action": action_scaler, "pcd": pcd_scaler, "goal": pcd_scaler}
    plan_config = swm.PlanConfig(
        horizon=args.horizon,
        receding_horizon=args.receding_horizon,
        history_len=args.history_len,
        action_block=args.action_block,
        warm_start=True,
    )

    completed = {}
    if args.output_json.exists():
        prior = json.loads(args.output_json.read_text())
        completed = {int(row["episode"]): row for row in prior.get("results", [])}

    results = []
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(args.trajectory_h5, "r") as source:
            for episode_meta in selected:
                episode_id = int(episode_meta["episode_id"])
                if episode_id in completed:
                    results.append(completed[episode_id])
                    continue
                started = time.time()
                trajectory = source[f"traj_{episode_id}"]
                states = trajectory_utils.dict_to_list_of_dicts(
                    trajectory["env_states"]
                )
                with np.load(
                    args.shard_dir / f"episode_{episode_id:04d}.npz",
                    allow_pickle=False,
                ) as shard:
                    goal_xyz = shard["pcd_xyz"][-1]
                    goal_rgb = shard["pcd_rgb"][-1]
                    goal_groups = shard["pcd_group"][-1] if args.goal_mask_noncube else None

                env.reset(**episode_meta["reset_kwargs"])
                env.unwrapped.set_state_dict(states[0])
                solver = CEMSolver(
                    cost=model,
                    batch_size=1,
                    num_samples=args.num_samples,
                    var_scale=args.var_scale,
                    n_steps=args.cem_steps,
                    topk=args.topk,
                    device="cuda",
                    seed=args.seed + episode_id,
                )
                policy = PCDWorldModelPolicy(
                    solver=solver,
                    config=plan_config,
                    process=process,
                    transform={},
                    use_rgb=True,
                )
                policy.set_env(policy_environment(env))
                previous_action = np.full(8, np.nan, dtype=np.float32)
                success = False
                final_info = env.unwrapped.evaluate()
                steps = 0

                for step in range(args.eval_budget):
                    if step >= args.eval_budget - args.final_release_steps:
                        qpos = env.unwrapped.agent.robot.get_qpos()[0].detach().cpu().numpy()
                        action = np.concatenate([qpos[:7], np.ones(1, dtype=np.float32)])
                    else:
                        observation = env.unwrapped.get_obs()
                        if args.sampling_protocol == "object_aware_v2":
                            pcd_xyz, pcd_rgb, pcd_groups = object_aware_pointcloud(
                                env,
                                observation,
                                renderer,
                                episode_id,
                                step,
                                args.num_points,
                                args.voxel_size,
                                return_groups=True,
                            )
                            if args.goal_mask_noncube:
                                goal_xyz_step, goal_rgb_step = mask_goal_to_cube(
                                    pcd_xyz, pcd_rgb, pcd_groups,
                                    goal_xyz, goal_rgb, goal_groups,
                                )
                            else:
                                goal_xyz_step, goal_rgb_step = goal_xyz, goal_rgb
                        else:
                            pointcloud = sensor_data_to_pointcloud(
                                observation, env.unwrapped._sensors
                            )["pointcloud"]
                            pcd_xyz, pcd_rgb = sample_point_cloud(
                                pointcloud["xyzw"][0].detach().cpu().numpy(),
                                pointcloud["rgb"][0].detach().cpu().numpy(),
                                num_points=args.num_points,
                                seed=3072 + episode_id * 1_000_003 + step,
                                bounds=((-0.8, 0.5), (-0.6, 0.6), (-0.05, 0.8)),
                            )
                            goal_xyz_step, goal_rgb_step = goal_xyz, goal_rgb
                        policy_info = build_policy_info(
                            pcd_xyz, pcd_rgb, goal_xyz_step, goal_rgb_step, previous_action
                        )
                        action = policy.get_action(policy_info)
                    action = clip_action(action, env.action_space.low, env.action_space.high)
                    _, _, terminated, truncated, final_info = env.step(action)
                    previous_action = action.reshape(-1)
                    steps = step + 1
                    success = bool(final_info["success"][0].detach().cpu().item())
                    done = bool(terminated[0].detach().cpu().item()) or bool(
                        truncated[0].detach().cpu().item()
                    )
                    if success or done:
                        break

                cube_pos = (
                    env.unwrapped.cube.pose.raw_pose[0, :3].detach().cpu().numpy()
                )
                goal_pos = env.unwrapped.goal_site.pose.raw_pose[0, :3].detach().cpu().numpy()
                row = {
                    "episode": episode_id,
                    "success": success,
                    "steps": steps,
                    "cube_goal_distance": float(np.linalg.norm(cube_pos - goal_pos)),
                    "is_obj_placed": bool(
                        final_info["is_obj_placed"][0].detach().cpu().item()
                    ),
                    "is_robot_static": bool(
                        final_info["is_robot_static"][0].detach().cpu().item()
                    ),
                    "elapsed_seconds": round(time.time() - started, 3),
                }
                results.append(row)
                payload = {
                    "status": "RUNNING",
                    "checkpoint": str(args.checkpoint),
                    "protocol": {
                        "start_episode": args.start_episode,
                        "num_episodes": args.num_episodes,
                        "eval_budget": args.eval_budget,
                        "horizon": args.horizon,
                        "history_len": args.history_len,
                        "receding_horizon": args.receding_horizon,
                        "action_block": args.action_block,
                        "num_samples": args.num_samples,
                        "cem_steps": args.cem_steps,
                        "topk": args.topk,
                        "var_scale": args.var_scale,
                        "goal_mask_noncube": args.goal_mask_noncube,
                        "final_release_steps": args.final_release_steps,
                        "sampling_protocol": args.sampling_protocol,
                        "extra_resolution": args.extra_resolution,
                        "voxel_size": args.voxel_size,
                    },
                    "completed": len(results),
                    "successes": int(sum(result["success"] for result in results)),
                    "results": results,
                }
                tmp_path = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
                tmp_path.write_text(json.dumps(payload, indent=2))
                tmp_path.replace(args.output_json)
                print(json.dumps(row), flush=True)
    finally:
        env.close()

    payload = {
        "status": "COMPLETE",
        "checkpoint": str(args.checkpoint),
        "protocol": {
            "start_episode": args.start_episode,
            "num_episodes": args.num_episodes,
            "eval_budget": args.eval_budget,
            "horizon": args.horizon,
            "history_len": args.history_len,
            "receding_horizon": args.receding_horizon,
            "action_block": args.action_block,
            "num_samples": args.num_samples,
            "cem_steps": args.cem_steps,
            "topk": args.topk,
            "var_scale": args.var_scale,
            "goal_mask_noncube": args.goal_mask_noncube,
            "final_release_steps": args.final_release_steps,
            "sampling_protocol": args.sampling_protocol,
            "extra_resolution": args.extra_resolution,
            "voxel_size": args.voxel_size,
        },
        "completed": len(results),
        "successes": int(sum(result["success"] for result in results)),
        "results": results,
    }
    payload["success_rate"] = payload["successes"] / args.num_episodes
    tmp_path = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(args.output_json)
    print(json.dumps({k: payload[k] for k in ("status", "completed", "successes", "success_rate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
