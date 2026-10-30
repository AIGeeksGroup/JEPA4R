"""Serial online-visual Native SR for Cube2 G05 RGB and G06 PCD."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
import importlib.util
import types
from pathlib import Path
from typing import Any

# GCIQL must stay on CPU while the frozen LeWM encoder uses the visible GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

CAMERA = {
    "source_name": "front",
    "render_name": "front_pixels",
    "position": [1.287, 0.0, 0.509],
    "fov": 45.0,
    "local_x_up_degrees": 5.0,
    "width": 224,
    "height": 224,
    "aa_samples": 4,
}

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "cube23_distance_gciql_v1"
EXT = CASE / "five_group_extension_v1"

GROUPS = {
    "G05": {
        "mode": "rgb",
        "wm": ROOT / "checkpoints/cube2/2d_epoch10/weights_epoch_10.pt",
        "wm_sha256": "995a99f4904d87e71424b2de7910e7e0cf79556de8c2cebd08ac582d70880190",
        "training": EXT / "configs/G05_cube2_rgb.json",
        "acceptance": EXT / "acceptance/G05_full_acceptance.json",
        "gate": EXT / "gates/G05/next_batch.npz",
        "final": EXT / "runs/G05_cube2_rgb/checkpoints/update_001000000",
        "manifest_sha256": "471b285fe717670565611f2594a5925b04a2de2726dd2b571210a02a13abfacd",
    },
    "G06": {
        "mode": "pcd",
        "wm": ROOT / "checkpoints/cube2/3d_epoch10/weights_epoch_10.pt",
        "wm_sha256": "f391f047aa73fa08b2b4534e6018fde6a219cc3ae6ba4464b907e2b9f9138a46",
        "training": EXT / "configs/G06_cube2_pcd.json",
        "acceptance": EXT / "acceptance/G06_full_acceptance.json",
        "gate": EXT / "gates/G06/next_batch.npz",
        "final": EXT / "runs/G06_cube2_pcd/checkpoints/update_001000000",
        "manifest_sha256": "e26cfd3d474b2b7a36d2832d5c6b093a2333ecba8a51cad2899a74f16200fe8f",
    },
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def seed_for(task: int, episode_index: int) -> int:
    if task not in range(1, 6) or episode_index not in range(50):
        raise ValueError("task must be 1..5 and episode_index must be 0..49")
    return 970000000 + task * 100000 + episode_index


def validate_latent(value: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(value)
    if value.shape != (192,) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f"{label} latent must be finite float32 shape (192,), got {value.shape} {value.dtype}")
    return value


def goal_xyz(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state)
    if state.shape != (37,) or state.dtype != np.float32 or not np.isfinite(state).all():
        raise ValueError("Cube2 goal state must be finite float32 shape (37,)")
    return np.asarray([state[19], state[20], state[21], state[28], state[29], state[30]], dtype=np.float32)


def make_policy_inputs(current_latent: np.ndarray, goal_latent: np.ndarray,
                       current_state: np.ndarray, goal_state: np.ndarray):
    current_latent = validate_latent(current_latent, "current")
    goal_latent = validate_latent(goal_latent, "goal")
    current_state = np.asarray(current_state, dtype=np.float32)
    if current_state.shape != (37,) or not np.isfinite(current_state).all():
        raise ValueError("current state must be finite float32 shape (37,)")
    return (
        np.concatenate([current_latent, current_state[:19]]).astype(np.float32),
        np.concatenate([goal_latent, goal_xyz(np.asarray(goal_state, dtype=np.float32))]).astype(np.float32),
    )


def configure_camera(base) -> dict:
    import mujoco

    model = base._model
    source = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
    target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_pixels")
    if source < 0 or target < 0:
        raise RuntimeError("official front/front_pixels cameras are missing")
    position = model.cam_pos[source].copy()
    original = model.cam_quat[source].copy()
    if not np.allclose(position, CAMERA["position"], atol=1e-12, rtol=0) or float(model.cam_fovy[source]) != 45.0:
        raise RuntimeError("official front camera binding changed")
    rotation = np.empty(4)
    actual = np.empty(4)
    mujoco.mju_axisAngle2Quat(rotation, np.array([1.0, 0.0, 0.0]), np.deg2rad(5.0))
    mujoco.mju_mulQuat(actual, original, rotation)
    model.cam_pos[target] = position
    model.cam_quat[target] = actual
    model.cam_fovy[target] = 45.0
    model.vis.quality.offsamples = 4
    return {**CAMERA, "actual_quaternion": actual.tolist(), "source_quaternion": original.tolist()}


class OnlineEncoder:
    def __init__(self, group: str):
        cfg = GROUPS[group]
        if sha(cfg["wm"]) != cfg["wm_sha256"]:
            raise RuntimeError("world-model checkpoint SHA mismatch")
        self.group, self.mode = group, cfg["mode"]
        sys.path[:0] = [str(ROOT), str(ROOT / "source/lewm_pcd"),
                        str(ROOT / "source/lewm_pcd/third_party/stable-worldmodel")]
        import torch
        from checkpoint_encoder import load_frozen, encode
        import importlib.util
        util_path = ROOT / "source/lewm_pcd/utils.py"
        spec = importlib.util.spec_from_file_location("_frozen_lewm_utils", util_path)
        lewm_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lewm_utils)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the online frozen encoder")
        self.torch = torch
        self.img_preprocess = lewm_utils.get_img_preprocessor("pixels", "pixels", 224)
        self.pcd_preprocess = lewm_utils.get_pcd_preprocessor(center=(.425, 0., .2), scale=4., use_rgb=True)
        self.encode_impl = encode
        self.model = load_frozen(cfg["wm"], device="cuda")
        self.calls = 0

    def encode(self, value: np.ndarray) -> np.ndarray:
        if self.mode == "rgb":
            if value.shape != (224, 224, 3) or value.dtype != np.uint8:
                raise ValueError(f"RGB online frame contract failed: {value.shape} {value.dtype}")
            batch = {"pixels": self.torch.from_numpy(np.ascontiguousarray(value[None])).permute(0, 3, 1, 2)}
            batch = self.img_preprocess(batch)
        else:
            if value.shape != (1200, 6) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"PCD online frame contract failed: {value.shape} {value.dtype}")
            batch = {
                "pcd_xyz": self.torch.from_numpy(np.ascontiguousarray(value[None, :, :3])),
                "pcd_rgb": self.torch.from_numpy(np.clip(np.rint(value[None, :, 3:]), 0, 255).astype(np.uint8)),
            }
            batch = self.pcd_preprocess(batch)
        inputs = batch["pixels" if self.mode == "rgb" else "pcd"].to("cuda")
        out = self.encode_impl(self.model, inputs)
        self.calls += 1
        return validate_latent(out[0].detach().cpu().numpy().astype(np.float32, copy=False), "online")


def load_agent(group: str):
    cfg = GROUPS[group]
    training = json.loads(cfg["training"].read_text())
    acceptance = json.loads(cfg["acceptance"].read_text())
    if acceptance["actual_exit_code"] != 0 or acceptance["actual_current_dim"] != 211 or acceptance["actual_goal_dim"] != 198:
        raise RuntimeError("training acceptance does not match visual contract")
    if acceptance["final_checkpoint"] != str(cfg["final"]):
        raise RuntimeError("final GCIQL checkpoint binding mismatch")
    manifest = json.loads((cfg["final"] / "manifest.json").read_text())
    if manifest["sha256"] if "sha256" in manifest else cfg["manifest_sha256"] != cfg["manifest_sha256"]:
        pass
    if sha(cfg["final"] / "manifest.json") != cfg["manifest_sha256"]:
        raise RuntimeError("final GCIQL manifest SHA mismatch")
    sys.path[:0] = [str(CASE), str(ROOT), str(ROOT / "source/ogbench/impls"), str(ROOT / "source/ogbench")]
    import gciql_boundary as boundary
    directory, _sidecar = boundary.restore_saved(training, Path(training["output"]), 1000000)
    with np.load(cfg["gate"], allow_pickle=False) as saved:
        example = {key: saved[key][:1].copy() for key in ("observations", "actions", "actor_goals")}
    agent = boundary.create_agent(boundary.model.GCIQLAgent, 0, example, boundary.agent_config(training))
    agent = boundary.restore_agent(agent, str(directory), 1000000)
    return agent


def make_env(group: str, seed: int):
    import gymnasium as gym
    sys.path.insert(0, str(ROOT / "source/ogbench"))
    import ogbench  # noqa: F401
    if group == "G06":
        from ogbench.manipspace.envs.cube_env import CubeEnv as OfficialCubeEnv
        # Load the repository's exact CubePCDEnv render hook while binding its
        # parent to the official CubeEnv used by the accepted dataset/runtime.
        package_names = ("stable_worldmodel.envs.ogbench",)
        for name in package_names:
            pkg = sys.modules.get(name) or types.ModuleType(name)
            pkg.__path__ = []
            sys.modules[name] = pkg
        parent_name = "stable_worldmodel.envs.ogbench.cube_env"
        parent = types.ModuleType(parent_name)
        parent.CubeEnv = OfficialCubeEnv
        sys.modules[parent_name] = parent
        cube_pcd_path = ROOT / "source/lewm_pcd/envs/cube_pcd_env.py"
        spec = importlib.util.spec_from_file_location("_native_sr_cube_pcd_env", cube_pcd_path)
        cube_pcd_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cube_pcd_module)
        pcd_wrapper_path = ROOT / "source/lewm_pcd/envs/pcd_wrapper.py"
        spec = importlib.util.spec_from_file_location("_native_sr_pcd_wrapper", pcd_wrapper_path)
        pcd_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pcd_module)
        env = cube_pcd_module.CubePCDEnv(env_type="double", render_mode="rgb_array", width=224, height=224,
                                         ob_type="states", visualize_info=False, pixel_transparent_arm=False)
        base_env = env
        env = pcd_module.PointCloudWrapper(base_env, num_points=1200, cameras=("front_pixels",),
            foreground_only=True, crop_box=((0.1, -0.5, -0.02), (0.75, 0.5, 0.6)),
            voxel_size=0.005, cube_fraction=0.25, keep_rgb=False)
    else:
        base_env = gym.make("cube-double-v0", render_mode="rgb_array", width=224, height=224,
                            ob_type="states", visualize_info=False, pixel_transparent_arm=False)
        env = base_env
    base = base_env.unwrapped
    base.reset(seed=0)
    camera = configure_camera(base)
    if group == "G05":
        original_render = base.render
        def render(*args, camera=None, **kwargs):
            if camera is None and getattr(base, "_render_goal", False):
                camera = "front_pixels"
            return original_render(*args, camera=camera, **kwargs)
        base.render = render
        return env, camera, None
    proof = {"callback_calls": 0, "physics_unchanged": True, "env_rng_unchanged": True}
    original_cloud = env._point_cloud
    def guarded(*args, **kwargs):
        before = {k: np.array(getattr(base._data, k), copy=True) for k in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat")}
        rng_before = copy.deepcopy(base.np_random.bit_generator.state)
        value = original_cloud(*args, **kwargs)
        proof["callback_calls"] += 1
        for key, expected in before.items():
            if not np.array_equal(getattr(base._data, key), expected):
                proof["physics_unchanged"] = False
        if base.np_random.bit_generator.state != rng_before:
            proof["env_rng_unchanged"] = False
        if not proof["physics_unchanged"] or not proof["env_rng_unchanged"]:
            raise RuntimeError("goal PCD capture changed physics or environment RNG")
        return value
    env._point_cloud = guarded
    return env, camera, proof


def run_episode(group: str, agent, encoder: OnlineEncoder, task: int, index: int, output: Path):
    import jax
    env, camera, proof = make_env(group, seed_for(task, index))
    seed = seed_for(task, index)
    started = time.time()
    try:
        observation, info = env.reset(seed=seed, options={"task_id": task, "render_goal": True})
        goal_state = np.asarray(info["goal"], dtype=np.float32)
        if group == "G05":
            goal_raw = np.asarray(info.get("goal_rendered"))
            if goal_raw.shape != (224, 224, 3) or goal_raw.dtype != np.uint8:
                raise RuntimeError("RGB goal render was not the real 224x224 uint8 frame")
        else:
            goal_raw = np.concatenate([np.asarray(info["goal_pcd_xyz"], np.float32),
                                       np.asarray(info["goal_pcd_rgb"], np.float32)], axis=1)
        goal_latent = encoder.encode(goal_raw)
        actions, rewards, successes = [], [], []
        rng = jax.random.PRNGKey(seed)
        done = False
        current_info = info
        while not done and len(actions) < 500:
            if group == "G05":
                current_raw = env.unwrapped.render(camera="front_pixels")
            else:
                current_raw = np.concatenate([np.asarray(current_info["pcd_xyz"], np.float32),
                                              np.asarray(current_info["pcd_rgb"], np.float32)], axis=1)
            current_latent = encoder.encode(current_raw)
            current, goal = make_policy_inputs(current_latent, goal_latent, np.asarray(observation, np.float32), goal_state)
            rng, key = jax.random.split(rng)
            action = np.asarray(agent.sample_actions(observations=current[None], goals=goal[None], seed=key, temperature=0.0))[0]
            if action.shape != (5,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.000001):
                raise RuntimeError("GCIQL action contract failed")
            observation, reward, terminated, truncated, step_info = env.step(action)
            current_info = step_info
            done = bool(terminated or truncated)
            actions.append(action.astype(np.float32)); rewards.append(float(reward)); successes.append(bool(step_info.get("success", False)))
        if not done or not actions:
            raise RuntimeError("episode did not terminate within 500 steps")
        result = {
            "actual_exit_code": 0, "group_id": group, "task_id": task, "episode_index": index,
            "seed": seed, "steps": len(actions), "success": int(successes[-1]),
            "online_encoder_calls": encoder.calls, "goal_online_encoder_calls": 1,
            "current_goal_contract": {"current_dim": 211, "goal_dim": 198, "synthetic_zero_padding": False},
            "camera": camera, "goal_capture_proof": proof, "elapsed_seconds": time.time() - started,
        }
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output / "steps.npz", actions=np.stack(actions), rewards=np.asarray(rewards), successes=np.asarray(successes))
        atomic_json(output / "result.json", result)
        (output / "attempt_exit_code").write_text("0\n")
        return result
    except BaseException as exc:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "failure.json", {"actual_exit_code": 1, "error": repr(exc)})
        (output / "attempt_exit_code").write_text("1\n")
        raise
    finally:
        try:
            env.close()
        except Exception:
            pass


def main(group: str, mode: str, output: Path):
    if group not in GROUPS:
        raise ValueError(group)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "status.json", {"stage": "loading", "group": group, "mode": mode, "completed_episodes": 0, "started": time.time()})
    encoder = agent = None
    code = 1
    try:
        agent = load_agent(group)
        encoder = OnlineEncoder(group)
        if mode == "diagnostic":
            tasks = [(1, 0)]
        else:
            tasks = [(task, index) for task in range(1, 6) for index in range(50)]
        results = []
        for task, index in tasks:
            item = run_episode(group, agent, encoder, task, index, output / "episodes" / f"task{task}" / f"episode_{index:03d}")
            results.append(item)
            atomic_json(output / "status.json", {"stage": "evaluating", "group": group, "mode": mode,
                "completed_episodes": len(results), "successes": sum(x["success"] for x in results), "updated": time.time()})
        expected = 1 if mode == "diagnostic" else 250
        if len(results) != expected or len({x["seed"] for x in results}) != expected:
            raise RuntimeError("episode/seed count mismatch")
        report = {"actual_exit_code": 0, "group_id": group, "mode": mode, "episodes": expected,
                  "successes": sum(x["success"] for x in results),
                  "success_rate": sum(x["success"] for x in results) / expected,
                  "unique_seeds": expected, "max_episode_steps": 500,
                  "temperature": 0.0, "gaussian_noise": None, "extra_policy_warmup": 0,
                  "online_encoder": True, "offline_embedding_cache_used_for_policy": False,
                  "current_dim": 211, "goal_dim": 198, "synthetic_zero_padding": False,
                  "completed": time.time()}
        atomic_json(output / "report.json", report)
        atomic_json(output / "status.json", {"stage": "complete", **report})
        code = 0
    except BaseException as exc:
        atomic_json(output / "failure.json", {"actual_exit_code": 1, "error": repr(exc)})
        atomic_json(output / "status.json", {"stage": "failed", "group": group, "mode": mode, "error": repr(exc), "updated": time.time()})
        raise
    finally:
        (output / "actual_exit_code").write_text(f"{code}\n")
        if encoder is not None:
            del encoder
        if agent is not None:
            del agent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    parser.add_argument("--mode", choices=("diagnostic", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.group, args.mode, args.output)
