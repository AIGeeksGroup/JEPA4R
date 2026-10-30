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
import inspect
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
for candidate in Path(__file__).resolve().parents:
    if (candidate / 'checkpoint_encoder.py').is_file():
        ROOT = candidate
        break
CASE = ROOT / "cube23_distance_gciql_v1"
EXT = CASE / "five_group_extension_v1"
G04_REFERENCE = CASE / 'state_priority_v1/native_sr_v1/configs/G04.json'

GROUPS = {
    "G05": {
        "cube_count": 2,
        "state_dim": 37,
        "current_dim": 211,
        "goal_dim": 198,
        "max_episode_steps": 500,
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
        "cube_count": 2,
        "state_dim": 37,
        "current_dim": 211,
        "goal_dim": 198,
        "max_episode_steps": 500,
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


def group_contract(group: str) -> dict:
    cfg = GROUPS[group]
    cubes = cfg.get("cube_count", 2)
    result = {
        "cube_count": cubes,
        "state_dim": cfg.get("state_dim", 19 + 9 * cubes),
        "current_dim": cfg.get("current_dim", 211),
        "goal_dim": cfg.get("goal_dim", 192 + 3 * cubes),
        "max_episode_steps": cfg.get("max_episode_steps", 500),
        "mode": cfg["mode"],
    }
    if cfg.get("record_capture_receipts", False):
        result["record_capture_receipts"] = True
    if cfg.get("pcd_crop_box") is not None:
        result["pcd_crop_box"] = cfg["pcd_crop_box"]
    return result


def camera_for_group(group):
    if group not in GROUPS:
        raise ValueError(group)
    return dict(CAMERA, **({'geometry_offsamples': 0}
                           if GROUPS[group].get('mode') == 'pcd' else {}))


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
    if state.ndim != 1 or state.size < 28 or (state.size - 19) % 9 or state.dtype != np.float32 or not np.isfinite(state).all():
        raise ValueError("goal state must be finite float32 with the official 19+9N layout")
    cubes = (state.size - 19) // 9
    return np.asarray([state[index] for cube in range(cubes)
                       for index in range(19 + 9 * cube, 22 + 9 * cube)], dtype=np.float32)


def make_policy_inputs(current_latent: np.ndarray, goal_latent: np.ndarray,
                       current_state: np.ndarray, goal_state: np.ndarray):
    current_latent = validate_latent(current_latent, "current")
    goal_latent = validate_latent(goal_latent, "goal")
    current_state = np.asarray(current_state, dtype=np.float32)
    if current_state.ndim != 1 or current_state.size < 28 or (current_state.size - 19) % 9 or not np.isfinite(current_state).all():
        raise ValueError("current state must use the official 19+9N layout")
    return (
        np.concatenate([current_latent, current_state[:19]]).astype(np.float32),
        np.concatenate([goal_latent, goal_xyz(np.asarray(goal_state, dtype=np.float32))]).astype(np.float32),
    )


def pcd_components(value: np.ndarray):
    value = np.asarray(value)
    if value.shape != (1200, 6) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("PCD must be finite float32 XYZRGB shape (1200,6)")
    return value[:, :3].copy(), np.clip(np.rint(value[:, 3:]), 0, 255).astype(np.uint8)


def configure_camera(base, group: str) -> dict:
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
    result = {**CAMERA, "actual_quaternion": actual.tolist(), "source_quaternion": original.tolist()}
    result.update(camera_for_group(group))
    return result


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
            xyz, rgb = pcd_components(value)
            batch = {"pcd_xyz": self.torch.from_numpy(xyz[None]), "pcd_rgb": self.torch.from_numpy(rgb[None])}
            batch = self.pcd_preprocess(batch)
        inputs = batch["pixels" if self.mode == "rgb" else "pcd"].to("cuda")
        out = self.encode_impl(self.model, inputs)
        self.calls += 1
        return validate_latent(out[0].detach().cpu().numpy().astype(np.float32, copy=False), "online")


def load_agent(group: str):
    cfg = GROUPS[group]
    contract = group_contract(group)
    training = json.loads(cfg["training"].read_text())
    acceptance = json.loads(cfg["acceptance"].read_text())
    for path_key, hash_key in (("training", "training_sha256"),
                               ("acceptance", "acceptance_sha256"),
                               ("gate", "gate_sha256")):
        if hash_key in cfg and sha(cfg[path_key]) != cfg[hash_key]:
            raise RuntimeError(f"{path_key} SHA mismatch")
    if (acceptance["actual_exit_code"] != 0 or
            acceptance["actual_current_dim"] != contract["current_dim"] or
            acceptance["actual_goal_dim"] != contract["goal_dim"]):
        raise RuntimeError("training acceptance does not match visual contract")
    if acceptance["final_checkpoint"] != str(cfg["final"]):
        raise RuntimeError("final GCIQL checkpoint binding mismatch")
    if sha(cfg["final"] / "manifest.json") != cfg["manifest_sha256"]:
        raise RuntimeError("final GCIQL manifest SHA mismatch")
    sys.path[:0] = [str(CASE), str(ROOT), str(ROOT / "source/ogbench/impls"), str(ROOT / "source/ogbench")]
    import gciql_boundary as boundary
    directory, _sidecar = boundary.restore_saved(training, Path(training["output"]), 1000000)
    with np.load(cfg["gate"], allow_pickle=False) as saved:
        example = {key: saved[key][:1].copy() for key in ("observations", "actions", "actor_goals")}
    agent = boundary.create_agent(boundary.model.GCIQLAgent, 0, example, boundary.agent_config(training))
    agent = boundary.restore_agent(agent, str(directory), 1000000)
    if boundary.jax.default_backend() != 'cpu' or boundary.fingerprint(agent) != acceptance['state']:
        raise RuntimeError('GCIQL CPU/backend or exact checkpoint state mismatch')
    return agent


def make_env(group: str, seed: int):
    import gymnasium as gym
    contract = group_contract(group)
    sys.path.insert(0, str(ROOT / "source/ogbench"))
    import ogbench  # noqa: F401
    if contract["mode"] == "pcd":
        from ogbench.manipspace.envs.cube_env import CubeEnv as OfficialCubeEnv
        # Reuse the repository's goal-render callback with the official CubeEnv base.
        shim_package = types.ModuleType('stable_worldmodel.envs.ogbench')
        shim_package.__path__ = []
        shim_parent = types.ModuleType('stable_worldmodel.envs.ogbench.cube_env')
        shim_parent.CubeEnv = OfficialCubeEnv
        sys.modules['stable_worldmodel.envs.ogbench'] = shim_package
        sys.modules['stable_worldmodel.envs.ogbench.cube_env'] = shim_parent
        path = ROOT / 'source/lewm_pcd/envs/cube_pcd_env.py'
        spec = importlib.util.spec_from_file_location('_sr_cube_pcd', path)
        pcd_env_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(pcd_env_module)
        env_type = {2: 'double', 3: 'triple'}[contract["cube_count"]]
        base = pcd_env_module.CubePCDEnv(env_type=env_type, render_mode='rgb_array', width=224, height=224,
            ob_type='states', visualize_info=False, pixel_transparent_arm=False)
        env = gym.wrappers.TimeLimit(base, max_episode_steps=contract["max_episode_steps"])
    else:
        play_env = {2: "cube-double-play-v0", 3: "cube-triple-play-v0"}[contract["cube_count"]]
        env = ogbench.make_env_and_datasets(play_env, env_only=True,
            render_mode="rgb_array", width=224, height=224, ob_type="states",
            visualize_info=False, pixel_transparent_arm=False)
        base = env.unwrapped
        env = gym.wrappers.TimeLimit(env, max_episode_steps=contract["max_episode_steps"])
    base.reset(seed=0)
    camera = configure_camera(base, group)
    if contract["mode"] == 'rgb':
        original_render = base.render
        def render(*args, camera=None, **kwargs):
            if camera is None and getattr(base, '_render_goal', False):
                camera = 'front_pixels'
            return original_render(*args, camera=camera, **kwargs)
        base.render = render
    proof = {'callback_calls': 0, 'goal_capture_calls': 0, 'current_capture_calls': 0,
             'physics_unchanged': True, 'env_rng_unchanged': True,
             'global_rng_unchanged': True, 'captures': [], 'RGB_offsamples': 4}
    if contract["mode"] == "pcd":
        proof["geometry_offsamples"] = 0
    return env, camera, proof


class OnlineVisualEnv:
    """Thin reset/step wrapper: force the true goal render and retain current visual info."""
    def __init__(self, env, group: str, encoder: OnlineEncoder, seed: int):
        self.env, self.group, self.encoder, self.seed = env, group, encoder, seed
        self.goal_latent = None
        self.goal_raw_sha256 = None
        self.encoder_calls_at_reset = 0
        self.last_info = None
        self.frame = None
        self.capture_proof = None

    def capture(self, rgb, phase):
        contract = group_contract(self.group)
        base = self.env.unwrapped
        keys = ('qpos', 'qvel', 'ctrl', 'act', 'mocap_pos', 'mocap_quat', 'qacc_warmstart', 'eq_active')
        before = {k: np.asarray(getattr(base._data, k)).copy() for k in keys if hasattr(base._data, k)}
        before_time = float(base._data.time)
        rng = copy.deepcopy(base.np_random.bit_generator.state)
        global_rng = np.random.get_state()
        cloud = self.frame.capture(base, base._data, rgb, phase)
        if before_time != base._data.time or any(not np.array_equal(v, getattr(base._data, k)) for k,v in before.items()):
            raise RuntimeError('PCD capture changed physics')
        if rng != base.np_random.bit_generator.state:
            raise RuntimeError('PCD capture changed environment RNG')
        after_rng = np.random.get_state()
        unchanged_global_rng = (global_rng[0] == after_rng[0] and
                                np.array_equal(global_rng[1], after_rng[1]) and global_rng[2:] == after_rng[2:])
        self.capture_proof["global_rng_unchanged"] &= unchanged_global_rng
        if not unchanged_global_rng:
            raise RuntimeError('PCD capture changed global RNG')
        self.capture_proof['callback_calls'] += 1
        self.capture_proof[f'{phase}_capture_calls'] += 1
        if contract.get("record_capture_receipts", False):
            receipt = dict(self.frame.last_receipts[phase])
            if "pcd_crop_box" in GROUPS[self.group]:
                receipt["training_crop_box"] = [list(row) for row in GROUPS[self.group]["pcd_crop_box"]]
            self.capture_proof["captures"].append(receipt)
        return cloud

    def reset(self, *, options=None, **kwargs):
        options = dict(options or {})
        options["render_goal"] = True
        if group_contract(self.group)["mode"] == 'pcd':
            sys.path.insert(0, str(ROOT/'external'))
            from native_pcd_frame import NativePCDFrame
            path = ROOT/'source/lewm_pcd/envs/pcd_wrapper.py'
            spec = importlib.util.spec_from_file_location('_sr_pcd_main', path)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            crop_box = GROUPS[self.group].get("pcd_crop_box")
            core_factory = (module.PointCloudWrapper if crop_box is None else
                            __import__("functools").partial(module.PointCloudWrapper, crop_box=crop_box))
            self.frame = NativePCDFrame(path, ROOT/'external/pcd_geometry.py', self.seed,
                core_factory=core_factory)
            base = self.env.unwrapped
            base._pcd_goal_capture = None
            base._pcd_goal_capture_callback = lambda: self.capture(base.render(camera='front_pixels'), 'goal')
        observation, info = self.env.reset(seed=self.seed, options=options)
        if int(self.env.unwrapped.np_random_seed) != self.seed:
            raise RuntimeError('environment did not apply the requested episode seed')
        self.last_info = info
        if group_contract(self.group)["mode"] == "rgb":
            goal_raw = np.asarray(info["goal_rendered"])
        else:
            goal_raw = self.env.unwrapped._pcd_goal_capture
            if goal_raw is None:
                raise RuntimeError('CubePCDEnv did not capture the official goal scene')
        self.goal_latent = self.encoder.encode(goal_raw)
        self.goal_raw_sha256 = hashlib.sha256(np.ascontiguousarray(goal_raw).tobytes()).hexdigest()
        self.encoder_calls_at_reset = self.encoder.calls
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.last_info = info
        return observation, reward, terminated, truncated, info

    def current_raw(self):
        if group_contract(self.group)["mode"] == "rgb":
            return self.env.unwrapped.render(camera="front_pixels")
        rgb = self.env.unwrapped.render(camera='front_pixels')
        return self.capture(rgb, 'current')

    def close(self):
        if self.frame is not None:
            self.frame.close()
        self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


class OnlineGCIQLPolicy:
    def __init__(self, agent, env: OnlineVisualEnv):
        self.agent, self.env = agent, env
        self.current_latents = []
        self.current_sha256 = []
        self.input_contracts = []

    def sample_actions(self, observations, goals=None, seed=None, temperature=0.0):
        observation = np.asarray(observations, dtype=np.float32)
        goal = np.asarray(goals, dtype=np.float32)
        if observation.shape == (1, group_contract(self.env.group)["state_dim"]):
            observation = observation[0]
        if goal.shape == (1, group_contract(self.env.group)["state_dim"]):
            goal = goal[0]
        raw = self.env.current_raw()
        latent = self.env.encoder.encode(raw)
        current, goal_input = make_policy_inputs(latent, self.env.goal_latent, observation, goal)
        self.current_latents.append(latent)
        self.current_sha256.append(hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest())
        self.input_contracts.append({"current_dim": int(current.shape[0]), "goal_dim": int(goal_input.shape[0]),
                                     "current_finite": bool(np.isfinite(current).all()),
                                     "goal_finite": bool(np.isfinite(goal_input).all())})
        return self.agent.sample_actions(observations=current, goals=goal_input,
                                         seed=seed, temperature=temperature)


def run_episode(group: str, agent, encoder: OnlineEncoder, task: int, index: int, output: Path):
    import jax
    import random
    import inspect
    from utils.evaluation import evaluate
    ref = json.loads(G04_REFERENCE.read_text())
    eval_path = Path(inspect.getfile(evaluate)).resolve()
    expected_eval = ref['source_pins'][str(ROOT/'source/ogbench/impls/utils/evaluation.py')]
    if eval_path != (ROOT/'source/ogbench/impls/utils/evaluation.py').resolve() or sha(eval_path) != expected_eval:
        raise RuntimeError('official G04 evaluator source binding changed')
    env, camera, proof = make_env(group, seed_for(task, index))
    seed = seed_for(task, index)
    start_encoder_calls = encoder.calls
    base = env.unwrapped
    contract = group_contract(group)
    policy_env = None
    started = time.time()
    try:
        if (base._num_cubes != contract["cube_count"] or len(base.task_infos) != 5 or
                base._permute_blocks is not True or base._use_oracle_rep is not False):
            raise RuntimeError('official task/environment contract changed')
        random.seed(seed)
        np.random.seed(seed)
        policy_env = OnlineVisualEnv(env, group, encoder, seed)
        policy_env.capture_proof = proof
        policy = OnlineGCIQLPolicy(agent, policy_env)
        train_config = json.loads(GROUPS[group]["training"].read_text())
        stats, trajectories, videos = evaluate(policy, policy_env, task_id=task,
            config=train_config["agent_config"], num_eval_episodes=1, num_video_episodes=0,
            eval_temperature=0, eval_gaussian=None)
        if len(trajectories) != 1 or videos:
            raise RuntimeError("official evaluator returned unexpected episode/video counts")
        traj = trajectories[0]
        actions = np.asarray(traj["action"], dtype=np.float32)
        rewards = np.asarray(traj["reward"], dtype=np.float32)
        successes = np.asarray([bool(row.get("success", False)) for row in traj["info"]], dtype=bool)
        contract = group_contract(group)
        if not 1 <= len(actions) <= contract["max_episode_steps"] or not np.isfinite(actions).all() or not np.isfinite(rewards).all():
            raise RuntimeError("official evaluator episode output failed finite/length checks")
        if actions.shape != (len(actions), 5) or np.any(np.abs(actions) > 1.000001):
            raise RuntimeError("GCIQL action contract failed")
        if not all(x["current_dim"] == contract["current_dim"] and x["goal_dim"] == contract["goal_dim"] and x["current_finite"] and x["goal_finite"]
                   for x in policy.input_contracts):
            raise RuntimeError("visual policy input contract failed during rollout")
        result = {
            "actual_exit_code": 0, "group_id": group, "task_id": task, "episode_index": index,
            "seed": seed, "steps": len(actions), "success": int(stats["success"]),
            "online_encoder_calls": encoder.calls - start_encoder_calls, "goal_online_encoder_calls": 1,
            "current_goal_contract": {"current_dim": contract["current_dim"], "goal_dim": contract["goal_dim"], "synthetic_zero_padding": False},
            "camera": camera, "goal_capture_proof": proof, "goal_raw_sha256": policy_env.goal_raw_sha256,
            "policy_input_contracts": policy.input_contracts,
            "current_raw_sha256": policy.current_sha256,
            "original_evaluate_reused": True, "extra_policy_warmup": 0,
        "evaluator_sha256": expected_eval,
            "temperature": 0.0, "gaussian_noise": None, "elapsed_seconds": time.time() - started,
        }
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output / "steps.npz", actions=actions, rewards=rewards, successes=successes)
        atomic_json(output / "result.json", result)
        (output / "attempt_exit_code").write_text("0\n")
        return result
    except BaseException as exc:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "failure.json", {"actual_exit_code": 1, "error": repr(exc)})
        (output / "attempt_exit_code").write_text("1\n")
        raise
    finally:
        try:
            if policy_env is not None:
                policy_env.close()
            else:
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
                  "unique_seeds": expected, "max_episode_steps": group_contract(group)["max_episode_steps"],
                  "temperature": 0.0, "gaussian_noise": None, "extra_policy_warmup": 0,
                  "online_encoder": True, "offline_embedding_cache_used_for_policy": False,
                  "current_dim": group_contract(group)["current_dim"], "goal_dim": group_contract(group)["goal_dim"], "synthetic_zero_padding": False,
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
