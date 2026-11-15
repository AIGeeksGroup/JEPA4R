"""Behavior cloning on the RH56 twin with CortexBench's BC loop (eai-vc's train_loop.bc_pvr_train_loop, with the
patch of third_party/eai-vc-relocate), the world model's encoder frozen as the embedding.

The encoder (wm_embeddings.load_lewm for route rgb, load_jepa4r for route pcd) embeds a history of 3 frames, fused with
the 13-D proprioception (arm 7 + active hand 6) into the 589-D input of the BC head (BatchNormMLP 256-256-256, ReLU,
MSE, Adam lr 1e-3, batch 256). The 100 demonstrations of the official initialisations (bc_demos_manifest.json),
100 epochs, 25 evaluation episodes every 5 epochs and at the last epoch on their own reset seeds (65000000 + 1000 x the
training seed + episode), the opposite camera. Metrics go to metrics.jsonl, the policies to policy/iterations.
usage: python -m jepa4r.envs.twins.rh56.bc_bridge --route rgb|pcd --checkpoint <dir> --demos-manifest <file> --out <dir>
       [--train-seed 100]"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np

from jepa4r.envs.twins.rh56._runtime import temporary_bindings

ENV_ID = "rh56-relocate-v0"
RUN_SPEC = dict(epochs=100, num_demos=100, eval_num_traj=25, eval_frequency=5, save_frequency=10, batch_size=256)


def evaluation_seed_base(train_seed):
    return 65000000 + int(train_seed) * 1000


class EvaluationSeedMap:
    """Map the official sampler's seed labels (train seed + episode) to the evaluation reset seeds."""

    def __init__(self, train_seed, physical_base, count):
        self.train_seed, self.physical_base, self.count = int(train_seed), int(physical_base), int(count)

    def __call__(self, requested):
        offset = int(requested) - self.train_seed
        if not 0 <= offset < self.count:
            raise ValueError(f"sampler seed {requested} is outside the evaluation range")
        return self.physical_base + offset


def prepare_demo(arrays, route):
    """One collected episode (T+1 states, T actions) in the official demonstration layout (states before each
    action); the images are RGB frames or point clouds (xyz and colour, 1200 x 6) by route."""
    observations = np.asarray(arrays["observations"])
    proprio = np.asarray(arrays["proprio"])
    raw_state = np.asarray(arrays["initial_state_json"]).item()
    state = json.loads(raw_state.decode() if isinstance(raw_state, bytes) else str(raw_state))
    if route == "rgb":
        images = np.asarray(arrays["pixels"])[:-1].copy()
    else:
        xyz, rgb = np.asarray(arrays["xyz"]), np.asarray(arrays["rgb"])
        images = np.concatenate((xyz[:-1], rgb[:-1]), axis=-1).astype(np.float32)
    return dict(
        images=images,
        observations=observations[:-1].copy(),
        actions=np.asarray(arrays["actions"]).copy(),
        rewards=np.asarray(arrays["rewards"]).copy(),
        env_infos={"proprio": proprio[:-1].copy()},
        init_state_dict=state,
    )


class RH56LegacyCore:
    """The old Gym API over NativeBridge (rendering and physics stay there).

    It has no reset_model on purpose: MJRL would call it on the unwrapped env and bypass the frozen visual observation
    wrappers.
    """

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 100}

    def __init__(self, seed, native_factory=None, route="rgb", seed_map=None):
        if native_factory is None:
            from jepa4r.envs.twins.rh56.native import NativeBridge

            native_factory = NativeBridge
        self.seed_map = seed_map
        physical_seed = int(seed) if seed_map is None else seed_map(seed)
        self.native = native_factory(seed=physical_seed, render=True)
        self.route = route
        self._next_seed = physical_seed
        self._last_obs = None
        self.on_state_restored = None
        self.spec = SimpleNamespace(id=ENV_ID, max_episode_steps=200)

    @property
    def unwrapped(self):
        return self

    def seed(self, seed=None):
        if seed is not None:
            self._next_seed = int(seed) if self.seed_map is None else self.seed_map(seed)
        return [self._next_seed]

    def reset(self, seed=None, **kwargs):
        self.seed(seed)
        active_seed = self._next_seed
        self._next_seed += 1
        self._last_obs, _ = self.native.reset(seed=active_seed)
        return self._last_obs.copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        executed = np.clip(action, -1.0, 1.0)
        self._last_obs, reward, terminated, truncated, info = self.native.step(executed)
        info = dict(info)
        info["proprio"] = self.get_proprio()
        info["TimeLimit.truncated"] = bool(truncated and not terminated)
        return self._last_obs.copy(), float(reward), bool(terminated or truncated), info

    def get_obs(self):
        if self._last_obs is None:
            return np.asarray(self.native.env._obs()).copy()
        return self._last_obs.copy()

    def get_proprio(self):
        return np.asarray(self.native.proprio(), dtype=np.float32).copy()

    def get_env_infos(self):
        # MJRL prefers this result over the step info when it is not empty; empty keeps the step's success / overload.
        return {}

    def get_env_state(self):
        return self.native.get_state()

    def set_env_state(self, value):
        if set(value) == {"seed"}:
            # a demonstration's state is its reset seed
            physical_seed = int(value["seed"])
            self._last_obs, _ = self.native.reset(seed=physical_seed)
            self._next_seed = physical_seed + 1
        else:
            self.native.set_state(value)
            self._last_obs = np.asarray(self.native.env._obs()).copy()
        if self.on_state_restored is not None:
            self.on_state_restored()

    def get_image(self):
        capture = self.native.capture(modality=self.route)
        if self.route == "rgb":
            return capture["pixels"]
        return np.concatenate((capture["pcd_xyz"], capture["pcd_rgb"].astype(np.float32)), axis=-1)

    def render(self, mode="rgb_array", **kwargs):
        return self.native.capture(modality="rgb")["pixels"]

    def close(self):
        if hasattr(self.native, "close"):
            self.native.close()

    @staticmethod
    def evaluate_success(paths):
        success = []
        for path in paths:
            info = path["env_infos"]
            achieved = np.asarray(info["success"], dtype=bool)
            overload = np.asarray(info.get("overload", False), dtype=bool)
            success.append(bool(achieved[-1]) and not bool(overload.any()))
        return 100.0 * float(np.mean(success))


def load_demos(manifest, route):
    """The demonstrations of bc_demos_manifest.json (its selected episodes), in its order."""
    path = Path(manifest).resolve()
    demos = []
    for record in json.loads(path.read_text(encoding="utf-8"))["selected"]:
        source = Path(record["npz"])
        with np.load(source if source.is_absolute() else path.parent / source, allow_pickle=False) as arrays:
            demos.append(prepare_demo(arrays, route))
    return demos


def install_bc_runtime(official, gym_wrapper, *, load_embedding, env_constructor, logger):
    """The RH56 encoder, environment, proprioception and metrics logger in the official loop, for the block."""
    return temporary_bindings(
        (gym_wrapper, "load_pretrained_model", load_embedding),
        (gym_wrapper, "get_proprioception", lambda env, suite: env.unwrapped.get_proprio()),
        (official, "load_pretrained_model", load_embedding),
        (official, "env_constructor", env_constructor),
        (official, "setup_wandb", lambda config: logger),
    )


def run_bc(args):
    """Prepare the demonstrations, the encoder and the environment, then run the official training loop."""
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    eval_seed_base = evaluation_seed_base(args.train_seed)
    original_cwd = Path.cwd()
    environments = []
    logger = None
    try:
        package, eai = Path(args.package_root).resolve(), Path(args.eai_root).resolve()
        vi = eai / "cortexbench/mujoco_vc/visual_imitation"
        for path in (
            package.parent,
            package / "policy/bc",
            eai,
            eai / "vc_models/src",
            eai / "cortexbench/mujoco_vc/src",
            vi,
        ):
            sys.path.insert(0, str(path))
        import torch
        import gym
        from PIL import Image
        from mjrl.utils.gym_env import GymEnv
        import mujoco_vc.gym_wrapper as gw
        import wm_embeddings

        official = importlib.import_module("train_loop")
        torch.set_num_threads(1)
        demos = out / "demos"
        demos.mkdir()
        with (demos / f"{ENV_ID}.pickle").open("xb") as stream:
            pickle.dump(load_demos(args.demos_manifest, args.route), stream, protocol=4)

        def load_embedding(embedding_name, *unused_args, **unused_kwargs):
            if args.route == "rgb":
                model, dimension, transform, metadata = wm_embeddings.load_lewm(
                    args.checkpoint, epoch=args.epoch, use_projector=True, seed_global_rng=True
                )

                def final_transform(frame):
                    return transform(Image.fromarray(np.asarray(frame, dtype=np.uint8))).unsqueeze(0)
            else:
                import yaml

                cfg = yaml.safe_load((Path(args.checkpoint) / "config.yaml").read_text())
                pre = cfg["data"]["preprocessor"]
                model, dimension, transform, metadata = wm_embeddings.load_jepa4r(
                    args.checkpoint, epoch=args.epoch, center=pre["center"], scale=pre["scale"], use_rgb=pre["use_rgb"]
                )

                def final_transform(cloud):
                    return transform(cloud).unsqueeze(0)

            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            return model, dimension, final_transform, metadata

        class LegacyEnv(RH56LegacyCore, gym.Env):
            def __init__(self, seed):
                from jepa4r.envs.twins.rh56.camera_preview import OppositeCameraBridge

                RH56LegacyCore.__init__(
                    self,
                    seed,
                    native_factory=OppositeCameraBridge,
                    route=args.route,
                    seed_map=EvaluationSeedMap(args.train_seed, eval_seed_base, RUN_SPEC["eval_num_traj"]),
                )
                self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
                self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(56,), dtype=np.float32)

        class PixelWrapper(gw.MuJoCoPixelObsWrapper):
            def get_image(self):
                return self.unwrapped.get_image()

        def env_constructor(**kwargs):
            core = LegacyEnv(kwargs["seed"])
            pixel = PixelWrapper(core, 256, 256, kwargs.get("camera_name", "vil_camera"))
            frozen = gw.FrozenEmbeddingWrapper(
                pixel,
                embedding_name=kwargs["embedding_name"],
                suite="adroit",
                history_window=3,
                fuse_embeddings=kwargs["fuse_embeddings"],
                device=args.device,
                seed=kwargs["seed"],
                add_proprio=True,
            )
            core.on_state_restored = lambda: frozen.embedding_buffer.clear()
            environment = GymEnv(frozen)
            environments.append(environment)
            return environment

        class JsonLogger:
            def __init__(self):
                self.stream = (out / "metrics.jsonl").open("x")

            def log(self, data=None, step=None, **kwargs):
                record = {key: float(value) for key, value in (data or {}).items()}
                if step is not None:
                    record["_step"] = int(step)
                self.stream.write(json.dumps(record) + "\n")
                self.stream.flush()

        logger = JsonLogger()
        config = dict(
            seed=args.train_seed,
            num_demos=RUN_SPEC["num_demos"],
            device=args.device,
            data_dir=str(demos),
            epochs=RUN_SPEC["epochs"],
            eval_frequency=RUN_SPEC["eval_frequency"],
            save_frequency=RUN_SPEC["save_frequency"],
            eval_num_traj=RUN_SPEC["eval_num_traj"],
            num_cpu=1,
            job_name="policy",
            algorithm="BC",
            pixel_based=True,
            env_kwargs=dict(
                env_name=ENV_ID,
                suite="adroit",
                device=args.device,
                image_width=256,
                image_height=256,
                camera_name="vil_camera",
                embedding_name="rh56_" + args.route,
                pixel_based=True,
                render_gpu_id=0,
                seed=args.train_seed,
                history_window=3,
                add_proprio=True,
                proprio_key="proprio",
            ),
            bc_kwargs=dict(
                hidden_sizes="(256, 256, 256)",
                nonlinearity="relu",
                loss_type="MSE",
                batch_size=RUN_SPEC["batch_size"],
                lr=0.001,
                dropout=0.0,
            ),
        )
        (out / "job_config.json").write_text(json.dumps(config, indent=2))
        os.chdir(out)
        with install_bc_runtime(
            official, gw, load_embedding=load_embedding, env_constructor=env_constructor, logger=logger
        ):
            official.bc_pvr_train_loop(config)
        return 0
    finally:
        os.chdir(original_cwd)
        if logger is not None and not logger.stream.closed:
            logger.stream.close()
        for env in environments:
            env.env.unwrapped.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--route", choices=("rgb", "pcd"), required=True)
    parser.add_argument("--checkpoint", required=True, help="the world model's checkpoint directory")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--demos-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--package-root",
        dest="package_root",
        default=str(Path(__file__).resolve().parents[3]),
        help="the jepa4r package root; defaults to this checkout",
    )
    parser.add_argument(
        "--eai-root",
        "--eai-vc-root",
        dest="eai_root",
        default=os.environ.get(
            "CORTEXBENCH_EAI_VC_ROOT", str(Path(__file__).resolve().parents[4] / "third_party/eai-vc")
        ),
        help="eai-vc checkout (with the patch of third_party/eai-vc-relocate); or CORTEXBENCH_EAI_VC_ROOT",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--train-seed", type=int, default=100)
    return run_bc(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
