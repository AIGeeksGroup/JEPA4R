"""The policy part of the Cube workflow, for the six encoders: encode the official play data with the frozen encoder,
train GCIQL on the cached latents, evaluate the policy online (the paper protocol, supp. sec:supp-policy: 1e6 updates
of batch 1024, five tasks x 50 episodes). Each stage runs in a fresh process:
python -m jepa4r.policy.gciql.workflow_downstream --stage <encode|gciql|evaluate> --config <run>/control/downstream.json
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import copy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

STAGES = ("encode", "gciql", "evaluate")
MODELS = {
    "lewm": ("jepa_wm", 192, "rgb"),
    "jepa4r": ("jepa_wm", 192, "pcd"),
    "pldm": ("pldm", 512, "rgb"),
    "lejepa": ("lejepa", 512, "rgb"),
    "visreg": ("visreg", 768, "rgb"),
    "vjepa": ("vjepa", 1024, "rgb"),
}
PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]  # this repository (jepa4r/, configs/)
CACHE_SEED_BASES = {"train": 0, "val": 1000000000}
PROTOCOL = dict(  # the paper protocol
    seed=0,
    batch_size=1024,
    total_updates=1000000,
    validation_interval=5000,
    save_interval=100000,
    eval_tasks=(1, 2, 3, 4, 5),
    eval_episodes_per_task=50,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf8")


def cache_sampling_seed(config, split, raw_id):
    """The point-sampling seed of one row of the official data: the split's base + the row's index."""
    return config["cache_sampling_seed_base"][split] + raw_id


def group_contract(model, cube_count):
    if model not in MODELS or cube_count not in (1, 2, 3, 4):
        raise ValueError("expected one of the six encoders and Cube 1-4")
    family, dim, modality = MODELS[model]
    return dict(
        model=model,
        cube_count=cube_count,
        family=family,
        modality=modality,
        input_contract=dict(latent_dim=dim, current_dim=dim + 19, goal_dim=dim + 3 * cube_count, zero_padding=False),
        native_max_episode_steps=(200, 500, 1000, 1000)[cube_count - 1],
    )


def prepare_recipe(run_config, encoder_record, output_path):
    """The run's downstream.json: model, Cube count, official data, camera and the frozen encoder -- the export of the
    run's train stage (wm/WM_EXPORT.json) or the released weights of its prepare-encoder stage (encoder/ENCODER.json).
    """
    model, cubes = run_config["model"], run_config["cube_count"]
    contract = group_contract(model, cubes)
    config = dict(
        model=model,
        cube_count=cubes,
        official_data=copy.deepcopy(run_config["official_data"]),
        camera=copy.deepcopy(run_config["camera"]),
        cache_sampling_seed_base=dict(CACHE_SEED_BASES),
        output_root=str(Path(run_config["downstream_output_root"]).resolve()),
    )
    record = read(encoder_record)
    if model in ("visreg", "vjepa"):
        config["encoder_binding"] = record["encoder_binding"]
    else:
        binding = dict(
            family=contract["family"],
            latent_dim=contract["input_contract"]["latent_dim"],
            checkpoint=record["checkpoint"],
            factory_config=record["config"],
            epoch=record["epochs"],
        )
        if model == "lejepa":
            pre_source = REPO / "jepa4r/policy/bc/jepa_baselines.py"
        elif model == "pldm":
            pre_source = REPO / "jepa4r/baselines/cube/workflow_pldm_encoder.py"
            binding["adapter"] = str(pre_source)
        else:
            binding["adapter"] = str(REPO / "jepa4r/models/cube/workflow_wm_encoder.py")
            pre_source = REPO / "jepa4r/data/clips.py"  # the world-model preprocessors
        binding["preprocessing"] = dict(
            source_path=str(pre_source),
            device="cpu",
            input_dtype="float32" if model == "jepa4r" else "uint8",
            layout="XYZRGB1200" if model == "jepa4r" else "HWC224",
        )
        if model == "jepa4r":
            binding["preprocessing"].update(center=[0.425, 0.0, 0.2], scale=4.0, use_rgb=True)
        config["encoder_binding"] = binding
    if "cuda_visible_devices" in run_config:
        config["cuda_visible_devices"] = run_config["cuda_visible_devices"]
    save(output_path, config)
    return config


class FrozenOfficialEncoder:
    """A released encoder (VISReg / V-JEPA 2.1), frozen, with its own preprocessing."""

    def __init__(self, binding):
        import numpy as np
        import torch

        spec = importlib.util.spec_from_file_location("_workflow_official_encoder", binding["adapter"])
        self.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter)
        self.model = self.adapter.load_encoder(binding["spec"], device="cuda")
        self.binding, self.torch, self.np, self.calls = binding, torch, np, 0
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("the released encoder must stay frozen")

    def encode(self, value):
        np, torch = self.np, self.torch
        if value.shape != (224, 224, 3) or value.dtype != np.uint8:
            raise ValueError("the released encoder takes uint8 HWC 224x224")
        with torch.inference_mode():
            features = self.adapter.encode_uint8(self.model, torch.from_numpy(np.ascontiguousarray(value[None])))
        latent = features[0].detach().cpu().numpy()
        if latent.shape != (self.binding["latent_dim"],) or latent.dtype != np.float32 or not np.isfinite(latent).all():
            raise ValueError("the released encoder did not return a finite FP32 latent of its width")
        self.calls += 1
        return latent


def _encoder(config):
    import torch

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if config["model"] in ("visreg", "vjepa"):
        return FrozenOfficialEncoder(config["encoder_binding"])
    from .downstream.native import OnlineEncoder

    return OnlineEncoder(dict(config, modality=MODELS[config["model"]][2]))


def _core(config):
    """The evaluation core (downstream/native_core.py) set up for this run: the run's camera, input widths and
    encoder; NumPy's BLAS limited to 4 threads and Torch to 1, as the state evaluator's IK math ran originally."""
    from .downstream import native_core as core
    from .downstream._source import VENDOR
    import torch

    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("_workflow_native_physics", VENDOR / "physics.py")
    physics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(physics)
    physics.configure_numpy_physics_runtime()
    contract = group_contract(config["model"], config["cube_count"])
    group = config["model"] + "_cube" + str(config["cube_count"])
    core.LATENT_DIM = contract["input_contract"]["latent_dim"]
    core.GROUPS = {
        group: dict(
            cube_count=config["cube_count"],
            state_dim=19 + 9 * config["cube_count"],
            current_dim=contract["input_contract"]["current_dim"],
            goal_dim=contract["input_contract"]["goal_dim"],
            mode=contract["modality"],
            max_episode_steps=contract["native_max_episode_steps"],
            camera=config["camera"],
            pcd_crop_box=config["camera"].get("pointcloud_crop_box"),
        )
    }
    core.OnlineEncoder = lambda _group: _encoder(config)
    return core, group


def _official_arrays(config, root):
    """The qpos / qvel of every row of the official play data (train, val), saved as .npy for the rendering loop;
    an episode is 1001 rows, the last one terminal."""
    import numpy as np

    data, arrays = {}, {}
    for split, path in config["official_data"].items():
        with np.load(path, allow_pickle=False) as raw:
            terminals = raw["terminals"]
            rows = len(terminals)
            episodes = rows // 1001
            if rows % 1001 or not np.array_equal(np.flatnonzero(terminals), np.arange(episodes) * 1001 + 1000):
                raise ValueError(f"{path}: the play data are not whole episodes of 1001 rows")
            values = {key: raw[key] for key in ("qpos", "qvel")}
        data[split] = dict(path=str(path), episodes=episodes, rows=rows)
        arrays[split] = {}
        for key, value in values.items():
            target = root / "source_arrays" / f"{split}_{key}.npy"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, value, allow_pickle=False)
            arrays[split][key] = target
    return data, arrays


def _runtime_config(config, root, data):
    """The GCIQL recipe (downstream/trainer.py): data, cache, input widths, the agent settings of
    configs/policy/gciql/workflow_agent.json and the budget of PROTOCOL."""
    contract = group_contract(config["model"], config["cube_count"])
    return dict(
        model=config["model"],
        cube_count=config["cube_count"],
        state_dim=19 + 9 * config["cube_count"],
        modality=contract["modality"],
        encoder_family=contract["family"],
        latent_dim=contract["input_contract"]["latent_dim"],
        input_contract=contract["input_contract"],
        agent_config=read(REPO / "configs/policy/gciql/workflow_agent.json")["agent_config"],
        seed=PROTOCOL["seed"],
        batch_size=PROTOCOL["batch_size"],
        total_updates=PROTOCOL["total_updates"],
        validation_interval=PROTOCOL["validation_interval"],
        save_interval=PROTOCOL["save_interval"],
        save_step1=True,
        feature_lookup_workers=4,
        data={split: source["path"] for split, source in data.items()},
        cache_dir=str(root / "cache"),
        output=str(root / "policy"),
    )


def encode(config, root):
    """Every row of the official play data set back into the simulator, rendered by the run's camera (and, for JEPA4R,
    turned into the camera's point cloud with a per-row sampling seed) and encoded by the frozen encoder:
    cache/<split>/episode_<i>/latent.npy (1001 x latent width). Writes the GCIQL recipe runtime_recipe.json."""
    import numpy as np

    importlib.import_module("jepa4r.policy.gciql.downstream.sampler")  # puts OGBench's impls on sys.path
    data, arrays = _official_arrays(config, root)
    core, group = _core(config)
    encoder = _encoder(config)
    contract = group_contract(config["model"], config["cube_count"])
    cache = root / "cache"
    cache.mkdir(exist_ok=False)
    for split, source in data.items():
        seed = cache_sampling_seed(config, split, 0)
        env, _camera = core.make_env(group, seed)
        wrapper = None
        try:
            if contract["modality"] == "pcd":
                wrapper = core.OnlineVisualEnv(env, group, encoder, seed)
                wrapper.reset(options={})
            base = env.unwrapped
            tables = {key: np.load(path, mmap_mode="r", allow_pickle=False) for key, path in arrays[split].items()}
            for episode in range(source["episodes"]):
                start = episode * 1001
                features = np.empty((1001, contract["input_contract"]["latent_dim"]), dtype=np.float32)
                for local_id, raw_id in enumerate(range(start, start + 1001)):
                    base.set_state(tables["qpos"][raw_id], tables["qvel"][raw_id])
                    image = base.render(camera="front_pixels")
                    if wrapper:
                        # every row has its own point-sampling RNG (the same for every encoder)
                        current_seed = cache_sampling_seed(config, split, raw_id)
                        wrapper.frame.seed = current_seed
                        wrapper.frame.cores["current"]._rng = np.random.default_rng(current_seed)
                    features[local_id] = encoder.encode(wrapper.capture(image, "current") if wrapper else image)
                directory = cache / split / f"episode_{episode:06d}"
                directory.mkdir(parents=True)
                np.save(directory / "latent.npy", features, allow_pickle=False)
        finally:
            if wrapper:
                wrapper.close()
            else:
                env.close()
    save(root / "runtime_recipe.json", _runtime_config(config, root, data))
    return dict(episodes={split: source["episodes"] for split, source in data.items()}, encoder_calls=encoder.calls)


def train(config, root):
    from .downstream import trainer

    trainer.run(root / "runtime_recipe.json")
    return dict(updates=PROTOCOL["total_updates"], checkpoints=str(root / "policy" / "checkpoints"))


def evaluate(config, root):
    """The online evaluation of the trained policy (downstream/native_core.py): native/episodes/task<t>/episode_<i>/
    result.json and native/report.json."""
    core, group = _core(config)
    runtime = read(root / "runtime_recipe.json")
    core.GROUPS[group].update(training=str(root / "runtime_recipe.json"), agent_config=runtime["agent_config"])
    return core.main(
        group,
        root / "native",
        tasks=PROTOCOL["eval_tasks"],
        episodes_per_task=PROTOCOL["eval_episodes_per_task"],
    )


def execute(config_path, stage):
    if "torch" in sys.modules or "jax" in sys.modules:
        raise RuntimeError("run each Cube policy stage in a fresh process (its environment is set before JAX / Torch)")
    config = read(config_path)
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    from .downstream.runtime import runtime_environment

    os.environ.update(runtime_environment(stage, config.get("cuda_visible_devices", "0")))
    os.environ["CUBE_WORKFLOW_FACTORY_CACHE"] = str(root / "factory_cache")
    os.environ.pop("JAX_DEFAULT_MATMUL_PRECISION", None)
    (root / "logs").mkdir(exist_ok=True)
    with (root / "logs" / f"{stage}.log").open("w", encoding="utf8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                result = dict(encode=encode, gciql=train, evaluate=evaluate)[stage](config, root)
            except BaseException:
                traceback.print_exc()
                return 1
    save(root / "stages" / f"{stage}.json", dict(stage=stage, result=result))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    return execute(args.config, args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
