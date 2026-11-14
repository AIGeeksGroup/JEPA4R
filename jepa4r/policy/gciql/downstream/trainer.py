"""GCIQL training of the Cube policy on the cached latents of a frozen encoder (a state-only run uses the states):
OGBench's GCIQL agent (vendor/gciql.py) with the distance-based Cube inputs of sampler.py.

The recipe (runtime_recipe.json, written by the encode stage) gives the data, the cache, the agent settings and the
budget: 1e6 updates of batch 1024, a validation loss every 5000 updates, a checkpoint every 100000 updates (and after
the first one). usage: python -m jepa4r.policy.gciql.downstream.trainer --config <runtime_recipe.json>"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

from ._source import VENDOR, install_impls

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_THREEFRY_PARTITIONABLE", "false")
install_impls()
sys.path[:0] = [str(VENDOR)]

import jax
import numpy as np
import ogbench
import torch
from utils.datasets import Dataset
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger
from .sampler import DistanceInputs, create_agent, load_cached_features


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


model = load_module("cube_policy_gciql", VENDOR / "gciql.py")


def agent_config(config):
    cfg = dict(config["agent_config"])
    # JSON stores sequences as lists; restore the original factory's tuple types.
    for key in ("actor_hidden_dims", "value_hidden_dims"):
        cfg[key] = tuple(cfg[key])
    return cfg


def prepare(config):
    """The train / val samplers: the official play data (readonly) and, for an encoder, its cached latents."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    samplers = {}
    try:
        for split in ("train", "val"):
            raw = ogbench.load_dataset(config["data"][split], compact_dataset=True, add_info=False)
            for value in raw.values():
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)
            features = None
            if config["modality"] != "state":
                episodes = len(raw["observations"]) // 1001
                features = load_cached_features(config["cache_dir"], split, episodes, config["latent_dim"])
            samplers[split] = DistanceInputs(
                Dataset.create(**raw),
                agent_config(config),
                cubes=config["cube_count"],
                mode=config["modality"],
                features=features,
                workers=config["feature_lookup_workers"],
                latent_dim=config["latent_dim"],
            )
        return samplers
    except BaseException:
        for sampler in samplers.values():
            sampler.close()
        raise


def initial_agent(config, samplers):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    # As in the single-cube code: one sample(1) before the agent is created (it advances the host RNG).
    batch = samplers["train"].sample(1)
    agent = create_agent(model.GCIQLAgent, config["seed"], batch, agent_config(config))
    return agent, batch


def update(agent, batch):
    agent, info = agent.update(batch)
    metrics = {key: float(value) for key, value in info.items()}
    if not all(np.isfinite(x) for x in metrics.values()) or metrics["grad/norm"] <= 0:
        raise ValueError("nonfinite update, loss or gradient, or a zero gradient norm")
    return agent, metrics


def save_checkpoint(agent, output, counter):
    directory = output / "checkpoints" / f"update_{counter:09d}"
    directory.mkdir(parents=True, exist_ok=False)
    save_agent(agent, str(directory), counter)
    return directory


def load_policy(config, counter=None):
    """The trained agent of a run (its checkpoint after ``counter`` updates, default the last one)."""
    output = Path(config["output"])
    counter = config["total_updates"] if counter is None else counter
    with np.load(output / "example_batch.npz", allow_pickle=False) as saved:
        example = {key: saved[key].copy() for key in saved.files}
    agent = create_agent(model.GCIQLAgent, config["seed"], example, agent_config(config))
    return restore_agent(agent, str(output / "checkpoints" / f"update_{counter:09d}"), counter)


def run(config_path):
    config = json.loads(Path(config_path).read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    samplers = prepare(config)
    logger = CsvLogger(str(output / "train.csv"))
    try:
        agent, example = initial_agent(config, samplers)
        np.savez(output / "example_batch.npz", **{k: example[k] for k in ("observations", "actions", "actor_goals")})
        counter = 0
        started = last_time = time.time()
        while counter < config["total_updates"]:
            batch = samplers["train"].sample(config["batch_size"])
            agent, info = update(agent, batch)
            counter += 1
            if counter % config["validation_interval"] == 0:
                val_batch = samplers["val"].sample(config["batch_size"])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                now = time.time()
                metrics = {f"training/{key}": value for key, value in info.items()}
                metrics.update({f"validation/{key}": float(value) for key, value in val_info.items()})
                metrics["time/epoch_time"] = (now - last_time) / config["validation_interval"]
                metrics["time/total_time"] = now - started
                logger.log(metrics, counter)
                last_time = now
            if (
                (counter == 1 and config["save_step1"])
                or counter % config["save_interval"] == 0
                or counter == config["total_updates"]
            ):
                save_checkpoint(agent, output, counter)
    finally:
        logger.close()
        for sampler in samplers.values():
            sampler.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    assert jax.default_backend() == "gpu", "GCIQL training runs JAX on the GPU"
    raise SystemExit(run(args.config))
