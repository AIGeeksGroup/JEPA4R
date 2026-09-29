"""Official CortexBench BC loop with Adroit Pen PCD observations.

This is a PCD extension track.  It keeps the pinned official train loop,
action/proprio contract, optimizer and evaluator, replacing only demo/runtime
images and the frozen embedding loader.
"""

from __future__ import annotations

import os
import json
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VI = os.path.join(HERE, "..", "..", "third_party", "eai-vc", "cortexbench", "mujoco_vc", "visual_imitation")
sys.path.insert(0, VI)

import hydra
import mujoco_vc.gym_wrapper as gw
import mujoco_vc.model_loading as ml
import numpy as np
import train_loop
from omegaconf import OmegaConf

from bench.cortexbench.adroit.pcd_runtime import get_adroit_pcd, seed_adroit_pcd  # noqa: E402
from bench.cortexbench.adroit.pcd import AdroitPCDConfig  # noqa: E402


_official_load = ml.load_pretrained_model


def load_pretrained_model(embedding_name, input_type=np.ndarray, *args, **kwargs):
    config_path = os.path.join(HERE, "conf", "model", embedding_name + ".yaml")
    config = OmegaConf.load(config_path)
    model, embedding_dim, transforms, metadata = hydra.utils.call(config)
    if metadata.get("input") != "pcd":
        return _official_load(embedding_name, input_type, *args, **kwargs)
    return model.eval(), embedding_dim, (lambda pcd: transforms(pcd).unsqueeze(0)), metadata


for module in (ml, gw, train_loop):
    module.load_pretrained_model = load_pretrained_model


class JsonlRun:
    def __init__(self, path):
        self.file = open(path, "a", encoding="utf-8")

    def log(self, data=None, step=None, **kwargs):
        record = {}
        for key, value in (data or {}).items():
            try:
                record[key] = float(value)
            except (TypeError, ValueError):
                record[key] = str(value)
        if step is not None:
            record["_step"] = int(step)
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()

    def finish(self):
        self.file.close()


train_loop.setup_wandb = lambda config: JsonlRun(os.path.join(os.getcwd(), "metrics.jsonl"))


def get_image(self):
    return get_adroit_pcd(self, config=getattr(self, "_adroit_pcd_config", AdroitPCDConfig()))


def seed(self, seed=None):
    return seed_adroit_pcd(self, seed)


gw.MuJoCoPixelObsWrapper.get_image = get_image
gw.MuJoCoPixelObsWrapper.seed = seed


def env_constructor(*args, **kwargs):
    kwargs["device"] = "cpu"
    env = _original_env_constructor(*args, **kwargs)
    env.env.env._adroit_pcd_config = AdroitPCDConfig()
    return env


_original_env_constructor = gw.env_constructor
gw.env_constructor = env_constructor
train_loop.env_constructor = env_constructor


sys.argv = [os.path.join(HERE, "run_bc.py")] + sys.argv[1:]
runpy.run_path(os.path.join(VI, "hydra_launcher.py"), run_name="__main__")
