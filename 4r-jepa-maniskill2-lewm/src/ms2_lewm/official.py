"""Load pinned upstream Python modules without editing their source trees."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
LEWM = ROOT / "vendor/le-wm"


def load_source(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


lewm_module = load_source("_ms2_official_lewm_module", LEWM / "module.py")
lewm_jepa = load_source("_ms2_official_lewm_jepa", LEWM / "jepa.py")


def lewm_config(data="ogb"):
    """Read, rather than duplicate, the official training/architecture recipe."""
    cfg = OmegaConf.load(LEWM / "config/train/lewm.yaml")
    cfg.model = OmegaConf.load(LEWM / "config/train/model/lewm.yaml")
    cfg.data = OmegaConf.load(LEWM / f"config/train/data/{data}.yaml")
    cfg.wandb = OmegaConf.load(LEWM / "config/train/launcher/local.yaml").wandb
    if not OmegaConf.has_resolver("eval"):
        # Only the arithmetic expression used by the pinned official configs.
        OmegaConf.register_new_resolver("eval", lambda expr: sum(int(x) for x in expr.split("+")))
    return cfg
