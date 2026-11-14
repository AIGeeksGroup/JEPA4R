"""Recipe and source loading and the local runtime paths of the Cube training."""

from contextlib import contextmanager
import importlib
import importlib.util
import json
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def code_root():
    for parent in PACKAGE.parents:
        if (parent / "jepa4r/models/world_model.py").is_file() and (parent / "third_party/stable-worldmodel").is_dir():
            return parent
    raise FileNotFoundError("this repository's model code and stable-worldmodel are missing")


def load_recipe(method):
    if method not in ("lewm", "jepa4r"):
        raise ValueError("the world-model recipes are lewm and jepa4r")
    recipe = PACKAGE.parents[2] / "configs" / "method" / "cube" / "recipes" / (method + ".json")
    return json.loads(recipe.read_text(encoding="utf8"))


def build_environment(data_root, output_root, environ=None):
    env = dict(os.environ if environ is None else environ)
    code = code_root()
    paths = [
        str(code),
        str(code / "third_party/stable-worldmodel"),
        os.environ.get("JEPA4R_OGBENCH") or str(code / "third_party/ogbench"),
    ]  # OGBench a977253 checkout
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["STABLEWM_HOME"] = str(Path(output_root).resolve())
    env["SPT_CACHE_DIR"] = str(Path(output_root).resolve() / "checkpoints/_spt_cache")
    return env


def load_tiny_trainer():
    return importlib.import_module("jepa4r.baselines.cube.native.tiny_train")


@contextmanager
def wm_trainer():
    """The world-model trainer (jepa4r.models.world_model: run_episode_split with the original output layout,
    lejepa_forward, SIGReg), with the module attributes its callers use and patch."""
    from functools import partial
    from types import SimpleNamespace
    import hydra
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from jepa4r.models import world_model

    yield SimpleNamespace(
        run=partial(world_model.run_episode_split, run_dir=None, original_layout=True),
        lejepa_forward=world_model.lejepa_forward,
        SIGReg=world_model.SIGReg,
        hydra=hydra,
        spt=spt,
        swm=swm,
        __file__=world_model.__file__,
    )
