"""Config composition: Hydra-style config groups on plain OmegaConf, plus run-directory layout.

Groups selected on the command line with ``group=name`` load ``configs/<group>/<name>.yaml``:

    env      task: simulator, camera, point-cloud pipeline, data recipe, default policy
    method   one encoder row of Tab. 1 / Tab. 2 and how it is obtained (trained by us / released weights)
    policy   policy learner; defaults to ``env.policy``
    decoder  latent-space reconstruction decoder (Sec. 4.4)

``method.encoder`` names a file in ``configs/encoder``. A file may inherit from files of its own group
with ``_base_: name`` (or a list of names); files whose name starts with ``_`` are bases only.
``???`` marks a value still to be filled (OmegaConf raises when it is read).

Composition order (later wins)::

    defaults.yaml -> selected groups -> method.per_env[<glob on env.name>] -> refs (method.encoder)
    -> command-line ``dotted.key=value`` overrides
"""
import fnmatch
import os
from pathlib import Path
from typing import Iterable

import yaml
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"

GROUPS = ("env", "method", "policy", "decoder")          # selectable with group=name
REFS = {"method.encoder": "encoder"}                      # string keys naming a file of another group

# Pipeline stages (jepa4r/stages) and the run keys that identify one run of each.
RUN_KEYS = {
    "collect_wm_data": ("env",),
    "prepare_policy_data": ("env",),
    "train_wm": ("env", "method"),
    "train_policy": ("env", "method"),
    "train_decoder": ("env", "decoder"),
    "eval_recon": ("env", "decoder"),
    "verify_twin": ("env",),                              # digital twin vs. vendor URDF
    "eval_preview": ("env", "decoder"),                   # decoded look-ahead of the policy in the twin
    "count_params": ("method", "decoder"),                # exactly one of them
}
STAGES = tuple(RUN_KEYS)
DATA_STAGES = {"collect_wm_data": "wm", "prepare_policy_data": "policy"}   # stage -> data/<subdir>


def available(group: str) -> list[str]:
    return sorted(p.stem for p in (CONFIG_ROOT / group).glob("*.yaml") if not p.stem.startswith("_"))


def load_group(group: str, name: str) -> DictConfig:
    """Load ``configs/<group>/<name>.yaml`` with ``_base_`` inheritance resolved."""
    path = CONFIG_ROOT / group / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"unknown {group} {name!r}; available: {available(group)}")
    cfg = OmegaConf.load(path)
    bases = cfg.pop("_base_", None)
    if bases is None:
        return cfg
    if isinstance(bases, str):
        bases = [bases]
    return OmegaConf.merge(*(load_group(group, str(b)) for b in bases), cfg)


def _parse_value(text: str):
    if text == "":
        return ""
    value = yaml.safe_load(text)
    if isinstance(value, str):          # YAML 1.1 reads 1e-4 as a string
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                pass
    return value


def parse_args(args: Iterable[str]) -> tuple[dict[str, str], list[tuple[str, object]]]:
    """Split ``key=value`` arguments into group selections and dotted overrides."""
    groups: dict[str, str] = {}
    overrides: list[tuple[str, object]] = []
    for arg in args:
        key, sep, value = arg.partition("=")
        if not sep or not key:
            raise ValueError(f"expected key=value, got {arg!r}")
        if key in GROUPS:
            groups[key] = value
        else:
            overrides.append((key, _parse_value(value)))
    return groups, overrides


def _resolve_refs(cfg: DictConfig) -> None:
    for key, group in REFS.items():
        node = OmegaConf.select(cfg, key, default=None)
        if isinstance(node, str):
            OmegaConf.update(cfg, key, load_group(group, node), merge=False)


def compose(args: Iterable[str] = ()) -> DictConfig:
    """Compose a run config from ``group=name`` selections and ``dotted.key=value`` overrides."""
    groups, overrides = parse_args(args)
    cfg = OmegaConf.load(CONFIG_ROOT / "defaults.yaml")
    for group in ("env", "method", "decoder"):
        if group in groups:
            cfg[group] = load_group(group, groups[group])
    policy = groups.get("policy") or OmegaConf.select(cfg, "env.policy", default=None, throw_on_missing=False)
    if policy:
        cfg.policy = load_group("policy", policy)
    if "method" in cfg:
        per_env = cfg.method.pop("per_env", None)
        if per_env and "env" in cfg:
            for pattern, patch in per_env.items():
                if fnmatch.fnmatchcase(cfg.env.name, str(pattern)):
                    cfg.method = OmegaConf.merge(cfg.method, patch)
    _resolve_refs(cfg)
    for key, value in overrides:
        if key in REFS and isinstance(value, str):
            OmegaConf.update(cfg, key, load_group(REFS[key], value), merge=False)
        else:
            OmegaConf.update(cfg, key, value, merge=True)
    return cfg


def paths() -> dict[str, Path]:
    """Storage roots (all git-ignored), overridable with JEPA4R_DATA / JEPA4R_OUTPUTS / JEPA4R_CHECKPOINTS:

    data/         collected datasets            outputs/      runs, report, figures
    checkpoints/  third-party pretrained weights
    """
    def get(var: str, default: str) -> Path:
        return Path(os.environ.get(var) or REPO_ROOT / default).expanduser()

    return {
        "data": get("JEPA4R_DATA", "data"),
        "outputs": get("JEPA4R_OUTPUTS", "outputs"),
        "checkpoints": get("JEPA4R_CHECKPOINTS", "checkpoints"),
    }


def run_key(cfg: DictConfig, stage: str) -> dict:
    """The fields that identify one run of ``stage``; written into every metrics.json."""
    key = {"stage": stage, "env": None, "method": None, "decoder": None,
           "tag": None if stage in DATA_STAGES else str(cfg.tag)}
    for group in RUN_KEYS[stage]:
        if group in cfg:
            key[group] = cfg[group].name
    return key


def stage_dir(stage: str, env=None, method=None, decoder=None, tag="default") -> Path:
    """data/<wm|policy>/<env> for data stages, else outputs/<stage>/[<env>/][<method>|<decoder>/]<tag>."""
    p = paths()
    if stage in DATA_STAGES:
        return p["data"] / DATA_STAGES[stage] / env
    return p["outputs"].joinpath(stage, *[x for x in (env, method, decoder) if x], tag)


def run_dir(cfg: DictConfig, stage: str) -> Path:
    key = run_key(cfg, stage)
    return stage_dir(stage, key["env"], key["method"], key["decoder"], key["tag"] or "default")
