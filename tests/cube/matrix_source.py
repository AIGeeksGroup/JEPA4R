"""For the model-matrix check: optionally take a case's encoder from another run (a shared encoder); off by
default, and the workflow's own validation is unchanged."""

import hashlib
import json
from pathlib import Path
import re


def select_source_assets(case, target_runtime, *, assets_loader=None):
    """Load the shared encoder's run (the source) after checking it; the target case's own inputs are kept.

    The caller checks the target's cube count, states, goals, camera and inputs. This function neither edits the
    target nor creates a world-model receipt for it. Returns what the workflow's loader returns for the source."""
    if assets_loader is None:
        from jepa4r.policy.gciql.workflow_downstream import _assets

        assets_loader = _assets
    pin = case.get("source_encoder_config")
    if pin is None:
        return dict(
            assets=assets_loader(target_runtime),
            source_runtime=target_runtime,
            provenance=dict(
                shared_encoder=False,
                source_config_pin=None,
                source_cube_count=target_runtime["cube_count"],
                target_cube_count=target_runtime["cube_count"],
            ),
        )
    if (
        not isinstance(pin, dict)
        or not isinstance(pin.get("path"), (str, Path))
        or not isinstance(pin.get("sha256"), str)
        or re.fullmatch("[0-9a-f]{64}", pin["sha256"]) is None
    ):
        raise ValueError("invalid shared encoder source config pin")
    raw = Path(pin["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError("shared encoder source config SHA differs")
    source = json.loads(raw.decode("utf-8"))
    if not isinstance(source, dict):
        raise ValueError("shared encoder source config must be an actual runtime object")
    for name in ("model", "mode"):
        if source.get(name) != target_runtime.get(name) or name not in source:
            raise ValueError("shared encoder source/target " + name + " differs")
    binding = source.get("encoder_binding")
    target_binding = target_runtime.get("encoder_binding")
    if (
        not isinstance(binding, dict)
        or not binding
        or binding != target_binding
        or not binding.get("family")
        or type(binding.get("latent_dim")) is not int
    ):
        raise ValueError("shared encoder complete binding/family/latent_dim differs")
    for name in ("family", "latent_dim"):
        if source.get(name, binding[name]) != target_runtime.get(name, target_binding[name]):
            raise ValueError("shared encoder source/target " + name + " differs")
    for name in ("encoder_receipt",):
        if name in target_runtime and target_runtime[name] != source.get(name):
            raise ValueError("shared encoder target " + name + " differs from actual source")
    # The workflow's loader checks the source's own data, episode length, weights,
    # factory, preprocessing and receipt; it is never given a made-up target.
    assets = assets_loader(source)
    return dict(
        assets=assets,
        source_runtime=source,
        provenance=dict(
            shared_encoder=True,
            source_config_pin=dict(path=str(pin["path"]), sha256=pin["sha256"]),
            source_cube_count=source["cube_count"],
            target_cube_count=target_runtime["cube_count"],
        ),
    )
