"""Default-off M2 shared encoder selection; formal workflow validation stays strict."""
import hashlib
import json
from pathlib import Path
import re


def select_source_assets(case, target_runtime, *, assets_loader=None):
    """Validate a real source runtime, retaining the target's real query contract.

    The caller independently validates target N/state/goal/camera/input provenance.
    This function neither edits the target nor manufactures a target WM receipt.
    Returned assets are the original loader result for the selected source.
    """
    if assets_loader is None:
        from .workflow_downstream import _assets
        assets_loader = _assets
    pin = case.get('source_encoder_config')
    if pin is None:
        return dict(assets=assets_loader(target_runtime), source_runtime=target_runtime,
                    provenance=dict(shared_encoder=False, source_config_pin=None,
                                    source_cube_count=target_runtime['cube_count'],
                                    target_cube_count=target_runtime['cube_count']))
    if (not isinstance(pin, dict) or not isinstance(pin.get('path'), (str, Path))
            or not isinstance(pin.get('sha256'), str)
            or re.fullmatch('[0-9a-f]{64}', pin['sha256']) is None):
        raise ValueError('invalid shared encoder source config pin')
    raw = Path(pin['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('shared encoder source config SHA differs')
    source = json.loads(raw.decode('utf-8'))
    if not isinstance(source, dict):
        raise ValueError('shared encoder source config must be an actual runtime object')
    for name in ('model', 'mode'):
        if source.get(name) != target_runtime.get(name) or name not in source:
            raise ValueError('shared encoder source/target ' + name + ' differs')
    binding = source.get('encoder_binding')
    target_binding = target_runtime.get('encoder_binding')
    if (not isinstance(binding, dict) or not binding or
            binding != target_binding or not binding.get('family') or
            type(binding.get('latent_dim')) is not int):
        raise ValueError('shared encoder complete binding/family/latent_dim differs')
    for name in ('family', 'latent_dim'):
        if source.get(name, binding[name]) != target_runtime.get(name, target_binding[name]):
            raise ValueError('shared encoder source/target ' + name + ' differs')
    for name in ('encoder_receipt', 'encoder_import'):
        if name in target_runtime and target_runtime[name] != source.get(name):
            raise ValueError('shared encoder target ' + name + ' differs from actual source')
    # Original strict loader checks the source's own actual producer/data/horizon,
    # weight/factory/preprocessing and receipt. It never sees a fabricated target.
    assets = assets_loader(source)
    return dict(assets=assets, source_runtime=source,
                provenance=dict(shared_encoder=True,
                                source_config_pin=dict(path=str(pin['path']), sha256=pin['sha256']),
                                source_cube_count=source['cube_count'],
                                target_cube_count=target_runtime['cube_count']))
