"""Explicit PLDM512 input width; preserve the original sampler and distance math."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import types

FAMILY, DIM = 'pldm', 512
CHECKPOINT_SHA = 'ada09ccd9508434f688455622491b5910f2ad2190189a5f67f60cac53d7a5ae5'
SOURCE_SHA = 'b9906dc2f4e7a38826f5ee05031713618a6ce408ee5555302801505452758f90'


def overlay_source(raw):
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    source = raw.decode()
    edits = [('192', '512', 9), ('211', '531', 1)]
    effective = source
    for before, after, count in edits:
        assert effective.count(before) == count
        effective = effective.replace(before, after)
    restored = effective
    for before, after, count in reversed(edits):
        assert restored.count(after) == count
        restored = restored.replace(after, before)
    assert restored == source
    def functions(text):
        tree = ast.parse(text)
        selected = [(node.name, ast.dump(node, include_attributes=False)) for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name in ('sample', 'sample_goals', 'object_xyz', 'create_agent')]
        return selected
    assert functions(source) == functions(effective)
    return effective, dict(source_sha256=SOURCE_SHA, edits=edits,
        effective_source_sha256=hashlib.sha256(effective.encode()).hexdigest(),
        reverse_source_exact=True, sampler_reward_factory_AST_exact=True)


def load_inputs(path):
    path = Path(path)
    source, proof = overlay_source(path.read_bytes())
    module = types.ModuleType('_PLDM512_original_inputs')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)
    original = module.DistanceInputs
    class VisualInputs(original):
        pending = False
        def __init__(self, *args, cubes, mode, features, workers, **kwargs):
            assert cubes == 2 and mode == 'rgb' and workers in (0, 4)
            super().__init__(*args, cubes=cubes, mode=mode, features=features, workers=workers, **kwargs)
        def sample(self, batch_size, idxs=None, evaluation=False, *, prefetch_next=True):
            return super().sample(batch_size, idxs=idxs, evaluation=evaluation)
        def barrier(self):
            pass
    module.VisualInputs, module.width_proof = VisualInputs, proof
    return module


def load_features(module, *, encoder_family, feature_dim, **arguments):
    assert encoder_family == FAMILY and feature_dim == DIM
    assert arguments['checkpoint_sha256'] == CHECKPOINT_SHA
    manifest = module.read_json(Path(arguments['cache_dir']) / 'cache_manifest.json', arguments['manifest_sha256'])
    assert manifest['protocol']['encoder_family'] == FAMILY and manifest['protocol']['latent_dim'] == DIM
    result, binding = module.load_cached_features(**arguments)
    binding.update(encoder_family=FAMILY, latent_dim=DIM)
    return result, binding
