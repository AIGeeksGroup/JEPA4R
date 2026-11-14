# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/training_runtime.py (verbatim except: imports / paths adapted
# to the package; lines wrapped at 120 columns, registration imports as importlib.import_module)
"""Current recipe/source loading and local runtime paths for Cube training."""
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent


def code_root():
    for parent in PACKAGE.parents:
        if (parent / 'jepa4r/models/world_model.py').is_file() and (parent / 'third_party/stable-worldmodel').is_dir():
            return parent
    raise FileNotFoundError('current model code and shared SWM runtime are missing')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources():
    manifest = PACKAGE.parents[2] / 'configs' / 'policy' / 'gciql' / 'SOURCE_MANIFEST.json'
    manifest = json.loads(manifest.read_text(encoding='utf8'))
    mismatches = []
    for root, entries in [(PACKAGE, manifest['package_files']), (code_root(), manifest['project_files'])]:
        for name, digest in entries.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or _sha(path) != digest:
                mismatches.append(name)
    return dict(checked_files=len(manifest['package_files']) + len(manifest['project_files']), mismatches=mismatches)


def require_sources():
    result = verify_sources()
    if result['mismatches']:
        raise ValueError('current training source mismatch: ' + ', '.join(result['mismatches']))


def load_recipe(method):
    if method not in ('lewm_rgb', 'lewm_pcd'):
        raise ValueError('current LeWM RGB/PCD recipe required')
    require_sources()
    recipe = PACKAGE.parents[2] / 'configs' / 'policy' / 'gciql' / 'recipes' / (method + '.json')
    return json.loads(recipe.read_text(encoding='utf8'))


def build_environment(data_root, output_root, environ=None):
    env = dict(os.environ if environ is None else environ)
    code = code_root()
    paths = [str(code), str(code / 'third_party/stable-worldmodel'),
             os.environ.get('JEPA4R_OGBENCH') or str(code / 'third_party/ogbench')]   # OGBench a977253 checkout
    if env.get('PYTHONPATH'):
        paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(paths)
    env['STABLEWM_HOME'] = str(Path(output_root).resolve())
    env['SPT_CACHE_DIR'] = str(Path(output_root).resolve() / 'checkpoints/_spt_cache')
    return env


def load_tiny_trainer():
    require_sources()
    return importlib.import_module('jepa4r.baselines.cube.native.tiny_train')


def _load(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def pinned_native():
    """The native LeWM trainer native/lewm_train.py, moved to jepa4r.models.world_model (run_episode_split with the
    source's own output layout, lejepa_forward, SIGReg), with the module attributes its callers use and patch."""
    require_sources()
    from functools import partial
    from types import SimpleNamespace
    import hydra
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from jepa4r.models import world_model
    yield SimpleNamespace(run=partial(world_model.run_episode_split, run_dir=None, original_layout=True),
                          lejepa_forward=world_model.lejepa_forward, SIGReg=world_model.SIGReg,
                          hydra=hydra, spt=spt, swm=swm, __file__=world_model.__file__)
