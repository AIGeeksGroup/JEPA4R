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
        if (parent / 'jepa.py').is_file() and (parent / 'third_party/stable-worldmodel').is_dir():
            return parent
    raise FileNotFoundError('current model code and shared SWM runtime are missing')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources():
    manifest = json.loads((PACKAGE / 'SOURCE_MANIFEST.json').read_text(encoding='utf8'))
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
    return json.loads((PACKAGE / 'recipes' / (method + '.json')).read_text(encoding='utf8'))


def build_environment(data_root, output_root, environ=None):
    env = dict(os.environ if environ is None else environ)
    code = code_root()
    paths = [str(code), str(code / 'third_party/stable-worldmodel'), str(code / 'third_party/ogbench')]
    if env.get('PYTHONPATH'):
        paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(paths)
    env['STABLEWM_HOME'] = str(Path(output_root).resolve())
    env['SPT_CACHE_DIR'] = str(Path(output_root).resolve() / 'checkpoints/_spt_cache')
    return env


def load_tiny_trainer():
    require_sources()
    return importlib.import_module('.native.tiny_train', __package__)


def _load(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def pinned_native():
    """Bind the unchanged model's bare factory imports to this code root."""
    require_sources()
    code = code_root()
    entry_name = __package__ + '.native.lewm_train'
    names = ('module', 'utils', 'jepa', 'encoders', entry_name)
    previous = {name: module for name, module in sys.modules.items()
                if name in names or name.startswith('encoders.')}
    for name in previous:
        del sys.modules[name]
    try:
        _load('module', code / 'module.py')
        _load('utils', code / 'utils.py')
        _load('encoders', code / 'encoders/__init__.py', package=True)
        _load('jepa', code / 'jepa.py')
        yield _load(entry_name, PACKAGE / 'native/lewm_train.py')
    finally:
        for name in list(sys.modules):
            if name in names or name.startswith('encoders.'):
                del sys.modules[name]
        sys.modules.update(previous)
