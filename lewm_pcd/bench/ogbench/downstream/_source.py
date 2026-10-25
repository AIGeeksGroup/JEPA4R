"""Load one byte-preserved source closure, with an explicit import guard."""
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys

VENDOR = Path(__file__).resolve().parent/'vendor'
IMPLS = VENDOR/'source/ogbench/impls'


def source_path(relative):
    path = PurePosixPath(relative)
    if '\\' in relative or ':' in relative or path.is_absolute() or '..' in path.parts:
        raise ValueError('Preserved source keys must be portable POSIX relative paths')
    return VENDOR.joinpath(*path.parts)


def verify_sources():
    manifest = json.loads((VENDOR/'source_manifest.json').read_text())
    for relative, record in manifest['files'].items():
        if hashlib.sha256(source_path(relative).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Preserved source changed: {relative}')
    return manifest


def install_impls():
    verify_sources()
    for name, module in list(sys.modules.items()):
        if name == 'utils' or name.startswith('utils.'):
            path = getattr(module, '__file__', None)
            paths = list(getattr(module, '__path__', []))
            if path and not Path(path).resolve().is_relative_to(IMPLS):
                raise RuntimeError(f'Conflicting OGBench import: {name} from {path}')
            if paths and not all(Path(p).resolve().is_relative_to(IMPLS) for p in paths):
                raise RuntimeError(f'Conflicting OGBench utils namespace: {paths}')
    if str(IMPLS) not in sys.path:
        sys.path.insert(0, str(IMPLS))


def load_source(name, relative):
    install_impls()
    spec = importlib.util.spec_from_file_location(name, VENDOR/relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
