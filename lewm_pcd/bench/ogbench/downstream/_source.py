"""Current pinned source roots shared by policy, official environment and PCD."""
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
VENDOR=Path(__file__).resolve().parent/'vendor'
LEWM=Path(__file__).resolve().parents[3]
OGBENCH_ROOT=LEWM/'third_party/ogbench'
IMPLS=OGBENCH_ROOT/'impls'
PCD_ROOT=Path(__file__).resolve().parents[1]/'collection/vendor/pcd'
def source_path(relative):
    name=PurePosixPath(relative)
    if name.is_absolute() or '..' in name.parts or '\\' in relative or ':' in relative:
        raise ValueError('Source keys must be safe POSIX relative names')
    manifest=json.loads((VENDOR/'source_manifest.json').read_text())
    target=PurePosixPath(manifest['files'][relative]['path'])
    if target.is_absolute() or '..' in target.parts:
        raise ValueError('Source manifest path escapes current package')
    path=LEWM.joinpath(*target.parts).resolve()
    if not path.is_relative_to(LEWM):raise ValueError('Source resolves outside current package')
    return path
def verify_sources():
    manifest=json.loads((VENDOR/'source_manifest.json').read_text())
    for relative,record in manifest['files'].items():
        if hashlib.sha256(source_path(relative).read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Current source changed: '+relative)
    return manifest
def install_impls():
    verify_sources()
    existing=sys.modules.get('ogbench')
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(OGBENCH_ROOT):
        raise RuntimeError('Conflicting OGBench environment package import')
    for name,module in list(sys.modules.items()):
        if name=='utils' or name.startswith('utils.'):
            path=getattr(module,'__file__',None)
            paths=list(getattr(module,'__path__',[]))
            if path and not Path(path).resolve().is_relative_to(IMPLS):raise RuntimeError('Conflicting OGBench utils import')
            if paths and not all(Path(p).resolve().is_relative_to(IMPLS) for p in paths):raise RuntimeError('Conflicting OGBench utils package')
    if str(OGBENCH_ROOT) not in sys.path:sys.path.insert(0,str(OGBENCH_ROOT))
    if str(IMPLS) not in sys.path:sys.path.insert(0,str(IMPLS))
def load_source(name,relative):
    install_impls()
    path=source_path(relative)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
