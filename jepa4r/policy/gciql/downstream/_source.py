"""Source roots shared by policy, official environment and PCD."""

import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys

VENDOR = Path(__file__).resolve().parent / "vendor"
LEWM = Path(__file__).resolve().parents[4]  # this repository
OGBENCH_ROOT = Path(os.environ.get("JEPA4R_OGBENCH") or LEWM / "third_party/ogbench")  # OGBench a977253 checkout
IMPLS = OGBENCH_ROOT / "impls"
PCD_ROOT = LEWM / "jepa4r/data/cube/collection/vendor/pcd"


def source_path(relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or "\\" in relative or ":" in relative:
        raise ValueError("Source keys must be safe POSIX relative names")
    manifest = json.loads((LEWM / "configs/policy/gciql/downstream/source_manifest.json").read_text())
    target = PurePosixPath(manifest["files"][relative]["path"])
    if target.is_absolute() or ".." in target.parts:
        raise ValueError("Source manifest path escapes current package")
    root = OGBENCH_ROOT if target.parts[:2] == ("third_party", "ogbench") else LEWM  # adapted: the OGBench checkout
    path = root.joinpath(*(target.parts[2:] if root is OGBENCH_ROOT else target.parts)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source resolves outside current package")
    return path


def install_impls():
    existing = sys.modules.get("ogbench")
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(OGBENCH_ROOT):
        raise RuntimeError("Conflicting OGBench environment package import")
    for name, module in list(sys.modules.items()):
        if name == "utils" or name.startswith("utils."):
            path = getattr(module, "__file__", None)
            paths = list(getattr(module, "__path__", []))
            if path and not Path(path).resolve().is_relative_to(IMPLS):
                raise RuntimeError("Conflicting OGBench utils import")
            if paths and not all(Path(p).resolve().is_relative_to(IMPLS) for p in paths):
                raise RuntimeError("Conflicting OGBench utils package")
    if str(OGBENCH_ROOT) not in sys.path:
        sys.path.insert(0, str(OGBENCH_ROOT))
    if str(IMPLS) not in sys.path:
        sys.path.insert(0, str(IMPLS))


def load_source(name, relative):
    install_impls()
    path = source_path(relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
