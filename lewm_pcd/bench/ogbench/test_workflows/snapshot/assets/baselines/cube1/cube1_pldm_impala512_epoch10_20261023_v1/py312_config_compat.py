"""Scoped import compatibility for two pinned PLDM dataclass declarations.

Preserve Python3.10 shared-default identity; do not alter model code on disk.
"""
import ast
from contextlib import contextmanager
from dataclasses import field
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import sys

FIELDS = {
    "pldm.models.jepa": {"backbone", "predictor"},
    "pldm.objectives": {"vicreg", "vicreg_obs", "vicreg_propio", "idm",
                         "prediction", "prediction_obs", "prediction_propio"},
}


def shared_default(value):
    return field(default_factory=lambda: value)


@contextmanager
def compatible_defaults(asset_root):
    if sys.version_info < (3, 11):
        yield
        return
    import json
    root = Path(asset_root).resolve()
    pins = json.loads((root / "MANIFEST.json").read_text())["members"]
    vendor = root / "source/vendor/pldm/source"

    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            path = Path(self.path)
            raw = path.read_bytes()
            name = path.relative_to(root).as_posix()
            if hashlib.sha256(raw).hexdigest() != pins[name]["sha256"]:
                raise ValueError("PLDM compatibility source pin differs")
            tree, changed = ast.parse(raw, filename=str(path)), set()
            for node in ast.walk(tree):
                if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                        and node.target.id in FIELDS[fullname]
                        and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)
                        and node.value.func.id.endswith("Config")
                        and not node.value.args and not node.value.keywords):
                    changed.add(node.target.id)
                    node.value = ast.Call(func=ast.Name(id="_pldm_shared_default", ctx=ast.Load()),
                                          args=[node.value], keywords=[])
            if changed != FIELDS[fullname]:
                raise ValueError("PLDM compatibility boundaries differ")
            return compile(ast.fix_missing_locations(tree), str(path), "exec")

        def exec_module(self, module):
            module.__dict__["_pldm_shared_default"] = shared_default
            super().exec_module(module)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in FIELDS:
                return None
            relative = fullname.replace(".", "/")
            source = vendor / (relative + ".py")
            package = not source.is_file()
            if package:
                source = vendor / relative / "__init__.py"
            loader = Loader(fullname, str(source))
            return importlib.util.spec_from_file_location(fullname, source,
                loader=loader, submodule_search_locations=[str(source.parent)] if package else None)

    finder = Finder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
