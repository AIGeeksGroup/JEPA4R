"""Paths of the code the Cube policy stages run: the vendored GCIQL agent, the OGBench checkout with its impls, and
the point-cloud code of the paired collection."""

import os
from pathlib import Path
import sys

VENDOR = Path(__file__).resolve().parent / "vendor"
REPO = Path(__file__).resolve().parents[4]  # this repository
OGBENCH_ROOT = Path(os.environ.get("JEPA4R_OGBENCH") or REPO / "third_party/ogbench")  # OGBench a977253 checkout
IMPLS = OGBENCH_ROOT / "impls"
PCD_ROOT = REPO / "jepa4r/data/cube/collection/vendor/pcd"


def install_impls():
    """Put the OGBench checkout and its impls (``utils.*``) first on sys.path; refuse another copy already imported."""
    existing = sys.modules.get("ogbench")
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(OGBENCH_ROOT):
        raise RuntimeError("another OGBench package is already imported")
    for name, module in list(sys.modules.items()):
        if name == "utils" or name.startswith("utils."):
            path = getattr(module, "__file__", None)
            paths = list(getattr(module, "__path__", []))
            if path and not Path(path).resolve().is_relative_to(IMPLS):
                raise RuntimeError("another utils module is already imported")
            if paths and not all(Path(p).resolve().is_relative_to(IMPLS) for p in paths):
                raise RuntimeError("another utils package is already imported")
    if str(OGBENCH_ROOT) not in sys.path:
        sys.path.insert(0, str(OGBENCH_ROOT))
    if str(IMPLS) not in sys.path:
        sys.path.insert(0, str(IMPLS))
