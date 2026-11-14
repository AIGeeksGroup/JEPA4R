"""Pose adjustment of the collection camera only (pulled back along its view axis), for the paired Cube collection."""

import argparse
from pathlib import Path
import runpy
import sys
from types import MethodType


def camera_position(position, rotation, distance):
    """The camera position pulled back by ``distance`` metres along the camera's local Z axis."""
    import numpy as np

    return (
        np.asarray(position, dtype=np.float64).copy()
        + distance * np.asarray(rotation, dtype=np.float64).reshape(3, 3)[:, 2]
    )


def install(root, distance):
    """Pull the collection camera of the quadruple-cube env back by ``distance`` at every reset (RGB and point cloud
    alike; orientation and field of view unchanged)."""
    if distance == 0.0:
        return False
    import mujoco
    import quadruple_collection as quad

    original_factory = quad.make_quadruple_wrapper

    def factory(*args, **kwargs):
        wrapped = original_factory(*args, **kwargs)
        capture = wrapped._capture

        def effective_capture(self, info):
            if self.steps == 0:
                u = self.unwrapped
                pos, rotation, _fovy = self.camera
                u._model.cam_pos[self.cid] = camera_position(pos, rotation, distance)
                mujoco.mj_camlight(u._model, u._data)
                self.camera = (
                    u._data.cam_xpos[self.cid].copy(),
                    u._data.cam_xmat[self.cid].copy(),
                    float(u._model.cam_fovy[self.cid]),
                )
            return capture(info)

        wrapped._capture = MethodType(effective_capture, wrapped)
        return wrapped

    quad.make_quadruple_wrapper = factory
    return True


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--base-root", type=Path, required=True)
    ap.add_argument("--camera-pullback-m", type=float, default=0.0)
    args, passthrough = ap.parse_known_args()
    if passthrough[:1] == ["--"]:
        passthrough = passthrough[1:]
    root = args.base_root.resolve()
    sys.path[:0] = [
        str(root / "code/collection"),
        str(root / "code/collection/scripts/data"),
        str(root / "code/collection/third_party/stable-worldmodel"),
        str(root / "third_party/ogbench"),
        str(root / "third_party/stable-pretraining"),
    ]
    install(root, args.camera_pullback_m)
    original = root / "code/collection/scripts/data/collect_cube_pcd.py"
    sys.argv = [str(original), *passthrough]
    runpy.run_path(str(original), run_name="__main__")


if __name__ == "__main__":
    main()
