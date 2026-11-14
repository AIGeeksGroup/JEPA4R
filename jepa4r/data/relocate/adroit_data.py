"""The official Relocate demonstrations for behavior cloning (the official-demos stage of the Adroit Relocate workflow).

Each official demonstration (the light copies of jepa4r/data/demos/prepare_official_assets.py) is replayed in mj_envs'
Relocate from its initial state with its own actions, rendering the fixed camera vil_camera as 256 x 256 RGB (MSAA 8)
and, in a second env, as point clouds (convert_adroit_demos_3d.convert_demo); written as <output>/rgb/relocate-v0.pickle
and <output>/pcd/relocate-v0.pickle, every other field unchanged.
usage: python -m jepa4r.data.relocate.adroit_data official --source <dir> --output <dir> [--count 100]"""

import argparse
from copy import deepcopy
from pathlib import Path
import pickle


def rgb_demo(demo, env, capture):
    """Replay one demonstration in env and return it with images = the RGB frames of capture()."""
    import numpy as np

    env.set_env_state(deepcopy(demo["init_state_dict"]))
    frames = []
    for action in np.asarray(demo["actions"]):
        frames.append(capture())
        env.step(action.copy())
    return dict(demo, images=np.asarray(frames))


def official(args):
    from jepa4r.data.relocate.convert_adroit_demos_3d import convert_demo, load_sources
    from jepa4r.policy.bc.run_adroit_bc3d import build_point_clouds
    from mj_envs.envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0

    _, read = load_sources(Path(args.source).expanduser().resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    rgb = RelocateEnvV0()
    pcd = RelocateEnvV0()
    converted_rgb = []
    converted_pcd = []
    try:
        rgb.seed(12345)
        rgb.reset()
        rgb.sim.render(width=256, height=256, camera_name="vil_camera", device_id=0)

        def capture():
            return rgb.sim.render(width=256, height=256, camera_name="vil_camera", device_id=0)[::-1].copy()

        sampler = build_point_clouds(pcd.sim, "relocate-v0")
        pcd.seed(12345)
        pcd.reset()
        sampler.seed(12345)
        sampler()
        for index in range(args.count):
            demo = read(index)
            sampler.seed(12345 + index)
            converted_rgb.append(rgb_demo(demo, rgb, capture))
            converted_pcd.append(convert_demo(demo, pcd, sampler))
        for route, demos in (("rgb", converted_rgb), ("pcd", converted_pcd)):
            path = args.output / route / "relocate-v0.pickle"
            path.parent.mkdir()
            with path.open("xb") as stream:
                pickle.dump(demos, stream, protocol=4)
    finally:
        rgb.close()
        pcd.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("official")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=100)
    official(parser.parse_args())


if __name__ == "__main__":
    main()
