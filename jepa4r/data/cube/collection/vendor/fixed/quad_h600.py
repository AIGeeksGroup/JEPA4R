"""Entry of the quadruple-cube collection: episodes of 600 steps (H600) on the shared fixed-task collector, with the
camera pulled back by 0.10 m (camera_pullback.py)."""

import argparse
import importlib.util
from pathlib import Path
import sys
from types import MethodType

HERE = Path(__file__).resolve().parent
GUARD = "if (triple_task or quadruple_task) and (not fixed_task or int(cfg.world.max_episode_steps) != 1000):"
H600_GUARD = (
    "if (triple_task and (not fixed_task or int(cfg.world.max_episode_steps) != 1000)) or (quadruple_task "
    "and (not fixed_task or int(cfg.world.max_episode_steps) != 600)):"
)


def effective_source(source, horizon):
    if horizon == 1000:
        return source
    if horizon != 600 or source.count(GUARD) != 1:
        raise ValueError("Expected explicit H600 and one unchanged collector budget guard")
    return source.replace(GUARD, H600_GUARD, 1).replace(
        "Triple/Quadruple collection requires a fixed task and 1000 actions",
        "Triple requires H1000; explicit Quadruple600 adapter requires H600",
        1,
    )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--horizon-actions", type=int, choices=(600,), default=600)
    parser.add_argument("--camera-pullback-m", type=float, choices=(0.10,), default=0.10)
    args, passthrough = parser.parse_known_args()
    if passthrough[:1] == ["--"]:
        passthrough = passthrough[1:]
    root = args.base_root.resolve()
    sys.path[:0] = [
        str(root / n)
        for n in (
            "code/collection",
            "code/collection/scripts/data",
            "code/collection/third_party/stable-worldmodel",
            "third_party/ogbench",
            "third_party/stable-pretraining",
        )
    ]
    camera_path = HERE.parent / "collection_adapters/camera_pullback.py"
    spec = importlib.util.spec_from_file_location("paper_cube_camera", camera_path)
    camera = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(camera)
    camera.install(root, args.camera_pullback_m)
    original = root / "code/collection/scripts/data/collect_cube_pcd.py"
    source = effective_source(original.read_text(), args.horizon_actions)
    if args.horizon_actions == 600:
        import gymnasium as gym
        import quadruple_collection as quad

        factory = quad.make_quadruple_wrapper

        def h600_factory(*factory_args, **kwargs):
            wrapped = factory(*factory_args, **kwargs)
            node, limits = wrapped, []
            while isinstance(node, gym.Wrapper):
                if isinstance(node, gym.wrappers.TimeLimit):
                    limits.append(int(node._max_episode_steps))
                node = node.env
            if limits != [600]:
                raise ValueError(f"Actual world TimeLimit must be H600: {limits}")
            capture = wrapped._capture

            def terminal_capture(self, info):
                result = capture(info)
                if self.steps == 600:
                    result["collection_action_source"] = 2
                if not 0 <= self.steps <= 600:
                    raise ValueError("H600 capture outside the episode budget")
                return result

            wrapped._capture = MethodType(terminal_capture, wrapped)
            return wrapped

        quad.make_quadruple_wrapper = h600_factory
    sys.argv = [str(original), *passthrough]
    # the collector runs from its file with the 600-step limit in place of its 1000-step check; the file is unchanged
    namespace = dict(__name__="__main__", __file__=str(original), __package__=None)
    exec(compile(source, str(original), "exec"), namespace)


if __name__ == "__main__":
    main()
