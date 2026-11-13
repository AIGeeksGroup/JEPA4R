"""Current four-Cube H600 entry over the shared fixed collector."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import MethodType

HERE = Path(__file__).resolve().parent
GUARD = "if (triple_task or quadruple_task) and (not fixed_task or int(cfg.world.max_episode_steps) != 1000):"
H600_GUARD = ("if (triple_task and (not fixed_task or int(cfg.world.max_episode_steps) != 1000)) or (quadruple_task "
              "and (not fixed_task or int(cfg.world.max_episode_steps) != 600)):")


def effective_source(source, horizon):
    if horizon == 1000:
        return source
    if horizon != 600 or source.count(GUARD) != 1:
        raise ValueError("Expected explicit H600 and one unchanged collector budget guard")
    return source.replace(GUARD, H600_GUARD, 1).replace(
        "Triple/Quadruple collection requires a fixed task and 1000 actions",
        "Triple requires H1000; explicit Quadruple600 adapter requires H600", 1)


def write_adapter_binding(output, collector, camera, horizon, pullback):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    files = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in (Path(__file__), Path(collector), Path(camera))}
    record = dict(schema='paper_cube_adapter_binding_v1', stage='source_binding_prepared',
                  files=files, raw_horizon=horizon, pullback_m=pullback)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    return record


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--horizon-actions", type=int, choices=(600,), default=600)
    parser.add_argument("--camera-pullback-m", type=float, choices=(.10,), default=.10)
    parser.add_argument("--adapter-binding", type=Path, required=True)
    args, passthrough = parser.parse_known_args()
    if passthrough[:1] == ["--"]:
        passthrough = passthrough[1:]
    root = args.base_root.resolve()
    sys.path[:0] = [
        str(root / n)
        for n in (
            "code/lewm_pcd",
            "code/lewm_pcd/scripts/data",
            "code/lewm_pcd/third_party/stable-worldmodel",
            "third_party/ogbench",
            "third_party/stable-pretraining",
        )
    ]
    camera_path = HERE.parent / "collection_adapters/camera_pullback.py"
    spec = importlib.util.spec_from_file_location("paper_cube_camera", camera_path)
    camera = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(camera)
    camera.install(root, args.camera_pullback_m)
    original = root / "code/lewm_pcd/scripts/data/collect_cube_pcd.py"
    source = effective_source(original.read_text(), args.horizon_actions)
    runtime_sha = hashlib.sha256(source.encode()).hexdigest()
    write_adapter_binding(args.adapter_binding, original, camera_path, args.horizon_actions, args.camera_pullback_m)
    if args.horizon_actions == 600:
        import gymnasium as gym
        import quadruple_collection as quad
        import collection_protocol as protocol
        factory, manifest_class = quad.make_quadruple_wrapper, protocol.CollectionManifest

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

        class HorizonManifest(manifest_class):
            def __init__(self, paths, config):
                config = dict(config)
                if config["horizon_actions"] != 600 or config["env_type"] != "quadruple":
                    raise ValueError("Explicit Quadruple H600 configuration required")
                config["horizon_override"] = dict(
                    actions=600, observations=601, terminal_source_enum=2,
                    effective_TimeLimit=600, adapter_path=str(Path(__file__).resolve()),
                    adapter_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    base_collector_SHA256=hashlib.sha256(original.read_bytes()).hexdigest(),
                    effective_collector_runtime_SHA256=runtime_sha,
                    original_ROOT_sources_unchanged=True)
                super().__init__(paths, config)

        quad.make_quadruple_wrapper = h600_factory
        protocol.CollectionManifest = HorizonManifest
    sys.argv = [str(original), *passthrough]
    # The manifest records both the original on-disk SHA and this explicit
    # overlay's runtime SHA; no source file is overwritten or relabeled.
    namespace = dict(__name__="__main__", __file__=str(original), __package__=None)
    exec(compile(source, str(original), "exec"), namespace)


if __name__ == "__main__":
    main()
