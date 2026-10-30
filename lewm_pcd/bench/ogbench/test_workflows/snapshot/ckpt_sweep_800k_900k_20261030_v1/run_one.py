"""Run one existing Native evaluator with an explicitly selected GCIQL update."""

from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "4"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["MUJOCO_GL"] = "egl"

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from registry import GROUPS
from sweep_common import ROOT, SWEEP_ROOT, atomic_json, install_selected_loader, read_json, sha


def _load_module(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _configure_lewm_physics():
    helper = (
        ROOT
        / "assets/imports/single_cube_rl3_101632_20261018_v1/source_layout/"
        "coordination/cached_lewm_state_gciql_xyz_109_20261012_v1/"
        "native_v1/locked_dependencies/native_runtime_helpers.py"
    )
    expected = "a83bcf181b00d57fe5b41ef4e61c42dded0561b1ec9d4dc2c7a1c64dfdb8bc3a"
    if sha(helper) != expected:
        raise RuntimeError("Native physics helper SHA changed")
    module = _load_module("_sweep_lewm_physics", helper)
    return module.configure_numpy_physics_runtime()


def _lewm_core(group: str):
    if group in {"G05", "G06", "G08", "G09"}:
        base = ROOT / "cube23_distance_gciql_v1/native_sr_visual_v1"
        sys.path[:0] = [str(base), str(base / "cube3_successor_v1")]
        import visual_sr
        import cube3_successor_v1.visual_sr_cube3 as adapter

        adapter.install(visual_sr)
        return visual_sr
    if group in {"G11", "G12"}:
        base = ROOT / "cube4_gciql_native_sr_v1"
        sys.path.insert(0, str(base))
        import quad_bridge

        return quad_bridge.load_core()
    raise ValueError(group)


def _foundation_core(config_path: Path, tiny: bool):
    functional = ROOT / "foundation_visual_gciql_v1/functional"
    sys.path.insert(0, str(functional))
    import native_sr

    cfg = read_json(config_path)
    return native_sr.install_core(
        cfg,
        config_path,
        tiny=tiny,
        gate_smoke=False,
    ), cfg


def _lejepa_core(config_path: Path, tiny: bool):
    cfg = read_json(config_path)
    case_dir = config_path.parents[2]
    if case_dir.name == "lejepa_tiny512_cube4_resource_v3":
        release = case_dir / "release_v1"
        resource_bridge = _load_module(
            "_sweep_lejepa_c4_resource_bridge",
            release / "c4_resource_bridge.py",
        )
        plan_path = (
            case_dir
            / "cache/groups/lejepa_tiny512_cube4/plan.json"
        )
        plan = read_json(plan_path)
        with resource_bridge.scoped_tiny_bridge(plan, plan_path) as bridge:
            modules = bridge.bootstrap()
            core = modules["native_sr"].install_core(
                cfg,
                config_path,
                tiny=tiny,
                gate_smoke=False,
            )
            overlay = bridge.recipe()
        return core, cfg, overlay
    release = ROOT / "lejepa_tiny512_cube14_gciql_v1/release_v1"
    module = _load_module("_sweep_lejepa_tiny_bridge", release / "tiny_bridge.py")
    modules = module.bootstrap()
    core = modules["native_sr"].install_core(
        cfg,
        config_path,
        tiny=tiny,
        gate_smoke=False,
    )
    return core, cfg, module.recipe()


def _pldm_binding_core(record: dict, tiny: bool):
    binding_path = Path(record["binding"])
    binding = read_json(binding_path)
    if binding.get("actual_exit_code") != 0 or binding.get("group_id") != record["group_id"]:
        raise RuntimeError("PLDM binding is not accepted for the requested group")
    case = binding_path.parent.parent
    functional = case / "functional_v1"
    sys.path.insert(0, str(functional))
    module = _load_module(f"_sweep_pldm_binding_{record['cube']}", functional / "native_sr.py")
    core = module.install_core(
        ROOT / "cube23_distance_gciql_v1/native_sr_visual_v1/visual_sr.py",
        binding["group"],
    )
    if tiny:
        core.GROUPS[record["group_id"]]["max_episode_steps"] = 4
    physics = module.configure_physics()
    return core, binding["group"], physics, binding


def _pldm14_core(config_path: Path, tiny: bool):
    case = ROOT / "pldm_cube14_gciql_v1"
    module = _load_module("_sweep_pldm14_functional", case / "functional.py")
    modules = module.bootstrap()
    cfg = read_json(config_path)
    core = modules["native_sr"].install_core(
        cfg,
        config_path,
        tiny=tiny,
        gate_smoke=False,
    )
    return core, cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    parser.add_argument("--update", type=int, choices=(800000, 900000, 1000000), required=True)
    parser.add_argument("--mode", choices=("tiny", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    record = GROUPS[args.group]
    config_path = Path(record["config"]).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if args.output.exists():
        raise FileExistsError(f"refusing existing output: {args.output}")

    family = record["family"]
    extra = {}
    if family == "lewm":
        core = _lewm_core(record["group_id"])
        if args.mode == "tiny":
            core.GROUPS[record["group_id"]]["max_episode_steps"] = 4
        extra["physics"] = _configure_lewm_physics()
    elif family == "foundation":
        core, _cfg = _foundation_core(config_path, args.mode == "tiny")
    elif family == "lejepa":
        core, _cfg, overlay = _lejepa_core(config_path, args.mode == "tiny")
        extra["lejepa_overlay"] = overlay
    elif family == "pldm_binding":
        core, _group, physics, source_binding = _pldm_binding_core(record, args.mode == "tiny")
        extra.update(
            physics=physics,
            pldm_native_binding=str(Path(record["binding"]).resolve()),
            pldm_native_binding_sha256=sha(Path(record["binding"])),
            pldm_binding_sha256=sha(Path(record["binding"])),
        )
    elif family == "pldm14":
        core, _cfg = _pldm14_core(config_path, args.mode == "tiny")
    else:
        raise ValueError(family)

    group = record["group_id"]
    if group not in core.GROUPS:
        raise RuntimeError(f"{group} was not registered by the existing evaluator")
    if Path(core.GROUPS[group]["training"]).resolve() != config_path:
        raise RuntimeError(
            f"{group}: evaluator config {core.GROUPS[group]['training']} != {config_path}"
        )
    core_contract = core.GROUPS[group]
    expected = (
        int(core_contract["current_dim"]),
        int(core_contract["goal_dim"]),
        int(core_contract["max_episode_steps"]),
        core_contract["mode"],
    )
    runtime_max_steps = 4 if args.mode == "tiny" else int(record["max_steps"])
    observed = (
        int(record["current_dim"]),
        int(record["goal_dim"]),
        runtime_max_steps,
        record["mode"],
    )
    if observed != expected:
        raise RuntimeError(f"{group}: registry/evaluator contract mismatch {observed} != {expected}")

    binding = install_selected_loader(core, group, args.update)
    source_paths = {
        Path(core.__file__).resolve(),
        ROOT / "cube23_distance_gciql_v1/gciql_boundary.py",
    }
    if family == "lewm":
        if record["group_id"] in {"G05", "G06", "G08", "G09"}:
            source_paths.add(
                ROOT
                / "cube23_distance_gciql_v1/native_sr_visual_v1/"
                "cube3_successor_v1/visual_sr_cube3.py"
            )
        else:
            source_paths.add(ROOT / "cube4_gciql_native_sr_v1/quad_bridge.py")
    elif family == "foundation":
        functional = ROOT / "foundation_visual_gciql_v1/functional"
        source_paths.update(
            functional / name
            for name in ("inputs.py", "prepare.py", "runtime.py", "native_sr.py", "audit_native.py")
        )
    elif family == "lejepa":
        functional = ROOT / "foundation_visual_gciql_v1/functional"
        release = ROOT / "lejepa_tiny512_cube14_gciql_v1/release_v1"
        source_paths.update(
            functional / name
            for name in ("inputs.py", "prepare.py", "runtime.py", "native_sr.py", "audit_native.py")
        )
        source_paths.update((release / "tiny_bridge.py", release / "tiny_encoder.py"))
        if record["cube"] == 4:
            c4 = ROOT / "lejepa_tiny512_cube4_resource_v3"
            source_paths.update(
                (
                    c4 / "release_v1/c4_resource_bridge.py",
                    c4 / "release_v1/resource_pool_v2.py",
                    c4 / "cache/groups/lejepa_tiny512_cube4/plan.json",
                )
            )
    elif family == "pldm_binding":
        case = Path(record["binding"]).parent.parent
        functional = case / "functional_v1"
        source_paths.update(
            (functional / "native_sr.py", functional / "pldm_inputs.py",
             functional / "pldm_runtime.py", functional / "audit_native.py")
        )
    elif family == "pldm14":
        source_paths.add(ROOT / "pldm_cube14_gciql_v1/functional.py")
    binding["evaluator_source_pins"] = {
        str(path): sha(path) for path in sorted(source_paths)
    }
    binding.update(
        runner="run_one.py",
        runner_sha256=sha(Path(__file__)),
        common_sha256=sha(Path(__file__).with_name("sweep_common.py")),
        registry_sha256=sha(Path(__file__).with_name("registry.py")),
        requested_mode=args.mode,
        model=record["model"],
        cube=record["cube"],
        family=family,
        evaluator_contract={
            "current_dim": record["current_dim"],
            "goal_dim": record["goal_dim"],
            "max_episode_steps": record["max_steps"],
            "mode": record["mode"],
        },
        runtime_environment={
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "JAX_PLATFORMS",
                "MUJOCO_GL",
            )
        },
        extra=extra,
    )

    core.main(
        group,
        "diagnostic" if args.mode == "tiny" else "full",
        args.output,
    )
    atomic_json(args.output / "native_source_binding.json", binding)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
