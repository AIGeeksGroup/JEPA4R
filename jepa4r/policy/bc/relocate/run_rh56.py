"""Entry point of the RH56 Relocate workflow (opt-in); the Adroit entry points do not use it."""

import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys


LEWM_ROOT = Path(__file__).resolve().parents[3]
MODULES = {
    "collect": "jepa4r.envs.twins.rh56.collect",
    "validate": "jepa4r.envs.twins.rh56.data_tools",
    "publish": "jepa4r.envs.twins.rh56.data_tools",
    "official-init": "jepa4r.envs.twins.rh56.official_init",
    "wm": "jepa4r.envs.twins.rh56.wm_bridge",
    "bc": "jepa4r.envs.twins.rh56.bc_bridge",
}


def parser():
    result = argparse.ArgumentParser(
        description=__doc__,
        epilog="Backend arguments are forwarded unchanged. No stage prints help only. "
        "Collection is paired RGB/PCD; learning requires an explicit route. "
        "See docs/experiments/rh56_native_20261025.md for complete examples.",
    )
    result.add_argument("stage", choices=tuple(MODULES), nargs="?")
    result.add_argument("--route", "--modality", choices=("rgb", "pcd"))
    result.add_argument(
        "--baseline-root", "--baseline", type=Path, help="Existing LeWM source/config root; defaults to this checkout"
    )
    result.add_argument(
        "--expert-root", type=Path, help="External native 56D-to-12D expert bundle; or RH56_EXPERT_ROOT"
    )
    result.add_argument(
        "--eai-root", "--eai-vc-root", type=Path, help="Existing eai-vc checkout for BC; or CORTEXBENCH_EAI_VC_ROOT"
    )
    result.add_argument(
        "--dry-run", action="store_true", help="Print dispatch and paths without importing or running a backend"
    )
    return result


def make_plan(args, forwarded, argument_parser):
    learning = args.stage in ("wm", "bc")
    if learning and args.route is None:
        argument_parser.error("wm/bc require an explicit --route rgb|pcd")
    if not learning and args.route is not None:
        argument_parser.error("--route applies only to wm/bc; collection is paired")
    baseline = Path(args.baseline_root or os.environ.get("RH56_BASELINE_ROOT") or LEWM_ROOT).expanduser().resolve()
    expert = args.expert_root or os.environ.get("RH56_EXPERT_ROOT")
    environment = {"RH56_BASELINE_ROOT": str(baseline)}
    if expert:
        environment["RH56_EXPERT_ROOT"] = str(Path(expert).expanduser().resolve())
    argv = list(forwarded)
    if args.stage in ("validate", "publish"):
        argv.insert(0, args.stage)
    if args.stage == "collect" and not any(a == "--camera" or a.startswith("--camera=") for a in argv):
        argv += ["--camera", "opposite"]
    if learning:
        argv += ["--route", args.route]
        argv += ["--baseline-root" if args.stage == "wm" else "--baseline", str(baseline)]
    if args.stage == "bc":
        eai = (
            Path(args.eai_root or os.environ.get("CORTEXBENCH_EAI_VC_ROOT") or LEWM_ROOT.parent / "third_party/eai-vc")
            .expanduser()
            .resolve()
        )
        environment["CORTEXBENCH_EAI_VC_ROOT"] = str(eai)
        argv += ["--eai-root", str(eai)]
    return {
        "embodiment": "rh56",
        "stage": args.stage,
        "module": MODULES[args.stage],
        "argv": argv,
        "environment": environment,
    }


def check_runtime_inputs(plan):
    baseline = Path(plan["environment"]["RH56_BASELINE_ROOT"])
    if not (
        baseline / "models/world_model.py"
    ).is_file():  # LeWM's train.py, in world_model.py (as in rh56/wm_bridge.tree_sha)
        raise FileNotFoundError(f"LeWM source root has no train.py: {baseline}")
    if plan["stage"] in ("collect", "official-init", "bc"):
        expert = plan["environment"].get("RH56_EXPERT_ROOT")
        if not expert:
            raise ValueError("Provide --expert-root or RH56_EXPERT_ROOT for the native RH56 bundle")
        missing = [
            name for name in ("env.py", "ppo.py", "experts/relocate_expert.pt") if not (Path(expert) / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Incomplete external RH56 expert bundle {expert}: {missing}")
    if plan["stage"] == "bc":
        eai = Path(plan["environment"]["CORTEXBENCH_EAI_VC_ROOT"])
        if not (eai / "cortexbench/mujoco_vc/visual_imitation/train_loop.py").is_file():
            raise FileNotFoundError(f"Provide an existing eai-vc checkout with --eai-vc-root: {eai}")


@contextmanager
def runtime_environment(values):
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def invoke(module, argv):
    path = str(LEWM_ROOT.parent)
    added = path not in sys.path
    if added:
        sys.path.insert(0, path)
    try:
        return importlib.import_module(module).main(argv)
    finally:
        if added:
            sys.path.remove(path)


def main(argv=None):
    argument_parser = parser()
    args, forwarded = argument_parser.parse_known_args(argv)
    if args.stage is None:
        argument_parser.print_help()
        return 0
    plan = make_plan(args, forwarded, argument_parser)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    check_runtime_inputs(plan)
    with runtime_environment(plan["environment"]):
        result = invoke(plan["module"], plan["argv"])
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
