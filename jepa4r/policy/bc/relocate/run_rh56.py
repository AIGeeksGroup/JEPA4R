"""Entry point of the RH56 Relocate workflow's stages (jepa4r/envs/twins/rh56): collect, publish, official-init, wm, bc;
the arguments after the stage go to its module unchanged.
usage: python jepa4r/policy/bc/relocate/run_rh56.py <stage> [--route rgb|pcd] [--expert-root <dir>] ..."""

import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULES = {
    "collect": "jepa4r.envs.twins.rh56.collect",
    "publish": "jepa4r.envs.twins.rh56.data_tools",
    "official-init": "jepa4r.envs.twins.rh56.official_init",
    "wm": "jepa4r.envs.twins.rh56.wm_bridge",
    "bc": "jepa4r.envs.twins.rh56.bc_bridge",
}


def parser():
    result = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Collection is paired RGB / point clouds; wm and bc take the route (rgb or pcd).",
    )
    result.add_argument("stage", choices=tuple(MODULES), nargs="?")
    result.add_argument("--route", "--modality", choices=("rgb", "pcd"))
    result.add_argument("--package-root", type=Path, help="the jepa4r package root; defaults to this checkout")
    result.add_argument(
        "--expert-root", type=Path, help="the external native RM75-B + RH56 bundle; or RH56_EXPERT_ROOT"
    )
    result.add_argument(
        "--eai-root", "--eai-vc-root", type=Path, help="eai-vc checkout for BC; or CORTEXBENCH_EAI_VC_ROOT"
    )
    result.add_argument(
        "--dry-run", action="store_true", help="print the stage's module, arguments and environment only"
    )
    return result


def make_plan(args, forwarded, argument_parser):
    learning = args.stage in ("wm", "bc")
    if learning and args.route is None:
        argument_parser.error("wm/bc require an explicit --route rgb|pcd")
    if not learning and args.route is not None:
        argument_parser.error("--route applies only to wm/bc; collection is paired")
    package = Path(args.package_root or os.environ.get("RH56_PACKAGE_ROOT") or PACKAGE_ROOT).expanduser().resolve()
    expert = args.expert_root or os.environ.get("RH56_EXPERT_ROOT")
    environment = {"RH56_PACKAGE_ROOT": str(package)}
    if expert:
        environment["RH56_EXPERT_ROOT"] = str(Path(expert).expanduser().resolve())
    argv = list(forwarded)
    if args.stage == "collect" and not any(a == "--camera" or a.startswith("--camera=") for a in argv):
        argv += ["--camera", "opposite"]
    if learning:
        argv += ["--route", args.route]
        argv += ["--package-root", str(package)]
    if args.stage == "bc":
        eai = (
            Path(
                args.eai_root or os.environ.get("CORTEXBENCH_EAI_VC_ROOT") or PACKAGE_ROOT.parent / "third_party/eai-vc"
            )
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
    package = Path(plan["environment"]["RH56_PACKAGE_ROOT"])
    if not (package / "models/world_model.py").is_file():  # LeWM's train.py
        raise FileNotFoundError(f"no models/world_model.py under {package}")
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
    path = str(PACKAGE_ROOT.parent)
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
