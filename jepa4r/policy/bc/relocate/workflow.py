"""The Relocate workflow runner: the stages of the Adroit Relocate or the RH56 twin plan (adroit_workflow.py,
rh56_workflow.py), each a child process with its log in <output>/logs; a stage that finished (exit code 0 recorded in
logs/<stage>.exit) is kept, so running again continues where it stopped.
usage: python -m jepa4r.policy.bc.relocate.workflow --env <adroit-relocate|rh56-relocate> --model <m> --assets <json>
       --output <dir> [--stage <name>] --execute"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
MODELS = {
    "adroit-relocate": ["jepa4r", "lewm", "pldm", "lejepa", "visreg", "vjepa"],
    "rh56-relocate": ["jepa4r", "lewm"],
}


def execute_plan(stages, output):
    """Run the stages in order; a finished stage is kept. Returns 0, or the exit code of the stage that failed."""
    output = Path(output).resolve()
    (output / "logs").mkdir(parents=True, exist_ok=True)
    (output / "PLAN.json").write_text(json.dumps(stages, indent=2) + "\n", encoding="utf-8")
    for stage in stages:
        name = stage["name"]
        exit_file = output / "logs" / f"{name}.exit"
        if exit_file.is_file() and exit_file.read_text(encoding="ascii").strip() == "0":
            continue
        environment = os.environ.copy()
        environment.update(stage.get("environment", {}))
        print(f"Running {name}", flush=True)
        with (output / "logs" / f"{name}.log").open("wb") as log:
            process = subprocess.run(
                stage["command"], cwd=stage["cwd"], env=environment, stdout=log, stderr=subprocess.STDOUT
            )
        exit_file.write_text(str(process.returncode) + "\n", encoding="ascii")
        if process.returncode != 0:
            return process.returncode
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--list", action="store_true", help="List the environments and their models")
    result.add_argument("--env", choices=tuple(MODELS), default="adroit-relocate")
    result.add_argument("--model", default="jepa4r")
    result.add_argument("--assets", type=Path, help="asset / environment JSON (see jepa4r.policy.bc.relocate_assets)")
    result.add_argument("--output", type=Path, help="absolute run directory outside the checkout")
    result.add_argument("--wm-python", help="world-model environment Python")
    result.add_argument("--sim-python", help="simulator / BC environment Python")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--stage", default="all", help="one planned stage name, or all")
    result.add_argument("--execute", action="store_true", help="run; otherwise print the plan only")
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.list:
        print(json.dumps(MODELS, indent=2))
        return 0
    if args.model not in MODELS[args.env]:
        argument_parser.error(f"{args.env} does not support model {args.model!r}")
    if args.assets is None or args.output is None or not args.output.is_absolute():
        argument_parser.error("--assets and an absolute --output are required")
    args.output = args.output.resolve()
    assets = json.loads(args.assets.read_text(encoding="utf-8-sig"))
    args.wm_python = args.wm_python if args.wm_python is not None else assets.get("wm_python")
    if args.sim_python is None:
        args.sim_python = assets.get(args.env.split("-")[0] + "_sim_python") or assets.get("sim_python")
    if not args.wm_python or not args.sim_python:
        argument_parser.error("Provide wm_python and sim_python through flags or the asset JSON")
    module_name = "adroit_workflow" if args.env == "adroit-relocate" else "rh56_workflow"
    stages = importlib.import_module(f"jepa4r.policy.bc.relocate.{module_name}").plan(args, assets)
    if args.stage != "all":
        stages = [stage for stage in stages if stage["name"] == args.stage]
        if not stages:
            argument_parser.error(f"Unknown stage {args.stage!r}")
    if not args.execute:
        print(json.dumps(stages, indent=2))
        return 0
    return execute_plan(stages, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
