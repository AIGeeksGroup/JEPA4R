"""Independently audit one selected-checkpoint Native250 result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from registry import GROUPS
from sweep_common import (
    SWEEP_ROOT,
    atomic_json,
    checkpoint_facts,
    independent_cpu_reload,
    read_json,
    sha,
    verify_output,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    parser.add_argument("--update", type=int, choices=(800000, 900000, 1000000), required=True)
    parser.add_argument("--mode", choices=("tiny", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_dir():
        raise FileNotFoundError(args.output)
    record = GROUPS[args.group]
    binding_path = args.output / "native_source_binding.json"
    binding = read_json(binding_path)
    if binding.get("actual_exit_code") != 0:
        raise RuntimeError("Native source binding is not successful")
    if binding.get("group_id") != record["group_id"]:
        raise RuntimeError("Native source binding group mismatch")
    if binding.get("selected_update") != args.update:
        raise RuntimeError("Native source binding update mismatch")

    config = read_json(Path(record["config"]))
    config["_config_path"] = record["config"]
    evaluator_config = read_json(Path(record["config"]))
    acceptance_path = Path(binding["source_acceptance"])
    acceptance = read_json(acceptance_path)
    facts = checkpoint_facts(evaluator_config, args.update, acceptance)
    saved_facts = binding["checkpoint"]
    for key in (
        "checkpoint",
        "manifest_sha256",
        "params_sha256",
        "run_state_sha256",
        "network_step",
        "adam_count",
        "run_state_binding_sha256",
    ):
        if facts[key] != saved_facts[key]:
            raise RuntimeError(f"checkpoint binding changed: {key}")
    reload_result = None
    if args.mode == "full":
        if not binding.get("evaluator_source_pins"):
            raise RuntimeError("formal job has no evaluator source pins")
        if "gate" not in binding:
            raise RuntimeError("formal job has no selected-checkpoint gate binding")
        for path, expected in binding["evaluator_source_pins"].items():
            if sha(Path(path)) != expected:
                raise RuntimeError(f"evaluator source changed: {path}")
        reload_result = independent_cpu_reload(
            Path(binding["training_config"]),
            Path(binding["gate"]),
            args.update,
        )
        if reload_result["agent_fingerprint"] != saved_facts["agent_fingerprint"]:
            raise RuntimeError("independent CPU reload fingerprint differs")

    result = verify_output(
        args.output,
        record["group_id"],
        args.mode,
        record["current_dim"],
        record["goal_dim"],
        record["max_steps"],
    )
    result.update(
        model=record["model"],
        cube=record["cube"],
        selected_update=args.update,
        selected_checkpoint=facts["checkpoint"],
        selected_manifest_sha256=facts["manifest_sha256"],
        selected_params_sha256=facts["params_sha256"],
        selected_run_state_sha256=facts["run_state_sha256"],
        native_source_binding_sha256=sha(binding_path),
        independent_cpu_reload=reload_result,
        audit_source_sha256=sha(Path(__file__)),
        common_source_sha256=sha(Path(__file__).with_name("sweep_common.py")),
        diagnostic_excluded_from_formal_SR=args.mode == "tiny",
    )
    atomic_json(args.output / "independent_audit.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
