"""Read-only validation of all planned configurations and selected checkpoints."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from registry import GROUPS, MODEL_ORDER, UPDATES, selected_groups
from sweep_common import (
    ROOT,
    SWEEP_ROOT,
    atomic_json,
    checkpoint_facts,
    read_json,
    sha,
)


def acceptance_path(key: str, record: dict) -> Path:
    group = record["group_id"]
    family = record["family"]
    if family == "lewm":
        if group in {"G05", "G06", "G08", "G09"}:
            return (
                ROOT
                / "cube23_distance_gciql_v1/five_group_extension_v1/acceptance"
                / f"{group}_full_acceptance.json"
            )
        return (
            ROOT
            / "cube4_visual_gciql_v1/acceptance"
            / f"{group}_full_acceptance.json"
        )
    if family == "pldm_binding":
        binding = read_json(Path(record["binding"]))
        return Path(binding["group"]["acceptance"])
    config = read_json(Path(record["config"]))
    return Path(config["execution_root"]) / "acceptance" / f"{group}.json"


def main() -> int:
    started = time.time()
    errors = []
    rows = []
    planned = selected_groups()
    model_counts = Counter(record["model"] for _, record in planned)
    expected_counts = {model: (3 if model in {"LeWM2D", "JEPA4R-PCD192"} else 4)
                       for model in MODEL_ORDER}
    if dict(model_counts) != expected_counts:
        errors.append(
            f"model group counts mismatch: {dict(model_counts)} != {expected_counts}"
        )

    for key, record in planned:
        config_path = Path(record["config"])
        try:
            family = record["family"]
            if not config_path.is_file():
                raise FileNotFoundError(config_path)
            config = read_json(config_path)
            if config.get("group_id") != record["group_id"]:
                raise RuntimeError(
                    f"config group id {config.get('group_id')} != {record['group_id']}"
                )
            if config["input_contract"]["current_dim"] != record["current_dim"]:
                raise RuntimeError("config current_dim mismatch")
            if config["input_contract"]["goal_dim"] != record["goal_dim"]:
                raise RuntimeError("config goal_dim mismatch")
            if config["modality"] != record["mode"]:
                raise RuntimeError("config modality mismatch")
            if "native_max_episode_steps" in config:
                if config["native_max_episode_steps"] != record["max_steps"]:
                    raise RuntimeError("config max_episode_steps mismatch")
            if family == "pldm_binding":
                binding = read_json(Path(record["binding"]))
                if binding["group"]["max_episode_steps"] != record["max_steps"]:
                    raise RuntimeError("PLDM binding max_episode_steps mismatch")
            if Path(config["output"]).name != "" and not Path(config["output"]).is_dir():
                raise FileNotFoundError(config["output"])
            acceptance_file = acceptance_path(key, record)
            acceptance = read_json(acceptance_file)
            if acceptance.get("actual_exit_code") != 0:
                raise RuntimeError("source 1M acceptance is not successful")
            checkpoint_rows = {}
            for update in UPDATES:
                checkpoint_rows[str(update)] = checkpoint_facts(
                    config, update, acceptance
                )
            one_m = checkpoint_facts(config, 1000000, acceptance)
            rows.append(
                {
                    "key": key,
                    "model": record["model"],
                    "cube": record["cube"],
                    "group_id": record["group_id"],
                    "config": str(config_path),
                    "config_sha256": sha(config_path),
                    "acceptance": str(acceptance_file),
                    "acceptance_sha256": sha(acceptance_file),
                    "updates": checkpoint_rows,
                    "one_m": one_m,
                }
            )
            print(
                f"PASS {record['model']} Cube{record['cube']} "
                f"{record['group_id']} {UPDATES}",
                flush=True,
            )
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
            print(f"FAIL {key}: {type(exc).__name__}: {exc}", flush=True)

    result = {
        "actual_exit_code": 0 if not errors else 1,
        "scope": "22 available groups x 2 selected checkpoints",
        "planned_groups": len(planned),
        "planned_evaluations": len(planned) * len(UPDATES),
        "planned_episodes_native250": len(planned) * len(UPDATES) * 250,
        "updates": list(UPDATES),
        "model_group_counts": dict(model_counts),
        "errors": errors,
        "groups": rows,
        "started": started,
        "completed": time.time(),
    }
    atomic_json(SWEEP_ROOT / "control/PREFLIGHT.json", result)
    print(json.dumps({key: result[key] for key in (
        "actual_exit_code",
        "planned_groups",
        "planned_evaluations",
        "planned_episodes_native250",
        "errors",
    )}, sort_keys=True))
    return int(result["actual_exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
