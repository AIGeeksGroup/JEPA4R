"""Aggregate the completed 800K/900K checkpoint sweep without extrapolation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from registry import GROUPS, MODEL_ORDER, UPDATES, selected_groups
from sweep_common import SWEEP_ROOT, atomic_json, read_json


REFERENCE_1M = {
    ("LeWM2D", 1): (61, 250),
    ("LeWM2D", 2): (3, 250),
    ("LeWM2D", 3): (0, 250),
    ("LeWM2D", 4): (0, 250),
    ("JEPA4R-PCD192", 1): (196, 250),
    ("JEPA4R-PCD192", 2): (104, 250),
    ("JEPA4R-PCD192", 3): (46, 250),
    ("JEPA4R-PCD192", 4): (0, 250),
    ("LeJEPA-Tiny512", 1): (0, 250),
    ("LeJEPA-Tiny512", 2): (0, 250),
    ("LeJEPA-Tiny512", 3): (0, 250),
    ("LeJEPA-Tiny512", 4): (0, 250),
    ("PLDM512", 1): (117, 250),
    ("PLDM512", 2): (9, 250),
    ("PLDM512", 3): (12, 250),
    ("PLDM512", 4): (0, 250),
    ("VISReg-ViT-B768", 1): (3, 250),
    ("VISReg-ViT-B768", 2): (0, 250),
    ("VISReg-ViT-B768", 3): (0, 250),
    ("VISReg-ViT-B768", 4): (0, 250),
    ("V-JEPA-ViT-L1024", 1): (15, 250),
    ("V-JEPA-ViT-L1024", 2): (1, 250),
    ("V-JEPA-ViT-L1024", 3): (0, 250),
    ("V-JEPA-ViT-L1024", 4): (0, 250),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SWEEP_ROOT / "FINAL_RESULTS.json")
    args = parser.parse_args()
    rows = []
    errors = []
    for key, record in selected_groups():
        for update in UPDATES:
            output = SWEEP_ROOT / "results/full" / key / f"u{update:09d}"
            audit_path = output / "independent_audit.json"
            binding_path = output / "native_source_binding.json"
            try:
                audit = read_json(audit_path)
                binding = read_json(binding_path)
                if audit.get("actual_exit_code") != 0:
                    raise RuntimeError("audit exit is not zero")
                if binding.get("actual_exit_code") != 0:
                    raise RuntimeError("binding exit is not zero")
                if audit.get("episodes") != 250 or audit.get("unique_seeds") != 250:
                    raise RuntimeError("Native250 count mismatch")
                if binding.get("selected_update") != update:
                    raise RuntimeError("selected update mismatch")
                if audit.get("actual_step_arrays_checked") is not True:
                    raise RuntimeError("step arrays not audited")
                rows.append(
                    {
                        "model": record["model"],
                        "cube": record["cube"],
                        "group_key": key,
                        "group_id": record["group_id"],
                        "update": update,
                        "episodes": audit["episodes"],
                        "unique_seeds": audit["unique_seeds"],
                        "successes": audit["successes"],
                        "success_rate": audit["success_rate"],
                        "successes_by_task": audit["successes_by_task"],
                        "selected_checkpoint": audit["selected_checkpoint"],
                        "selected_manifest_sha256": audit["selected_manifest_sha256"],
                        "native_source_binding_sha256": audit[
                            "native_source_binding_sha256"
                        ],
                    }
                )
            except Exception as exc:
                errors.append(f"{key} update={update}: {type(exc).__name__}: {exc}")

    missing_models = {
        ("LeWM2D", 1),
        ("JEPA4R-PCD192", 1),
    }
    expected = {
        (record["model"], record["cube"], update)
        for _, record in selected_groups()
        for update in UPDATES
    }
    observed = {(row["model"], row["cube"], row["update"]) for row in rows}
    if observed != expected:
        errors.append(
            f"result key mismatch: missing={sorted(expected-observed)} "
            f"unexpected={sorted(observed-expected)}"
        )

    matrix = []
    for model in MODEL_ORDER:
        for cube in range(1, 5):
            values = {
                str(row["update"]): row
                for row in rows
                if row["model"] == model and row["cube"] == cube
            }
            ref = REFERENCE_1M[(model, cube)]
            matrix.append(
                {
                    "model": model,
                    "cube": cube,
                    "800000": values.get("800000"),
                    "900000": values.get("900000"),
                    "1000000_reference": {
                        "successes": ref[0],
                        "episodes": ref[1],
                        "success_rate": ref[0] / ref[1],
                        "source": "accepted prior 1M result; not rerun in this sweep",
                    },
                    "checkpoint_availability": (
                        "NOT_AVAILABLE_IN_CURRENT_SERVER"
                        if (model, cube) in missing_models
                        else "COMPLETE"
                    ),
                }
            )
    result = {
        "actual_exit_code": 0 if not errors else 1,
        "scope": "22 available groups; 800K/900K Native250; current+goal online per policy call",
        "formal_evaluations": len(rows),
        "formal_episodes": sum(row["episodes"] for row in rows),
        "expected_evaluations": len(expected),
        "expected_episodes": len(expected) * 250,
        "updates": list(UPDATES),
        "missing_checkpoint_groups": [
            {"model": model, "cube": cube}
            for model, cube in sorted(missing_models)
        ],
        "errors": errors,
        "rows": rows,
        "matrix": matrix,
        "diagnostics_excluded": True,
        "old_1m_results_overwritten": False,
    }
    atomic_json(args.output, result)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "model",
                "cube",
                "group_id",
                "update",
                "successes",
                "episodes",
                "success_rate",
                "selected_checkpoint",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    print(json.dumps({key: result[key] for key in (
        "actual_exit_code",
        "formal_evaluations",
        "formal_episodes",
        "expected_evaluations",
        "expected_episodes",
        "errors",
    )}, sort_keys=True))
    return int(result["actual_exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
