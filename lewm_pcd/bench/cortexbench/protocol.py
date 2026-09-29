"""Small, testable pieces of the official CortexBench scheduling contract."""

import math
from typing import Any, Dict, Iterable, Optional


_SUITE_NUM_DEMOS = {
    "adroit": 100,
    "metaworld": 25,
}


def build_job_contract(suite: str, env: str, model: str, seed: int, data_dir: str, run_dir: str,
                       is_pcd: bool) -> Dict[str, Any]:
    return {
        "model": model,
        "suite": suite,
        "env": env,
        "seed": seed,
        "demos": data_dir,
        "run": run_dir,
        "num_demos": num_demos_for_suite(suite),
        "epochs": 100,
        "eval_frequency": 5,
        "eval_num_traj": 25,
        "result_scope": "PCD extension" if is_pcd else "official RGB",
    }


def num_demos_for_suite(suite: str, default: Optional[int] = None) -> Optional[int]:
    """Return the explicit demo count for a suite, preserving other defaults."""
    return _SUITE_NUM_DEMOS.get(suite, default)


def expected_evaluator_records(epochs: int, eval_frequency: int) -> int:
    """Match the official train loop's periodic plus final evaluation schedule."""
    return sum(
        1
        for epoch in range(epochs)
        if (epoch % eval_frequency == 0 and epoch > 0) or epoch == epochs - 1
    )


def expected_evaluator_epochs(epochs: int, eval_frequency: int):
    return [
        epoch
        for epoch in range(epochs)
        if (epoch % eval_frequency == 0 and epoch > 0) or epoch == epochs - 1
    ]


def validate_metric_records(
    records: Iterable[Dict[str, Any]], epochs: int, eval_frequency: int
) -> Dict[str, Any]:
    """Validate the official logger output and extract best-of-evaluation metrics."""
    records = list(records)
    losses = [record for record in records if "epoch_loss" in record]
    evaluations = [record for record in records if "eval/epoch" in record]
    if len(losses) != epochs:
        raise ValueError(f"expected {epochs} loss records, found {len(losses)}")
    expected_epochs = expected_evaluator_epochs(epochs, eval_frequency)
    if len(evaluations) != len(expected_epochs):
        raise ValueError(
            f"expected {len(expected_epochs)} evaluator records, found {len(evaluations)}"
        )
    actual_epochs = [int(float(record["eval/epoch"])) for record in evaluations]
    if actual_epochs != expected_epochs:
        raise ValueError(
            f"evaluator epochs mismatch: expected {expected_epochs}, found {actual_epochs}"
        )
    for record in losses:
        if not math.isfinite(float(record["epoch_loss"])):
            raise ValueError("epoch loss contains NaN or Inf")
    required = ("eval/success", "eval/highest_success", "eval/score_mean", "eval/highest_score")
    for record in evaluations:
        for key in required:
            if key not in record or not math.isfinite(float(record[key])):
                raise ValueError(f"evaluation record has missing/non-finite {key}")
    return {
        "loss_records": len(losses),
        "eval_records": len(evaluations),
        "best_success": max(float(record["eval/success"]) for record in evaluations),
        "final_success": float(evaluations[-1]["eval/success"]),
        "best_score": max(float(record["eval/score_mean"]) for record in evaluations),
        "final_score": float(evaluations[-1]["eval/score_mean"]),
    }
