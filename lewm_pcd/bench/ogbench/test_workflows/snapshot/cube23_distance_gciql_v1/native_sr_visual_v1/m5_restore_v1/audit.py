"""Reuse the existing small result auditor with CubeN dimensions and true2N checks."""
from pathlib import Path
import json

# Original continuation_controller.verify_report; scheduler is not copied.
def verify_report(base: Path, group: str, mode: str, output: Path, expected: int, goal_dim: int, max_steps: int) -> dict:
    report = json.loads((output / "report.json").read_text())
    if (int((output / "actual_exit_code").read_text()) != 0 or
            report.get("actual_exit_code") != 0 or report.get("stage") not in (None, "complete") or
            report.get("group_id") != group or report.get("mode") != mode or
            report.get("episodes") != expected or report.get("unique_seeds") != expected or
            report.get("current_dim") != 211 or report.get("goal_dim") != goal_dim or report.get("max_episode_steps") != max_steps or
            report.get("encoding_protocol") != 'fresh_current_and_goal_each_policy_call_2N' or report.get("online_encoder") is not True or
            report.get("offline_embedding_cache_used_for_policy") is not False or
            report.get("synthetic_zero_padding") is not False):
        raise RuntimeError(f"{group} {mode} aggregate report failed contract checks")
    successes = 0
    successes_by_task = {}
    seeds = set()
    for task in range(1, 6):
        count = min(50, expected - 50 * (task - 1))
        if count <= 0:
            break
        task_successes = 0
        for index in range(count):
            episode = output / "episodes" / f"task{task}" / f"episode_{index:03d}"
            item = json.loads((episode / "result.json").read_text())
            seed = 970000000 + task * 100000 + index
            if (int((episode / "attempt_exit_code").read_text()) != 0 or
                    item.get("actual_exit_code") != 0 or item.get("group_id") != group or
                    item.get("task_id") != task or item.get("episode_index") != index or
                    item.get("seed") != seed or not 1 <= item.get("steps", 0) <= max_steps or
                    item.get("current_goal_contract") != {"current_dim": 211, "goal_dim": goal_dim,
                                                           "synthetic_zero_padding": False} or
                    item.get("temperature") != 0.0 or item.get("gaussian_noise") is not None or
                    item.get("goal_online_encoder_calls") != item['steps'] or item.get("online_encoder_calls") != 2 * item['steps'] or item.get("encoding_protocol") != 'fresh_current_and_goal_each_policy_call_2N' or
                    item.get("original_evaluate_reused") is not True):
                raise RuntimeError(f"{group} episode contract failed: task={task} index={index}")
            if len(item.get("policy_input_contracts", [])) != item["steps"]:
                raise RuntimeError(f"{group} input-contract count mismatch: task={task} index={index}")
            for row in item["policy_input_contracts"]:
                if (row.get("current_dim") != 211 or row.get("goal_dim") != goal_dim or
                        row.get("current_finite") is not True or row.get("goal_finite") is not True):
                    raise RuntimeError(f"{group} non-finite/wrong-width policy input")
            if group in ('G06', 'G09'):
                proof = item.get("goal_capture_proof", {})
                if (proof.get("physics_unchanged") is not True or
                        proof.get("env_rng_unchanged") is not True or
                        proof.get("geometry_offsamples") != 0 or
                        item.get("camera", {}).get("geometry_offsamples") != 0):
                    raise RuntimeError("G06 PCD goal-capture invariants failed")
            seeds.add(seed)
            task_successes += int(item["success"])
            successes += int(item["success"])
        successes_by_task[str(task)] = task_successes
    if len(seeds) != expected or successes != report.get("successes"):
        raise RuntimeError(f"{group} per-episode count/seed/success summary mismatch")
    if abs(successes / expected - float(report.get("success_rate", -1))) > 1e-12:
        raise RuntimeError(f"{group} success-rate summary mismatch")
    return {"episodes": expected, "successes": successes,
            "success_rate": successes / expected, "unique_seeds": len(seeds),
            "successes_by_task": successes_by_task}
