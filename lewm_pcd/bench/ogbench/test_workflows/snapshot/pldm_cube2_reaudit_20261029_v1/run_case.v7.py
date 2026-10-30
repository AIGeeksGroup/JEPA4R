"""Serial, no-retry runner for the independent PLDM Cube2 re-audit case."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1")
CASE = ROOT / "pldm_cube2_reaudit_20261029_v1"
PY = ROOT / "env_runtime/bin/python"
GPU_LEASE = ROOT / "cube4_visual_gciql_v1/control_v1/gpu_lease.py"
GROUP = "P2_cube2_pldm_reaudit_rgb"
CONFIG = CASE / "functional_v1/configs/P2_cube2_pldm_reaudit_rgb.v1.json"
TRAIN = CASE / "runs" / GROUP
RUNTIME = CASE / "control/run/runtime"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def env(kind: str) -> dict[str, str]:
    result = os.environ.copy()
    result.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        JAX_THREEFRY_PARTITIONABLE="false",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_FLAGS="--xla_gpu_autotune_level=0",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
    )
    runtime_lib = str(ROOT / "runtime/lib")
    prior_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    result["LD_LIBRARY_PATH"] = runtime_lib + (
        ":" + prior_library_path if prior_library_path else ""
    )
    if kind in ("cpu", "native"):
        result["JAX_PLATFORMS"] = "cpu"
    if kind == "cpu":
        result["CUDA_VISIBLE_DEVICES"] = ""
    if kind == "gpu":
        result["CUDA_VISIBLE_DEVICES"] = "0"
        result["JAX_PLATFORMS"] = "cuda,cpu"
        result["MUJOCO_GL"] = "egl"
    if kind == "native":
        result["CUDA_VISIBLE_DEVICES"] = "0"
        result["MUJOCO_GL"] = "egl"
    return result


def stage_commands() -> dict[str, tuple[list[str], str, dict]]:
    cache = CASE / "cache_v1"
    func = CASE / "functional_v1"
    plan = cache / "producer_plan.json"
    cache_dir = cache / "cache"
    checks = CASE / "acceptance"
    native = CASE / "native_sr_v1"
    common = [str(PY), "-B"]
    return {
        "cache-prepare": (
            common + [str(cache / "prepare_plan.py"), "--case", str(CASE)],
            "cpu",
            {"path": cache / "PREPARED.json", "equals": {"actual_exit_code": 0, "episodes": 1100, "canonical_rows": 1101100}},
        ),
        "cache-pilot": (
            common + [str(GPU_LEASE), "--"] + common + [str(cache / "producer_bridge.py"), "--plan", str(plan), "--execute", "--pilot"],
            "gpu",
            {"path": cache_dir / "pilot_receipt.json", "equals": {"actual_exit_code": 0}},
        ),
        "cache-full": (
            common + [str(GPU_LEASE), "--"] + common + [str(cache / "producer_bridge.py"), "--plan", str(plan), "--execute"],
            "gpu",
            {"path": cache_dir / "cache_seal.json", "equals": {"actual_exit_code": 0, "train_rows": 1001000, "val_rows": 100100, "full_raw_ID_coverage": True}},
        ),
        "cache-audit": (
            common + [str(cache / "audit_bridge.py"), "--cache-dir", str(cache_dir), "--plan", str(plan), "--output-dir", str(cache / "independent_audit")],
            "cpu",
            {"path": cache / "independent_audit/report.json", "equals": {"actual_exit_code": 0, "episodes_verified": 1100, "raw_rows_verified": 1101100, "main_rows_verified": 1100000, "latent_dim": 512}},
        ),
        "verify-hooks": (
            common + [str(func / "pldm_runtime.py"), "verify-hooks", "--report", str(func / "CPU_HOOK_VERIFY_v1.json")],
            "cpu",
            {"path": func / "CPU_HOOK_VERIFY_v1.json", "equals": {"actual_exit_code": 0, "boundary_prepare_identity": True, "boundary_get_batch_identity": True, "backend": "cpu", "sampler_called": False, "agent_or_encoder_created": False}},
        ),
        "prepare-train": (
            common + [str(func / "prepare.py"), "train", "--case", str(CASE)],
            "cpu",
            {"path": func / "TRAIN_CONFIG_PREPARED_v1.json", "equals": {"actual_exit_code": 0, "current_dim": 531, "goal_dim": 518, "cache_episodes": 1100, "cache_raw_rows": 1101100}},
        ),
        "gate": (
            common + [str(GPU_LEASE), "--"] + common + [str(func / "pldm_runtime.py"), "gate", "--config", str(CONFIG), "--output", str(CASE / "gates" / GROUP)],
            "gpu",
            {"path": CASE / "gates" / GROUP / "report.json", "equals": {"actual_exit_code": 0, "updates": 2, "network_step": 3, "adam_count": 2, "batch_size": 1024, "encoder_calls": 0}},
        ),
        "reload": (
            common + [str(func / "pldm_runtime.py"), "reload", "--config", str(CONFIG), "--output", str(CASE / "gates" / GROUP), "--report", str(checks / f"{GROUP}_gate_reload.json")],
            "cpu",
            {"path": checks / f"{GROUP}_gate_reload.json", "equals": {"actual_exit_code": 0, "full153_state_exact": True, "next_batch_and_rawIDs_exact": True, "finite_actions": True}},
        ),
        "train": (
            common + [str(GPU_LEASE), "--"] + common + [str(func / "pldm_runtime.py"), "train", "--config", str(CONFIG)],
            "gpu",
            {"path": TRAIN / "status.json", "equals": {"exit_code": 0, "completed_updates": 1000000, "adam_count": 1000000, "network_step": 1000001, "validation_count": 200}},
        ),
        "accept": (
            common + [str(func / "pldm_runtime.py"), "accept", "--config", str(CONFIG), "--output", str(TRAIN), "--report", str(checks / f"{GROUP}_full_acceptance.json")],
            "cpu",
            {"path": checks / f"{GROUP}_full_acceptance.json", "equals": {"actual_exit_code": 0, "full153_state_exact": True, "Adam_count": 1000000, "network_step": 1000001, "CSV_validation_records": 200, "actual_current_dim": 531, "actual_goal_dim": 518, "synthetic_zero_padding": False}},
        ),
        "prepare-native": (
            common + [str(func / "prepare.py"), "native", "--case", str(CASE)],
            "cpu",
            {"path": func / "NATIVE_PREPARED_v1.json", "equals": {"actual_exit_code": 0, "final_counter": 1000000, "current_dim": 531, "goal_dim": 518, "checkpoint_transactions": 11}},
        ),
        "native-tiny": (
            common + [str(GPU_LEASE), "--"] + common + [str(func / "native_sr.py"), "--case", str(CASE), "--mode", "tiny", "--output", str(native / "tiny")],
            "native",
            {"path": native / "tiny/report.json", "equals": {"actual_exit_code": 0, "episodes": 1, "unique_seeds": 1, "current_dim": 531, "goal_dim": 518, "max_episode_steps": 4, "offline_embedding_cache_used_for_policy": False}},
        ),
        "native-tiny-audit": (
            common + [str(func / "audit_native.py"), "--case", str(CASE), "--mode", "tiny", "--output", str(native / "tiny")],
            "cpu",
            {"path": native / "tiny/independent_audit.json", "equals": {"actual_exit_code": 0, "episodes": 1, "unique_seeds": 1, "current_dim": 531, "goal_dim": 518, "diagnostic_excluded_from_formal_SR": True, "fresh_goal_then_current_encoding_required": True}},
        ),
        "native-full": (
            common + [str(GPU_LEASE), "--"] + common + [str(func / "native_sr.py"), "--case", str(CASE), "--mode", "full", "--output", str(native / "full")],
            "native",
            {"path": native / "full/report.json", "equals": {"actual_exit_code": 0, "episodes": 250, "unique_seeds": 250, "current_dim": 531, "goal_dim": 518, "max_episode_steps": 500, "offline_embedding_cache_used_for_policy": False}},
        ),
        "native-full-audit": (
            common + [str(func / "audit_native.py"), "--case", str(CASE), "--mode", "full", "--output", str(native / "full")],
            "cpu",
            {"path": native / "full/independent_audit.json", "equals": {"actual_exit_code": 0, "episodes": 250, "unique_seeds": 250, "current_dim": 531, "goal_dim": 518, "actual_step_arrays_checked": True}},
        ),
    }


STAGES = list(stage_commands())


def check(path: Path, expected: dict) -> None:
    value = read_json(path)
    for key, wanted in expected.items():
        actual = value.get(key)
        if actual != wanted:
            raise ValueError(f"{path}: {key}={actual!r}, expected {wanted!r}")


def run_one(name: str) -> None:
    argv, kind, check_spec = stage_commands()[name]
    directory = RUNTIME / name
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite preserved stage evidence: {directory}")
    directory.mkdir(parents=True)
    write_json_exclusive(directory / "argv.json", argv)
    write_json_exclusive(directory / "environment.json", env(kind))
    started = time.time()
    with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
        completed = subprocess.run(argv, cwd=ROOT, env=env(kind), stdout=stdout, stderr=stderr)
    write_json_exclusive(
        directory / "result.json",
        {"stage": name, "started": started, "completed": time.time(), "actual_exit_code": completed.returncode, "automatic_retry": False},
    )
    (directory / "actual_exit_code").write_text(str(completed.returncode) + "\n", encoding="ascii")
    if completed.returncode:
        raise SystemExit(completed.returncode)
    check(check_spec["path"], check_spec["equals"])


def finalize() -> None:
    report = read_json(CASE / "native_sr_v1/full/report.json")
    audit = read_json(CASE / "native_sr_v1/full/independent_audit.json")
    acceptance = read_json(CASE / "acceptance" / f"{GROUP}_full_acceptance.json")
    if not (report["episodes"] == audit["episodes"] == 250):
        raise ValueError("formal episode counts disagree")
    if not audit["actual_step_arrays_checked"]:
        raise ValueError("full step-array audit did not actually check arrays")
    result = {
        "schema_version": 1,
        "actual_exit_code": 0,
        "group_id": GROUP,
        "cube_count": 2,
        "training_updates": 1000000,
        "formal_episodes": 250,
        "successes": report["successes"],
        "success_rate": report["success_rate"],
        "current_dim": 531,
        "goal_dim": 518,
        "native_max_episode_steps": 500,
        "online_current_and_goal_2N": True,
        "actual_step_arrays_checked": True,
        "cpu_acceptance": acceptance,
        "final_report_sha256": hashlib.sha256((CASE / "native_sr_v1/full/report.json").read_bytes()).hexdigest(),
        "independent_audit_sha256": hashlib.sha256((CASE / "native_sr_v1/full/independent_audit.json").read_bytes()).hexdigest(),
        "D_backup_acceptance_pending": True,
    }
    write_json_exclusive(CASE / "FINAL_RESULTS.json", result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES + ["all", "finalize"], required=True)
    parser.add_argument("--single-lock", action="store_true")
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = CASE / "control/run/single_run.lock"
    if args.single_lock:
        with lock.open("x", encoding="ascii") as stream:
            stream.write(str(os.getpid()) + "\n")
    if args.stage == "all":
        for name in STAGES:
            run_one(name)
        finalize()
    elif args.stage == "finalize":
        finalize()
    else:
        run_one(args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
