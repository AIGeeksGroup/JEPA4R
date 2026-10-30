"""Persistent serial driver for 44 selected-checkpoint evaluations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from registry import UPDATES, selected_groups
from sweep_common import ROOT, SWEEP_ROOT, atomic_json, read_json

PYTHON = ROOT / "env_runtime/bin/python"
CONTROL = SWEEP_ROOT / "control"
RESULTS = SWEEP_ROOT / "results"
LOGS = SWEEP_ROOT / "logs"


def jobs():
    return [
        (key, record, update)
        for key, record in selected_groups()
        for update in UPDATES
    ]


def job_name(key: str, update: int) -> str:
    return f"{key}_u{update:09d}"


def environment() -> dict:
    value = os.environ.copy()
    value.update(
        LD_LIBRARY_PATH=f"{ROOT}/runtime/lib:/usr/lib/x86_64-linux-gnu",
        MUJOCO_GL="egl",
        JAX_PLATFORMS="cpu",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return value


def invoke(argv: list[str], stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("x") as stdout, stderr_path.open("x") as stderr:
        process = subprocess.run(
            argv,
            cwd=str(SWEEP_ROOT),
            env=environment(),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return int(process.returncode)


def phase_paths(phase: str, key: str, update: int) -> tuple[Path, Path, Path]:
    name = job_name(key, update)
    output = RESULTS / phase / key / f"u{update:09d}"
    stdout = LOGS / phase / f"{name}.stdout.log"
    stderr = LOGS / phase / f"{name}.stderr.log"
    return output, stdout, stderr


def preserve_previous_logs(phase: str, name: str) -> None:
    candidates = [
        LOGS / phase / f"{name}.stdout.log",
        LOGS / phase / f"{name}.stderr.log",
        LOGS / phase / f"{name}.audit.stdout.log",
        LOGS / phase / f"{name}.audit.stderr.log",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return
    destination = LOGS / "failed_attempts" / phase / name
    if destination.exists():
        raise FileExistsError(f"failed-attempt log preservation exists: {destination}")
    destination.mkdir(parents=True)
    for path in existing:
        path.rename(destination / path.name)


def existing_success(output: Path, key: str, update: int, phase: str) -> bool:
    audit_path = output / "independent_audit.json"
    binding_path = output / "native_source_binding.json"
    if not audit_path.is_file() or not binding_path.is_file():
        return False
    audit = read_json(audit_path)
    binding = read_json(binding_path)
    expected_mode = "tiny" if phase == "tiny" else "full"
    return (
        audit.get("actual_exit_code") == 0
        and audit.get("episodes") == (1 if phase == "tiny" else 250)
        and audit.get("unique_seeds") == (1 if phase == "tiny" else 250)
        and audit.get("actual_step_arrays_checked") is True
        and binding.get("actual_exit_code") == 0
        and binding.get("selected_update") == update
        and binding.get("requested_mode") == expected_mode
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("tiny", "full"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    CONTROL.mkdir(parents=True, exist_ok=True)
    phase_control = CONTROL / args.phase
    status_path = phase_control / "STATUS.json"
    planned = jobs()
    if phase_control.exists():
        if not args.resume or not status_path.is_file():
            raise FileExistsError(f"refusing existing phase control: {phase_control}")
        status = read_json(status_path)
        if status.get("phase") != args.phase or status.get("state") not in ("failed", "running"):
            raise RuntimeError("existing phase status is not resumable")
        status.update(
            actual_exit_code=None,
            state="running",
            updated=time.time(),
            resumed_at=time.time(),
        )
        status.pop("error", None)
        status.pop("error_type", None)
        status.pop("traceback", None)
    else:
        phase_control.mkdir(parents=True, exist_ok=False)
        status = {
            "actual_exit_code": None,
            "state": "running",
            "phase": args.phase,
            "started": time.time(),
            "updated": time.time(),
            "completed_jobs": 0,
            "failed_jobs": 0,
            "total_jobs": len(planned),
            "last_job": None,
            "automatic_retry": False,
        }
    atomic_json(status_path, status)
    try:
        preflight = read_json(CONTROL / "PREFLIGHT.json")
        if preflight.get("actual_exit_code") != 0 or preflight.get("planned_evaluations") != len(planned):
            raise RuntimeError("preflight is not successful")
        for index, (key, record, update) in enumerate(planned, 1):
            output, stdout, stderr = phase_paths(args.phase, key, update)
            name = job_name(key, update)
            status.update(
                state="running",
                updated=time.time(),
                current_job=name,
                current_index=index,
                current_model=record["model"],
                current_cube=record["cube"],
                current_update=update,
                current_output=str(output),
            )
            atomic_json(status_path, status)
            if existing_success(output, key, update, args.phase):
                status.update(
                    completed_jobs=index,
                    last_job=name,
                    last_actual_exit_code=0,
                    updated=time.time(),
                    resumed_existing=True,
                )
                atomic_json(status_path, status)
                continue
            if output.exists():
                raise RuntimeError(f"existing incomplete output must be inspected: {output}")
            preserve_previous_logs(args.phase, name)

            run_argv = [
                str(PYTHON),
                "-B",
                str(SWEEP_ROOT / "run_one.py"),
                "--group",
                key,
                "--update",
                str(update),
                "--mode",
                args.phase,
                "--output",
                str(output),
            ]
            run_code = invoke(run_argv, stdout, stderr)
            atomic_json(
                phase_control / "jobs" / f"{name}.run_exit.json",
                {
                    "actual_exit_code": run_code,
                    "argv": run_argv,
                    "stdout": str(stdout),
                    "stderr": str(stderr),
                    "completed": time.time(),
                },
            )
            if run_code != 0:
                raise RuntimeError(f"{name}: evaluator actual_exit_code={run_code}")

            audit_argv = [
                str(PYTHON),
                "-B",
                str(SWEEP_ROOT / "audit_one.py"),
                "--group",
                key,
                "--update",
                str(update),
                "--mode",
                args.phase,
                "--output",
                str(output),
            ]
            audit_code = invoke(
                audit_argv,
                LOGS / args.phase / f"{name}.audit.stdout.log",
                LOGS / args.phase / f"{name}.audit.stderr.log",
            )
            atomic_json(
                phase_control / "jobs" / f"{name}.audit_exit.json",
                {
                    "actual_exit_code": audit_code,
                    "argv": audit_argv,
                    "completed": time.time(),
                },
            )
            if audit_code != 0:
                raise RuntimeError(f"{name}: independent audit actual_exit_code={audit_code}")

            audit = read_json(output / "independent_audit.json")
            status.update(
                completed_jobs=index,
                last_job=name,
                last_actual_exit_code=0,
                last_successes=audit["successes"],
                last_success_rate=audit["success_rate"],
                updated=time.time(),
                resumed_existing=False,
            )
            atomic_json(status_path, status)
        if args.phase == "full":
            summary_stdout = LOGS / "full.summary.stdout.log"
            summary_stderr = LOGS / "full.summary.stderr.log"
            summary_code = invoke(
                [
                    str(PYTHON),
                    "-B",
                    str(SWEEP_ROOT / "summarize.py"),
                    "--output",
                    str(SWEEP_ROOT / "FINAL_RESULTS.json"),
                ],
                summary_stdout,
                summary_stderr,
            )
            if summary_code != 0:
                raise RuntimeError(
                    f"FINAL_RESULTS aggregation actual_exit_code={summary_code}"
                )
        status.update(
            actual_exit_code=0,
            state="complete",
            updated=time.time(),
            completed=time.time(),
        )
        atomic_json(status_path, status)
        return 0
    except BaseException as exc:
        status.update(
            actual_exit_code=1,
            state="failed",
            updated=time.time(),
            failed_jobs=status.get("failed_jobs", 0) + 1,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
