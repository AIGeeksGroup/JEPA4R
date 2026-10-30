"""Supervise one frozen PLDM stage with persistent status, logs, and exit code."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf8")
    temp.replace(path)


def resource_sample(pid):
    record = {"at": now(), "pid": pid}
    try:
        record["gpu"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
    except Exception as error:
        record["gpu_error"] = repr(error)
    try:
        record["process"] = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "etime=,%cpu=,%mem=,rss=,args="],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
    except Exception as error:
        record["process_error"] = repr(error)
    try:
        record["disk_tmp"] = subprocess.check_output(
            ["df", "-h", "/root/autodl-tmp"],
            text=True,
            timeout=10,
        ).splitlines()[-1].strip()
    except Exception as error:
        record["disk_error"] = repr(error)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("gate", "formal"), required=True)
    parser.add_argument("--case-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--interval", type=int, default=15)
    args = parser.parse_args()

    case = args.case_root.resolve()
    source = args.source_root.resolve()
    data = args.data_root.resolve()
    stage_root = case / "stages" / args.stage
    state_root = case / "run_state" / args.stage
    native_output = stage_root / "native_output"
    if state_root.exists() or stage_root.exists():
        raise FileExistsError(f"preserving existing stage path: {stage_root}")
    state_root.mkdir(parents=True)
    stage_root.mkdir(parents=True)

    args_file = state_root / "config.json"
    status_file = state_root / "status.json"
    exit_file = state_root / "actual_exit_code"
    log_file = state_root / "stdout_stderr.log"
    samples_file = state_root / "resources.jsonl"
    source_freeze = source / "source_freeze_v1.json"
    run_tag = f"single_cube_seed0_{args.stage}_v1"
    argv = [
        str(args.python),
        "-B",
        str(source / "train.py"),
        "--root",
        str(source),
        "--data-root",
        str(data),
        "--output",
        str(native_output),
        "--method",
        "pldm",
        "--run-tag",
        run_tag,
        "--phase",
        args.stage,
        "--mode",
        "train",
    ]
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "HYDRA_FULL_ERROR": "1",
            "PYTHONPATH": os.pathsep.join(
                [
                    str(source),
                    str(source / "vendor" / "pldm" / "source"),
                    env.get("PYTHONPATH", ""),
                ]
            ),
        }
    )
    config = {
        "schema": "single_cube_pldm_rgb_stage_config_v1",
        "stage": args.stage,
        "created_at": now(),
        "argv": argv,
        "cwd": str(source),
        "data_root": str(data),
        "source_freeze_sha256": digest(source_freeze),
        "data_manifest_sha256": digest(data / "data_views" / "training_view_index.json"),
        "native_output": str(native_output),
        "python": str(args.python),
        "sampling_interval_seconds": args.interval,
        "seed": 0,
        "formal_uses_gate_checkpoint": False,
        "environment": {
            key: env[key]
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "HYDRA_FULL_ERROR",
                "PYTHONPATH",
            )
        },
    }
    atomic_json(args_file, config)
    state = {
        "stage": f"pldm_{args.stage}_starting",
        "started_at": now(),
        "updated_at": now(),
        "actual_exit_code": None,
        "pid": None,
        "config_path": str(args_file),
        "config_sha256": digest(args_file),
        "runner_sha256": digest(__file__),
        "source_freeze_sha256": config["source_freeze_sha256"],
        "data_manifest_sha256": config["data_manifest_sha256"],
        "argv": argv,
        "failed_stages": 0,
        "error": None,
    }
    atomic_json(status_file, state)
    try:
        with log_file.open("ab", buffering=0) as log:
            child = subprocess.Popen(
                argv,
                cwd=source,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            state["pid"] = child.pid
            state["pid_created_at"] = Path(f"/proc/{child.pid}").stat().st_ctime
            state["stage"] = f"pldm_{args.stage}_running"
            state["updated_at"] = now()
            atomic_json(status_file, state)
            with samples_file.open("a", encoding="utf8") as samples:
                while True:
                    samples.write(
                        json.dumps(resource_sample(child.pid), sort_keys=True) + "\n"
                    )
                    samples.flush()
                    code = child.poll()
                    if code is not None:
                        state["actual_exit_code"] = int(code)
                        state["stage"] = (
                            f"pldm_{args.stage}_{'completed' if code == 0 else 'failed'}"
                        )
                        state["finished_at"] = now()
                        state["updated_at"] = state["finished_at"]
                        atomic_json(status_file, state)
                        exit_file.write_text(str(code) + "\n", encoding="ascii")
                        return code
                    state["updated_at"] = now()
                    atomic_json(status_file, state)
                    time.sleep(max(1, args.interval))
    except Exception:
        state.update(
            stage=f"pldm_{args.stage}_supervisor_failed",
            actual_exit_code=1,
            failed_stages=1,
            error=traceback.format_exc(),
            finished_at=now(),
        )
        atomic_json(status_file, state)
        exit_file.write_text("1\n", encoding="ascii")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
