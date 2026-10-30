"""Run gate, gate acceptance, formal 10 epochs, then formal acceptance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
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


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_logged(argv, *, cwd, env, stdout_path, stderr_path):
    with Path(stdout_path).open("wb") as stdout, Path(stderr_path).open("wb") as stderr:
        result = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return result.returncode


def run_acceptance(phase, *, case, source, data, python, environment):
    accept_root = case / "stages" / f"accept_{phase}"
    state_root = case / "run_state" / f"accept_{phase}"
    native_output = accept_root / "native_output"
    view_root = accept_root / f"{phase}_view"
    require(not accept_root.exists(), f"preserving existing acceptance root: {accept_root}")
    state_root.mkdir(parents=True, exist_ok=False)
    view_root.mkdir(parents=True, exist_ok=False)

    native = case / "stages" / phase / "native_output"
    run_state = case / "run_state" / phase
    os.symlink(native, view_root / "result", target_is_directory=True)
    os.symlink(
        native / "actual_exit_code",
        view_root / "actual_exit_code",
    )
    os.symlink(
        run_state / "stdout_stderr.log",
        view_root / "stdout.log",
    )

    argv = [
        str(python),
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
        f"single_cube_seed0_{phase}_v1_accept",
        "--phase",
        phase,
        "--mode",
        "accept",
        "--training-output",
        str(view_root),
    ]
    config = {
        "schema": "single_cube_pldm_rgb_accept_config_v1",
        "phase": phase,
        "created_at": now(),
        "argv": argv,
        "cwd": str(source),
        "training_output": str(view_root),
        "native_output": str(native_output),
    }
    atomic_json(state_root / "config.json", config)
    code = run_logged(
        argv,
        cwd=source,
        env=environment,
        stdout_path=state_root / "stdout.log",
        stderr_path=state_root / "stderr.log",
    )
    (state_root / "actual_exit_code").write_text(str(code) + "\n", encoding="ascii")
    status_path = native_output / "status.json"
    require(status_path.is_file(), f"{phase} acceptance did not write status.json")
    status = json.loads(status_path.read_text(encoding="utf8"))
    require(code == 0, f"{phase} acceptance exit={code}")
    require(
        status.get("actual_exit_code") == 0
        and status.get("stage") == f"baseline_{phase}_accepted",
        f"{phase} acceptance status differs: {status}",
    )
    return {
        "phase": phase,
        "actual_exit_code": code,
        "status_path": str(status_path),
        "status_sha256": digest(status_path),
        "result": status,
    }


def persist_completion(case):
    persistent = Path(
        "/autodl-fs/data/lewm/single_cube_pldm_rgb10ep_10225090_20261022_v1"
    )
    require(not persistent.exists(), f"refusing existing persistent target: {persistent}")
    persistent.mkdir(parents=True)
    for name in ("source", "run_state", "stages", "data_views", "reference"):
        source = case / name
        if source.exists():
            shutil.copytree(source, persistent / name)
    key_files = [
        case / "FINAL_ACCEPT.json",
        case / "source" / "source_freeze_v1.json",
        case / "source" / "data_contract.json",
        case / "data_views" / "training_view_index.json",
    ]
    manifest = []
    for path in key_files:
        require(path.is_file(), f"missing completion artifact: {path}")
        manifest.append(
            {
                "path": str(persistent / path.relative_to(case)),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
    atomic_json(
        persistent / "PERSISTENT_MANIFEST.json",
        {
            "schema": "single_cube_pldm_rgb_persistent_manifest_v1",
            "created_at": now(),
            "case_root": str(case),
            "persistent_root": str(persistent),
            "artifacts": manifest,
        },
    )
    return str(persistent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    args = parser.parse_args()

    case = args.case_root.resolve()
    source = args.source_root.resolve()
    data = args.data_root.resolve()
    state_root = case / "run_state" / "pipeline"
    require(not state_root.exists(), f"preserving existing pipeline state: {state_root}")
    state_root.mkdir(parents=True)
    status_path = state_root / "status.json"
    state = {
        "schema": "single_cube_pldm_rgb_pipeline_status_v1",
        "stage": "starting",
        "started_at": now(),
        "updated_at": now(),
        "actual_exit_code": None,
        "case_root": str(case),
        "source_root": str(source),
        "data_root": str(data),
        "seed": 0,
        "epochs": 10,
        "method": "pldm",
        "modality": "rgb",
        "formal_uses_gate_checkpoint": False,
    }
    atomic_json(status_path, state)
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
    try:
        state["stage"] = "gate_train"
        state["updated_at"] = now()
        atomic_json(status_path, state)
        gate_code = run_logged(
            [
                str(args.python),
                str(case / "launch_stage.py"),
                "--stage",
                "gate",
                "--case-root",
                str(case),
                "--source-root",
                str(source),
                "--data-root",
                str(data),
                "--python",
                str(args.python),
            ],
            cwd=case,
            env=env,
            stdout_path=state_root / "gate_supervisor.stdout.log",
            stderr_path=state_root / "gate_supervisor.stderr.log",
        )
        state["gate_supervisor_exit"] = gate_code
        require(gate_code == 0, f"gate trainer failed: {gate_code}")

        state["stage"] = "gate_accept"
        state["updated_at"] = now()
        atomic_json(status_path, state)
        gate_accept = run_acceptance(
            "gate",
            case=case,
            source=source,
            data=data,
            python=args.python,
            environment=env,
        )
        state["gate_accept"] = gate_accept
        atomic_json(status_path, state)

        state["stage"] = "formal_train_10_epochs"
        state["updated_at"] = now()
        atomic_json(status_path, state)
        formal_code = run_logged(
            [
                str(args.python),
                str(case / "launch_stage.py"),
                "--stage",
                "formal",
                "--case-root",
                str(case),
                "--source-root",
                str(source),
                "--data-root",
                str(data),
                "--python",
                str(args.python),
            ],
            cwd=case,
            env=env,
            stdout_path=state_root / "formal_supervisor.stdout.log",
            stderr_path=state_root / "formal_supervisor.stderr.log",
        )
        state["formal_supervisor_exit"] = formal_code
        require(formal_code == 0, f"formal trainer failed: {formal_code}")

        state["stage"] = "formal_accept"
        state["updated_at"] = now()
        atomic_json(status_path, state)
        formal_accept = run_acceptance(
            "formal",
            case=case,
            source=source,
            data=data,
            python=args.python,
            environment=env,
        )
        state["formal_accept"] = formal_accept
        final = {
            "schema": "single_cube_pldm_rgb_final_accept_v1",
            "created_at": now(),
            "actual_exit_code": 0,
            "case_root": str(case),
            "source_freeze_sha256": digest(source / "source_freeze_v1.json"),
            "data_manifest_sha256": digest(
                data / "data_views" / "training_view_index.json"
            ),
            "gate_accept": gate_accept,
            "formal_accept": formal_accept,
            "checkpoint_exports": formal_accept["result"]["epoch_exports"],
            "strict_factory_reload": formal_accept["result"][
                "strict_factory_reload"
            ],
            "full_bare_exact": formal_accept["result"]["full_bare_exact"],
            "extra_optimizer_updates": formal_accept["result"][
                "extra_optimizer_updates"
            ],
        }
        atomic_json(case / "FINAL_ACCEPT.json", final)
        persistent = persist_completion(case)
        state["persistent_root"] = persistent
        state["final_accept_sha256"] = digest(case / "FINAL_ACCEPT.json")
        state["stage"] = "completed"
        state["actual_exit_code"] = 0
        state["finished_at"] = now()
        state["updated_at"] = state["finished_at"]
        atomic_json(status_path, state)
        return 0
    except Exception:
        state.update(
            stage="failed",
            actual_exit_code=1,
            error=traceback.format_exc(),
            finished_at=now(),
            updated_at=now(),
        )
        atomic_json(status_path, state)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
