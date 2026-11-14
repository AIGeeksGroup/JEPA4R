"""The Cube 1-4 workflow of the paper: prints its plan by default, runs the stages only when asked."""

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


PACKAGE = Path(__file__).resolve().parent
LEWM = PACKAGE.parents[2]  # this repository (the code root: jepa4r/, third_party/)
REPO = LEWM
MODELS = ("lewm_rgb", "lewm_pcd", "pldm", "lejepa", "visreg", "vjepa")
ALIASES = {"lewm": "lewm_rgb", "jepa4r": "lewm_pcd"}
STAGES = (
    "prepare-data",
    "collect",
    "views",
    "train",
    "prepare-encoder",
    "camera",
    "bind",
    "encode",
    "cache-audit",
    "gate",
    "reload",
    "gciql",
    "accept",
    "native-sr",
    "native-audit",
)
ALLOWED = {
    "schema",
    "cube_count",
    "model",
    "mode",
    "output_root",
    "python",
    "workers",
    "prefetch",
    "official_root",
    "collection_root",
    "data_root",
    "cuda_visible_devices",
    "collection_seed",
    "assets_root",
    "weights",
}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save(path, value, *, new=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if new:
        with path.open("xb") as stream:
            stream.write(data)
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)


def normalize(value):
    value = dict(value)
    if set(value) - ALLOWED:
        raise ValueError("unknown workflow fields: " + ",".join(sorted(set(value) - ALLOWED)))
    if value.get("schema") != "cube_workflow_v1":
        raise ValueError("workflow schema must be cube_workflow_v1")
    cube = value.get("cube_count")
    if type(cube) is not int or cube not in (1, 2, 3, 4):
        raise ValueError("cube_count must be 1..4")
    value["model"] = ALIASES.get(value.get("model"), value.get("model"))
    if value["model"] not in MODELS or value.get("mode") not in ("smoke", "formal"):
        raise ValueError("choose a current paper model and explicit smoke/formal mode")
    if not Path(value.get("output_root", "")).is_absolute():
        raise ValueError("an absolute output_root is required")
    python = value.get("python", {})
    if set(python) != {"world_model", "downstream"} or any(not Path(p).is_absolute() for p in python.values()):
        raise ValueError("explicit absolute world_model/downstream Python paths required")
    root = Path(value["output_root"]).resolve()
    value["output_root"] = str(root)
    value.setdefault("official_root", str(root.parent / "official" / ("cube" + str(cube))))
    value.setdefault("collection_root", str(root.parent / "collected" / value["mode"] / ("cube" + str(cube))))
    value.setdefault("data_root", str(root.parent / "views" / value["mode"] / ("cube" + str(cube))))
    value.setdefault("assets_root", str(root.parent / "assets"))
    for key in ("official_root", "collection_root", "data_root", "assets_root"):
        if not Path(value[key]).is_absolute():
            raise ValueError(key + " must be absolute")
        value[key] = str(Path(value[key]).resolve())
    for key, default in (("workers", 2), ("prefetch", 2)):
        value.setdefault(key, default)
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(key + " must be a positive integer")
    value.setdefault("collection_seed", 0)
    if type(value["collection_seed"]) is not int or value["collection_seed"] != 0:
        raise ValueError("the current paper collection seed is fixed at zero")
    return value


def build_plan(config_path, stage="all"):
    if isinstance(config_path, dict):
        config = normalize(config_path)
        config_path = None
    else:
        config_path = Path(config_path).resolve()
        config = normalize(read(config_path))
    root = Path(config["output_root"])
    cube, model, mode = config["cube_count"], config["model"], config["mode"]
    wm, down = config["python"]["world_model"], config["python"]["downstream"]
    control = root / "control"
    paths = dict(
        official=Path(config["official_root"]) / "OFFICIAL_DATA.json",
        collection=Path(config["collection_root"]) / "COLLECTION.json",
        data=Path(config["data_root"]) / "data_manifest.json",
        wm=root / "wm" / "WM_ACCEPTANCE.json",
        camera=control / "CAMERA.json",
        recipe=control / "downstream.json",
    )
    calls = {
        "prepare-data": (
            wm,
            [
                "-m",
                "jepa4r.data.cube.official_data",
                "--cube-count",
                str(cube),
                "--output",
                config["official_root"],
                "--execute",
            ],
            paths["official"],
        ),
        "collect": (
            wm,
            [
                "-m",
                "jepa4r.data.cube.collection",
                "--cube-count",
                str(cube),
                "--mode",
                mode,
                "--output",
                config["collection_root"],
                "--collection-seed",
                str(config["collection_seed"]),
                "--python",
                wm,
                "--execute",
            ],
            paths["collection"],
        ),
        "views": (
            wm,
            [
                "-m",
                "jepa4r.data.cube.fresh_data",
                "--collection-manifest",
                str(paths["collection"]),
                "--output",
                config["data_root"],
            ],
            paths["data"],
        ),
        "train": (
            wm,
            [
                "-m",
                "jepa4r.models.cube.fresh_train",
                "--method",
                model,
                "--data-manifest",
                str(paths["data"]),
                "--output",
                str(root / "wm"),
                "--mode",
                mode,
                "--workers",
                str(config["workers"]),
                "--prefetch",
                str(config["prefetch"]),
                "--execute",
            ],
            paths["wm"],
        ),
        "prepare-encoder": (
            down,
            [
                "-m",
                "jepa4r.baselines.cube.foundation_assets",
                "--model",
                model,
                "--cube-count",
                str(cube),
                "--mode",
                mode,
                "--assets-root",
                config["assets_root"],
                "--output",
                str(root / "encoder"),
                "--execute",
                "--download",
            ]
            + (["--weights", config["weights"]] if config.get("weights") else []),
            root / "encoder" / "ENCODER.json",
        ),
        "camera": (
            wm,
            ["-m", "jepa4r.policy.gciql.workflow", "--internal", "camera", "--config", str(control / "RUN.json")],
            paths["camera"],
        ),
        "bind": (
            down,
            ["-m", "jepa4r.policy.gciql.workflow", "--internal", "bind", "--config", str(control / "RUN.json")],
            paths["recipe"],
        ),
    }
    for name in ("encode", "cache-audit", "gate", "reload", "gciql", "accept", "native-sr", "native-audit"):
        calls[name] = (
            down,
            [
                "-m",
                "jepa4r.policy.gciql.workflow_downstream",
                "--stage",
                name,
                "--config",
                str(paths["recipe"]),
                "--execute",
            ],
            root / "downstream" / "stages" / (name + ".json"),
        )
    if stage == "all":
        sequence = list(STAGES)
        if model in ("visreg", "vjepa"):
            sequence = [s for s in sequence if s not in ("collect", "views", "train")]
        if model not in ("visreg", "vjepa"):
            sequence = [s for s in sequence if s != "prepare-encoder"]
        if mode == "smoke":
            sequence = [s for s in sequence if s not in ("gciql", "accept")]
    elif stage in STAGES:
        if stage == "train" and model in ("visreg", "vjepa"):
            raise ValueError("this model uses released frozen weights; no world-model train stage")
        if stage == "prepare-encoder" and model not in ("visreg", "vjepa"):
            raise ValueError("this model trains its own encoder")
        sequence = [stage]
    else:
        raise ValueError("unknown workflow stage: " + stage)
    rows = [
        dict(stage=name, command=[calls[name][0], *calls[name][1]], receipt=str(calls[name][2])) for name in sequence
    ]
    return dict(
        schema="cube_workflow_plan_v1",
        config=config,
        config_path=str(config_path) if config_path else None,
        config_sha256=digest(config),
        requested_stage=stage,
        stages=rows,
        paper_protocol=dict(
            wm_seed=0,
            policy_seed=0,
            formal_policy_updates=1000000,
            formal_native_episodes=250,
            raw_horizon=(200, 500, 1000, 600)[cube - 1],
            wm_training_rows=(200, 500, 1000, 600)[cube - 1],
            train_episodes=10000,
            validation_episodes=1000,
        ),
        executed=False,
    )


def _receipt(path, config, stage=None):
    path = Path(path)
    value = read(path)
    if stage == "bind":
        if (value.get("schema"), value.get("model"), value.get("cube_count"), value.get("mode")) != (
            "cube_workflow_downstream_v1",
            config["model"],
            config["cube_count"],
            config["mode"],
        ):
            raise ValueError("bound downstream recipe identity differs")
    elif type(value.get("actual_exit_code")) is not int or value["actual_exit_code"] != 0:
        raise ValueError("stage receipt reports a failure: " + str(path))
    if "cube_count" in value and value["cube_count"] != config["cube_count"]:
        raise ValueError("stage output belongs to another Cube")
    if "mode" in value and value["mode"] != config["mode"]:
        raise ValueError("stage output mode differs")
    return dict(path=str(path), sha256=sha(path))


def _reuse_input(row, config):
    if row["stage"] not in ("prepare-data", "collect", "views") or not Path(row["receipt"]).is_file():
        return None
    record = _receipt(row["receipt"], config, row["stage"])
    if row["stage"] == "prepare-data":
        value = read(row["receipt"])
        for split in ("train", "val"):
            if sha(value[split]["path"]) != value[split]["sha256"]:
                raise ValueError("reused official source changed")
    elif row["stage"] == "views":
        from jepa4r.data.cube.fresh_data import load_binding

        load_binding(row["receipt"])
    else:
        value = read(row["receipt"])
        for name, expected in value["producer_binding"]["files"].items():
            if sha(name) != expected:
                raise ValueError("reused collection producer changed")
        for name, item in value["data_files"].items():
            if sha(name) != item["sha256"]:
                raise ValueError("reused collection source changed")
    return record


def _verify_artifacts(path, checked=None, verified=None):
    """Recheck the checksums a completed stage's receipt recorded (and those of the receipts it points to) and the
    GCIQL checkpoints, without loading any model."""
    path = Path(path).resolve()
    checked = set() if checked is None else checked
    verified = {} if verified is None else verified
    if path in checked:
        return
    checked.add(path)
    value = read(path)
    if value.get("schema") == "cube_current_training_v1":
        from jepa4r.data.cube.fresh_data import load_binding

        load_binding(path)
        return

    def pin(record):
        target = Path(record["path"]).resolve()
        expected = record["sha256"]
        if target in verified and verified[target] != expected:
            raise ValueError("completed artifact has conflicting checksums: " + str(target))
        if target not in verified and sha(target) != expected:
            raise ValueError("completed artifact checksum changed: " + str(target))
        verified[target] = expected
        if target.suffix == ".json":
            _verify_artifacts(target, checked, verified)
        return target

    def visit(item):
        if isinstance(item, dict):
            if "path" in item and "sha256" in item:
                pin(item)
            if "receipt_path" in item and "receipt_sha256" in item:
                pin(dict(path=item["receipt_path"], sha256=item["receipt_sha256"]))
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    # gate reports point to a checkpoint directory, checked through its manifest;
    # reload reports keep the gate's action checksum for comparison, a reference
    # that is not next to reload.json.
    for field in ("checkpoint", "final_checkpoint"):
        checkpoint = value.get(field)
        if not isinstance(checkpoint, str):
            continue
        checkpoint = Path(checkpoint)
        if field == "checkpoint" and "updates" not in value:
            continue
        if field == "checkpoint":
            for key, name in (("full_state_sha256", "full_state.npz"), ("action_reference_sha256", "next_actions.npy")):
                if key in value:
                    pin(dict(path=path.parent / name, sha256=value[key]))
        manifest = checkpoint / "manifest.json"
        if field == "final_checkpoint" and "final_manifest_sha256" in value:
            pin(dict(path=manifest, sha256=value["final_manifest_sha256"]))
        transaction = read(manifest)
        counter = transaction["counter"]
        if type(counter) is not int or counter != value.get("updates", counter):
            raise ValueError("completed checkpoint counter differs")
        pin(dict(path=checkpoint / f"params_{counter}.pkl", sha256=transaction["agent_sha256"]))
        pin(dict(path=checkpoint / "run_state.pkl", sha256=transaction["sidecar_sha256"]))


def _verify_completed(row, config, previous, root):
    if Path(previous["path"]).resolve() != Path(row["receipt"]).resolve():
        raise ValueError("completed receipt differs from the canonical stage path")
    if _receipt(row["receipt"], config, row["stage"]) != previous:
        raise ValueError("completed output changed: " + row["stage"])
    driver = root / "dispatch" / row["stage"]
    candidates = [driver, *sorted((driver / "attempts").glob("attempt_*"))]
    accepted = []
    for candidate in candidates:
        if not (candidate / "COMPLETED.json").is_file():
            continue
        completion = read(candidate / "COMPLETED.json")
        if completion.get("receipt") == previous:
            if (
                type(completion.get("actual_exit_code")) is not int
                or completion["actual_exit_code"] != 0
                or type(completion.get("reused_input")) is not bool
            ):
                raise ValueError("completed dispatch acceptance identity differs")
            accepted.append(candidate)
    if len(accepted) != 1 or (accepted[0] / "actual_exit_code").read_text().strip() != "0":
        raise ValueError("completed dispatch has no unique successful actual exit")
    dispatched = read(accepted[0] / "plan.json")
    commands = [row["command"]]
    if row["stage"] == "collect":
        commands.append([*row["command"], "--resume"])
    if dispatched.get("command") not in commands or any(
        dispatched.get(k) != v for k, v in row.items() if k != "command"
    ):
        raise ValueError("completed dispatch differs from the canonical stage plan")
    if _reuse_input(row, config) is None:
        _verify_artifacts(row["receipt"])


def child_environment(cache, *, platform=None):
    cache = Path(cache)
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            (
                str(LEWM),
                str(LEWM / "third_party/stable-worldmodel"),
                str(os.environ.get("JEPA4R_OGBENCH") or LEWM / "third_party/ogbench"),
            )
        ),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        WANDB_MODE="offline",
    )
    for variable, folder in (
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
        ("CUDA_CACHE_PATH", "cuda"),
        ("JAX_COMPILATION_CACHE_DIR", "jax"),
        ("HF_HOME", "huggingface"),
        ("TORCH_HOME", "torch"),
    ):
        target = cache / folder
        target.mkdir(parents=True, exist_ok=True)
        env[variable] = str(target)
    # multiprocessing embeds TMPDIR in AF_UNIX names, whose limit is 108 bytes.
    # (adapted: a caller's short JEPA4R_TMPDIR / TMPDIR is kept, so that nothing lands on a small system disk; /tmp
    # only without one. pymp-XXXXXXXX/listener-XXXXXXXX adds 33 characters.)
    short = os.environ.get("JEPA4R_TMPDIR") or os.environ.get("TMPDIR")
    if (platform or os.name) == "posix" and short and Path(short).is_absolute() and len(short) <= 64:
        env["TMPDIR"] = short
    elif (platform or os.name) == "posix":
        env["TMPDIR"] = "/tmp"
    else:
        temporary = cache / "tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(temporary)
    return env


def execute(plan, *, resume=False):
    config = normalize(read(plan["config_path"])) if plan["config_path"] else normalize(plan["config"])
    if plan.get("input_config_path") and sha(plan["input_config_path"]) != plan["input_config_sha256"]:
        raise ValueError("input machine configuration changed after planning")
    if digest(config) != plan["config_sha256"]:
        raise ValueError("run config changed after planning")
    root = Path(config["output_root"])
    run_file = root / "control" / "RUN.json"
    if root.exists():
        if not run_file.is_file() or digest(read(run_file)) != digest(config):
            raise FileExistsError("output root is not this run: " + str(root))
    else:
        root.mkdir(parents=True, exist_ok=False)
        save(run_file, config, new=True)
    state_path = root / "STATUS.json"
    state = (
        read(state_path)
        if state_path.exists()
        else dict(
            stage="ready",
            started_at=datetime.now(timezone.utc).isoformat(),
            completed={},
            completed_count=0,
            failed_count=0,
            actual_exit_code=None,
            config_sha256=plan["config_sha256"],
        )
    )
    for row in plan["stages"]:
        name = row["stage"]
        if name in state["completed"]:
            previous = state["completed"][name]
            if not resume:
                raise FileExistsError("stage already completed; explicit --resume is required: " + name)
            try:
                _verify_completed(row, config, previous, root)
            except BaseException:
                state.update(
                    stage="failed",
                    failed_count=state["failed_count"] + 1,
                    actual_exit_code=1,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    error=traceback.format_exc(),
                )
                save(state_path, state)
                return 1
            continue
        driver = root / "dispatch" / name
        command = list(row["command"])
        if name == "collect" and resume:
            command.append("--resume")
        if driver.exists():
            exit_path = driver / "actual_exit_code"
            if not (resume and name == "collect" and exit_path.is_file()):
                raise FileExistsError(
                    "prior stage attempt exists; inspect its exit before any recovery: " + str(driver)
                )
            attempts = driver / "attempts"
            attempts.mkdir(exist_ok=True)
            index = 1
            while (attempts / f"attempt_{index:03d}").exists():
                if not (attempts / f"attempt_{index:03d}" / "actual_exit_code").is_file():
                    raise FileExistsError("prior collection retry has no exit receipt; inspect it before recovery")
                index += 1
            driver = attempts / f"attempt_{index:03d}"
        driver.mkdir(parents=True, exist_ok=False)
        save(driver / "plan.json", dict(row, command=command), new=True)
        state.update(stage=name, updated_at=datetime.now(timezone.utc).isoformat(), actual_exit_code=None)
        save(state_path, state)
        code = 1
        try:
            reused = _reuse_input(row, config)
            if reused is None:
                env = child_environment(root.parent / ".runtime_cache")
                with (driver / "stdout_stderr.log").open("xb") as log:
                    child = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
                code = child.returncode
                if code != 0:
                    raise RuntimeError("child exited " + str(code))
                record = _receipt(row["receipt"], config, name)
            else:
                (driver / "stdout_stderr.log").write_bytes(
                    b"Existing immutable input validated and reused; no child dispatched.\n"
                )
                record = reused
            code = 0
            state["completed"][name] = record
            state["completed_count"] = len(state["completed"])
            save(
                driver / "COMPLETED.json",
                dict(actual_exit_code=0, receipt=record, reused_input=reused is not None),
                new=True,
            )
        except BaseException:
            if code == 0:
                code = 1
            state.update(stage="failed", failed_count=state["failed_count"] + 1, error=traceback.format_exc())
        finally:
            (driver / "actual_exit_code").write_bytes((str(code) + "\n").encode())
            state.update(actual_exit_code=code, updated_at=datetime.now(timezone.utc).isoformat())
            save(state_path, state)
        if code:
            return code
    state.update(stage="stages_completed", actual_exit_code=0)
    save(state_path, state)
    if plan["requested_stage"] == "all":
        completion = root / "PIPELINE_COMPLETE.json"
        if not completion.exists():
            save(
                completion,
                dict(
                    schema="cube_pipeline_completion_v1",
                    actual_exit_code=0,
                    mode=config["mode"],
                    model=config["model"],
                    cube_count=config["cube_count"],
                    formal_acceptance=config["mode"] == "formal",
                    stages=state["completed"],
                ),
                new=True,
            )
    return 0


def bind(config_path):
    config = normalize(read(config_path))
    root = Path(config["output_root"])
    official = read(Path(config["official_root"]) / "OFFICIAL_DATA.json")
    camera = read(root / "control" / "CAMERA.json")
    actual_camera = camera.get("camera", camera)
    default_receipt = (
        root / "encoder" / "ENCODER.json"
        if config["model"] in ("visreg", "vjepa")
        else root / "wm" / "WM_ACCEPTANCE.json"
    )
    receipt = str(default_receipt)
    requested = dict(
        model=config["model"],
        cube_count=config["cube_count"],
        mode=config["mode"],
        official_data={s: {"path": official[s]["path"], "sha256": official[s]["sha256"]} for s in ("train", "val")},
        camera=actual_camera,
        downstream_output_root=str(root / "downstream"),
        cache_sampling_seed_base={"train": 0, "val": 1000000000},
    )
    for key in ("cuda_visible_devices",):
        if key in config:
            requested[key] = config[key]
    if config["model"] in ("visreg", "vjepa") and "encoder_binding" not in requested:
        requested["encoder_binding"] = read(receipt)["encoder_binding"]
    from .workflow_downstream import prepare_recipe

    prepare_recipe(requested, receipt, root / "control" / "downstream.json")


def camera(config_path):
    config = normalize(read(config_path))
    control = Path(config["output_root"]) / "control"
    from jepa4r.data.cube.collection import materialize
    from jepa4r.data.cube.collection import camera as probe

    runtime = control / "camera_runtime"
    binding = materialize(runtime)
    binding_path = control / "camera_source.json"
    save(binding_path, binding, new=True)
    plan = probe.build_plan(
        config["cube_count"],
        runtime,
        binding_path,
        control / "CAMERA.json",
        LEWM / "third_party/stable-worldmodel",
        config["collection_seed"],
    )
    return subprocess.run(probe.command(plan, python=sys.executable), check=False).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", "--method", choices=MODELS + tuple(ALIASES))
    parser.add_argument("--cube-count", "--cubes", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--mode", choices=("formal", "smoke"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wm-python")
    parser.add_argument("--rl-python")
    parser.add_argument("--stage", choices=("all",) + STAGES, default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--internal", choices=("bind", "camera"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.list:
        print(
            json.dumps(
                dict(
                    models=list(MODELS),
                    aliases=ALIASES,
                    cube_counts=[1, 2, 3, 4],
                    stages=["all", *STAGES],
                    modes=["smoke", "formal"],
                )
            )
        )
        return 0
    if args.internal:
        if args.config is None:
            parser.error("internal stage requires its bound run config")
        if args.internal == "bind":
            bind(args.config)
            return 0
        return camera(args.config)
    value = read(args.config) if args.config else dict(schema="cube_workflow_v1")
    for field in ("model", "cube_count", "mode"):
        if getattr(args, field) is not None:
            value[field] = getattr(args, field)
    if args.output:
        value["output_root"] = str(args.output.resolve())
    value.setdefault("mode", "formal")
    python = value.setdefault("python", {})
    for key, option, variable in (
        ("world_model", args.wm_python, "CUBE_WM_PYTHON"),
        ("downstream", args.rl_python, "CUBE_RL_PYTHON"),
    ):
        selected = option or python.get(key) or os.environ.get(variable) or sys.executable
        # Resolving a venv's executable symlink selects its base environment.
        resolved = (
            os.path.abspath(os.path.expanduser(selected)) if Path(selected).is_absolute() else shutil.which(selected)
        )
        if resolved is None:
            parser.error("Python executable not found: " + selected)
        python[key] = resolved
    plan = build_plan(value, args.stage)
    if args.config:
        plan.update(input_config_path=str(args.config.resolve()), input_config_sha256=sha(args.config))
    print(json.dumps(plan, indent=2), flush=True)
    return execute(plan, resume=args.resume) if args.execute else 0


if __name__ == "__main__":
    raise SystemExit(main())
