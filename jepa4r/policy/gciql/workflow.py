"""The Cube 1-4 workflow of the paper: its stages run one after another, each as a child process in its interpreter
(collection and world models: the world-model Python; encoding, GCIQL and evaluation: the policy Python). A stage
whose output exists is kept, so running the workflow again continues where it stopped.

  prepare-data     OGBench's official play data (jepa4r/data/cube/official_data.py)
  collect, views   the paired RGB / point-cloud collection and its training views (jepa4r/data/cube)
  train            the world model (jepa4r/models/cube/fresh_train.py); not for the released encoders
  prepare-encoder  the released VISReg / V-JEPA 2.1 weights (jepa4r/baselines/cube/foundation_assets.py)
  camera, bind     the run's camera, and its downstream.json (encoder, data, camera)
  encode, gciql, evaluate   the policy stages (workflow_downstream.py)

usage: python -m jepa4r.policy.gciql.workflow --model <m> --cube-count <n> --output <run root> --execute"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]  # this repository (jepa4r/, third_party/)
MODELS = ("jepa4r", "lewm", "pldm", "lejepa", "visreg", "vjepa")
RELEASED = ("visreg", "vjepa")
STAGES = (
    "prepare-data",
    "collect",
    "views",
    "train",
    "prepare-encoder",
    "camera",
    "bind",
    "encode",
    "gciql",
    "evaluate",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf8")


def normalize(value):
    value = dict(value)
    cube = value.get("cube_count")
    if cube not in (1, 2, 3, 4):
        raise ValueError("cube_count must be 1..4")
    if value.get("model") not in MODELS:
        raise ValueError(f"unknown model {value['model']!r} (one of {MODELS})")
    if not Path(value.get("output_root", "")).is_absolute():
        raise ValueError("an absolute output_root is required")
    python = value.get("python", {})
    if set(python) != {"world_model", "downstream"} or any(not Path(p).is_absolute() for p in python.values()):
        raise ValueError("absolute world_model / downstream Python paths are required")
    root = Path(value["output_root"]).resolve()
    value["output_root"] = str(root)
    value.setdefault("official_root", str(root.parent / "official" / f"cube{cube}"))
    value.setdefault("collection_root", str(root.parent / "collected" / f"cube{cube}"))
    value.setdefault("data_root", str(root.parent / "views" / f"cube{cube}"))
    value.setdefault("assets_root", str(root.parent / "assets"))
    for key in ("official_root", "collection_root", "data_root", "assets_root"):
        value[key] = str(Path(value[key]).resolve())
    value.setdefault("workers", 2)
    value.setdefault("prefetch", 2)
    value.setdefault("collection_seed", 0)
    return value


def build_plan(config, stage="all"):
    config = normalize(config)
    root = Path(config["output_root"])
    cube, model = config["cube_count"], config["model"]
    wm, down = config["python"]["world_model"], config["python"]["downstream"]
    control = root / "control"
    internal = ["-m", "jepa4r.policy.gciql.workflow", "--config", str(control / "RUN.json"), "--internal"]
    calls = {
        "prepare-data": (
            wm,
            ["-m", "jepa4r.data.cube.official_data", "--cube-count", str(cube), "--output", config["official_root"]],
            Path(config["official_root"]) / "OFFICIAL_DATA.json",
        ),
        "collect": (
            wm,
            [
                "-m",
                "jepa4r.data.cube.collection",
                "--cube-count",
                str(cube),
                "--output",
                config["collection_root"],
                "--collection-seed",
                str(config["collection_seed"]),
                "--python",
                wm,
                "--execute",
            ],
            Path(config["collection_root"]) / "COLLECTION.json",
        ),
        "views": (
            wm,
            [
                "-m",
                "jepa4r.data.cube.fresh_data",
                "--collection-manifest",
                str(Path(config["collection_root"]) / "COLLECTION.json"),
                "--output",
                config["data_root"],
            ],
            Path(config["data_root"]) / "data_manifest.json",
        ),
        "train": (
            wm,
            [
                "-m",
                "jepa4r.models.cube.fresh_train",
                "--method",
                model,
                "--data-manifest",
                str(Path(config["data_root"]) / "data_manifest.json"),
                "--output",
                str(root / "wm"),
                "--workers",
                str(config["workers"]),
                "--prefetch",
                str(config["prefetch"]),
                "--execute",
            ],
            root / "wm" / "WM_EXPORT.json",
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
                "--assets-root",
                config["assets_root"],
                "--output",
                str(root / "encoder"),
                "--download",
            ]
            + (["--weights", config["weights"]] if config.get("weights") else []),
            root / "encoder" / "ENCODER.json",
        ),
        "camera": (wm, [*internal, "camera"], control / "CAMERA.json"),
        "bind": (down, [*internal, "bind"], control / "downstream.json"),
    }
    for name in ("encode", "gciql", "evaluate"):
        calls[name] = (
            down,
            [
                "-m",
                "jepa4r.policy.gciql.workflow_downstream",
                "--stage",
                name,
                "--config",
                str(control / "downstream.json"),
            ],
            root / "downstream" / "stages" / f"{name}.json",
        )
    if stage == "all":
        sequence = [
            s for s in STAGES if s not in (("collect", "views", "train") if model in RELEASED else ("prepare-encoder",))
        ]
    elif stage in STAGES:
        if stage in ("collect", "views", "train") and model in RELEASED:
            raise ValueError("the released encoders are not trained here")
        if stage == "prepare-encoder" and model not in RELEASED:
            raise ValueError("this model trains its own encoder")
        sequence = [stage]
    else:
        raise ValueError("unknown workflow stage: " + stage)
    rows = [
        dict(stage=name, command=[calls[name][0], *calls[name][1]], output=str(calls[name][2])) for name in sequence
    ]
    return dict(config=config, stages=rows)


def child_environment(cache, *, platform=None):
    cache = Path(cache)
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            (
                str(REPO),
                str(REPO / "third_party/stable-worldmodel"),
                str(os.environ.get("JEPA4R_OGBENCH") or REPO / "third_party/ogbench"),
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
    # multiprocessing puts TMPDIR into AF_UNIX socket names, which are limited to 108 bytes (pymp-XXXXXXXX/
    # listener-XXXXXXXX adds 33 characters): a short JEPA4R_TMPDIR / TMPDIR of the caller is kept, else /tmp.
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


def execute(plan):
    """Run the plan's stages in order; a stage whose output exists is kept. Returns 0, or the failed child's exit
    code (its log: <run root>/logs/<stage>.log)."""
    config = plan["config"]
    root = Path(config["output_root"])
    save(root / "control" / "RUN.json", config)
    for row in plan["stages"]:
        if Path(row["output"]).is_file():
            print(f"{row['stage']}: done ({row['output']})", flush=True)
            continue
        print(f"{row['stage']}: running", flush=True)
        (root / "logs").mkdir(parents=True, exist_ok=True)
        with (root / "logs" / f"{row['stage']}.log").open("ab") as log:
            env = child_environment(root.parent / ".runtime_cache")
            code = subprocess.run(row["command"], env=env, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
        if code != 0 or not Path(row["output"]).is_file():
            print(f"{row['stage']}: failed (exit {code}); log {root / 'logs' / (row['stage'] + '.log')}", flush=True)
            return code or 1
    return 0


def bind(config_path):
    """The run's downstream.json (workflow_downstream.prepare_recipe): its encoder, the official data, its camera."""
    config = normalize(read(config_path))
    root = Path(config["output_root"])
    official = read(Path(config["official_root"]) / "OFFICIAL_DATA.json")
    camera = read(root / "control" / "CAMERA.json")
    record = root / ("encoder/ENCODER.json" if config["model"] in RELEASED else "wm/WM_EXPORT.json")
    requested = dict(
        model=config["model"],
        cube_count=config["cube_count"],
        official_data={split: official[split]["path"] for split in ("train", "val")},
        camera=camera["camera"],
        downstream_output_root=str(root / "downstream"),
    )
    if "cuda_visible_devices" in config:
        requested["cuda_visible_devices"] = config["cuda_visible_devices"]
    from .workflow_downstream import prepare_recipe

    prepare_recipe(requested, record, root / "control" / "downstream.json")
    return 0


def camera(config_path):
    """The run's camera (jepa4r/data/cube/collection/camera.py): the collection camera of this Cube count, read from
    the simulator in a child process."""
    config = normalize(read(config_path))
    control = Path(config["output_root"]) / "control"
    from jepa4r.data.cube.collection import materialize
    from jepa4r.data.cube.collection import camera as probe

    runtime = control / "camera_runtime"
    materialize(runtime)
    plan = probe.build_plan(
        config["cube_count"],
        runtime,
        control / "CAMERA.json",
        REPO / "third_party/stable-worldmodel",
        config["collection_seed"],
    )
    return subprocess.run(probe.command(plan, python=sys.executable), check=False).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="a run config (JSON); the options below override it")
    parser.add_argument("--model", "--method", choices=MODELS)
    parser.add_argument("--cube-count", "--cubes", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wm-python")
    parser.add_argument("--rl-python")
    parser.add_argument("--stage", choices=("all",) + STAGES, default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--internal", choices=("bind", "camera"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.internal:
        return dict(bind=bind, camera=camera)[args.internal](args.config)
    value = read(args.config) if args.config else {}
    for field in ("model", "cube_count"):
        if getattr(args, field) is not None:
            value[field] = getattr(args, field)
    if args.output:
        value["output_root"] = str(args.output.resolve())
    python = value.setdefault("python", {})
    for key, option, variable in (
        ("world_model", args.wm_python, "JEPA4R_PY_CUBE_WM"),
        ("downstream", args.rl_python, "JEPA4R_PY_CUBE_RL"),
    ):
        selected = option or python.get(key) or os.environ.get(variable) or sys.executable
        resolved = (
            os.path.abspath(os.path.expanduser(selected)) if Path(selected).is_absolute() else shutil.which(selected)
        )
        if resolved is None:
            parser.error("Python executable not found: " + selected)
        python[key] = resolved
    plan = build_plan(value, args.stage)
    print(json.dumps(plan, indent=2), flush=True)
    return execute(plan) if args.execute else 0


if __name__ == "__main__":
    raise SystemExit(main())
