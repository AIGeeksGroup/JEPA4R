"""The RM75-B + RH56 Relocate workflow in simulation: its stages and their command lines (planning starts no process).

rh56_prepare         the collection plans (batches of 250 episodes: train 9000, seeds 61000000 + i; val 1000, seeds
                     62000000 + i; coverage "protocol") and a copy of the official initialisations, in rh56/inputs
rh56_collect_<i>     batch i (jepa4r/envs/twins/rh56/collect.py, the opposite camera)
rh56_publish         the paired dataset over the batches (data_tools.py)
rh56_wm              world-model training (wm_bridge.py; jepa4r: point clouds, lewm: RGB), 10 epochs, seed 3072
rh56_official_demo   the 100 BC demonstrations from the official initialisations (official_init.py)
rh56_bc_<seed>       behavior cloning, seeds 100 / 200 / 300 (bc_bridge.py), on the epoch-10 world model
usage (stage prepare): python jepa4r/policy/bc/relocate/rh56_workflow.py prepare --config-json <json>"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
ENTRY = PACKAGE_ROOT / "policy/bc/relocate/run_rh56.py"
BATCH_SIZE = 250
SPLITS = (("train", 9000, 61000000), ("val", 1000, 62000000))  # split, episodes, first reset seed


def _episode_batches():
    episodes = [
        dict(split=split, seed=base + index, coverage="protocol")
        for split, count, base in SPLITS
        for index in range(count)
    ]
    return [episodes[start : start + BATCH_SIZE] for start in range(0, len(episodes), BATCH_SIZE)]


def plan(args, assets: dict) -> list[dict]:
    """The stages of one (env, model) run as {name, command, cwd, environment}."""
    if args.model not in ("jepa4r", "lewm"):
        raise ValueError("the RH56 workflow runs jepa4r and lewm only")
    for key in ("rh56_expert_root", "eai_vc_root", "rh56_official_task_init"):
        if not assets.get(key):
            raise ValueError("missing RH56 asset: " + key)
    route = "pcd" if args.model == "jepa4r" else "rgb"
    base = Path(args.output).resolve() / "rh56"
    sim_python = str(args.sim_python)
    bc_python = str(assets.get("rh56_bc_python", args.sim_python))
    environment = {str(k): str(v) for k, v in assets.get("environment", {}).items()}
    paths = [str(PACKAGE_ROOT.parent)] + [str(p) for p in assets.get("pythonpath", [])]
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYOPENGL_PLATFORM", "egl")
    common = ["--package-root", str(PACKAGE_ROOT)]
    expert = ["--expert-root", str(assets["rh56_expert_root"])]
    stages = []

    def stage(name, python, arguments, *, entry=ENTRY):
        stages.append(
            dict(
                name=name,
                command=[str(python), str(entry), *map(str, arguments)],
                cwd=str(PACKAGE_ROOT.parent),
                environment=dict(environment),
            )
        )

    configuration = dict(output=str(base), official_task_init=str(assets["rh56_official_task_init"]))
    stage("rh56_prepare", sim_python, ["prepare", "--config-json", json.dumps(configuration)], entry=Path(__file__))
    batch_dirs = []
    for index in range(len(_episode_batches())):
        directory = base / "batches" / f"batch_{index:03d}"
        batch_dirs.append(str(directory))
        arguments = ["collect", *common, *expert, "--camera", "opposite"]
        arguments += ["--plan", base / "inputs" / f"plan_{index:03d}.json", "--output", directory]
        stage(f"rh56_collect_{index:03d}", sim_python, arguments)
    stage("rh56_publish", sim_python, ["publish", *common, "--batches", *batch_dirs, "--output", base / "data"])
    wm_arguments = ["wm", *common, "--route", route, "--device", "cuda", "--seed", "3072"]
    wm_arguments += ["--manifest", base / "data/manifest.json", "--dataset", base / f"data/paired_{route}.h5"]
    wm_arguments += ["--output", base / "wm", "--workers", str(assets.get("rh56_wm_workers", 24))]
    wm_arguments += ["--prefetch-factor", "2"]
    stage("rh56_wm", args.wm_python, wm_arguments)
    demo_arguments = ["official-init", *common, *expert, "--init-source", base / "inputs/official_task_init.json"]
    demo_arguments += ["--count", "100", "--start-demo", "0", "--seed-base", "64500000", "--output", base / "demos"]
    stage("rh56_official_demo", sim_python, demo_arguments)
    checkpoint = base / f"wm/worldmodel_cache/checkpoints/rh56_relocate_{route}"
    for seed in (100, 200, 300):
        arguments = ["bc", *common, *expert, "--eai-vc-root", assets["eai_vc_root"], "--route", route]
        arguments += ["--device", "cuda", "--checkpoint", checkpoint, "--epoch", "10"]
        arguments += ["--demos-manifest", base / "demos/bc_demos_manifest.json", "--train-seed", str(seed)]
        arguments += ["--out", base / f"bc_seed{seed}"]
        stage(f"rh56_bc_{seed}", bc_python, arguments)
    return stages


def prepare(configuration):
    """Write the collection plans and copy the official initialisations into <output>/inputs."""
    directory = Path(configuration["output"]) / "inputs"
    directory.mkdir(parents=True, exist_ok=True)
    for index, episodes in enumerate(_episode_batches()):
        (directory / f"plan_{index:03d}.json").write_text(json.dumps(dict(episodes=episodes), indent=2) + "\n")
    shutil.copyfile(configuration["official_task_init"], directory / "official_task_init.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare").add_argument("--config-json", required=True)
    args = parser.parse_args(argv)
    prepare(json.loads(args.config_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
