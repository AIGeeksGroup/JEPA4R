"""The paired RGB / point-cloud collection of Cube 1-4 (the collect stage of the Cube workflow).

The collection scripts (vendor/) run in a copy of their source tree (configs/env/cube/collection/sources.json lists
the files and where they are in this repository), one child process per part of at most 100 episodes: for each of the
five tasks, the five behaviour strategies (expert with low / high noise, expert then random from step 100 or 45,
random) with their train / validation episode counts -- 10000 train and 1000 validation episodes in all, every episode
with its own reset seed. A part that finished is kept; the parts are then joined into one virtual HDF5 dataset per
split and modality, recorded in COLLECTION.json.
usage: python -m jepa4r.data.cube.collection --cube-count <n> --output <dir> [--execute]"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]  # this repository (jepa4r/, third_party/)
SOURCES = REPO / "configs" / "env" / "cube" / "collection" / "sources.json"
OGBENCH = Path(os.environ.get("JEPA4R_OGBENCH") or REPO / "third_party/ogbench")  # OGBench a977253 checkout
ENV_TYPES = ("single", "double", "triple", "quadruple")
HORIZONS = (200, 500, 1000, 600)
STRATEGIES = (  # strategy, train episodes, validation episodes (per task)
    ("expert_low_noise", 400, 40),
    ("expert_high_noise", 800, 80),
    ("expert_random_branch100", 200, 20),
    ("expert_random_branch45", 200, 20),
    ("random", 400, 40),
)


def _save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")
    temporary.replace(path)


def _inside(root, name):
    pure = PurePosixPath(str(name))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("a source path must stay inside its root: " + str(name))
    return Path(root) / str(name)


def materialize(output):
    """Copy the collection scripts into their source tree under ``output``."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    for destination, row in json.loads(SOURCES.read_text(encoding="utf8"))["files"].items():
        name = row["repository_path"]
        source = (
            _inside(OGBENCH, name[len("third_party/ogbench/") :])
            if name.startswith("third_party/ogbench/")
            else _inside(REPO, name)
        )
        target = _inside(output, destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return output


def _columns(handle):
    import h5py

    result = []

    def walk(group, prefix=""):
        for name in sorted(group):
            value = group[name]
            if isinstance(value, h5py.Group):
                walk(value, prefix + name + "/")
            else:
                result.append(prefix + name)

    walk(handle)
    return [name for name in result if name not in ("ep_len", "ep_offset")]


def merge_raw(shards, output):
    """Join the parts' datasets into one virtual dataset (VDS) per column; the parts' files stay where they are."""
    import h5py
    import numpy as np

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    (sources, lengths, keys, schemas) = ([], [], None, None)
    for path in map(Path, shards):
        with h5py.File(path, "r") as file:
            columns = _columns(file)
            schema = {name: (file[name].shape[1:], file[name].dtype) for name in columns}
            if keys is not None and (columns != keys or schema != schemas):
                raise ValueError("the parts have different columns")
            (keys, schemas) = (columns, schema)
            lens = file["ep_len"][:]
            sources.append((path.resolve(), int(lens.sum())))
            lengths.extend(map(int, lens))
    if not sources:
        raise ValueError("no parts to join")
    output.parent.mkdir(parents=True, exist_ok=True)
    total = sum((rows for (_, rows) in sources))
    with h5py.File(output, "x", libver="latest") as file:
        for name in keys:
            (shape, dtype) = schemas[name]
            layout = h5py.VirtualLayout((total, *shape), dtype=dtype)
            offset = 0
            for path, rows in sources:
                source = h5py.VirtualSource(str(path), name, shape=(rows, *shape))
                layout[offset : offset + rows] = source
                offset += rows
            file.create_virtual_dataset(name, layout)
        file["ep_len"] = np.asarray(lengths, dtype=np.int32)
        file["ep_offset"] = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
    return str(output.resolve())


def collection_jobs():
    """(task, strategy, split, episodes) of the collection, in order: task; strategy; train then validation."""
    return [
        (task, strategy, split, count)
        for task in range(1, 6)
        for (strategy, train, val) in STRATEGIES
        for (split, count) in (("train", train), ("validation", val))
    ]


def build_plan(cube_count, output_root, python=None, collection_seed=0):
    """The parts of the collection (collection_jobs(), split into parts of at most 100 episodes), with their reset
    seeds (world_seed: 1e6 x cube count + collection seed + the episodes before it + 100 x the job index + the part
    offset) and policy seeds (1e8 + 1e6 x cube count + collection seed + 17 x part index)."""
    if cube_count not in (1, 2, 3, 4):
        raise ValueError("cube_count must be 1..4")
    root = Path(output_root).resolve()
    world_base = 1000000 * cube_count + collection_seed
    policy_base = 100000000 + 1000000 * cube_count + collection_seed
    (parts, preceding) = ([], 0)
    for logical, (task, strategy, split, count) in enumerate(collection_jobs()):
        for offset in range(0, count, 100):
            index = len(parts)
            directory = root / "shards" / f"part_{index:03d}"
            parts.append(
                dict(
                    index=index,
                    task=task,
                    strategy=strategy,
                    split=split,
                    episodes=min(100, count - offset),
                    world_seed=world_base + preceding + 100 * logical + offset,
                    policy_seed=policy_base + 17 * index,
                    directory=str(directory),
                    rgb=str(directory / "datasets/pair_pix.h5"),
                    pcd=str(directory / "datasets/pair_pcd.h5"),
                )
            )
        preceding += count
    return dict(
        cube_count=cube_count,
        env_type=ENV_TYPES[cube_count - 1],
        collection_seed=collection_seed,
        raw_horizon=HORIZONS[cube_count - 1],
        split_counts={
            split: sum(p["episodes"] for p in parts if p["split"] == split) for split in ("train", "validation")
        },
        output_root=str(root),
        python=str(python or sys.executable),
        parts=parts,
    )


def run(plan):
    """Run the parts that have not finished (a part's DONE file marks it), then join them (COLLECTION.json)."""
    root = Path(plan["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    _save(root / "plan.json", plan)
    runtime = root / "runtime"
    if not runtime.exists():
        materialize(runtime)
    code = runtime / "code/collection"
    env = dict(
        os.environ,
        CUBE_COLLECTION_RUNTIME_ROOT=str(runtime),
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        EGL_PLATFORM="surfaceless",
        STABLEWM_HOME=str(root / "native_outputs"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        (
            str(code),
            str(code / "scripts/data"),
            str(REPO / "third_party/stable-worldmodel"),
            str(runtime / "third_party/ogbench"),
        )
    )
    for part in plan["parts"]:
        directory = Path(part["directory"])
        if (directory / "DONE").exists():
            continue
        if directory.exists():  # an unfinished part starts over
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        with (directory / "stdout_stderr.log").open("xb") as log:
            result = subprocess.run(
                child_command(plan, part, runtime), cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        if result.returncode != 0:
            raise RuntimeError(f"collection part {part['index']} exited {result.returncode} (log: {directory})")
        (directory / "DONE").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    paths = {}
    merged = root / "datasets"
    if merged.exists():
        shutil.rmtree(merged)
    for split in ("train", "validation"):
        for modality in ("rgb", "pcd"):
            key = split + "_" + modality
            selected = [p[modality] for p in plan["parts"] if p["split"] == split]
            paths[key] = merge_raw(selected, merged / (key + ".h5"))
    manifest = dict(
        cube_count=plan["cube_count"],
        raw_horizon=plan["raw_horizon"],
        collection_seed=plan["collection_seed"],
        train_episodes=plan["split_counts"]["train"],
        validation_episodes=plan["split_counts"]["validation"],
        paths=paths,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    _save(root / "COLLECTION.json", manifest)
    return manifest


def child_command(plan, part, runtime):
    """The collector's command line for one part: one config (paper_cube) for the four Cube tasks, the episode length
    given explicitly."""
    runtime = Path(runtime)
    cube = plan["cube_count"]
    command = [plan["python"], str(runtime / "code/collection/scripts/data/collect_cube_pcd.py")]
    if cube == 4:
        command = [
            plan["python"],
            str(runtime / "collection_adapters/quad_h600.py"),
            "--base-root",
            str(runtime),
            "--horizon-actions",
            "600",
            "--camera-pullback-m",
            "0.10",
            "--",
        ]
    command += [
        "--config-name",
        "paper_cube",
        "env_type=" + plan["env_type"],
        f"fixed_task_id={part['task']}",
        "strategy=" + part["strategy"],
        "split=" + part["split"],
        f"num_traj={part['episodes']}",
        f"seed={part['policy_seed']}",
        f"world_seed={part['world_seed']}",
        "cache_dir=" + part["directory"],
        "dataset_name=pair_pcd.h5",
        "pix_dataset_name=pair_pix.h5",
        "pcd.keep_rgb=false",
        "world.num_envs=1",
        f"world.max_episode_steps={plan['raw_horizon']}",
        "hydra.run.dir=" + str(Path(part["directory"]) / "hydra"),
    ]
    if cube > 1:
        command += [
            "pcd_capture.frame_source_path=" + str(runtime / "external/native_pcd_frame.py"),
            "pcd_capture.provider_path=" + str(runtime / "external/pcd_geometry.py"),
            "pcd_capture.core_path=" + str(runtime / "code/collection/envs/pcd_wrapper.py"),
        ]
    if cube > 2:
        command += ["pcd.crop_box=[[-0.25,-0.5,-0.02],[0.75,0.5,0.9]]"]
    return command
