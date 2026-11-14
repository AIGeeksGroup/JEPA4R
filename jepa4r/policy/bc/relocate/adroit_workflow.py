"""The Adroit Relocate workflow: its stages and their command lines (collect, world-model training, the official
demonstrations, BC per seed); planning starts no process."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULES = {
    "adroit_collect": "jepa4r.data.relocate.adroit_collect",  # stage modules outside this package
    "adroit_data": "jepa4r.data.relocate.adroit_data",
}


def plan(args, assets):
    output = Path(args.output).resolve()
    model = args.model
    if model not in ("jepa4r", "lewm", "pldm", "lejepa", "visreg", "vjepa"):
        raise ValueError("unsupported native Relocate model: " + model)
    common = {str(key): str(value) for key, value in assets.get("environment", {}).items()}
    for key, value in {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        common.setdefault(key, value)
    paths = [
        ROOT.parent,
        ROOT / "policy/bc",
        assets.get("stable_worldmodel_root", ""),
        assets.get("mjrl_root", ""),
        assets.get("mj_envs_root", ""),
    ]
    if assets.get("eai_vc_root"):
        eai = Path(assets["eai_vc_root"])
        paths.extend(
            [eai / "cortexbench/mujoco_vc/src", eai / "vc_models/src", eai / "cortexbench", eai / "vc_models", eai]
        )
    paths.extend(assets.get("pythonpath", []))
    paths.extend(common.get("PYTHONPATH", "").split(os.pathsep))
    paths.extend(os.environ.get("PYTHONPATH", "").split(os.pathsep))
    unique = []
    seen = set()
    for path in paths:
        value = str(path)
        if not value:
            continue
        key = os.path.normcase(os.path.normpath(value))
        if key not in seen:
            seen.add(key)
            unique.append(value)
    common["PYTHONPATH"] = os.pathsep.join(unique)
    sampling = ROOT.parent / "configs/env/relocate/adroit_relocate_sampling.json"
    common.update(ADROIT_FRESH_EVAL="1", ADROIT_PCD_CONFIG=str(sampling))
    for key, env in (
        ("eai_vc_root", "CORTEXBENCH_EAI_VC_ROOT"),
        ("mujoco_path", "MUJOCO_PY_MUJOCO_PATH"),
        ("mujoco_key", "MUJOCO_PY_MJKEY_PATH"),
    ):
        if assets.get(key):
            common[env] = str(assets[key])
    if assets.get("mujoco_path"):
        common.setdefault(
            "LD_LIBRARY_PATH",
            str(Path(assets["mujoco_path"]) / "bin") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""),
        )
    if assets.get("wm_extra_site"):
        common["ADROIT_WM_EXTRA_SITE"] = str(assets["wm_extra_site"])
    stages = []

    def add(name, python, module, arguments):
        stages.append(
            {
                "name": name,
                "command": [
                    str(python),
                    "-B",
                    "-m",
                    MODULES.get(module, "jepa4r.policy.bc.relocate." + module),
                    *map(str, arguments),
                ],
                "cwd": str(ROOT.parent),
                "environment": dict(common),
            }
        )

    data = output / "data"
    add(
        "collect",
        args.sim_python,
        "adroit_collect",
        ["--output", data, "--expert", assets.get("adroit_expert", "REQUIRED_ADROIT_EXPERT")]
        + ["--train-episodes", 10000, "--test-episodes", 150, "--seed", args.seed],
    )
    checkpoint = output / "wm/checkpoints/relocate"
    epoch = 10
    if model in ("lewm", "jepa4r", "pldm", "lejepa"):
        add(
            "wm-train",
            args.wm_python,
            "adroit_learning",
            ["train", "--model", model, "--data-root", data, "--output", output / "wm"],
        )
    else:
        checkpoint = Path(assets.get(model + "_checkpoint", "REQUIRED_" + model.upper() + "_CHECKPOINT"))
    official = output / "official"
    add(
        "official-demos",
        args.sim_python,
        "adroit_data",
        ["official", "--source", assets.get("adroit_official_demos", "REQUIRED_OFFICIAL_DEMOS")]
        + ["--output", official, "--count", 100],
    )
    for seed in (100, 200, 300):
        bc = output / ("bc-seed" + str(seed))
        add(
            "bc-seed" + str(seed),
            args.sim_python,
            "adroit_learning",
            [
                "bc",
                "--model",
                model,
                "--checkpoint",
                checkpoint,
                "--epoch",
                epoch,
                "--output",
                bc,
                "--",
                "--config-name",
                "Adroit_BC_config",
                "env=relocate-v0",
                "seed=" + str(seed),
                "data_dir=" + str(official / ("pcd" if model == "jepa4r" else "rgb")),
                "num_demos=100",
                "epochs=100",
                "eval_num_traj=25",
                "eval_frequency=5",
                "save_frequency=10",
                "num_cpu=1",
                "bc_kwargs.batch_size=256",
                "hydra.run.dir=" + str(bc / "run"),
                "hydra.job.chdir=true",
                "job_name=policy",
                "wandb.project=local",
                "wandb.entity=local",
            ],
        )
    if assets.get("stable_worldmodel_root"):
        for s in stages:
            s["environment"]["ADROIT_FRAMEWORK_ROOT"] = str(assets["stable_worldmodel_root"])
    for key in ("visreg_checkpoint", "vjepa_checkpoint", "vjepa_root"):
        if assets.get(key):
            for s in stages:
                s["environment"]["ADROIT_" + key.upper()] = str(assets[key])
    return stages
