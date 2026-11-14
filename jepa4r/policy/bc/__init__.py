"""CortexBench behaviour cloning [majumdar2023vc1] (supp. sec:supp-policy, configs/policy/bc.yaml).

MetaWorld and Adroit Relocate follow the official CortexBench imitation protocol; OGBench Scene reuses the same
code on its five evaluation tasks, and so do four MetaWorld tasks on the RM75-B twin (Sec. 5.6), on their own demos.
100 epochs on env.policy_data.demos_per_task demonstrations per task (25; 100 on Relocate); success = best of 20
evaluations (jepa4r/eval/cortexbench.py).

Modules (the CortexBench harness of the paper runs; eai-vc / the embedding configs / V-JEPA 2 are found through the
environment variables below):
  run_bc.py        eai-vc 76fe35e's hydra_launcher + train_loop (unchanged on disk) with three runtime patches that do
                   not change the numbers (local embedding configs, metrics.jsonl logging, reuse of the evaluation paths
                   for the logging-only train/* rollouts)
  run_bc3d.py      run_bc.py with the camera image replaced by the same camera's point cloud (pcd_fork.MWPointClouds)
  run_bc_scene.py  run_bc.py on OGBench scene's five tasks (scene_env.SceneCBEnv in a scene_server.py subprocess of the
                   MuJoCo 3 / OGBench python), proprioception = the first 6 of 8 entries (SCENE_PROPRIO_DIMS)
  run_bc_rm75.py   run_bc.py / run_bc3d.py on the RM75-B twin's MetaWorld tasks (rm75_env.RM75CBEnv in a rm75_server.py
                   subprocess of the MuJoCo 3 / metaworld 3.0 python given by MW_PY), on their own demos
                   (make_rm75_demos.py)
  wm_embeddings.py   the frozen LeWM / JEPA4R encoders (encoder + projector, epoch 10) as CortexBench embeddings
  jepa_baselines.py  PLDM, LeJEPA, VISReg, V-JEPA 2.1 embeddings
  enc_server.py    the frozen encoder in one shared GPU process for run_bc_scene.py (ENC_SOCKET)
  run_adroit_bc3d.py  run_bc.py on Relocate point clouds
  relocate/        the Relocate workflow on Adroit Relocate and the RH56 twin (collection, world-model training,
                   official demonstrations, behaviour cloning; its data in jepa4r/data/relocate), see relocate_stage
run_bc.py, wm_embeddings.py and jepa_baselines.py also serve the Relocate runs (their docstrings list the switches). The
embedding configs of the paper runs, one per encoder kind (embedding_config fills in each run's checkpoint), and those
of the Relocate runs: configs/policy/bc/conf/model/.
``train`` runs, for each evaluation task of the env and each seed of cfg.policy.eval.seeds, one runner process with the
arguments the paper's job runners gave it, then reports success = mean over tasks of the mean over seeds of each run's
highest success among its 20 evaluations, as the paper's summaries compute it.
Interpreters / paths (environment variables): JEPA4R_PY_CORTEXBENCH (eai-vc + MuJoCo 2.0 + mujoco-py python),
JEPA4R_PY_COLLECT (MuJoCo 3 + OGBench / metaworld 3.0 python for the scene env and the RM75-B twin's MetaWorld env),
JEPA4R_EAI_VC (eai-vc checkout), JEPA4R_VJEPA2.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONF_MODEL = HERE.parents[2] / "configs" / "policy" / "bc" / "conf" / "model"  # the embedding configs
SUITES = {  # suite -> (hydra config of eai-vc, 2D demo dir, 3D demo dir)   source grid.py:39-44, scene_grid.py:79-80
    "metaworld": ("Metaworld_BC_config", "metaworld-expert-v1.0", "metaworld-expert-v1.0-pcd"),
    "ogbench": ("Metaworld_BC_config", "scene-expert", "scene-expert-pcd"),
    # source grid.py:46-48
    "metaworld_rm75": ("Metaworld_BC_config", "metaworld-rm75-expert", "metaworld-rm75-expert-pcd"),
    "adroit": ("Adroit_BC_config", "adroit-expert-v1.0", "adroit-expert-v1.0-pcd"),
}
TEMPLATES = {  # method kind / name -> conf/model template of the paper runs (checkpoint filled in per run)
    "pointcloud": "jepa4r_mw_hammer.yaml",
    "image": "lewm_mw_hammer.yaml",
    "pldm": "pldm_mw_hammer.yaml",
    "lejepa": "lejepa_mw_hammer.yaml",
    "visreg": "visreg_vitb.yaml",
    "vjepa21": "vjepa21_vitb.yaml",
    "random_vitb": "rand_vitb.yaml",
}


def tasks(env_cfg):
    """CortexBench env ids evaluated for this env: the task itself, or Scene's five evaluation tasks."""
    if env_cfg.suite == "ogbench":
        return [f"scene-task{k}-v0" for k in range(1, 6)]
    return [str(env_cfg.env_id)]


def embedding_config(cfg):
    """The conf/model/<embedding>.yaml of this run: the paper runs' template with the checkpoint of cfg.method."""
    from jepa4r.config import paths, stage_dir

    m = cfg.method
    key = m.name if m.name in TEMPLATES else (m.kind if m.kind in TEMPLATES else m.obs)
    conf = yaml.safe_load((CONF_MODEL / TEMPLATES[key]).read_text())
    if "checkpoint_dir" in conf:
        conf["checkpoint_dir"] = str(stage_dir("train_wm", cfg.env.name, m.name, None, str(cfg.tag)) / "checkpoints")
        conf["epoch"] = int(m.get("policy_epoch", 10))
    if "center" in conf:
        conf["center"] = [float(c) for c in cfg.env.pointcloud.affine.center]
        conf["scale"] = float(cfg.env.pointcloud.affine.scale)
    if "checkpoint" in conf:
        conf["checkpoint"] = str(paths()["checkpoints"] / str(m.weights))
    return conf


def run_command(cfg, task, seed, run_dir, emb_name, demos):
    """The command the source's grid.py (MetaWorld, Adroit) / scene_grid.py (Scene, hand6 proprioception) gave one
    job."""
    suite = cfg.env.suite
    config_name = SUITES[suite][0]
    pcd = cfg.method.obs == "pointcloud"
    py = os.environ.get("JEPA4R_PY_CORTEXBENCH", sys.executable)
    if suite == "ogbench":
        launcher = HERE / "run_bc_scene.py"
        extra = ["camera=front_pixels", "env_kwargs.add_proprio=True", "env_kwargs.proprio_key=gripper_proprio"]
    # source grid.py:159-160: run_bc_rm75.py for image and point-cloud embeddings alike
    elif suite == "metaworld_rm75":
        launcher = HERE / "run_bc_rm75.py"
        extra = []
    else:
        launcher = HERE / ("run_bc3d.py" if pcd else "run_bc.py")
        extra = []
    return [
        py,
        str(launcher),
        "--config-name",
        config_name,
        f"env={task}",
        f"embedding={emb_name}",
        f"seed={seed}",
        f"data_dir={demos}/",
        *extra,
        "wandb.project=local",
        "wandb.entity=local",
        f"hydra.run.dir={run_dir}",
    ]


def evals_of(run_dir):
    """The 20 evaluation records of one run (source grid.py / summarize_scene.py: lines with eval/epoch in
    metrics.jsonl)."""
    m = Path(run_dir) / "metrics.jsonl"
    return [json.loads(line) for line in m.read_text().splitlines() if '"eval/epoch"' in line] if m.exists() else []


def aggregate(runs):
    """runs: {task: {seed: [eval records]}} -> the train_policy metrics. Per run the highest eval/success of its 20
    evaluations; per task the mean over seeds; success = the mean over tasks."""
    per_task = {
        t: float(sum(max(e["eval/success"] for e in ev) for ev in s.values()) / len(s)) for t, s in runs.items()
    }
    evals = [
        {"task": t, "seed": int(seed), "epoch": int(e["eval/epoch"]), "success": float(e["eval/success"])}
        for t, s in runs.items()
        for seed, ev in s.items()
        for e in ev
    ]
    return {"success": sum(per_task.values()) / len(per_task), "success_per_task": per_task, "evals": evals}


def train(cfg, out_dir) -> dict:
    from jepa4r.config import stage_dir

    if cfg.env.suite in RELOCATE_ENVS:
        return relocate_policy(cfg, out_dir)
    out_dir = Path(out_dir)
    emb_name = f"{cfg.method.name}_{cfg.env.name}"
    conf_dir = out_dir / "conf" / "model"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / f"{emb_name}.yaml").write_text(yaml.safe_dump(embedding_config(cfg), sort_keys=False))
    demos = (
        stage_dir("prepare_policy_data", cfg.env.name)
        / SUITES[cfg.env.suite][2 if cfg.method.obs == "pointcloud" else 1]
    )
    env = dict(
        os.environ,
        JEPA4R_BC_CONF_HOME=str(out_dir),
        MUJOCO_GL="egl",
        WANDB_MODE="disabled",
        PYTHONPATH=os.pathsep.join([str(HERE.parents[2]), str(HERE)] + [os.environ.get("PYTHONPATH", "")]),
    )
    if cfg.env.suite == "ogbench":
        env.update(
            SCENE_PROPRIO_DIMS=str(cfg.policy.proprio.dims), MW_PY=os.environ.get("JEPA4R_PY_COLLECT", sys.executable)
        )
    # source grid.py:177-178: the RM75 env runs in the MuJoCo 3 / metaworld 3.0 python
    if cfg.env.suite == "metaworld_rm75":
        env.update(MW_PY=os.environ.get("JEPA4R_PY_COLLECT", sys.executable))
    runs = {}
    for seed in cfg.policy.eval.seeds:  # seed by seed, as the source's grid.py
        for task in tasks(cfg.env):
            run_dir = out_dir / task / f"s{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(f"{run_dir}.log", "w") as log:
                subprocess.run(
                    run_command(cfg, task, seed, run_dir, emb_name, demos),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=run_dir,
                    env=env,
                    check=True,
                )
            runs.setdefault(task, {})[int(seed)] = evals_of(run_dir)
    return aggregate(runs)


# ------------------------------------------------------------------------------------ Adroit Relocate and the RH56 twin
# The Relocate workflow (policy/bc/relocate, its data in data/relocate): one plan per (env, model) -- collection,
# world-model training, the official demonstrations, BC per seed -- run stage by stage by its runner
# (relocate/workflow.py). Each stage of this repository runs its share of that plan with the workflow's commands
# unchanged; the outputs stay at the workflow's paths under one run root per (env, method, tag), with the shared
# collection / demonstrations linked in; the runner's logs go to the stage dir.
RELOCATE_ENVS = {"adroit": "adroit-relocate", "rh56": "rh56-relocate"}  # env.suite -> the workflow's --env
RELOCATE_MODELS = {
    "jepa4r": "jepa4r",
    "lewm": "lewm",
    "pldm": "pldm",
    "lejepa": "lejepa",
    "visreg": "visreg",
    "vjepa21": "vjepa",
}  # method -> the workflow's --model
RELOCATE_STAGES = {  # stage -> the workflow's stage names (prefixes)
    "adroit": {
        "collect_wm_data": ("collect",),
        "train_wm": ("wm-train",),
        "prepare_policy_data": ("official-demos",),
        "train_policy": ("bc-seed",),
    },
    "rh56": {
        "collect_wm_data": ("rh56_prepare", "rh56_collect_", "rh56_publish"),
        "train_wm": ("rh56_wm",),
        "prepare_policy_data": ("rh56_official_demo",),
        "train_policy": ("rh56_bc_",),
    },
}
RELOCATE_EVAL_EPISODES = 25  # episodes per BC evaluation round (eval_num_traj of the workflow's BC stages)
RELOCATE_LINKS = {  # artifacts a run root takes from the shared stages: (owner stage, path under the run root)
    "adroit": {
        "train_wm": [("collect_wm_data", "data")],
        "train_policy": [("collect_wm_data", "data"), ("prepare_policy_data", "official")],
    },
    "rh56": {
        "train_wm": [("collect_wm_data", "rh56/data")],
        "prepare_policy_data": [("collect_wm_data", "rh56/inputs")],
        "train_policy": [
            ("collect_wm_data", "rh56/data"),
            ("collect_wm_data", "rh56/inputs"),
            ("prepare_policy_data", "rh56/demos"),
        ],
    },
}


def official_assets_dir():
    """The light copy of the official Relocate demonstrations + the RH56 initialisations
    (prepare_official_assets.py)."""
    from jepa4r.config import stage_dir

    return stage_dir("prepare_policy_data", "adroit_relocate") / "official_assets"


def ensure_official_assets():
    """Prepare (once) the light copy of the 100 official Relocate demos and the RH56 task initialisations from the
    original relocate-v0.pickle (JEPA4R_ADROIT_DEMOS)."""
    from jepa4r.config import paths

    official = official_assets_dir()
    if not (official / "manifest.json").is_file():
        raw = Path(
            os.environ.get("JEPA4R_ADROIT_DEMOS", paths()["data"] / "raw" / "adroit-expert-v1.0" / "relocate-v0.pickle")
        )
        official.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                os.environ.get("JEPA4R_PY_RELOCATE_WM", sys.executable),
                str(HERE.parents[1] / "data" / "demos" / "prepare_official_assets.py"),
                "--source",
                str(raw),
                "--output",
                str(official),
            ],
            check=True,
        )
    return official


def relocate_assets(cfg):
    """The workflow's asset / environment JSON from this repository's variables:
    JEPA4R_PY_RELOCATE_WM (WM stack), JEPA4R_PY_ADROIT_SIM (eai-vc + MuJoCo 2.0 / mujoco-py), JEPA4R_PY_RH56_SIM
    (MuJoCo 3.12), JEPA4R_SWM (stable-worldmodel 6f1e499 checkout), JEPA4R_EAI_VC_RELOCATE (eai-vc 76fe35e with
    third_party/eai-vc-relocate/eai-vc-train-loop.patch applied), JEPA4R_VJEPA2, JEPA4R_RH56_BUNDLE (the native
    RM75-B/RH56 bundle, configs/policy/bc/relocate/rh56_native_bundle_v1.json), MUJOCO_PY_MUJOCO_PATH /
    MUJOCO_PY_MJKEY_PATH; weights under checkpoints/."""
    from jepa4r.config import paths

    env = os.environ
    ck = paths()["checkpoints"]
    eai = Path(env.get("JEPA4R_EAI_VC_RELOCATE", HERE.parents[2] / "third_party" / "eai-vc"))
    swm = Path(env.get("JEPA4R_SWM", HERE.parents[2] / "third_party" / "stable-worldmodel"))
    mujoco = env.get("MUJOCO_PY_MUJOCO_PATH", "")
    official = official_assets_dir()
    assets = {
        "wm_python": env.get("JEPA4R_PY_RELOCATE_WM", sys.executable),
        "adroit_sim_python": env.get("JEPA4R_PY_ADROIT_SIM", sys.executable),
        "rh56_sim_python": env.get("JEPA4R_PY_RH56_SIM", sys.executable),
        "rh56_bc_python": env.get("JEPA4R_PY_RH56_SIM", sys.executable),
        "stable_worldmodel_root": str(swm),
        "eai_vc_root": str(eai),
        "mjrl_root": str(eai / "third_party" / "mjrl"),
        "mj_envs_root": str(eai / "third_party" / "mj_envs"),
        "mujoco_path": mujoco,
        "mujoco_key": env.get("MUJOCO_PY_MJKEY_PATH", ""),
        "adroit_expert": str(ck / "adroit" / "relocate-v0.pickle"),
        "adroit_official_demos": str(official / "demos"),
        "rh56_expert_root": env.get("JEPA4R_RH56_BUNDLE", str(ck / "rh56" / "adroit_rm75_rh56")),
        "rh56_official_task_init": str(official / "rh56_official_task_init.json"),
        "visreg_checkpoint": str(ck / "visreg" / "visreg-vit-b-inet1k.pth"),
        "vjepa_checkpoint": str(ck / "cortexbench_baselines" / "vjepa2_1_vitb_dist_vitG_384.pt"),
        "vjepa_root": env.get("JEPA4R_VJEPA2", str(HERE.parents[2] / "third_party" / "vjepa2")),
        "pythonpath": [str(swm), str(eai / "cortexbench" / "mujoco_vc" / "src"), str(eai / "vc_models" / "src")],
        "environment": {
            "MUJOCO_PY_MUJOCO_PATH": mujoco,
            "MUJOCO_PY_MJKEY_PATH": env.get("MUJOCO_PY_MJKEY_PATH", ""),
            "LD_LIBRARY_PATH": os.pathsep.join(
                x for x in (str(Path(mujoco) / "bin") if mujoco else "", env.get("LD_LIBRARY_PATH", "")) if x
            ),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
        },
    }
    if env.get("JEPA4R_DMC_IMPORT_OVERLAY"):  # DEPENDENCIES.json dmc2gym import_overlay
        assets["pythonpath"].append(env["JEPA4R_DMC_IMPORT_OVERLAY"])
    return assets


def relocate_run_root(cfg, stage):
    """Where the workflow's outputs of this stage live: data/<wm|policy>/<env>/workflow for the shared stages, the
    train_wm run dir (shared by train_policy) for the per-method stages."""
    from jepa4r.config import stage_dir

    if stage in ("collect_wm_data", "prepare_policy_data"):
        return stage_dir(stage, cfg.env.name) / "workflow"
    return stage_dir("train_wm", cfg.env.name, cfg.method.name, None, str(cfg.tag)) / "workflow"


def relocate_stage(cfg, stage, out_dir):
    """Run the workflow's stages that make up ``stage`` for (cfg.env, cfg.method); their logs go to out_dir/workflow.
    -> the run root of the stage's outputs"""
    from types import SimpleNamespace
    from jepa4r.policy.bc.relocate import adroit_workflow, rh56_workflow, workflow

    suite = cfg.env.suite
    if "method" in cfg and cfg.method.name not in RELOCATE_MODELS:
        raise SystemExit(
            f"the Relocate workflow has no {cfg.method.name!r} model (it runs {sorted(RELOCATE_MODELS)}); "
            "the paper's Random (ViT-B) row on Relocate is the VC-1 study's Table 8 value"
        )
    model = RELOCATE_MODELS[cfg.method.name] if "method" in cfg else "jepa4r"  # the shared stages do not depend on it
    if suite == "rh56" or stage == "prepare_policy_data":
        ensure_official_assets()  # RH56 collection starts from the official task initialisations
    run_root = relocate_run_root(cfg, stage)
    for owner, rel in RELOCATE_LINKS[suite].get(stage, []):
        target, link = relocate_run_root(cfg, owner) / rel, run_root / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
    assets = relocate_assets(cfg)
    args = SimpleNamespace(
        env=RELOCATE_ENVS[suite],
        model=model,
        seed=int(cfg.env.workflow.seed) if suite == "adroit" else 0,  # the Adroit collection's seed
        output=run_root,
        wm_python=assets["wm_python"],
        sim_python=assets[f"{suite}_sim_python"],
    )
    backend = adroit_workflow if suite == "adroit" else rh56_workflow
    stages = [s for s in backend.plan(args, assets) if s["name"].startswith(RELOCATE_STAGES[suite][stage])]
    logs = Path(out_dir) / "workflow"
    code = workflow.execute_plan(stages, logs)
    if code != 0:
        raise RuntimeError(f"{suite} {stage}: a workflow stage exited {code}; logs in {logs / 'logs'}")
    return str(run_root)


def relocate_policy(cfg, out_dir):
    """train_policy on Adroit Relocate / the RH56 twin: the workflow's BC per seed (100 / 200 / 300), then the paper's
    rule -- per seed the highest success of its 20 evaluations (25 episodes each, at epochs 5 .. 95, 99), the mean over
    seeds (Tab. 2) -- through the aggregation of the evaluation rounds (jepa4r.eval.aggregation.bc_summary), whose
    summary (per seed best, final and their means, the sample standard deviation of the bests) goes to
    out_dir/bc_summary.json."""
    from jepa4r.eval.aggregation import BC_EPOCHS, bc_summary

    relocate_stage(cfg, "train_policy", out_dir)
    root = relocate_run_root(cfg, "train_policy")
    seeds = tuple(int(s) for s in cfg.policy.eval.seeds)
    episodes = RELOCATE_EVAL_EPISODES
    runs, rounds = {}, []
    for seed in seeds:  # the workflow's BC record: bc-seed<s>/run/ (Adroit), rh56/bc_seed<s>/ (RH56)
        run = root / (f"bc-seed{seed}/run" if cfg.env.suite == "adroit" else f"rh56/bc_seed{seed}")
        records = evals_of(run)
        if records:
            runs.setdefault(str(cfg.env.env_id), {})[seed] = records
        for record in records:
            successes = round(record["eval/success"] * episodes / 100)
            rounds.append(dict(seed=seed, epoch=int(record["eval/epoch"]), successes=successes, episodes=episodes))
    if not runs:
        raise RuntimeError(f"{cfg.env.name} train_policy: no BC evaluation records under {root}")
    summary = bc_summary(rounds, seeds=seeds, epochs=BC_EPOCHS, episodes_per_round=episodes)
    (Path(out_dir) / "bc_summary.json").write_text(json.dumps(summary, indent=2))
    return aggregate(runs)
