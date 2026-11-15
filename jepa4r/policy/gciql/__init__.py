"""GCIQL [park2025ogbench] on OGBench Cube (supp. sec:supp-policy, configs/policy/gciql.yaml).

Offline goal-conditioned RL on OGBench's official play data for 1e6 steps. Input: frozen latent and robot state;
goal: target state of the objects. Evaluation: 50 episodes on each of the five official tasks (the workflow's
evaluate stage).

The Cube 1-4 workflow of the paper's Cube runs: the controller (workflow.py) and the policy stages
(workflow_downstream.py, downstream/) here; collection and official play data in jepa4r/data/cube, world-model training
in jepa4r/models/cube, PLDM, LeJEPA and the released VISReg / V-JEPA 2.1 weights in jepa4r/baselines/cube
(docs/architecture.md). The controller runs each stage as a child process (python -m <module>) and keeps the stages
whose output exists. downstream/vendor/gciql.py is OGBench's GCIQL agent. The stages of this repository run the
workflow's stages for (cfg.env, cfg.method), in the workflow's own order:

  stage                workflow stages                                                  run root (the workflow's
  collect_wm_data      collect, views                                                   output_root)
                                                                                        data/wm/<env>/workflow
  prepare_policy_data  prepare-data                                                     data/policy/<env>/workflow
  train_wm             collect, views (kept when done), train                           outputs/train_wm/<env>/
                       (JEPA4R, LeWM, PLDM, LeJEPA)                                     <method>/<tag>/workflow
  train_policy         prepare-data (kept), prepare-encoder (VISReg, V-JEPA 2.1),       the same run root
                       camera, bind, encode, gciql, evaluate

The train_policy stages: prepare-encoder fetches the released weights; camera reads the collection camera from the
simulator; bind writes the run's downstream.json (encoder, camera, data); encode turns the official play data into
latents (the feature cache); gciql trains (1e6 updates); evaluate runs the policy online in the simulator (5 tasks x
50 episodes).

Shared inputs of every run root of an env: official play data data/policy/<env>/official/cube<N>, the paired collection
data/wm/<env>/collected/cube<N> and its training views data/wm/<env>/views/cube<N>, the released weights
checkpoints/<visreg|vjepa>/ (prepare-encoder downloads a missing file from the URL in
configs/method/cube/released_models.json). Interpreters: JEPA4R_PY_CUBE_WM (world model, collection, camera; the paper
runs: Python 3.10, NumPy 2.2.6, MuJoCo 3.3.7) and JEPA4R_PY_CUBE_RL (GCIQL, encoding, evaluation; JAX 0.7.2,
Flax 0.11.2, Torch). Checkouts: JEPA4R_OGBENCH (OGBench a977253, default third_party/ogbench), JEPA4R_VJEPA2
(V-JEPA 2 204698b, default third_party/vjepa2); stable-worldmodel is the vendored third_party/stable-worldmodel.
"""

import json
import os
import sys
from pathlib import Path

MODELS = {
    "jepa4r": "jepa4r",
    "lewm": "lewm",
    "pldm": "pldm",
    "lejepa": "lejepa",
    "visreg": "visreg",
    "vjepa21": "vjepa",
}  # this repository's methods -> the workflow's models (workflow.MODELS)
STAGES = {
    "collect_wm_data": ("collect", "views"),
    "prepare_policy_data": ("prepare-data",),
    "train_wm": ("collect", "views", "train"),
    "train_policy": ("prepare-data", "prepare-encoder", "camera", "bind", "encode", "gciql", "evaluate"),
}
SHARED_MODEL = "jepa4r"  # the model named in the method-independent run roots (the stages they run ignore it)


def is_cube_workflow(cfg):
    """An env this workflow runs: Cube 1-4 (configs/env/cube_*.yaml with wm_data.collector cube_workflow; not the
    reconstruction env cube_single_recon)."""
    from omegaconf import OmegaConf

    return OmegaConf.select(cfg, "env.wm_data.collector") == "cube_workflow"


def cube_count(cfg):
    return {"cube_single": 1, "cube_double": 2, "cube_triple": 3, "cube_quadruple": 4}[str(cfg.env.name)]


def run_root(cfg, stage):
    """The workflow output_root of this stage: one per env for the method-independent stages, one per (env, method,
    tag) for train_wm and train_policy."""
    from jepa4r.config import stage_dir

    if stage in ("collect_wm_data", "prepare_policy_data"):
        return stage_dir(stage, cfg.env.name) / "workflow"
    return stage_dir("train_wm", cfg.env.name, cfg.method.name, None, str(cfg.tag)) / "workflow"


def workflow_config(cfg, stage):
    """The workflow's run config of this stage's run root."""
    from jepa4r.config import paths, stage_dir

    n = cube_count(cfg)
    if stage in ("collect_wm_data", "prepare_policy_data"):
        model = SHARED_MODEL
    else:
        if cfg.method.name not in MODELS:
            raise SystemExit(f"the Cube workflow has no {cfg.method.name!r} model (it runs {sorted(MODELS)})")
        model = MODELS[cfg.method.name]
    wm = stage_dir("collect_wm_data", cfg.env.name)
    config = dict(
        cube_count=n,
        model=model,
        output_root=str(run_root(cfg, stage).resolve()),
        python=dict(
            world_model=os.path.abspath(os.environ.get("JEPA4R_PY_CUBE_WM") or sys.executable),
            downstream=os.path.abspath(os.environ.get("JEPA4R_PY_CUBE_RL") or sys.executable),
        ),
        official_root=str((stage_dir("prepare_policy_data", cfg.env.name) / "official" / f"cube{n}").resolve()),
        collection_root=str((wm / "collected" / f"cube{n}").resolve()),
        data_root=str((wm / "views" / f"cube{n}").resolve()),
        assets_root=str(paths()["checkpoints"].resolve()),
        workers=int(cfg.env.workflow.workers),
        prefetch=int(cfg.env.workflow.prefetch),
        collection_seed=0,
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        config["cuda_visible_devices"] = os.environ["CUDA_VISIBLE_DEVICES"]
    return config


def cube_stage(cfg, stage, out_dir=None):
    """Run the workflow stages that make up this repository's ``stage`` for (cfg.env, cfg.method); the stages whose
    output exists are kept. -> the stages and their outputs"""
    from jepa4r.policy.gciql import workflow

    config = workflow_config(cfg, stage)
    if os.environ.get("JEPA4R_TMPDIR"):  # the children's TMPDIR (workflow.child_environment), off the system disk
        Path(os.environ["JEPA4R_TMPDIR"]).mkdir(parents=True, exist_ok=True)
    plan = workflow.build_plan(config, "all")
    plan["stages"] = [row for row in plan["stages"] if row["stage"] in STAGES[stage]]
    print(json.dumps(plan, indent=2), flush=True)
    code = workflow.execute(plan)
    if code != 0:
        raise RuntimeError(
            f"{cfg.env.name} {stage}: a workflow stage failed (exit {code}); logs in {run_root(cfg, stage)}"
        )
    return {row["stage"]: row["output"] for row in plan["stages"]}


def collect_cube(cfg, out_dir):
    """collect_wm_data on Cube 1-4: the workflow's paired RGB / PCD collection and its training views."""
    return {"outputs": cube_stage(cfg, "collect_wm_data", out_dir)}


def prepare_official(cfg, out_dir):
    """prepare_policy_data on Cube 1-4: OGBench's public play data (official_data.py,
    configs/env/cube/official_sources.json)."""
    cube_stage(cfg, "prepare_policy_data", out_dir)
    official = json.loads(
        (Path(workflow_config(cfg, "prepare_policy_data")["official_root"]) / "OFFICIAL_DATA.json").read_text()
    )
    return {"dataset": official["train"]["path"], "val": official["val"]["path"]}


def train_world_model(cfg, out_dir):
    """train_wm on Cube 1-4 (JEPA4R, LeWM, PLDM, LeJEPA): the workflow's own trainers (fresh_train.py)."""
    return {"outputs": cube_stage(cfg, "train_wm", out_dir)}


def native_events(root):
    """The online evaluation of a run root, one event per episode (downstream/native/episodes/task*/episode_*/
    result.json): task, episode, success and the evaluated model."""
    events = []
    for path in sorted((Path(root) / "downstream" / "native" / "episodes").glob("task*/episode_*/result.json")):
        result = json.loads(path.read_text())
        events.append(
            dict(
                task=int(result["task_id"]),
                episode_id=int(result["episode_index"]),
                success=bool(result["success"]),
                model=result["model"],
            )
        )
    return events


def native_summary(cfg, root):
    """The success of a run root's online evaluation through the raw-event aggregation (jepa4r.eval.aggregation
    .cube_summary) over the tasks and episodes the evaluation ran (downstream/native/report.json; the paper protocol:
    5 tasks x 50 episodes): a missing task, a duplicated episode, a wrong denominator or a second model raises."""
    from jepa4r.eval.aggregation import cube_summary

    report = json.loads((Path(root) / "downstream" / "native" / "report.json").read_text())
    return cube_summary(
        native_events(root), tasks=tuple(report["tasks"]), episodes_per_task=report["episodes_per_task"]
    )


def train(cfg, out_dir) -> dict:
    """train_policy on Cube 1-4: encode the official play data with the frozen encoder, GCIQL (1e6 updates, seed 0),
    then the online evaluation of the final policy (5 tasks x 50 episodes)."""
    from .workflow_downstream import PROTOCOL

    cube_stage(cfg, "train_policy", out_dir)
    summary = native_summary(cfg, run_root(cfg, "train_policy"))
    success = summary["success_percent"]
    return {
        "success": success,
        "success_per_task": {f"task{row['task']}": row["success_percent"] for row in summary["tasks"]},
        "evals": [{"step": PROTOCOL["total_updates"], "success": success}],
    }


def wm_export(cfg):
    """The world-model export of train_wm (the run root's wm/WM_EXPORT.json): the directory of weights_epoch_<n>.pt
    and config.json from which the workflow's encode and evaluate stages read the frozen encoder."""
    export = json.loads((run_root(cfg, "train_wm") / "wm" / "WM_EXPORT.json").read_text())
    return Path(export["checkpoint"]).parent
