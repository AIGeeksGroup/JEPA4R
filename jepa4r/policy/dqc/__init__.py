"""Decoupled Q-chunking (DQC) [li2025dqc] on OGBench Puzzle (supp. sec:supp-policy, configs/policy/dqc.yaml).

Offline goal-conditioned RL on the official play data, 1e6 steps. Input: frozen latent and button lights;
goal: target lights. Ten evaluations during training (five tasks x 50 episodes); report the best.

Modules (the DQC code of the paper runs):
  run_dqc.py       dqc df89825's training loop (the checkout, unchanged, at JEPA4R_DQC or third_party/dqc) with the
                   frozen encoder's latent + the true lights as observation; restartable (resume.pkl); metrics to
                   train.jsonl / eval.jsonl; the agent settings of the run in flags.json
  wm_obs.py        the frozen encoders as CortexBench loads them (jepa4r/policy/bc/wm_embeddings.py, jepa_baselines.py)
                   and the evaluation-time observations (WMObs, BaselineObs)
  extract_feats.py, extract_feats_baselines.py   every row of the official play data set back into the simulator,
                   rendered (front_pixels 224; point sampling seeded per frame with (20261011, split, row)) and encoded
                   (epoch 10) -> one latent per row
``train``: the latents of the official training set (``cache_latents``), one run_dqc.py run with the board's agent
settings (``agent_flags``), then the train_policy metrics (``aggregate``): success = the best of the ten evaluations
during training (steps 100k, 200k, ..., 1M). run_dqc.py, unchanged, also evaluates the untrained agent at step 0: that
evaluation draws from numpy's global RNG, which also drives the batch sampling, so it stays in the run (dropping it
would change the training) and is listed in ``evals`` but not counted. In all 18 Puzzle runs of the paper it is 0.0,
so the best of ten equals the best of the eleven records.
Interpreter: JEPA4R_PY_DQC (JAX 0.6.2 / flax 0.10.7, MuJoCo 3 + OGBench, torch for the encoders).
"""

import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOADERS = ("vjepa21", "visreg", "lejepa", "pldm")  # wm_obs.BaselineEncoder.LOADERS (run_dqc.py --obs <name>)


def obs_kind(method_cfg):
    """run_dqc.py --obs of an encoder row: jepa4r / lewm (LeWM recipe on point clouds / images) or the baseline."""
    if method_cfg.kind == "jepa_wm":
        return "jepa4r" if method_cfg.obs == "pointcloud" else "lewm"
    if method_cfg.name in LOADERS:
        return str(method_cfg.name)
    raise ValueError(f"no DQC input for method {method_cfg.name!r}")


def board(env_cfg):
    """puzzle-3x3-v0 -> 3x3 (extract_feats.py --env-type)."""
    return str(env_cfg.env_id).split("-")[1]


def task_env(env_cfg):
    """DQC's env name: the play dataset with OGBench's oracle goal representation (the goal = the lights' target state),
    puzzle-<b>-play-v0 -> puzzle-<b>-play-oraclerep-v0."""
    head, _, version = str(env_cfg.policy_data.dataset).rpartition("-")
    return f"{head}-oraclerep-{version}"


def agent_flags(cfg):
    """run_dqc.py --agent.* : cfg.policy.hparams.agent, then the board's entry of hparams.per_env (glob on env.name);
    every other setting is dqc's agents/dqc.py default."""
    from omegaconf import OmegaConf

    hp = dict(OmegaConf.to_container(cfg.policy.hparams.agent, resolve=True))
    for pattern, patch in (cfg.policy.hparams.get("per_env") or {}).items():
        if fnmatch.fnmatchcase(cfg.env.name, str(pattern)):
            hp.update(OmegaConf.to_container(patch, resolve=True))
    return [f"--agent.{k}={v}" for k, v in hp.items()]


def _env():
    from jepa4r.config import paths

    # STABLEWM_HOME: read by the scripts only for defaults that every call here overrides (--data, --dataset_path)
    return dict(
        os.environ,
        MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        STABLEWM_HOME=os.environ.get("STABLEWM_HOME") or str(paths()["data"]),
        PYTHONPATH=os.pathsep.join([str(HERE.parents[2]), os.environ.get("PYTHONPATH", "")]),
    )


def cache_latents(cfg, encoder, out_path):
    """The latents of the official training set through the frozen encoder read from ``encoder`` (a checkpoint
    directory, or the released weights file of VISReg / V-JEPA 2.1): extract_feats.py (LeWM recipe) or
    extract_feats_baselines.py (the baselines), restartable -> {kind: path of train_z<kind>.npy}."""
    from jepa4r.config import stage_dir

    kind, out_path = obs_kind(cfg.method), Path(out_path)
    epoch = str(int(cfg.method.get("policy_epoch", 10)))
    if kind in LOADERS:
        script, args, z = "extract_feats_baselines.py", ["--encoder", f"{kind}={encoder}"], kind
    else:
        script, args, z = "extract_feats.py", [f"--ckpt-{kind}", str(encoder)], kind
    done = out_path / (f"train_done_{kind}.npy" if kind in LOADERS else "train_done.npy")
    if not done.exists() or not _all_done(done):
        cmd = [
            os.environ.get("JEPA4R_PY_DQC", sys.executable),
            str(HERE / script),
            *args,
            "--epoch",
            epoch,
            "--env-type",
            board(cfg.env),
            "--data",
            str(stage_dir("prepare_policy_data", cfg.env.name)),
            "--out",
            str(out_path),
            "--split",
            "train",
            "--workers",
            str(int(cfg.policy.latents.workers)),
        ]
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "extract.log", "a") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE, env=_env(), check=True)
    return {kind: str(out_path / f"train_z{z}.npy")}


def _all_done(path):
    import numpy as np

    return bool(np.load(path).all())


def run_command(cfg, encoder, feats_dir, run_dir):
    """The run_dqc.py command of one run (the paper runs' arguments)."""
    from jepa4r.config import stage_dir

    p = cfg.policy
    data = stage_dir("prepare_policy_data", cfg.env.name) / f"{cfg.env.policy_data.dataset}.npz"
    return [
        os.environ.get("JEPA4R_PY_DQC", sys.executable),
        str(HERE / "run_dqc.py"),
        "--obs",
        obs_kind(cfg.method),
        "--ckpt",
        str(encoder),
        "--wm_epoch",
        str(int(cfg.method.get("policy_epoch", 10))),
        "--feats_dir",
        str(feats_dir),
        "--run_dir",
        str(run_dir),
        *agent_flags(cfg),
        f"--env_name={task_env(cfg.env)}",
        f"--dataset_path={data}",
        f"--offline_steps={int(p.train_steps)}",
        f"--eval_interval={int(p.eval.interval)}",
        f"--eval_episodes={int(p.eval.episodes)}",
        f"--save_every={int(p.save_every)}",
        f"--seed={int(p.seed)}",
    ]


def aggregate(records, num_evals=10, interval=100000):
    """eval.jsonl records -> the train_policy metrics: success = the highest overall success (%) among the evaluations
    during training (steps interval, ..., num_evals x interval); success_per_task at that evaluation; every record
    (incl. step 0) in evals."""
    during = [r for r in records if 0 < int(r["step"]) <= num_evals * interval]
    if len(during) != num_evals:
        raise RuntimeError(f"{len(during)} of {num_evals} evaluations during training found")
    best = max(during, key=lambda r: r["overall_success"])
    per_task = {
        k[: -len("_success")]: 100 * float(v)
        for k, v in best.items()
        if k.endswith("_success") and k != "overall_success"
    }
    return {
        "success": 100 * float(best["overall_success"]),
        "success_per_task": per_task,
        "evals": [{"step": int(r["step"]), "success": 100 * float(r["overall_success"])} for r in records],
    }


def train(cfg, out_dir) -> dict:
    from jepa4r.policy.frozen import checkpoint

    out_dir = Path(out_dir)
    encoder = checkpoint(cfg)
    feats_dir, run_dir = out_dir / "feats", out_dir / "run"
    cache_latents(cfg, encoder, feats_dir)
    (run_dir / ".tmp").mkdir(parents=True, exist_ok=True)
    env = dict(_env(), TMPDIR=str(run_dir / ".tmp"), XLA_PYTHON_CLIENT_PREALLOCATE="false")
    if not (run_dir / "DONE").exists():  # run_dqc.py resumes from resume.pkl
        with open(out_dir / "run.log", "a") as log:
            subprocess.run(
                run_command(cfg, encoder, feats_dir, run_dir),
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=HERE,
                env=env,
                check=True,
            )
    records = [json.loads(line) for line in (run_dir / "eval.jsonl").read_text().splitlines() if line.strip()]
    return aggregate(records, int(cfg.policy.eval.num_evals), int(cfg.policy.eval.interval))
