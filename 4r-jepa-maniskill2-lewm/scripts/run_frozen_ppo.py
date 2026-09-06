#!/usr/bin/env python
"""Prepare/run stage 3 PPO from a trained consecutive-frame world-model encoder.

Preparation exports only encoder-through-latent2, snapshots its config and
records hashes. Actual rollout/training/evaluation use the upstream PPO CLI.
--prepare-only allocates no simulator or GPU and is safe while other jobs run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "vendor/ManiSkill2-Learn", ROOT / "vendor/pytorch3d",
          ROOT / "vendor/le-wm", ROOT / "vendor/stable-worldmodel-lewm-compatible"):
    sys.path.insert(0, str(p))


def prepare(args):
    from maniskill2_learn.utils.meta import Config
    from ms2_lewm.frozen_ppo import FrozenWorldModelEncoder, export_encoder_bundle

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    work_dir = Path(args.work_dir or ROOT / "artifacts/ppo_frozen_wm" / stamp).expanduser().resolve()
    if not work_dir.is_relative_to(ROOT):
        raise ValueError("Stage 3 outputs must remain inside this isolated task directory.")
    if args.resume_from and not Path(args.resume_from).is_file():
        raise FileNotFoundError(args.resume_from)
    if args.evaluation and not args.resume_from:
        raise ValueError("Evaluation requires --resume-from PPO_CHECKPOINT.")
    if work_dir.exists() and any(work_dir.iterdir()) and not (args.resume_from or args.evaluation):
        raise FileExistsError("Choose a new work directory or explicitly resume; existing outputs are preserved.")
    prepared = ROOT / "artifacts/frozen_ppo_prepared" / stamp
    prepared.mkdir(parents=True, exist_ok=False)
    bundle = export_encoder_bundle(args.wm_checkpoint, prepared / "encoder.pt")
    bundle_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    frozen = FrozenWorldModelEncoder(bundle, expected_sha256=bundle_sha)
    if args.resume_from:
        import torch
        resumed = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        state = resumed.get("state_dict", resumed)
        fingerprint = state.get("actor.backbone.visual_nn.source_sha256_bytes")
        if fingerprint is None or not torch.equal(fingerprint.cpu(), frozen.source_sha256_bytes.cpu()):
            raise ValueError("PPO checkpoint belongs to a different or unrecorded frozen world-model source.")
    test_source = Path(args.wm_checkpoint).resolve().is_relative_to(ROOT / "artifacts/tests")
    if test_source and not (args.smoke or args.prepare_only):
        raise ValueError("Synthetic test encoders are only for --smoke/--prepare-only, not production PPO.")
    cfg = Config.fromfile(str(ROOT / "configs/ppo_frozen_wm.py"))
    cfg.agent_cfg.actor_cfg.nn_cfg.visual_nn_cfg.checkpoint = str(bundle)
    cfg.agent_cfg.actor_cfg.nn_cfg.visual_nn_cfg.expected_sha256 = bundle_sha
    cfg.env_cfg.update(env_name="PickCube-v0", obs_mode="pointcloud", n_points=1200,
                       obs_frame="world", control_mode="pd_ee_delta_pose")
    cfg.env_cfg.extra_wrappers = [dict(type="PointCloudHistory", history_size=frozen.history_size, history_stride=1)]
    cfg.rollout_cfg.num_procs = args.num_envs
    cfg.eval_cfg.num_procs = args.num_envs
    if args.smoke:
        cfg.env_cfg.horizon = 16
        cfg.train_cfg.update(total_steps=32, n_steps=32, n_checkpoint=32, n_log=32, n_eval=None)
        cfg.replay_cfg.capacity = 32
        cfg.agent_cfg.batch_size = 16
        cfg.eval_cfg.update(num=2, save_video=False)
    if not args.evaluation:
        # Evaluation has its own explicit invocation, keeping one live renderer.
        cfg.eval_cfg = None
    provenance = dict(stage="ppo_with_frozen_latent2", source=frozen.provenance,
                      encoder_bundle=str(bundle), encoder_bundle_sha256=bundle_sha,
                      representation="latent2", latent_dim=frozen.output_dim,
                      history_size=frozen.history_size, history_stride=1,
                      action_dim=7, test_source=test_source, work_dir=str(work_dir),
                      actor_mlp=[512, 256, 256, 7], critic_mlp=[512, 256, 256, 1])
    cfg.frozen_encoder_provenance = provenance
    config_file = prepared / "ppo.py"
    cfg.dump(str(config_file))
    command = [sys.executable, str(ROOT / "scripts/launch_official_rl.py"), str(config_file),
               "--work-dir", str(work_dir), "--seed", str(args.seed)]
    command += ["--num-gpus", "0"] if args.device == "cpu" else ["--gpu-ids", "0"]
    if args.resume_from:
        command += ["--resume-from", str(Path(args.resume_from).resolve())]
    if args.evaluation:
        command += ["--evaluation"]
    provenance["command"] = command
    provenance["config"] = str(config_file)
    (prepared / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return cfg, provenance, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wm-checkpoint", required=True)
    parser.add_argument("--work-dir")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--resume-from")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    _, provenance, command = prepare(args)
    print(json.dumps(provenance, indent=2), flush=True)
    print("Prepared command: " + shlex.join(command), flush=True)
    if args.prepare_only:
        return
    from ms2_lewm.environment import ensure_renderer_headroom
    ensure_renderer_headroom(args.num_envs, extra_model_mib=8192 if args.device == "cuda" else 0)
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Select a CUDA-enabled environment for --device cuda.")
    work_dir = Path(provenance["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "frozen_encoder_provenance.json").write_text(json.dumps(provenance, indent=2))
    if hasattr(os, "nice") and os.nice(0) < 10:
        os.nice(10 - os.nice(0))
    raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()
