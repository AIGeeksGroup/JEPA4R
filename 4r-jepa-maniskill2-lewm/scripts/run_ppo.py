#!/usr/bin/env python3
"""GPU production PPO; explicit CPU compatibility modes, never silent fallback."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def build_command(mode="train", extra_args=()):
    if mode not in {"train", "train-cpu", "smoke", "eval-smoke"}:
        raise ValueError(f"Unknown PPO mode: {mode}")
    gpu = mode == "train"
    executable = ROOT / (".venv-rl-gpu" if gpu else ".venv-rl") / "bin/python"
    run_name = {"train": "rl_identity_train", "train-cpu": "rl_identity_train_cpu",
                "smoke": "rl_identity_smoke", "eval-smoke": "rl_identity_smoke"}[mode]
    cmd = [str(executable), str(ROOT / "scripts/launch_official_rl.py"),
           str(ROOT / "configs/ppo_pointnet_identity.py"),
           "--work-dir", str(ROOT / "artifacts" / run_name), "--seed", "20260906"]
    cmd += ["--gpu-ids", "0"] if gpu else ["--num-gpus", "0"]
    if mode == "eval-smoke":
        cmd += ["--evaluation", "--resume-from", str(ROOT / "artifacts/rl_identity_smoke/models/model_final.ckpt")]
    cmd += ["--cfg-options", "env_cfg.env_name=PickCube-v0", "env_cfg.obs_mode=pointcloud",
            "env_cfg.control_mode=pd_ee_delta_pose", "env_cfg.obs_frame=world", "env_cfg.n_points=1200"]
    if mode == "smoke":
        cmd += ["env_cfg.horizon=16", "rollout_cfg.num_procs=1", "train_cfg.total_steps=32",
                "train_cfg.n_steps=32", "train_cfg.n_checkpoint=32", "train_cfg.n_log=32",
                "train_cfg.n_eval=None", "replay_cfg.capacity=32", "agent_cfg.batch_size=16", "eval_cfg=None"]
    elif mode == "eval-smoke":
        cmd += ["env_cfg.horizon=16", "eval_cfg.num=2", "eval_cfg.num_procs=1", "eval_cfg.save_video=False"]
    else:
        # Production has exclusive resources and retains upstream's five workers.
        workers = 5 if gpu else 1
        cmd += [f"rollout_cfg.num_procs={workers}", "eval_cfg=None"]
    return cmd + list(extra_args)


def require_headroom(command):
    """Read memory before any CUDA probe/renderer; never stop another process."""
    gpu = "--gpu-ids" in command
    workers = 1
    for arg in command:
        if arg.startswith(("rollout_cfg.num_procs=", "eval_cfg.num_procs=")):
            workers = max(workers, int(arg.split("=", 1)[1]))
    required = max(2048, 1024 + 256 * workers) + (8192 if gpu else 0)
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                                  text=True, timeout=10)
    devices = [{"index": int(row.split(",")[0]), "free_mib": int(row.split(",")[1])}
               for row in raw.strip().splitlines()]
    report = {"devices": devices, "required_free_mib": required,
              "passed": bool(devices) and all(d["free_mib"] >= required for d in devices)}
    if not report["passed"]:
        raise RuntimeError("Insufficient free VRAM; no training/renderer started: " + json.dumps(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="train", choices=["train", "train-cpu", "smoke", "eval-smoke"])
    parser.add_argument("--print-command", action="store_true", help="Print only; no runtime/environment/GPU creation")
    args, extra = parser.parse_known_args(argv)
    command = build_command(args.mode, extra)
    if args.print_command:
        print(shlex.join(command))
        return
    if not Path(command[0]).is_file():
        setup = "setup_rl_gpu.sh" if args.mode == "train" else "setup_rl.sh"
        raise SystemExit(f"Missing selected environment. Run bash scripts/{setup}; CPU fallback is not automatic.")
    print(json.dumps(require_headroom(command)), flush=True)
    if args.mode == "train":
        subprocess.run([command[0], "-c", "import torch; assert torch.version.cuda and torch.cuda.is_available(), 'CUDA runtime unavailable'"], check=True)
    raise SystemExit(subprocess.call(["nice", "-n", "10", *command], cwd=ROOT))


if __name__ == "__main__":
    main()
