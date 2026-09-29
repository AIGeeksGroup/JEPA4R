"""Resumable sharded paired Adroit Pen collection.

This orchestrator reuses run_adroit_paired_smoke.py for each independent shard,
then merges the paired pixel and point-cloud HDF5 files only after every shard
has a PASS report. It is intentionally separate from the bounded smoke output.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
LEWM_ROOT = HERE.parent.parent
SMOKE = LEWM_ROOT / "scripts" / "checks" / "run_adroit_paired_smoke.py"
MERGER = HERE / "merge_h5_shards.py"
ADD_EVAL = HERE / "add_eval_columns.py"


def now():
    return datetime.now(timezone.utc).isoformat()


def shard_counts(total: int, shards: int) -> list[int]:
    if total < 0 or shards <= 0:
        raise ValueError(f"total must be nonnegative and shards positive, got {total}, {shards}")
    base, remainder = divmod(total, shards)
    return [base + (1 if i < remainder else 0) for i in range(shards)]


def shard_seed_ranges(seed: int, counts: list[int]) -> list[tuple[int, int]]:
    ranges = []
    cursor = int(seed)
    for count in counts:
        end = cursor + int(count)
        ranges.append((cursor, end))
        cursor = end
    return ranges


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def dataset_files(split_dir: Path):
    return split_dir / "pen.h5", split_dir / "pen_pcd.h5"


def valid_shard(shard_dir: Path, expected_episodes: int) -> bool:
    report = load_json(shard_dir / "smoke_report.json")
    pix = shard_dir / "pen_smoke.h5"
    pcd = shard_dir / "pen_smoke_pcd.h5"
    if not report or report.get("status") != "PASS" or not pix.exists() or not pcd.exists():
        return False
    if int(report.get("episodes", -1)) != int(expected_episodes):
        return False
    with h5py.File(pix, "r") as a, h5py.File(pcd, "r") as b:
        return len(a["ep_len"]) == expected_episodes and len(b["ep_len"]) == expected_episodes


def check_paired(pixel_path: Path, pcd_path: Path, expected_episodes: int | None = None) -> dict:
    shared = ("ep_len", "ep_offset", "action", "observation", "qpos", "qvel", "sample_type",
              "rand_action", "noise_std", "branch_t", "branch_len", "seed", "step_idx", "reward", "goal_achieved")
    with h5py.File(pixel_path, "r") as pix, h5py.File(pcd_path, "r") as pcd:
        if expected_episodes is not None and len(pix["ep_len"]) != expected_episodes:
            raise ValueError(f"{pixel_path}: expected {expected_episodes} episodes, got {len(pix['ep_len'])}")
        for key in shared:
            if not np.array_equal(pix[key][:], pcd[key][:]):
                raise ValueError(f"paired mismatch in {key}")
        xyz = pcd["pcd_xyz"]
        if xyz.shape[1:] != (1200, 3) or not np.isfinite(xyz[:]).all():
            raise ValueError(f"invalid pcd_xyz shape/finite values: {xyz.shape}")
        if pix["pixels"].shape[1:] != (256, 256, 3):
            raise ValueError(f"invalid pixels shape: {pix['pixels'].shape}")
        lengths = pix["ep_len"][:].astype(np.int64)
        offsets = pix["ep_offset"][:].astype(np.int64)
        if not np.array_equal(offsets, np.concatenate(([0], np.cumsum(lengths)[:-1]))):
            raise ValueError("episode offsets are inconsistent")
        return {"episodes": int(len(lengths)), "rows_2d": int(len(pix["action"])),
                "rows_3d": int(len(pcd["action"])), "action_equal": True,
                "metadata_equal": True, "pcd_finite": True}


def dataset_qc(pixel_path: Path, pcd_path: Path, expected_episodes: int) -> dict:
    pairing = check_paired(pixel_path, pcd_path, expected_episodes)
    with h5py.File(pixel_path, "r") as f:
        lengths = f["ep_len"][:].astype(np.int64)
        offsets = f["ep_offset"][:].astype(np.int64)
        types = np.asarray([int(f["sample_type"][int(o)]) for o in offsets], dtype=np.int64)
        goals = int(f["goal_achieved"][:].sum())
        counts = np.bincount(types, minlength=3)
        return {**pairing, "episode_len_min": int(lengths.min()), "episode_len_max": int(lengths.max()),
                "sample_type_counts": counts.tolist(), "goal_rows": goals,
                "rows_expected_from_lengths": int(lengths.sum())}


def run_process(cmd: list[str], log_path: Path, env: dict) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(LEWM_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        return int(proc.wait())


def merge_split(split_dir: Path, output_dir: Path, expected_episodes: int) -> dict:
    shards_dir = split_dir / "shards"
    shard_files = sorted(shards_dir.glob("shard_*"))
    pixels = [p / "pen_smoke.h5" for p in shard_files if (p / "pen_smoke.h5").exists()]
    pcd = [p / "pen_smoke_pcd.h5" for p in shard_files if (p / "pen_smoke_pcd.h5").exists()]
    if len(pixels) != len(pcd) or not pixels:
        raise ValueError(f"incomplete shard outputs under {shards_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pixel_out, pcd_out = dataset_files(output_dir)
    for out in (pixel_out, pcd_out):
        if out.exists():
            raise FileExistsError(f"refusing to overwrite {out}")
    subprocess.run([sys.executable, str(MERGER), "--out", str(pixel_out), "--delete", *map(str, pixels)],
                   cwd=str(LEWM_ROOT), check=True)
    subprocess.run([sys.executable, str(MERGER), "--out", str(pcd_out), "--delete", *map(str, pcd)],
                   cwd=str(LEWM_ROOT), check=True)
    qc = dataset_qc(pixel_out, pcd_out, expected_episodes)
    for path in (pixel_out, pcd_out):
        subprocess.run([sys.executable, str(ADD_EVAL), str(path)], cwd=str(LEWM_ROOT), check=True)
    return qc


def run_split(split: str, split_dir: Path, output_dir: Path, episodes: int, shards: int, seed: int,
              episode_len: int, max_parallel: int, policy: Path, state: dict, state_path: Path, env: dict):
    counts = shard_counts(episodes, shards)
    ranges = shard_seed_ranges(seed, counts)
    shards_dir = split_dir / "shards"
    logs_dir = split_dir / "logs"
    shards_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    state[split] = {"stage": "running", "episodes": episodes, "shards": shards, "completed_shards": [],
                    "failed_shards": [], "seed_ranges": ranges, "started_at": now()}
    atomic_json(state_path, state)
    pending = list(range(shards))
    running = {}
    while pending or running:
        while pending and len(running) < max_parallel:
            i = pending.pop(0)
            shard_dir = shards_dir / f"shard_{i:03d}"
            if valid_shard(shard_dir, counts[i]):
                state[split]["completed_shards"].append(i)
                atomic_json(state_path, state)
                continue
            if shard_dir.exists() and not valid_shard(shard_dir, counts[i]):
                stale = shard_dir.with_name(shard_dir.name + ".incomplete")
                suffix = 1
                while stale.exists():
                    stale = shard_dir.with_name(shard_dir.name + f".incomplete{suffix}")
                    suffix += 1
                shard_dir.rename(stale)
            cmd = [sys.executable, str(SMOKE), "--episodes", str(counts[i]), "--episode-len", str(episode_len),
                   "--replay-episodes", "0", "--seed", str(ranges[i][0]), "--output-dir", str(shard_dir),
                   "--official-policy", str(policy)]
            proc = subprocess.Popen(cmd, cwd=str(LEWM_ROOT), env=env,
                                    stdout=(logs_dir / f"shard_{i:03d}.log").open("w"),
                                    stderr=subprocess.STDOUT)
            running[i] = (proc, shard_dir)
            state[split]["last_started_shard"] = i
            atomic_json(state_path, state)
        time.sleep(2)
        for i, (proc, shard_dir) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            del running[i]
            if code == 0 and valid_shard(shard_dir, counts[i]):
                state[split]["completed_shards"].append(i)
            else:
                state[split]["failed_shards"].append({"shard": i, "exit_code": code})
                atomic_json(state_path, state)
                raise RuntimeError(f"{split} shard {i} failed with exit code {code}")
            state[split]["completed_shards"] = sorted(set(state[split]["completed_shards"]))
            state[split]["completed_episodes"] = int(sum(counts[j] for j in state[split]["completed_shards"]))
            state[split]["updated_at"] = now()
            atomic_json(state_path, state)
    state[split]["stage"] = "shards_finished"
    state[split]["updated_at"] = now()
    atomic_json(state_path, state)
    qc = merge_split(split_dir, output_dir, episodes)
    state[split]["stage"] = "finished"
    state[split]["qc"] = qc
    state[split]["output_dir"] = str(output_dir)
    state[split]["updated_at"] = now()
    atomic_json(state_path, state)
    return qc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--test-episodes", type=int, default=150)
    ap.add_argument("--shards", type=int, default=40)
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3072)
    ap.add_argument("--test-seed", type=int, default=2_000_000)
    ap.add_argument("--episode-len", type=int, default=100)
    ap.add_argument("--official-policy", type=Path, required=True)
    args = ap.parse_args()
    if args.output_root.exists() and (args.output_root / "state.json").exists():
        state = load_json(args.output_root / "state.json")
        if state.get("stage") == "finished":
            print(json.dumps(state, indent=2))
            return
    elif args.output_root.exists() and any(p.name not in ("launch_command.txt", "launcher.log", "launcher.pid") for p in args.output_root.iterdir()):
        raise FileExistsError(f"output root is nonempty: {args.output_root}")
    else:
        args.output_root.mkdir(parents=True, exist_ok=True)
        state = {"stage": "initialized", "started_at": now(), "config": dict(vars(args), output_root=str(args.output_root), official_policy=str(args.official_policy))}
    state_path = args.output_root / "state.json"
    env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    atomic_json(state_path, state)
    try:
        state["stage"] = "test"
        atomic_json(state_path, state)
        test_qc = run_split("test", args.output_root / "test", args.output_root / "datasets" / "test",
                            args.test_episodes, 1, args.test_seed, args.episode_len, 1, args.official_policy,
                            state, state_path, env)
        state["stage"] = "train"
        atomic_json(state_path, state)
        train_qc = run_split("train", args.output_root / "train", args.output_root / "datasets" / "train",
                             args.episodes, args.shards, args.seed, args.episode_len, args.max_parallel,
                             args.official_policy, state, state_path, env)
        state["stage"] = "finished"
        state["exit_code"] = 0
        state["test_qc"] = test_qc
        state["train_qc"] = train_qc
        state["finished_at"] = now()
        atomic_json(state_path, state)
        print(json.dumps(state, indent=2))
    except Exception as exc:
        state["stage"] = "failed"
        state["exit_code"] = 1
        state["error"] = repr(exc)
        state["failed_at"] = now()
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    main()