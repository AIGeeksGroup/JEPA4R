"""Collect fixed-task UR5e accepted episodes in recoverable process shards."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.accepted_episode_h5 import validate_labeled_dataset
from scripts.data.merge_h5_shards import append_shard


def episode_counts(total: int, shards: int) -> list[int]:
    if total <= 0 or shards <= 0 or shards > total:
        raise ValueError("episodes and shards must be positive, with shards <= episodes")
    base, remainder = divmod(total, shards)
    return [base + int(index < remainder) for index in range(shards)]


def shard_seeds(
    base_seed: int, counts: list[int], attempts_factor: int
) -> tuple[list[int], list[list[int]]]:
    if attempts_factor <= 0:
        raise ValueError("attempts_factor must be positive")
    stride = max(counts) * attempts_factor + 1024
    seeds = [int(base_seed) + index * stride for index in range(len(counts))]
    ranges = [
        [seed, seed + count * attempts_factor]
        for seed, count in zip(seeds, counts)
    ]
    return seeds, ranges


def shard_directory(output: Path) -> Path:
    return output.with_name(output.name + ".shards")


def shard_path(output: Path, index: int) -> Path:
    return shard_directory(output) / f"shard_{index:03d}.h5"


def manifest_path(output: Path) -> Path:
    return shard_directory(output) / "collection_manifest.json"


def merge_partial_path(output: Path) -> Path:
    return output.with_name(output.name + ".merge.partial.h5")


def make_manifest(args, output: Path) -> dict:
    output = output.resolve()
    counts = episode_counts(int(args.episodes), int(args.shards))
    seeds, ranges = shard_seeds(int(args.seed), counts, int(args.attempts_factor))
    return {
        "version": 1,
        "out": str(output),
        "family": args.family,
        "task_id": int(args.task_id),
        "policy": args.policy,
        "desired_outcome": "success" if args.policy == "expert" else "failure",
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "shards": int(args.shards),
        "max_parallel": int(args.max_parallel),
        "num_envs": int(args.num_envs),
        "batch_size": int(args.batch_size),
        "attempts_factor": int(args.attempts_factor),
        "threads": int(args.threads),
        "episode_counts": counts,
        "shard_seeds": seeds,
        "seed_ranges": ranges,
    }


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def assert_resume_matches(manifest: dict, args, output: Path) -> None:
    if manifest != make_manifest(args, output):
        raise ValueError("resume arguments do not match collection manifest")


def acquire_lock(directory: Path):
    lock = open(directory / ".collection.lock", "a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError(f"another launcher holds {directory}") from exc
    return lock


def release_lock(lock) -> None:
    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    lock.close()


def _policy_counts(policy: str, episodes: int) -> dict[str, int]:
    return {
        "expert": episodes if policy == "expert" else 0,
        "random": episodes if policy == "random" else 0,
    }


def validate_shard(manifest: dict, output: Path, index: int) -> dict:
    episodes = int(manifest["episode_counts"][index])
    return validate_labeled_dataset(
        shard_path(output, index),
        family=manifest["family"],
        task_id=int(manifest["task_id"]),
        expected_episodes=episodes,
        expected_policy_counts=_policy_counts(manifest["policy"], episodes),
    )


def recover_completed_shards(manifest: dict, output: Path) -> set[int]:
    completed = set()
    for index in range(int(manifest["shards"])):
        path = shard_path(output, index)
        if path.exists():
            validate_shard(manifest, output, index)
            completed.add(index)
    return completed


def remove_stale_merge(output: Path, *, resume: bool) -> None:
    if resume:
        merge_partial_path(output).unlink(missing_ok=True)


def build_child_command(manifest: dict, output: Path, index: int) -> list[str]:
    count = int(manifest["episode_counts"][index])
    child_output = shard_path(output, index)
    command = [
        sys.executable,
        str(Path(__file__).with_name("collect_fixed_ur5e_task_pcd.py")),
        "--family",
        manifest["family"],
        "--task-id",
        str(manifest["task_id"]),
        "--policy",
        manifest["policy"],
        "--episodes",
        str(count),
        "--seed",
        str(manifest["shard_seeds"][index]),
        "--out",
        str(child_output),
        "--num-envs",
        str(manifest["num_envs"]),
        "--batch-size",
        str(manifest["batch_size"]),
        "--max-attempts",
        str(count * int(manifest["attempts_factor"])),
        "--cleanup-work",
    ]
    if child_output.with_name(child_output.name + ".work").is_dir():
        command.append("--resume")
    return command


def _terminate_processes(active: dict[int, tuple[object, object]]) -> None:
    for process, _log in active.values():
        if process.poll() is None:
            process.terminate()
    for process, _log in active.values():
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def raise_interrupted(_signum, _frame) -> None:
    raise KeyboardInterrupt


def run_shards(
    manifest: dict,
    output: Path,
    *,
    completed: set[int],
    popen=subprocess.Popen,
    sleep=time.sleep,
) -> None:
    directory = shard_directory(output)
    log_directory = directory / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    pending = [
        index for index in range(int(manifest["shards"])) if index not in completed
    ]
    active: dict[int, tuple[object, object]] = {}
    env = dict(
        os.environ,
        MUJOCO_GL="egl",
        OMP_NUM_THREADS=str(manifest["threads"]),
        MKL_NUM_THREADS=str(manifest["threads"]),
        OPENBLAS_NUM_THREADS=str(manifest["threads"]),
    )
    try:
        while pending or active:
            while pending and len(active) < int(manifest["max_parallel"]):
                index = pending.pop(0)
                log = open(log_directory / f"shard_{index:03d}.log", "a")
                process = popen(
                    build_child_command(manifest, output, index),
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[index] = (process, log)

            progressed = False
            for index, (process, log) in list(active.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                log.close()
                del active[index]
                progressed = True
                if returncode != 0:
                    raise RuntimeError(
                        f"shard {index:03d} failed with exit code {returncode}"
                    )
                validate_shard(manifest, output, index)
            if active and not progressed:
                sleep(0.2)
    except BaseException:
        _terminate_processes(active)
        raise
    finally:
        for _process, log in active.values():
            if not log.closed:
                log.close()


def merge_shards_atomic(manifest: dict, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    partial = merge_partial_path(output)
    if partial.exists():
        raise FileExistsError(partial)
    sources = [
        shard_path(output, index) for index in range(int(manifest["shards"]))
    ]
    shutil.copy2(sources[0], partial)
    for source in sources[1:]:
        append_shard(partial, source)
    episodes = int(manifest["episodes"])
    report = validate_labeled_dataset(
        partial,
        family=manifest["family"],
        task_id=int(manifest["task_id"]),
        expected_episodes=episodes,
        expected_policy_counts=_policy_counts(manifest["policy"], episodes),
    )
    os.replace(partial, output)
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=["double", "triple"])
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--policy", required=True, choices=["expert", "random"])
    parser.add_argument("--episodes", required=True, type=_positive_int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--shards", required=True, type=_positive_int)
    parser.add_argument("--max-parallel", required=True, type=_positive_int)
    parser.add_argument("--num-envs", type=_positive_int, default=1)
    parser.add_argument("--batch-size", type=_positive_int, default=20)
    parser.add_argument("--attempts-factor", type=_positive_int, default=10)
    parser.add_argument("--threads", type=_positive_int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser


def _commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    args = build_parser().parse_args()
    output = args.out.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    directory = shard_directory(output)
    if args.resume:
        if not manifest_path(output).is_file():
            raise SystemExit(f"cannot resume without {manifest_path(output)}")
        manifest = load_manifest(manifest_path(output))
        try:
            assert_resume_matches(manifest, args, output)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        if directory.exists():
            raise SystemExit(f"refusing to reuse existing shard directory: {directory}")
        directory.mkdir(parents=True)
        manifest = make_manifest(args, output)
        write_json_atomic(manifest_path(output), manifest)

    try:
        lock = acquire_lock(directory)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    old_handlers = {
        signum: signal.signal(signum, raise_interrupted)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        remove_stale_merge(output, resume=args.resume)
        completed = recover_completed_shards(manifest, output)
        run_shards(manifest, output, completed=completed)
        report = merge_shards_atomic(manifest, output)
        final_manifest = {
            **manifest,
            "code_commit": _commit(),
            "shard_paths": [
                str(shard_path(output, index))
                for index in range(int(manifest["shards"]))
            ],
            "report": report,
        }
        write_json_atomic(output.with_suffix(output.suffix + ".manifest.json"), final_manifest)
        print(json.dumps(final_manifest, indent=2, sort_keys=True))
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        release_lock(lock)


if __name__ == "__main__":
    main()
