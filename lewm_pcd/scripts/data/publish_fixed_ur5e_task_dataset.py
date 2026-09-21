"""Atomically publish one fixed-task train/holdout dataset pair."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py

from scripts.data.accepted_episode_h5 import validate_labeled_dataset
from scripts.data.merge_h5_shards import append_shard


def _write_json_atomic(path: Path, value):
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def publish_task_datasets(
    *,
    expert_train,
    random_failures,
    holdout,
    train_out,
    family,
    task_id,
    expert_train_episodes,
    random_failure_episodes,
    holdout_episodes,
):
    expert_train = Path(expert_train)
    random_failures = Path(random_failures)
    holdout = Path(holdout)
    train_out = Path(train_out)
    if train_out.exists():
        raise FileExistsError(train_out)

    expert_report = validate_labeled_dataset(
        expert_train,
        family=family,
        task_id=task_id,
        expected_episodes=expert_train_episodes,
        expected_policy_counts={"expert": expert_train_episodes, "random": 0},
    )
    random_report = validate_labeled_dataset(
        random_failures,
        family=family,
        task_id=task_id,
        expected_episodes=random_failure_episodes,
        expected_policy_counts={"expert": 0, "random": random_failure_episodes},
    )
    holdout_report = validate_labeled_dataset(
        holdout,
        family=family,
        task_id=task_id,
        expected_episodes=holdout_episodes,
        expected_policy_counts={"expert": holdout_episodes, "random": 0},
    )

    train_out.parent.mkdir(parents=True, exist_ok=True)
    partial = train_out.with_name(train_out.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        shutil.copy2(expert_train, partial)
        append_shard(partial, random_failures)
        with h5py.File(partial, "r+") as handle:
            handle.attrs["source_policy"] = "mixed"
            handle.attrs["accepted_outcome"] = "mixed"
        train_report = validate_labeled_dataset(
            partial,
            family=family,
            task_id=task_id,
            expected_episodes=expert_train_episodes + random_failure_episodes,
            expected_policy_counts={
                "expert": expert_train_episodes,
                "random": random_failure_episodes,
            },
        )
        os.replace(partial, train_out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    report = {
        "version": 1,
        "family": family,
        "task_id": int(task_id),
        "train_path": str(train_out.resolve()),
        "holdout_path": str(holdout.resolve()),
        "train": train_report,
        "holdout": holdout_report,
        "sources": {"expert_train": expert_report, "random_failures": random_report},
    }
    _write_json_atomic(train_out.with_suffix(train_out.suffix + ".manifest.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-train", required=True, type=Path)
    parser.add_argument("--random-failures", required=True, type=Path)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--train-out", required=True, type=Path)
    parser.add_argument("--family", required=True, choices=["double", "triple"])
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--expert-train-episodes", required=True, type=int)
    parser.add_argument("--random-failure-episodes", required=True, type=int)
    parser.add_argument("--holdout-episodes", required=True, type=int)
    args = parser.parse_args()
    report = publish_task_datasets(**vars(args))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
