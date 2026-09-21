"""Select and validate one labeled subset from a fixed-task UR5e HDF5."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.accepted_episode_h5 import (
    copy_accepted_episodes,
    validate_labeled_dataset,
)


def _write_json_atomic(path: Path, value: dict) -> None:
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def select_fixed_episodes(
    *,
    source: str | Path,
    output: str | Path,
    family: str,
    task_id: int,
    outcome: str,
    policy: str,
    episodes: int,
) -> dict:
    source = Path(source).resolve()
    output = Path(output).resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(output if output.exists() else manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        selected = copy_accepted_episodes(
            source,
            partial,
            outcome=outcome,
            limit=episodes,
            family=family,
            task_id=task_id,
            policy_kind=policy,
        )
        if selected != episodes:
            raise RuntimeError(f"selected {selected}/{episodes} requested episodes")
        policy_counts = {
            "expert": episodes if policy == "expert" else 0,
            "random": episodes if policy == "random" else 0,
        }
        report = validate_labeled_dataset(
            partial,
            family=family,
            task_id=task_id,
            expected_episodes=episodes,
            expected_policy_counts=policy_counts,
        )
        os.replace(partial, output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    _write_json_atomic(
        manifest_path,
        {
            "version": 1,
            "source": str(source),
            "output": str(output),
            "family": family,
            "task_id": int(task_id),
            "outcome": outcome,
            "policy": policy,
            "episodes": int(episodes),
            "report": report,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--family", required=True, choices=["double", "triple"])
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--outcome", required=True, choices=["success", "failure"])
    parser.add_argument("--policy", required=True, choices=["expert", "random"])
    parser.add_argument("--episodes", required=True, type=int)
    args = parser.parse_args()
    report = select_fixed_episodes(**vars(args))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
