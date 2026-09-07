"""Report all collected operations without filtering successful or failed data."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def episode_coverage(episode):
    length = len(episode["pointcloud"])
    if length == 0:
        raise ValueError("Cannot describe an empty episode.")

    def column(name):
        if name not in episode:
            return None
        values = np.asarray(episode[name])
        if values.size != length or values.dtype.kind not in "biuf" or not np.isfinite(values).all():
            return None
        return values.reshape(length)

    def threshold(name, operation):
        values = column(name)
        return None if values is None else operation(values)

    grasp = threshold("object_grasped", lambda values: values.astype(bool))
    height = column("object_height_above_table")
    displacement = column("object_displacement")
    success = threshold("success", lambda values: values.astype(bool))
    stages = {
        "reached_object": threshold("tcp_object_distance", lambda values: values <= 0.06),
        "grasped": grasp,
        "lifted": None if grasp is None or height is None else grasp & (height >= 0.05),
        "carried": None if grasp is None or displacement is None else grasp & (displacement >= 0.08),
        # Geometric diagnostics remain measurable for older datasets whose
        # NumPy boolean fields were omitted by the official collection loop.
        # They must never be substituted for native grasp or task success.
        "object_elevated": None if height is None else height >= 0.05,
        "object_displaced": None if displacement is None else displacement >= 0.08,
        "at_goal": threshold("object_goal_distance", lambda values: values <= 0.025),
        "native_success": success,
    }
    lost_grasp = None if grasp is None else bool(np.any(grasp[:-1] & ~grasp[1:]))
    seed = column("seed")
    return {
        "seed": None if seed is None else int(seed[0]), "frames": length,
        "available": {name: values is not None for name, values in stages.items()},
        "episode_has": {name: None if values is None else bool(values.any()) for name, values in stages.items()},
        "stage_frames": {name: None if values is None else int(values.sum()) for name, values in stages.items()},
        "lost_grasp_after_grasp": lost_grasp,
        "final_transition_success": None if success is None else bool(success[-1]),
        "no_grasp": None if grasp is None else not bool(grasp.any()),
        "no_native_success": None if success is None else not bool(success.any()),
    }


class CoverageWriter:
    """Observe the official episode stream and forward every episode unchanged."""
    def __init__(self, writer, report_path):
        self.writer = writer
        self.report_path = Path(report_path)
        self.episodes_path = self.report_path.with_suffix(".episodes.jsonl")
        self.episodes = 0
        self.frames = 0
        self.episode_counts = Counter()
        self.frame_counts = Counter()
        self.episodes_available = Counter()
        self.last_episode = None
        self.status = "collecting"

    def __enter__(self):
        self.writer.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.status = "complete" if exc is None else "interrupted"
        try:
            return self.writer.__exit__(exc_type, exc, tb)
        finally:
            self._save()

    def _observe(self, episode):
        row = episode_coverage(episode)
        self.episodes += 1
        self.frames += row["frames"]
        for name, present in row["episode_has"].items():
            self.episodes_available[name] += int(present is not None)
            if present is not None:
                self.episode_counts[name] += int(present)
        for name in ("lost_grasp_after_grasp", "no_grasp", "no_native_success"):
            self.episodes_available[name] += int(row[name] is not None)
            if row[name] is not None:
                self.episode_counts[name] += int(row[name])
        self.frame_counts.update({name: count for name, count in row["stage_frames"].items() if count is not None})
        row["episode_number"] = self.episodes
        self.last_episode = row
        self.episodes_path.parent.mkdir(parents=True, exist_ok=True)
        with self.episodes_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if self.episodes <= 10 or self.episodes % 10 == 0:
            self._save()
        # No success/failure quota, rejection, trimming or resampling.
        return episode

    def _save(self):
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        counts = {name: self.episode_counts[name] if available else None
                  for name, available in self.episodes_available.items()}
        stage_counts = {
            name: self.frame_counts[name] if self.episodes_available[name] else None
            for name in (self.last_episode or {}).get("stage_frames", {})
        }
        report = {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "status": self.status, "episodes_observed": self.episodes,
            "frames_observed": self.frames,
            "episodes_with": counts,
            "stage_frame_counts": stage_counts,
            "available": {name: bool(count) for name, count in self.episodes_available.items()},
            "episodes_available": dict(self.episodes_available),
            "episodes_unknown": {name: self.episodes - count for name, count in self.episodes_available.items()},
            "last_episode": self.last_episode,
            "success_definition": "any native success transition during the episode, not only its last frame",
            "filtering": "none; all successes, failures and intermediate transitions are retained",
            "missing_data": "null means unknown; geometric elevation/goal proximity are not native grasp/success",
        }
        temporary = self.report_path.with_suffix(self.report_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(self.report_path)
        print("COLLECTION_COVERAGE " + json.dumps({
            "episodes": self.episodes, "frames": self.frames,
            "episodes_with": counts,
            "episodes_available": dict(self.episodes_available),
        }), flush=True)

    def write_episodes(self, episodes):
        return self.writer.write_episodes(self._observe(ep) for ep in episodes)

    def write_episode(self, episode):
        return self.writer.write_episode(self._observe(episode))
