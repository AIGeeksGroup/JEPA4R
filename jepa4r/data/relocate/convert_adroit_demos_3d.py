"""Point-cloud copies of the official Relocate demonstrations: each demonstration is replayed from its initial state
with its own actions, and its images are replaced by the visible-surface point clouds of the same camera (1200 points,
xyz and colour); every other field stays as it is."""

from copy import deepcopy
import json
import pickle

import numpy as np


def load_sources(source):
    """The official demonstrations, one at a time: a directory of per-demo files (prepare_official_assets.py, listed
    in the manifest.json next to it) or one pickle file. Returns (count, read) with read(index) -> demonstration."""
    if source.is_dir():
        records = json.loads((source.parent / "manifest.json").read_text(encoding="utf-8"))["demo_files"]

        def read(index):
            return pickle.loads((source.parent / records[index]["relative_path"]).read_bytes())

        return len(records), read
    demos = pickle.loads(source.read_bytes())
    return len(demos), lambda index: demos[index]


def convert_demo(demo, env, capture):
    """Replay one demonstration in env and return it with images = the point clouds (T x 1200 x 6) of capture()."""
    env.set_env_state(deepcopy(demo["init_state_dict"]))
    frames = []
    for action in np.asarray(demo["actions"]):
        xyz, rgb = capture()
        frames.append(np.concatenate([xyz, rgb.astype(np.float32)], axis=1))
        env.step(action.copy())
    return dict(demo, images=np.asarray(frames, dtype=np.float32))
