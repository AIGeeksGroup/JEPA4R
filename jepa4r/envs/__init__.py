"""Simulator wrappers with paired observations from one fixed eye-to-hand camera (Sec. 4.1).

``make_env(env_cfg)`` imports ``env_cfg.impl`` and calls its ``make(env_cfg)``. Every wrapper provides, for the
camera that renders the benchmark's pixels (``env_cfg.camera.name``):
  rgb    uint8   [224, 224, 3]  input of the 2D encoders
  depth  float32 [H, W], metres the same camera treated as a time-of-flight sensor
  seg    int     [H, W]         renderer segmentation (drop table, floor, fixtures, invisible goal markers)
  camera intrinsics and extrinsics, to lift depth to world-frame points
plus what the policy learners read (proprioception, robot state, button lights, goals) and the evaluation tasks
with their success signal.
"""

import importlib


def make_env(env_cfg, **kwargs):
    return importlib.import_module(env_cfg.impl).make(env_cfg, **kwargs)
