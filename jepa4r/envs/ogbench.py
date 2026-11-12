"""OGBench manipulation: Cube (1-4 cubes), Scene, Puzzle 3x3 / 4x4 / 4x5 [park2025ogbench].

TODO(integrate): wrapper that renders rgb + depth + segmentation from the camera behind OGBench's pixel
observations, exposes the robot state (Cube) and button lights (Puzzle), runs the five official evaluation tasks
(env_cfg.eval) and reports success, and provides the expert control used by the data mixtures.
"""


def make(env_cfg, **kwargs):
    raise NotImplementedError("TODO(integrate): OGBench paired-observation wrapper")
