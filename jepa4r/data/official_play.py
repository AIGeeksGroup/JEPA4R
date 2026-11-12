"""OGBench's official play data (Cube, Puzzle) as training data for the policy learners (supp. sec:supp-policy):
besides the frozen latent, the input carries the robot state (Cube) or the button lights (Puzzle), and the goal
gives the target state of the objects or the lights.
"""


def prepare_official_play(cfg, out_dir):
    raise NotImplementedError("TODO(integrate): official play data of cfg.env into data/policy/<env>")
