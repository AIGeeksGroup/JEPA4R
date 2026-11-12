"""GCIQL [park2025ogbench] on OGBench Cube (supp. sec:supp-policy, configs/policy/gciql.yaml).

Offline goal-conditioned RL on OGBench's official play data for 1e6 steps. Input: frozen latent and robot state;
goal: target state of the objects. Evaluation: 50 episodes on each of the five official tasks
(jepa4r/eval/ogbench.py).
"""


def train(cfg, out_dir) -> dict:
    raise NotImplementedError("TODO(integrate): GCIQL on frozen latents")
