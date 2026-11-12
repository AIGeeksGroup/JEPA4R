"""Decoupled Q-chunking (DQC) [li2025dqc] on OGBench Puzzle (supp. sec:supp-policy, configs/policy/dqc.yaml).

Offline goal-conditioned RL on the official play data, 1e6 steps. Input: frozen latent and button lights;
goal: target lights. Ten evaluations during training (five tasks x 50 episodes); report the best.
"""


def train(cfg, out_dir) -> dict:
    raise NotImplementedError("TODO(integrate): DQC on frozen latents")
