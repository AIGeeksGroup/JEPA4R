"""Policy heads pi_omega on frozen latents (Sec. 4.3). ``train(cfg, out_dir)`` dispatches to ``cfg.policy.impl``:

gciql.py  OGBench Cube    GCIQL, offline goal-conditioned RL, 1e6 steps, latent + robot state
dqc.py    OGBench Puzzle  decoupled Q-chunking, 1e6 steps, latent + button lights, best of 10 evaluations
bc.py     Scene, MetaWorld, Adroit Relocate   CortexBench behaviour cloning, 100 epochs, best of 20 evaluations
All of them read the encoder through jepa4r/policy/frozen.py and return the train_policy metrics
(see jepa4r/stages/train_policy.py).
"""
import importlib


def train(cfg, out_dir) -> dict:
    return importlib.import_module(cfg.policy.impl).train(cfg, out_dir)
