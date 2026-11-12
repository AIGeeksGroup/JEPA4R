"""CortexBench behaviour cloning [majumdar2023vc1] (supp. sec:supp-policy, configs/policy/bc.yaml).

MetaWorld and Adroit Relocate follow the official CortexBench imitation protocol; OGBench Scene reuses the same
code on its five evaluation tasks. 100 epochs on env.policy_data.demos_per_task demonstrations per task
(25; 100 on Relocate); success = best of 20 evaluations (jepa4r/eval/cortexbench.py).
"""


def train(cfg, out_dir) -> dict:
    raise NotImplementedError("TODO(integrate): CortexBench BC on frozen latents")
