"""Behaviour-cloning demonstrations (supp. sec:supp-policy): 25 per task on MetaWorld and on the five Scene
evaluation tasks, 100 on Adroit Relocate (cfg.env.policy_data.demos_per_task).
"""


def prepare_demos(cfg, out_dir):
    raise NotImplementedError("TODO(integrate): demonstrations of cfg.env into data/policy/<env>")
