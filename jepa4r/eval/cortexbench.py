"""CortexBench evaluation (MetaWorld, Adroit Relocate; reused on Scene): roll out the BC policy at each of the
20 evaluations and report the best (cfg.policy.eval). Episodes / seeds per evaluation: cfg.policy.eval.
"""


def evaluate_policy(cfg, policy, encoder) -> dict:
    """-> {"success": %, ...} for one evaluation."""
    raise NotImplementedError("TODO(integrate)")
