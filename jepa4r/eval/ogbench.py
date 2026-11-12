"""OGBench evaluation: the five official tasks x 50 episodes (cfg.env.eval); success in % over all 250."""


def evaluate_policy(cfg, policy, encoder) -> dict:
    """-> {"success": %, "success_per_task": {task: %}}"""
    raise NotImplementedError("TODO(integrate)")
