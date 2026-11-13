"""OGBench evaluation: the five official tasks x 50 episodes (cfg.env.eval); success in % over all 250.

Puzzle: DQC evaluates during training (run_dqc.py every 100k steps; jepa4r/policy/dqc); ``evaluate_policy`` evaluates a
finished run again with the same function (jepa4r.policy.dqc.evaluate_run; DQC interpreter). Cube: the Cube workflow's
native evaluation (stages native-sr, native-audit of train_policy; jepa4r.policy.gciql.evaluate_run reads it, and runs it
first when it has not run).
"""


def evaluate_policy(cfg, policy, encoder) -> dict:
    """policy: a run_dqc.py run directory (flags.json, resume.pkl); encoder: where the frozen encoder is read from
    (jepa4r.policy.frozen.checkpoint) -> {"success": %, "success_per_task": {task: %}}"""
    from jepa4r.policy.gciql import is_cube_workflow

    if is_cube_workflow(cfg):
        from jepa4r.policy.gciql import evaluate_run as evaluate_cube

        return evaluate_cube(cfg, policy, encoder)
    from jepa4r.policy.dqc import evaluate_run

    return evaluate_run(cfg, policy, encoder)
