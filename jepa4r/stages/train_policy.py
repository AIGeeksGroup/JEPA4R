"""Stage ``train_policy`` -- policy head on the frozen latents of cfg.method; produces the success rates.

Paper: Sec. 4.3, Sec. 5.1, Sec. 5.3, supp. sec:supp-policy. Numbers: Tab. 1, Tab. 2, Tab. 4b-d.
in : the frozen encoder of cfg.method (outputs/train_wm/<env>/<method>/<tag>/, or released weights) and
     data/policy/<env>/
out: outputs/train_policy/<env>/<method>/<tag>/metrics.json
       success           float, %  -- the table entry (final or best evaluation, cfg.policy.eval.report)
       success_per_task  {task: %}
       evals             [{"step" | "epoch": int, "success": %}, ...]
Dispatch on cfg.policy (default cfg.env.policy): gciql (Cube), dqc (Puzzle), bc (Scene, MetaWorld, Adroit).
The encoder stays frozen; every encoder shares the same heads, data and evaluation episodes.
"""


def run(cfg, out_dir):
    from jepa4r.policy import train
    return train(cfg, out_dir)
