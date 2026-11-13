"""Stage ``prepare_policy_data`` -- data for the policy head on frozen latents (supp. sec:supp-policy).

cfg.env.policy_data.source
  official_play  OGBench Cube / Puzzle: OGBench's official play data, with the learner inputs besides the latent
                 (robot state on Cube, button lights on Puzzle) and the goals (target state of the objects or
                 lights)                                                              -> data/official_play.py
  demos          Scene, MetaWorld, Adroit Relocate: demonstrations for behaviour cloning,
                 cfg.env.policy_data.demos_per_task per task (25; 100 on Relocate)   -> data/demos/
out: data/policy/<env>/
"""


def run(cfg, out_dir):
    source = cfg.env.policy_data.source
    if source == "official_play":
        from jepa4r.data.official_play import prepare_official_play
        return prepare_official_play(cfg, out_dir)
    if source == "demos":
        from jepa4r.data.demos import prepare_demos
        return prepare_demos(cfg, out_dir)
    raise ValueError(f"unknown env.policy_data.source {source!r}")
