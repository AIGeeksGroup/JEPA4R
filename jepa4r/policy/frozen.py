"""Where the frozen encoder of cfg.method is read from. The integrated learners (CortexBench, DQC, GCIQL) load the
encoder in their own interpreter (no gradient, eval mode) from ``checkpoint(cfg)``.
"""


def checkpoint(cfg):
    """The train_wm run's checkpoints/ (kinds trained by us: config.json + weights_epoch_<n>.pt, the epoch read is
    cfg.method.policy_epoch, default 10), the released weights checkpoints/<cfg.method.weights> (pretrained kinds), or
    None (random initialisation)."""
    from jepa4r.config import paths, stage_dir
    from jepa4r.methods import trained_by_us

    m = cfg.method
    if trained_by_us(m):
        from jepa4r.policy.gciql import is_cube_workflow, wm_export

        if is_cube_workflow(cfg):  # Cube 1-4: the export the Cube workflow's train stage accepted
            return wm_export(cfg)
        return stage_dir("train_wm", cfg.env.name, m.name, None, str(cfg.tag)) / "checkpoints"
    if m.kind == "pretrained":
        return paths()["checkpoints"] / str(m.weights)
    return None
