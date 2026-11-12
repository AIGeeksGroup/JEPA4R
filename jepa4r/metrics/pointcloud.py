"""Point-cloud metrics of Tab. 3 / Tab. 4a, in the world frame (undo the affine normalisation), reported in cm.

nn_stats(pred, gt)       bidirectional nearest-neighbour distances between decoded and observed points
                         -> mean, median, p90, outlier_pct (share above 3 cm)
                         (confirm: pooled over both directions vs. mean of the two directional means)
cube_position_error      distance between the cube positions read from the two clouds
                         (rule: cfg.decoder.eval.cube_position)
"""


def nn_stats(pred_xyz, gt_xyz, outlier_cm=3.0) -> dict:
    raise NotImplementedError("TODO(integrate)")


def cube_position_error(pred, gt, rule) -> float:
    raise NotImplementedError("TODO(integrate)")
