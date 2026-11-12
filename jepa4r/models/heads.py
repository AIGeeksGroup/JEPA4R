"""Projector (part of f_theta) and action encoder, inherited from LeWM (Sec. 3).

projector       encoder output -> z_t in R^192
action_encoder  action block (5 concatenated actions) -> embedding for the predictor's AdaLN-zero
"""


def build_projector(method_cfg):
    raise NotImplementedError("TODO(integrate): LeWM projector")


def build_action_encoder(method_cfg, action_dim):
    raise NotImplementedError("TODO(integrate): LeWM action encoder")
