"""Predictor g_phi (Sec. 3): six-block transformer, actions through AdaLN-zero [peebles2023dit], causal
attention over at most ``history`` (= 3) past latents; z_hat_{t+1} = g_phi(z_{1:t}, a_{1:t}). Inherited from LeWM.
"""


def build_predictor(method_cfg):
    raise NotImplementedError("TODO(integrate): LeWM predictor")
