"""Latents for the policy learners.

cache_latents  encode data/policy/<env> once with a frozen encoder (offline training input)
LatentObs      env wrapper that encodes observations on the fly during evaluation, with the same
               point-cloud / image pipeline as the training data
"""


def cache_latents(cfg, encoder, out_path):
    raise NotImplementedError("TODO(integrate)")


class LatentObs:
    def __init__(self, env, encoder, cfg):
        raise NotImplementedError("TODO(integrate)")
