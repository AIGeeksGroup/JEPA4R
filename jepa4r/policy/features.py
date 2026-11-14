"""Latents for the policy learners.

cache_latents  encode data/policy/<env> once with a frozen encoder (offline training input)
LatentObs      env wrapper that encodes observations on the fly during evaluation, with the same
               point-cloud / image pipeline as the training data

The integrated learners load the frozen encoder in their own interpreter (JAX for DQC / GCIQL, eai-vc for CortexBench),
so ``encoder`` is where it is read from (jepa4r.policy.frozen.checkpoint) and both dispatch to the learner's package
(cfg.policy.impl):
  dqc    jepa4r/policy/dqc: extract_feats.py / extract_feats_baselines.py (cache), wm_obs.WMObs / BaselineObs
  gciql  jepa4r/policy/gciql: the Cube workflow's encode stage (cache) and its online evaluation's
         OnlineVisualEnv + OnlineEncoder (on the fly)
  bc     nothing cached: CortexBench embeds every frame on the fly (eai-vc's env wrappers with the embedding of
         jepa4r/policy/bc/wm_embeddings.py / jepa_baselines.py)
"""

import importlib


def cache_latents(cfg, encoder, out_path):
    """-> {name: path of the cached latents}"""
    return importlib.import_module(cfg.policy.impl).cache_latents(cfg, encoder, out_path)


class LatentObs:
    """Call it to get the learner's observation (latent + the task's extra inputs) of env's current state."""

    def __init__(self, env, encoder, cfg):
        self._obs = importlib.import_module(cfg.policy.impl).latent_obs(env, encoder, cfg)

    def __call__(self):
        return self._obs()
