"""CortexBench Adroit Relocate [rajeswaran2018dapg].

TODO(integrate): wrapper around the CortexBench Adroit env ``env_cfg.env_id`` that renders rgb + depth +
segmentation from the CortexBench camera; the visible goal marker is kept as a task object (Sec. 4.1).
"""


def make(env_cfg, **kwargs):
    raise NotImplementedError("TODO(integrate): CortexBench Adroit paired-observation wrapper")
