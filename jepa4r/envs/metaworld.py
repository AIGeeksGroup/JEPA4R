"""CortexBench MetaWorld: Assembly, Bin-pick, Button, Drawer, Hammer [majumdar2023vc1, yu2019metaworld].

TODO(integrate): wrapper that renders rgb + depth + segmentation from the CortexBench camera, exposes
proprioception and success, and provides the expert control used by the data mixtures.
"""


def make(env_cfg, **kwargs):
    raise NotImplementedError("TODO(integrate): CortexBench MetaWorld paired-observation wrapper")
