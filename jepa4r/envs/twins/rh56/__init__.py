"""Opt-in native RH56 interfaces; importing this package loads no simulator.

The original Adroit entry points do not import this package. Expert source,
robot assets and the 56-D -> 12-D expert checkpoint are external artifacts.
"""


# The env interface (configs/env/rm75b_rh56_relocate.yaml impl).
def make(env_cfg, seed=0, **kwargs):
    """The RM75-B + RH56 twin of the external native bundle (RH56_EXPERT_ROOT) through the harness's bridge
    (native.NativeBridge: the bundle's env and expert, rendered for the opposite camera); loads the simulator now."""
    from .native import NativeBridge
    return NativeBridge(seed, **kwargs)
