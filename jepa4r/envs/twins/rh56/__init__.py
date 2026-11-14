"""The RM75-B + RH56 twin of Relocate, on an external native bundle (RH56_EXPERT_ROOT: its env, robot assets and the
56-D -> 12-D expert); importing this package loads no simulator.

native.py          the bundle's env and expert, rendered (RGB and point clouds of the fixed camera)
camera_preview.py  the opposite camera
collect.py         the paired RGB / point-cloud collection (world-model data and BC demonstrations)
official_init.py   the BC demonstrations from Adroit Relocate's official initialisations
data_tools.py      the published dataset over the collection batches
wm_bridge.py       world-model training (jepa4r/models/world_model.py)
bc_bridge.py       behavior cloning with CortexBench's BC loop
The stages and their command lines: jepa4r/policy/bc/relocate/rh56_workflow.py.
"""


# The env interface (configs/env/rm75b_rh56_relocate.yaml impl).
def make(env_cfg, seed=0, **kwargs):
    """The twin's env (native.NativeBridge: the bundle's env and expert, rendered); loads the simulator now."""
    from .native import NativeBridge

    return NativeBridge(seed, **kwargs)
