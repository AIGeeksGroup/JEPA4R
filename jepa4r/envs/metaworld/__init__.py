"""CortexBench MetaWorld: Assembly, Bin-pick, Button, Drawer, Hammer [majumdar2023vc1, yu2019metaworld].

Wrapper that renders rgb + depth + segmentation from the CortexBench camera, exposes proprioception and success, and
provides the expert control used by the data mixtures.

Modules (one per task because their module-level constants -- WAYPOINTS, HOLD, GRASP_OFF, ... -- differ; pcd.py's
multisample flag selects the render of the paper data, see configs/env/mw_*.yaml):
  env            MWTaskEnv (metaworld 3.0 <task>-v3-goal-observable, free camera reproducing CortexBench's top_cap2,
                 per-episode three-part plan) and MWBinPickEnv
  policy         ThreePartPolicy (backbone / branch / random) for bin-picking
  assembly, button_press, drawer, hammer   MW<Task>Env + <Task>ThreePartPolicy
  pcd            MWPointCloudWrapper (jepa4r.data.pointcloud.PointCloudWrapper with the task's fixtures)
World-model data use metaworld 3.0 (conda env lewm_mw); the CortexBench demos and BC evaluation use eai-vc's metaworld
fork on MuJoCo 2.0 (jepa4r.data.demos, jepa4r.policy.bc). The RM75-B + EG2-4C2 variants of four tasks (the
RM75-B digital twin, Sec. 5.6) are in jepa4r.envs.twins.metaworld_rm75; their env ids are registered here.
"""

from importlib import import_module

# Sawyer tasks
ENV_IDS = {
    "swm/MWBinPick-v0": "jepa4r.envs.metaworld.env:MWBinPickEnv",
    "swm/MWAssembly-v0": "jepa4r.envs.metaworld.assembly:MWAssemblyEnv",
    "swm/MWButtonPress-v0": "jepa4r.envs.metaworld.button_press:MWButtonPressEnv",
    "swm/MWDrawer-v0": "jepa4r.envs.metaworld.drawer:MWDrawerEnv",
    "swm/MWHammer-v0": "jepa4r.envs.metaworld.hammer:MWHammerEnv",
    # RealMan RM75-B + EG2-4C2 instead of the Sawyer (jepa4r.envs.twins.metaworld_rm75)
    "swm/MWBinPickRM75-v0": "jepa4r.envs.twins.metaworld_rm75:RM75BinPickEnv",
    "swm/MWButtonPressRM75-v0": "jepa4r.envs.twins.metaworld_rm75:RM75ButtonPressEnv",
    "swm/MWDrawerRM75-v0": "jepa4r.envs.twins.metaworld_rm75:RM75DrawerEnv",
    "swm/MWAssemblyRM75-v0": "jepa4r.envs.twins.metaworld_rm75:RM75AssemblyEnv",
}
_EXPORTS = {
    "MWTaskEnv": "env",
    "MWBinPickEnv": "env",
    "SAMPLE_TYPES": "env",
    "ThreePartPolicy": "policy",
    "MWAssemblyEnv": "assembly",
    "AssemblyThreePartPolicy": "assembly",
    "MWButtonPressEnv": "button_press",
    "ButtonThreePartPolicy": "button_press",
    "MWDrawerEnv": "drawer",
    "DrawerThreePartPolicy": "drawer",
    "MWHammerEnv": "hammer",
    "HammerThreePartPolicy": "hammer",
    "MWPointCloudWrapper": "pcd",
}
__all__ = list(_EXPORTS)


def register_envs():
    try:
        from gymnasium.envs.registration import register, registry
    except ModuleNotFoundError:  # gymnasium is needed to run the envs, not to import this package
        return
    for env_id, entry in ENV_IDS.items():
        if env_id not in registry:
            register(id=env_id, entry_point=entry)


register_envs()


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


def make(env_cfg, **kwargs):
    """One collection env (``env_cfg.wm_data.env_id``, a swm/MW*-v0 id) with its point-cloud wrapper
    (``env_cfg.wm_data.pcd``)."""
    import gymnasium

    from omegaconf import OmegaConf

    wm = env_cfg.wm_data
    kw = wm.get("env_kwargs")
    kw = OmegaConf.to_container(kw, resolve=True) if OmegaConf.is_config(kw) else dict(kw or {})
    env = gymnasium.make(str(wm.env_id), **{**kw, **kwargs})
    pcd = OmegaConf.to_container(wm.pcd, resolve=True)
    module, _, name = pcd.pop("_target_", "jepa4r.data.pointcloud.PointCloudWrapper").rpartition(".")
    return getattr(import_module(module), name)(env, **pcd)
