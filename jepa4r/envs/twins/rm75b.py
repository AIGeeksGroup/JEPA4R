"""RM75-B digital twin in OGBench Cube (Sec. 5.6): a MuJoCo twin of the RealMan RM75-B arm built from its vendor
URDF, matching the URDF in forward kinematics (2,000 joint configurations), joint limits, masses and gripper
mechanics; it replaces the UR5e in an OGBench Cube workspace scaled by 0.5 about its center (610 mm reach).
Same paired-observation contract as jepa4r/envs/__init__.py.

Modules:
  cube_rm75_env.py   RM75ManipSpaceEnv (OGBench's robot layer with the RM75-B + the
                     Inspire EG2-4C2 gripper of jepa4r/envs/twins/hand.py), RM75CubeEnv / RM75CubePCDEnv
                     (swm/OGBCubeRM75-v0 / swm/OGBCubeRM75PCD-v0), base at (0.15, 0, 0), boxes scaled by 0.5 about
                     (0.425, 0, 0.02), point-cloud crop x_min 0.21
  descriptions/      the MJCF models (realman_rm75b/rm75b.xml, inspire_eg2_4c2/eg2_4c2.xml) and their STL meshes
  verify_rm75.py     the acceptance check against the vendor URDFs (RealMan
                     rm_models RM75-B and EG2-4C2), datasheets, the UR5e env and the two point-cloud datasets
"""

import os
import tempfile
from importlib import import_module
from pathlib import Path

ENV_IDS = {  # gymnasium registrations
    "swm/OGBCubeRM75-v0": "jepa4r.envs.twins.cube_rm75_env:RM75CubeEnv",
    "swm/OGBCubeRM75PCD-v0": "jepa4r.envs.twins.cube_rm75_env:RM75CubePCDEnv",
}


def register_envs():
    try:
        from gymnasium.envs.registration import register, registry
    except ModuleNotFoundError:  # gymnasium is needed to run the twin, not to import this module
        return
    for env_id, entry in ENV_IDS.items():
        if env_id not in registry:
            register(id=env_id, entry_point=entry)


register_envs()


def make(env_cfg, **kwargs):
    """The twin env ``env_cfg.wm_data.env_id`` with the single-depth-camera PointCloudWrapper (``env_cfg.wm_data.pcd``,
    crop box x_min 0.21)."""
    import gymnasium

    from omegaconf import OmegaConf

    wm = env_cfg.wm_data
    kw = wm.get("env_kwargs")
    kw = OmegaConf.to_container(kw, resolve=True) if OmegaConf.is_config(kw) else dict(kw or {})
    env = gymnasium.make(str(wm.env_id), **{**kw, **kwargs})
    pcd = OmegaConf.to_container(wm.pcd, resolve=True)
    module, _, name = pcd.pop("_target_", "jepa4r.data.pointcloud.PointCloudWrapper").rpartition(".")
    return getattr(import_module(module), name)(env, **pcd)


def check_against_urdf(env_cfg, out_dir=None) -> dict:
    """Stage verify_twin: compare the twin with the vendor URDF (env_cfg.twin_check).

    Runs the acceptance script verify_rm75.py (sections A arm, B gripper, C interface, D behaviour with the plan
    oracle, E datasets) and returns its table: {"n_pass", "n_fail", "n_info", "checks": {"<section> | <name>":
    {...}}}."""
    import runpy

    from jepa4r.config import paths, stage_dir

    tc = env_cfg.twin_check
    vendor = tc.get("vendor_dir") or paths()["checkpoints"] / "vendor" / "realman"
    report = Path(out_dir or tempfile.mkdtemp()) / "verify_twin.txt"
    env = {
        "JEPA4R_TWIN_VENDOR": str(Path(vendor).expanduser()),
        "JEPA4R_TWIN_REPORT": str(report),
        "JEPA4R_TWIN_RM75_DS": str(stage_dir("collect_wm_data", env_cfg.name) / str(tc.datasets.twin)),
        "JEPA4R_TWIN_UR5E_DS": str(stage_dir("collect_wm_data", str(tc.datasets.ur5e_env)) / str(tc.datasets.ur5e)),
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        g = runpy.run_module("jepa4r.envs.twins.verify_rm75", run_name="verify_rm75")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    results = g["RESULTS"]
    checks = {
        f"{sec} | {name}": {"status": tag, "measured": str(meas), "expected": str(exp)}
        for tag, sec, name, meas, exp in results
    }
    return {
        "n_pass": sum(r[0] == "PASS" for r in results),
        "n_fail": sum(r[0] == "FAIL" for r in results),
        "n_info": sum(r[0] == "INFO" for r in results),
        "report": str(report),
        "checks": checks,
    }
