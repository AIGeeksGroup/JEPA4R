"""Stage ``verify_twin`` -- check a digital twin against the vendor URDF (Sec. 5.6): forward kinematics on 2,000
joint configurations, joint limits, masses and gripper mechanics (cfg.env.twin_check).

out: outputs/verify_twin/<env>/<tag>/metrics.json with the per-check errors / pass flags
     (and verify_twin.txt, the line-by-line report of the integrated check, cortexbench scripts/checks/verify_rm75.py).
"""


def run(cfg, out_dir):
    from jepa4r.envs.twins.rm75b import check_against_urdf

    return check_against_urdf(cfg.env, out_dir)
