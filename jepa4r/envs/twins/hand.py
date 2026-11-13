"""Twin of the hand / end effector mounted on the RM75-B (paper, Sec. 5.6: Inspire EG2-4C2 gripper; the twin
matches the URDF in gripper mechanics). Model files and details come from the integrated code.

The paper's twin "carries either end effector" (Sec. 5.6, Robot Setup): the Inspire EG2-4C2 parallel gripper (picking
and pushing a cube; OGBench Cube in the twin, configs/env/rm75b_cube.yaml) and the Inspire RH56 dexterous hand
(picking an object; the RM75-B + RH56 Relocate simulation of the C4 branch, configs/env/rm75b_rh56_relocate.yaml).
  eg2_4c2  descriptions/inspire_eg2_4c2/eg2_4c2.xml (+ STL meshes), cortexbench@9331494:lewm_pcd/envs/descriptions/:
           the vendor URDF's links / inertials / meshes (RealMan rm_models, EG2-4C2 + B-series adapter plate) on
           OGBench's 2f85.xml template, keeping OGBench's slot / entity names (robotiq, pinch, right_driver_joint, ...)
           so that the unchanged ManipSpaceEnv / CubeEnv code finds the gripper; attached to the arm flange exactly as
           RM75ManipSpaceEnv.build_mjcf_model does
  rh56     jepa4r.envs.twins.rh56 (OGBench-cube@4fbb0c6:lewm_pcd/rh56_adapter): adapter of the external native RM75-B +
           RH56 simulation bundle (its env.py, robot assets and expert are not part of any source branch)
"""
from pathlib import Path

DESCRIPTIONS = Path(__file__).resolve().parent / 'descriptions'
END_EFFECTORS = {  # cortexbench@9331494:lewm_pcd/envs/cube_rm75_env.py:46 (GRIPPER_XML)
    'eg2_4c2': DESCRIPTIONS / 'inspire_eg2_4c2' / 'eg2_4c2.xml',
}


def build(env_cfg):
    """The end effector of ``env_cfg.robot.end_effector``: the EG2-4C2 MJCF element ready to be attached to the arm's
    ``attachment_site`` (twin of the gripper), or the RH56 native-bundle adapter module (twin of the hand)."""
    ee = str(env_cfg.robot.end_effector)
    if ee == 'eg2_4c2':
        from dm_control import mjcf

        GRIPPER_XML = END_EFFECTORS['eg2_4c2']
        # Moved from cortexbench@9331494:lewm_pcd/envs/cube_rm75_env.py:RM75ManipSpaceEnv.build_mjcf_model (lines 108-109; verbatim)
        gripper_mjcf = mjcf.from_path(GRIPPER_XML.as_posix(), escape_separators=True)
        gripper_mjcf.model = 'robotiq'
        return gripper_mjcf
    if ee == 'rh56':  # the hand lives in the external native bundle; its adapter (RH56_EXPERT_ROOT, MuJoCo 3.12.0)
        from jepa4r.envs.twins.rh56 import native
        return native
    raise ValueError(f'unknown end effector {ee!r}; expected one of eg2_4c2, rh56')
