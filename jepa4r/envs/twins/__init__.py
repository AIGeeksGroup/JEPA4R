"""Digital twins of the deployment hardware (Sec. 5.6). All world models, policy heads and decoders used on the
real hardware are trained only in these twins; the real-arm runs themselves are not part of this repository.

rm75b.py  RealMan RM75-B arm, MuJoCo twin from the vendor URDF, placed in OGBench Cube (configs/env/rm75b_cube.yaml)
cube_rm75_env.py, descriptions/, verify_rm75.py   the twin's Cube env with the Inspire EG2-4C2 gripper (paper), its
MJCF models and the URDF check (stage verify_twin)
metaworld_rm75/  four CortexBench MetaWorld tasks on the twin with the EG2-4C2 (configs/env/rm75b_mw_*.yaml)
rh56/     the twin with the Inspire RH56 hand: Relocate (configs/env/rm75b_rh56_relocate.yaml)
"""
