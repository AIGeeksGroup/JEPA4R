"""MetaWorld tasks of CortexBench on the lab robot: RealMan RM75-B arm + Inspire EG2-4C2 gripper instead of MetaWorld's
Sawyer, as envs/cube_rm75_env.py does for the OGBench cube. Only the robot layer changes; the task (layout, observation,
reward, success), the camera, the collection plan and the eval hooks are envs.metaworld's.

    scene.py   the task's MuJoCo model with the RM75 in place of the Sawyer (on a pedestal on the table's +x side)
    robot.py   metaworld's task env on that model: RM75Mixin replaces the robot layer (IK -> joint servos, gripper)
    env.py     envs.metaworld's task envs on robot.py's env:
                 RM75BinPickEnv       (MWBinPickEnv)       id swm/MWBinPickRM75-v0
                 RM75ButtonPressEnv   (MWButtonPressEnv)   id swm/MWButtonPressRM75-v0
                 RM75DrawerEnv        (MWDrawerEnv)        id swm/MWDrawerRM75-v0
                 RM75AssemblyEnv      (MWAssemblyEnv)      id swm/MWAssemblyRM75-v0
    policy.py  the three-part collection policies made for the RM75: RM75BinPickThreePartPolicy (bin-picking),
               RM75DrawerThreePartPolicy (drawer-open), RM75AssemblyThreePartPolicy (assembly); button-press-topdown
               collects with ButtonThreePartPolicy unchanged
    expert.py  metaworld's scripted experts made for the RM75 (the CortexBench demos): RM75BinPickingPolicy,
               RM75DrawerOpenPolicy, RM75AssemblyPolicy; button-press-topdown's demos use metaworld's script unchanged
               (bench/cortexbench/make_rm75_demos.py)

Needs metaworld 3.0 (conda env ``lewm_mw``) like envs.metaworld; the env id is registered in envs/metaworld/__init__.py.
"""

from .env import RM75AssemblyEnv, RM75BinPickEnv, RM75ButtonPressEnv, RM75DrawerEnv, RM75TaskEnv
from .expert import RM75AssemblyPolicy, RM75BinPickingPolicy, RM75DrawerOpenPolicy
from .policy import RM75AssemblyThreePartPolicy, RM75BinPickThreePartPolicy, RM75DrawerThreePartPolicy

__all__ = [
    "RM75TaskEnv",
    "RM75BinPickEnv",
    "RM75AssemblyEnv",
    "RM75ButtonPressEnv",
    "RM75DrawerEnv",
    "RM75BinPickThreePartPolicy",
    "RM75DrawerThreePartPolicy",
    "RM75AssemblyThreePartPolicy",
    "RM75BinPickingPolicy",
    "RM75DrawerOpenPolicy",
    "RM75AssemblyPolicy",
]
