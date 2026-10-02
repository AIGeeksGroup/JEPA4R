"""MetaWorld bin-picking (a CortexBench task) on the lab robot: RealMan RM75-B arm + Inspire EG2-4C2 gripper instead of
MetaWorld's Sawyer, as envs/cube_rm75_env.py does for the OGBench cube. Only the robot layer changes; the task (layout,
observation, reward, success), the camera, the collection plan and the eval hooks are envs.metaworld's.

    scene.py   the task's MuJoCo model with the RM75 in place of the Sawyer (on a pedestal on the table's +x side)
    robot.py   metaworld's task env on that model: RM75Mixin replaces the robot layer (IK -> joint servos, gripper)
    env.py     RM75BinPickEnv: envs.metaworld's MWBinPickEnv on robot.py's env           id swm/MWBinPickRM75-v0
    policy.py  RM75BinPickThreePartPolicy: the three-part collection policy, its pick-and-place oracle made for the RM75
    expert.py  RM75BinPickingPolicy: metaworld's scripted bin-picking expert made for the RM75 (the CortexBench demos)

Needs metaworld 3.0 (conda env ``lewm_mw``) like envs.metaworld; the env id is registered in envs/metaworld/__init__.py.
"""

from .env import RM75BinPickEnv, RM75TaskEnv
from .expert import RM75BinPickingPolicy
from .policy import RM75BinPickThreePartPolicy

__all__ = ['RM75TaskEnv', 'RM75BinPickEnv', 'RM75BinPickThreePartPolicy', 'RM75BinPickingPolicy']
