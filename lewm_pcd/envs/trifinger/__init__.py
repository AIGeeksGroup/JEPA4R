"""CortexBench's TriFinger tasks (move_cube, reach_cube) as a stable-worldmodel env: CortexBench's own MoveCubeEnv /
CubeReachEnv with camera60's frame, the single-depth-camera point cloud and the eval hooks (env.TriFingerEnv,
env.TriFingerPCDWrapper), the experts of CortexBench's demos, reproduced (expert.MoveExpert / ReachExpert), and the
three-part collection policy around them (policy.TriFingerThreePartPolicy).

Needs pybullet and trifinger_simulation (conda env ``lewm_tf``, see bench/cortexbench/README.md), so the parent package
``envs`` does not import this one. Importing it registers the env id; configs name the env as
``envs.trifinger:swm/TriFinger-v0`` (gymnasium imports the module first) or instantiate
``envs.trifinger.TriFingerPCDWrapper``, which imports it too.
"""

from gymnasium.envs.registration import register

from .env import ACTION_MAX, EPISODE_STEPS, Camera60PCD, TriFingerEnv, TriFingerPCDWrapper
from .expert import MoveExpert, ReachExpert
from .policy import TriFingerThreePartPolicy

register(id='swm/TriFinger-v0', entry_point='envs.trifinger.env:TriFingerEnv')

__all__ = ['TriFingerEnv', 'TriFingerPCDWrapper', 'Camera60PCD', 'TriFingerThreePartPolicy', 'MoveExpert',
           'ReachExpert', 'EPISODE_STEPS', 'ACTION_MAX']
