"""FourCube switch over the original fixed-task collector, camera and plan oracle."""
import importlib
from triple_collection import (TripleStages, TripleFixedPlanPolicyMixin, make_official_triple_env,
                               make_triple_wrapper)

QUADRUPLE_ENV_ID = 'swm/OGBCubeOfficialFixedQuadruple-v0'

class QuadrupleStages(TripleStages):
    def __init__(self, init_xyzs, goal_xyzs, *, task_id):
        super().__init__(init_xyzs, goal_xyzs, task_id=task_id, num_cubes=4)

QuadrupleFixedPlanPolicyMixin = TripleFixedPlanPolicyMixin

def make_quadruple_wrapper(env, *, task_id, strategy, mixed_switch_step, capture):
    return make_triple_wrapper(env, task_id=task_id, strategy=strategy,
                               mixed_switch_step=mixed_switch_step, capture=capture,
                               num_cubes=4, stages_type=QuadrupleStages)

def register_quadruple_environment():
    import gymnasium as gym
    importlib.import_module('ogbench.manipspace')   # registers the OGBench manipulation envs
    if QUADRUPLE_ENV_ID not in gym.registry:
        gym.register(QUADRUPLE_ENV_ID, entry_point=make_official_triple_env, max_episode_steps=1000)
