"""Bit-exact trajectory equivalence: paper collection config vs. point-cloud config.

    python TMP/check_equivalence.py paper <out.h5>
    python TMP/check_equivalence.py pcd   <out.h5>

Both runs seed numpy's global RNG (used by the OGBench plan oracle) and the World
identically, so qpos / qvel / action / cube pose must be identical if the extra
cameras, the wrapper, ob_type='states' and add_pixels=False change nothing.
"""
import os, sys
from functools import partial
os.environ['MUJOCO_GL'] = 'egl'
import numpy as np
import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench import ExpertPolicy
from envs import PointCloudWrapper, PCDExpertPolicy  # registers swm/OGBCubePCD-v0

mode, out = sys.argv[1], sys.argv[2]
np.random.seed(20260909)

# upstream swm CubeEnv.reset() re-seeds its variation space from OS entropy on every
# reset (cube start/goal position come from there); give it a deterministic sequence
# instead so both configs draw identical episodes. Test-only patch.
import itertools
import stable_worldmodel.spaces as swm_spaces
_orig_reset_vs, _ctr = swm_spaces.reset_variation_space, itertools.count(5000)
def _det_reset_vs(variation_space, seed=None, options=None, default_variations=None):
    return _orig_reset_vs(variation_space, seed=next(_ctr), options=options, default_variations=default_variations)
swm_spaces.reset_variation_space = _det_reset_vs
common = dict(num_envs=2, max_episode_steps=200, image_shape=[224, 224], env_type='single',
              multiview=False, width=224, height=224, visualize_info=False,
              terminate_at_goal=False, mode='data_collection')
if mode == 'paper':
    world = swm.World('swm/OGBCube-v0', ob_type='pixels', **common)
    world.set_policy(ExpertPolicy(policy_type='plan_oracle'))
else:
    world = swm.World('swm/OGBCubePCD-v0', ob_type='states', add_pixels=False,
                      pre_wrappers=[partial(PointCloudWrapper)], **common)
    world.set_policy(PCDExpertPolicy(policy_type='plan_oracle'))
# upstream swm CubeEnv.reset() drops the `seed` argument, so the env RNG would be
# entropy-seeded; pin it explicitly (same in both configs) to allow a bit-exact test.
for i, e in enumerate(world.envs.envs):
    e.unwrapped.np_random = np.random.default_rng(1000 + i)
if os.path.exists(out): os.remove(out)
world.collect(out, episodes=4, seed=777, format='hdf5', progress=False)
print(mode, 'done ->', out)
