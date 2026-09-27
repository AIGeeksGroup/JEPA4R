"""Three-part collection policy for the MetaWorld envs: follows each episode's plan (sample type, noise level, random
steps). ``ThreePartPolicy`` is the bin-picking one; a task subclass replaces the backbone controller (``_oracle``, its
memory ``_new_mem``) and the lowest hand height of random actions (``LOW_Z``), e.g. assembly.AssemblyThreePartPolicy.

* backbone (type 0): a pick-and-place oracle written like metaworld's SawyerBinPickingV3Policy (P gain 25, travel
  above the bin walls, grasp-point y clamp, grab effort 0.6 / -1), but with the place target as a parameter: after each
  release the env draws a new random target (the other bin with probability 0.7), so an episode chains pick-and-place
  sub-tasks the way OGBench's data collection does. Travel heights are 0.15 m instead of the metaworld policy's
  0.18-0.20 m so the hand stays inside top_cap2's view. Gaussian noise with the episode's std (drawn in [0, 0.2],
  up to the 0.1 x action range metaworld's own scripted-policy tests use) is added to every action.
* branch (type 1): the same, but from ``branch_t`` on the actions are uniform random, either to the end of the
  episode (tail) or for ``branch_len`` steps after which the oracle takes over again (burst).
* random (type 2): uniform random actions in [-1, 1]^4 every step.
Random actions are uniform, except that the hand is kept inside the camera's view: a random xyz step that would
leave it (or go below the lowest hand height) is flipped in the offending components. Without this, a random walk
from a random start leaves the frame and both the image and the point cloud (same camera) lose the robot.
The env reports per row whether that row's action is random (``rand_action``), so the policy only reads the plan.
"""

from __future__ import annotations

import numpy as np
from stable_worldmodel.policy import BasePolicy

P_GAIN = 25.0
APPROACH_DZ = 0.10  # approach from this far above the grasp point (metaworld policy: 0.15)
TRAVEL_Z = 0.15     # carry height (metaworld policy: 0.18)
LIFT_Z = 0.12       # lift straight up below this before travelling (metaworld policy: 0.15)
PLACE_Z = 0.07      # lowest reachable hand height (mocap_low z)
GRAB, RELEASE = 0.6, -1.0
STEP = 0.01         # metaworld moves the hand target by action * 1 cm


def _grasp_point(cube, clamp=True):
    p = cube + np.array([0.0, 0.0, 0.03])
    if clamp and 0.64 < p[1] < 0.76:  # inside a bin: aim centrally in y so the fingers do not catch the wall
        p[1] = min(max(p[1], 0.675), 0.725)
    return p


class ThreePartPolicy(BasePolicy):
    LOW_Z = PLACE_Z                          # a random action never heads below this hand height
    VIEW_CENTRE = np.array([0.0, 0.70, 0.14])  # random steps leaving the view are reflected towards this point

    def __init__(self, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.type = 'three_part'
        self.rng = np.random.default_rng(seed)
        self._mem = []

    def set_env(self, env):
        self.env = env
        self._mem = [None] * env.num_envs

    @staticmethod
    def _new_mem():
        return {'phase': 'pick', 'n': 0, 'fails': 0}

    def get_action(self, info_dict, **kwargs):
        envs = [e.unwrapped for e in self.env.envs]
        actions = np.zeros((len(envs), 4), dtype=np.float32)
        for i, env in enumerate(envs):
            info = {k: v[i][0] for k, v in info_dict.items() if not k.startswith('_')}
            if int(info['step_idx']) == 0 or self._mem[i] is None:
                self._mem[i] = self._new_mem()
            if bool(info['rand_action']):
                a = self._random_action(env, np.asarray(info['hand_pos'], dtype=np.float64))
            else:
                a = self._oracle(self._mem[i], env, info)
                a = a + self.rng.normal(0.0, float(info['noise_std']), 4)
            actions[i] = np.clip(a, -1.0, 1.0)
        return actions

    def _random_action(self, env, hand):
        a = self.rng.uniform(-1.0, 1.0, 4)
        ahead = hand + a[:3] * STEP * 3  # where three such steps would take the hand
        if env.in_view(ahead) and ahead[2] >= self.LOW_Z:
            return a
        away = (self.VIEW_CENTRE - hand) * a[:3] < 0  # reflect: flip every component that points away from the view centre
        a[:3][away] = -a[:3][away]
        return a

    @staticmethod
    def _oracle(m, env, info):
        hand = np.asarray(info['hand_pos'], dtype=np.float64)
        cube = np.asarray(info['obj_pos'], dtype=np.float64)
        target = np.asarray(info['place_target'], dtype=np.float64)
        grasp = _grasp_point(cube, clamp=m['fails'] % 2 == 0)  # after a failed grasp, try the unclamped point
        grab = RELEASE
        phase = m['phase']

        if phase == 'carry' and np.linalg.norm(hand - cube) > 0.06:  # dropped it (noise, a random burst): pick again
            phase, m['fails'] = 'pick', m['fails'] + 1
        if phase == 'pick':
            if np.linalg.norm(hand[:2] - grasp[:2]) > 0.02:
                desired = grasp + np.array([0.0, 0.0, APPROACH_DZ])
            elif abs(hand[2] - grasp[2]) > 0.01:
                desired = grasp
                grab = GRAB if abs(hand[2] - grasp[2]) <= 0.02 else RELEASE
            else:
                desired, grab = grasp, GRAB
                m['n'] += 1
                if m['n'] >= 6:  # fingers closed for a few steps: lift and carry
                    phase, m['n'] = 'carry', 0
        if phase == 'carry':
            grab = GRAB
            if np.linalg.norm(hand[:2] - target[:2]) > 0.02:
                desired = hand + np.array([0.0, 0.0, 0.1]) if hand[2] < LIFT_Z else np.array([target[0], target[1], TRAVEL_Z])
            else:
                desired = np.array([target[0], target[1], PLACE_Z])
                if hand[2] < PLACE_Z + 0.02:
                    phase, m['n'] = 'release', 0
        if phase == 'release':
            desired, grab = hand.copy(), RELEASE
            m['n'] += 1
            if m['n'] >= 6:
                env.new_place_target()
                phase, m['n'], m['fails'] = 'retreat', 0, 0
        if phase == 'retreat':
            desired, grab = np.array([hand[0], hand[1], TRAVEL_Z]), RELEASE
            if hand[2] > LIFT_Z:
                phase = 'pick'
        m['phase'] = phase
        return np.concatenate([(desired - hand) * P_GAIN, [grab]])
