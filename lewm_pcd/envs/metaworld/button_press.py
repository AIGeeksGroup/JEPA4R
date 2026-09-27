"""MetaWorld button-press-topdown (a CortexBench task): press the red button of the box from above.

* Layout: metaworld 3.0 ``button-press-topdown-v3-goal-observable``; every reset places the box anew (x -0.1..0.1,
  y 0.8..0.9). The button's return spring is too weak to lift it back within an episode (released at -0.077 it rises
  3 mm in 100 steps), so the task cannot be repeated: the first press of an episode is the real one; later presses
  push on the already pressed button (contact, no state change).
* Targets: 'press' (the button) or a waypoint in the air in front of the box ('wp'). After a press the next target is
  a waypoint with probability 0.7, else another press; after a waypoint it is a press with probability 0.7, else
  another waypoint. So a backbone episode chains presses with moves around the scene.
* info adds the box position (a fixture the reset moves), the button's depth (its joint position), whether the
  button counts as pressed (metaworld's success: its top within 2.4 cm of the hole), and the target (xyz; the 4th
  number 1 = press).
* Evaluation: set_state also restores the box (model data, not qpos) and the target position that depends on it.
* Backbone controller (ButtonThreePartPolicy): above the button, press down until it counts as pressed, hold a few
  steps, lift; waypoints are reached directly (after rising above the box first when the hand is below its top).
  Fingers closed throughout, as metaworld's SawyerButtonPressTopdownV3Policy. Branch / random steps: the base policy's.
"""

from __future__ import annotations

import numpy as np

from .env import MWTaskEnv
from .policy import P_GAIN, ThreePartPolicy

BUTTON_TOP = 0.193            # the button's top above the box origin (metaworld's object position), unpressed
BOX_HALF = 0.12              # the box's collision extent around its origin in x and y (its front edge: y - 0.12 >= 0.68)
WAYPOINTS = ((-0.20, 0.45, 0.15), (0.20, 0.63, 0.35))  # the hand's targets between presses (in view, 5 cm in front of any box)
HOVER_DZ = 0.08               # approach / lift height above the unpressed button top
HOLD = 4                      # steps the pressed button is held down


class MWButtonPressEnv(MWTaskEnv):
    TASK = 'button-press-topdown'
    HAND_INIT = WAYPOINTS  # in view and in front of the box

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._box_body = self._model.body('box').id
        self._target = ('press', np.zeros(3))

    def box_pos(self):
        return self._data.body('box').xpos.copy()

    def pressed(self) -> bool:
        return bool(abs(self._mw._target_pos[2] - self._obj_pos()[2]) <= 0.024)

    def _draw_target(self, rng, after_press: bool):
        if (rng.uniform() < 0.7) == after_press:  # after a press: waypoint w.p. 0.7; after a waypoint: press w.p. 0.7
            for _ in range(100):
                p = rng.uniform(WAYPOINTS[0], WAYPOINTS[1])
                if self.in_view(p):
                    return ('wp', p)
            return ('wp', p)
        return ('press', self.box_pos() + np.array([0.0, 0.0, BUTTON_TOP]))

    def new_place_target(self):
        """Called by the policy when a press or a waypoint is done: the next target."""
        self._target = self._draw_target(self.np_random, self._target[0] == 'press')

    def _reset_target(self, rng, obs):
        self._target = self._draw_target(rng, after_press=False)

    def _task_info(self):
        kind, xyz = self._target
        return {
            'box_pos': self.box_pos(),
            'button_depth': np.float64(self._data.qpos[9]),
            'pressed': np.int64(self.pressed()),
            'place_target': np.array([*xyz, 1.0 if kind == 'press' else 0.0]),  # 4th = 1: press the button
        }

    def set_state(self, qpos, qvel, mocap_pos=None, mocap_quat=None, box_pos=None):
        if box_pos is not None:  # the box is a fixture placed at reset: model data, not qpos
            import mujoco
            self._model.body_pos[self._box_body] = np.asarray(box_pos, dtype=np.float64)
            mujoco.mj_forward(self._model, self._data)
            self._mw._target_pos = self._mw._get_site_pos('hole')
        super().set_state(qpos, qvel, mocap_pos, mocap_quat)


class ButtonThreePartPolicy(ThreePartPolicy):
    LOW_Z = 0.10

    @staticmethod
    def _new_mem():
        return {'phase': 'approach', 'n': 0, 'zs': []}

    @staticmethod
    def _oracle(m, env, info):
        hand = np.asarray(info['hand_pos'], dtype=np.float64)
        top = np.asarray(info['obj_pos'], dtype=np.float64)  # the button's top (moves down when pressed)
        top0 = float(info['box_pos'][2]) + BUTTON_TOP      # ... and where it is when released
        target = np.asarray(info['place_target'], dtype=np.float64)
        grab = 1.0
        if target[3] < 0.5:  # waypoint: over the box and below its top, first back off upwards
            if hand[1] > float(info['box_pos'][1]) - BOX_HALF - 0.02 and hand[2] < top0 + 0.03:
                desired = np.array([hand[0], hand[1] - 0.05, top0 + HOVER_DZ])
            else:
                desired = target[:3]
            dist = float(np.linalg.norm(hand - target[:3]))
            m['zs'].append(dist)  # no progress for 15 steps (e.g. the arm sags a little low near its base): done too
            stalled = len(m['zs']) >= 15 and m['zs'][-15] - dist < 0.002
            if dist < 0.03 or stalled:
                m['n'] += 1
                if m['n'] >= 3 or stalled:
                    env.new_place_target()
                    m['n'], m['phase'], m['zs'] = 0, 'approach', []
            return np.concatenate([(desired - hand) * P_GAIN, [grab]])
        phase = m['phase']
        above = np.array([top[0], top[1], top0 + HOVER_DZ])
        if phase == 'approach':
            if np.linalg.norm(hand[:2] - top[:2]) > 0.03 and hand[2] < top0 + 0.02:
                desired = hand + np.array([0.0, 0.0, 0.1])  # rise before moving over the box
            else:
                desired = above
            if np.linalg.norm(hand[:2] - top[:2]) < 0.015 and hand[2] > top0 + 0.03:  # over the button (the hover height
                phase, m['zs'] = 'press', []                                             # is out of reach for far boxes)
        if phase == 'press':
            desired = np.array([top[0], top[1], top[2] - 0.05])
            m['zs'].append(hand[2])
            stuck = len(m['zs']) >= 12 and m['zs'][-12] - hand[2] < 0.002
            if info['pressed'] and hand[2] - top[2] < 0.05:  # pressed, and the fingers are on the button top
                m['n'] += 1
                if m['n'] >= HOLD:
                    phase, m['n'] = 'lift', 0
            elif stuck:
                phase, m['n'] = 'lift', 0
        if phase == 'lift':
            desired = np.array([hand[0], hand[1], top0 + HOVER_DZ])
            if hand[2] > top0 + HOVER_DZ - 0.02:
                env.new_place_target()
                phase, m['zs'] = 'approach', []
        m['phase'] = phase
        return np.concatenate([(desired - hand) * P_GAIN, [grab]])
