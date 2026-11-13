# Moved from cortexbench@9331494:lewm_pcd/envs/metaworld/drawer.py (verbatim)
"""MetaWorld drawer-open (a CortexBench task): bring the drawer to target openings, pulling it out or pushing it in.

* Layout: metaworld 3.0 ``drawer-open-v3-goal-observable``; every reset places the drawer case anew (x -0.1..0.1, y 0.9).
  The drawer is a slide joint (0 .. 0.16 m out, damping, no spring): it stays where it is left, so it can be opened and
  closed again and again.
* Targets: 'drawer' (bring the drawer to a random opening in [0, 0.16] m) or a waypoint in the air in front of the case
  ('wp'). After a drawer target the next one is another opening with probability 0.6, else a waypoint; after a waypoint
  it is an opening with probability 0.8.
* info adds the case position (a fixture the reset moves), the drawer's opening, and the target (x, y, z / opening; the
  4th number 1 = drawer). obj_pos is metaworld's object position, the handle.
* Evaluation: set_state also restores the case (model data, not qpos) and the target that depends on it.
* The green goal marker stays at the xml position, as in CortexBench's metaworld v2 (v3 moves it in front of the case).
* Backbone controller (DrawerThreePartPolicy): fingers open throughout, as metaworld's SawyerDrawerOpenV3Policy. Come
  down around the handle bar; the far finger then pulls the bar (open further) or pushes the drawer front (close). The
  hand's y goes through the mocap (the hand lags it under load) to where the drawer should end, less the distance the
  drawer coasts on after the hand stops; hold until the drawer stops, lift, next target. No retries: a missed target is
  followed by the next one. After random steps (a branch burst) the controller starts over from the approach.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .env import MWTaskEnv
from .policy import P_GAIN, STEP, ThreePartPolicy

OPEN_MAX = 0.16
GOAL_SITE_V2 = np.array([0.0, 0.74, 0.05])  # the goal marker where CortexBench's v2 env leaves it (v3 moves it to the target)
WAYPOINTS = ((-0.20, 0.45, 0.15), (0.20, 0.55, 0.30))  # in view, in front of a fully open drawer (handle y ~0.57)
ABOVE = 0.08       # approach height above the handle
TOL = 0.01         # m: target opening reached
VMAX = 0.006       # m/step: drawer move speed
K_SLOW = 1.5       # steps: slow down over the last VMAX * K_SLOW
HOLD_MAX = 15      # steps at most to wait for the drawer to stop
PULL_START, PUSH_START = 0.015, -0.01  # the hand goes down this far in front of the handle point
PULL_OFF = 0.025   # hand this far in front of the handle point when the far finger pulls the bar
PUSH_OFF = 0.02    # hand this far behind it when the far finger pushes the drawer front
PULL_COAST, PUSH_COAST = 0.02, 0.032 # stop the hand this much early: the drawer coasts on


class MWDrawerEnv(MWTaskEnv):
    TASK = 'drawer-open'
    HAND_INIT = WAYPOINTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        m = self._model
        self._case_body = m.body('drawer').id
        self._slide_q = m.jnt_qposadr[m.joint('goal_slidey').id]
        self._target = ('drawer', 0.0)

    def case_pos(self):
        return self._data.body('drawer').xpos.copy()

    def opening(self) -> float:
        return float(-self._data.qpos[self._slide_q])

    def _draw_target(self, rng, after_drawer: bool):
        p_wp = 0.4 if after_drawer else 0.2
        if rng.uniform() < p_wp:
            for _ in range(100):
                p = rng.uniform(WAYPOINTS[0], WAYPOINTS[1])
                if self.in_view(p):
                    return ('wp', p)
            return ('wp', p)
        return ('drawer', float(rng.uniform(0.0, OPEN_MAX)))

    def new_place_target(self):
        self._target = self._draw_target(self.np_random, self._target[0] == 'drawer')

    def _reset_target(self, rng, obs):
        # metaworld's reset_model moves the case (model data) after the last forward pass, so the reset row's frame, point
        # cloud and observation would show it elsewhere: one more forward pass, and the reset observation redone (both
        # stacked frames the current one, as metaworld's reset does)
        self._model.site('goal').pos = GOAL_SITE_V2
        mujoco.mj_forward(self._model, self._data)
        cur = self._mw._get_curr_obs_combined_no_goal()
        self._mw._prev_obs = cur.copy()
        obs[:18], obs[18:36] = cur, cur
        self._target = self._draw_target(rng, after_drawer=False)

    def _task_info(self):
        kind, v = self._target
        tgt = np.array([*v, 0.0]) if kind == 'wp' else np.array([0.0, 0.0, float(v), 1.0])  # drawer: (0, 0, opening, 1)
        return {
            'case_pos': self.case_pos(),
            'drawer_open': np.float64(self.opening()),
            'place_target': tgt,
        }

    def set_state(self, qpos, qvel, mocap_pos=None, mocap_quat=None, case_pos=None):
        if case_pos is not None:  # the case is a fixture placed at reset: model data, not qpos
            pos = np.asarray(case_pos, dtype=np.float64)
            self._model.body_pos[self._case_body] = pos
            self._mw.obj_init_pos = pos.copy()
            self._mw._target_pos = pos + np.array([0.0, -0.16 - self._mw.maxDist, 0.09])
        super().set_state(qpos, qvel, mocap_pos, mocap_quat)


class DrawerThreePartPolicy(ThreePartPolicy):
    @staticmethod
    def _new_mem():
        return {'phase': 'go', 'n': 0, 'hist': [], 'last': None}

    @staticmethod
    def _oracle(m, env, info):
        hand = np.asarray(info['hand_pos'], dtype=np.float64)
        h = np.asarray(info['obj_pos'], dtype=np.float64) + np.array([0.0, 0.0, -0.02])  # hook point (as metaworld)
        target = np.asarray(info['place_target'], dtype=np.float64)
        opening = float(info['drawer_open'])
        t = int(info['step_idx'])
        if m['last'] is not None and t != m['last'] + 1:  # back from random steps: start over
            m['phase'], m['n'], m['hist'] = 'go', 0, []
        m['last'] = t
        m['hist'] = (m['hist'] + [opening])[-2:]

        def act(desired):
            return np.concatenate([(desired - hand) * P_GAIN, [-1.0]])

        def next_target():
            env.new_place_target()
            m['phase'], m['n'] = 'go', 0

        m['n'] += 1
        if target[3] < 0.5:  # waypoint: from low near the drawer, back off and go up first
            wp = target[:3]
            low_near = hand[1] > h[1] - 0.06 and hand[2] < h[2] + 0.06
            desired = np.array([hand[0], hand[1] - 0.03, h[2] + ABOVE]) if low_near else wp
            if np.linalg.norm(hand - wp) < 0.03 or m['n'] > 60:
                next_target()
            return act(desired)

        err = float(target[2]) - opening  # > 0: pull further out, < 0: push in
        phase = m['phase']
        vy = None
        if phase == 'go':
            if abs(err) < TOL or m['n'] > 60:  # already there / cannot get there
                next_target()
                return act(hand)
            spot = h + np.array([0.0, -PULL_START if err > 0 else PUSH_START, 0.0])  # where the hand goes down
            m['spot_dy'] = spot[1] - h[1]
            if hand[2] < h[2] + 0.06 and np.linalg.norm(hand[:2] - spot[:2]) > 0.02:  # low: back off and rise first
                desired = np.array([hand[0], min(hand[1], h[1] - 0.06), h[2] + ABOVE])
            else:
                desired = spot + np.array([0.0, 0.0, ABOVE])
                if np.linalg.norm(hand[:2] - spot[:2]) < 0.015:
                    phase, m['n'] = 'down', 0
        if phase == 'down':  # fingers open around the handle bar
            desired = h + np.array([0.0, m['spot_dy'], 0.0])
            if hand[2] - h[2] < 0.02:
                # the hand's end y: the handle's y at the target opening, less the far finger's offset from it
                # (pull: behind the bar; push: against the drawer front), less the drawer's coasting after a pull
                y_t = h[1] - err
                m['y_end'] = y_t - PULL_OFF + PULL_COAST if err > 0 else y_t + PUSH_OFF - PUSH_COAST
                phase, m['n'] = 'move', 0
            elif m['n'] > 25:  # blocked (e.g. on top of the handle): give this target up
                phase, m['n'] = 'lift', 0
        if phase == 'move':  # y through the mocap (the hand lags it under load)
            desired = np.array([h[0], hand[1], h[2]])
            d = m['y_end'] - float(env._data.mocap_pos[0, 1])
            vy = np.sign(d) * min(VMAX, abs(d) / K_SLOW)
            if abs(d) < 0.002 or m['n'] > 60:
                phase, m['n'] = 'hold', 0
        if phase == 'hold':  # the drawer coasts on until the finger's other side stops it
            desired = np.array([h[0], hand[1], h[2]])
            vy = 0.0
            still = len(m['hist']) > 1 and abs(m['hist'][-1] - m['hist'][-2]) < 0.0005
            if (still and m['n'] >= 3) or m['n'] > HOLD_MAX:
                phase, m['n'] = 'lift', 0
        if phase == 'lift':
            desired = np.array([hand[0], hand[1], h[2] + ABOVE])
            vy = 0.0
            if hand[2] > h[2] + ABOVE - 0.02 or m['n'] > 30:
                next_target()
                return act(desired)
        m['phase'] = phase
        a = act(desired)
        if vy is not None:
            a[1] = vy / STEP
        return a
