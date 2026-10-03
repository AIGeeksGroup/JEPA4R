"""The three-part collection policies for the RM75. RM75DrawerThreePartPolicy (drawer-open): at the end of this file.

envs.metaworld.ThreePartPolicy (bin-picking) for the RM75 (RM75BinPickThreePartPolicy): the plan (backbone / branch / random), the noise and the
random actions (kept in the camera's view) unchanged; the backbone's pick-and-place oracle keeps its sub-tasks
(approach above the grasp point, down, grasp, lift, carry at TRAVEL_Z to the env's place target, put down, release,
retreat, next target) and changes what the RM75 / EG2-4C2 needs:
* P gain 20 instead of 25 (10 is slower and fails more grasps at the bin walls);
* the grasp: all the way down (8 mm), close, and lift once the fingers stopped moving (the EG2-4C2 takes ~13 steps to
  close on the cube; the Sawyer version lifts after 6 steps at the grasp point);
* near a start-bin y wall the fingers approach pre-closed to ~50 mm and centred on the cube (21 mm thick fingers); a
  descent that stalls above the grasp point (a finger on the wall) keeps pushing for 20 steps and then closes (that can
  drag the cube in between); closed on nothing: up and again (the attempt may have moved the cube);
* over the place target: hysteresis (down within 2 cm, up again beyond 4 cm) instead of one 2 cm window;
* the release: until the fingers are open (the EG2-4C2 opens as slowly as it closes), then retreat;
* the grasp height is kept at bin-picking's hand floor (0.07; cube + 3 cm is 0.067 for a cube resting on a bin floor),
  as the Sawyer version's 1 cm window allows implicitly;
* after 4 failed grasps (a cube within ~2 cm of a start-bin wall: the fingers are wider than the gap) the backbone
  gives the cube up and moves the hand between random in-view points for the rest of the episode."""

from __future__ import annotations

import numpy as np

from ..drawer import ABOVE, HOLD_MAX, K_SLOW, PULL_COAST, PULL_OFF, PULL_START, PUSH_COAST, PUSH_OFF, PUSH_START, TOL, VMAX
from ..drawer import DrawerThreePartPolicy
from ..policy import LIFT_Z, P_GAIN as SAWYER_P_GAIN, PLACE_Z, STEP, TRAVEL_Z, ThreePartPolicy
from .expert import DRAWER_DZ

P_GAIN = 20.0
APPROACH_DZ = 0.10
GRAB, RELEASE, PRESHAPE = 0.6, -1.0, 0.25  # EG2-4C2 command: PRESHAPE ~50 mm between the pads
GIVE_UP = 4                                 # failed grasps before the backbone gives the cube up
WALL_Y = (0.665, 0.735)  # start bin: a cube outside this y range is within ~2 cm of a wall
WIN_IN, WIN_OUT = 0.02, 0.04


def _grasp_point(cube, near_wall):
    p = cube + np.array([0.0, 0.0, 0.03])
    p[2] = max(p[2], PLACE_Z + 0.001)  # bin-picking's hand floor (mocap_low z 0.07) is above cube + 3 cm
    if not near_wall and 0.64 < p[1] < 0.76:  # inside a bin, away from its y walls: aim centrally in y (13 mm at most)
        p[1] = cube[1] + np.clip(min(max(p[1], 0.675), 0.725) - cube[1], -0.012, 0.012)
    return p


class RM75BinPickThreePartPolicy(ThreePartPolicy):
    @staticmethod
    def _new_mem():
        return {'phase': 'pick', 'sub': 'above', 'n': 0, 'hist': [], 'fails': 0, 'over': False, 'zs': [], 'wp': None,
                'stall': 0}

    @staticmethod
    def _oracle(m, env, info):
        hand = np.asarray(info['hand_pos'], dtype=np.float64)
        cube = np.asarray(info['obj_pos'], dtype=np.float64)
        target = np.asarray(info['place_target'], dtype=np.float64)
        grip = float(info['gripper'])
        in_bin = 0.60 < cube[1] < 0.80 and abs(abs(cube[0]) - 0.12) < 0.10
        near_wall = in_bin and not (WALL_Y[0] <= cube[1] <= WALL_Y[1])
        grasp = _grasp_point(cube, near_wall)
        phase = m['phase']
        grab = RELEASE
        m['n'] += 1

        if phase == 'pick' and m['fails'] >= GIVE_UP:  # a cube the fingers cannot get around (at a wall): give it up
            phase, m['n'] = 'wander', 0
        if phase == 'carry' and np.linalg.norm(hand - (cube + [0, 0, 0.03])) > 0.05:  # dropped it: pick again
            phase, m['sub'], m['n'], m['fails'] = 'pick', 'above', 0, m['fails'] + 1
        pre = PRESHAPE if near_wall else RELEASE
        if phase == 'pick':
            if m['sub'] == 'above':
                desired = grasp + np.array([0.0, 0.0, APPROACH_DZ])
                grab = pre
                if np.linalg.norm(hand[:2] - grasp[:2]) < 0.01:
                    m['sub'], m['n'], m['zs'], m['stall'] = 'down', 0, [], 0
            if m['sub'] == 'down':
                desired = grasp
                grab = pre
                m['zs'].append(hand[2])
                if len(m['zs']) > 10 and m['zs'][-6] - hand[2] < 0.002 and hand[2] - grasp[2] > 0.015:
                    m['stall'] += 1  # a finger on the wall of the bin (cube near it): keep pushing for a while
                if np.linalg.norm(hand[:2] - grasp[:2]) > 0.03:  # pushed off (noise): back up
                    m['sub'], m['n'] = 'above', 0
                elif m['stall'] > 20 or np.linalg.norm(hand - grasp) < 0.003 or m['n'] > 45:
                    # closing from a stall can still drag the cube in between the fingers
                    m['sub'], m['n'], m['hist'] = 'close', 0, []
            if m['sub'] == 'close':
                desired, grab = grasp, GRAB
                m['hist'].append(grip)
                if m['n'] >= 8 and abs(m['hist'][-1] - m['hist'][-4]) < 0.004:
                    if grip < 0.2:  # closed on nothing: up and again (the attempt may have moved the cube)
                        m['sub'], m['n'], m['fails'] = 'above', 0, m['fails'] + 1
                    else:
                        phase, m['n'], m['over'] = 'carry', 0, False
        if phase == 'carry':
            grab = GRAB
            xy = np.linalg.norm(hand[:2] - target[:2])
            m['over'] = xy < (WIN_OUT if m['over'] else WIN_IN)
            if not m['over']:
                desired = np.array([hand[0], hand[1], hand[2] + 0.1]) if hand[2] < LIFT_Z else np.array([target[0], target[1], TRAVEL_Z])
            else:
                desired = np.array([target[0], target[1], PLACE_Z])
                if hand[2] < PLACE_Z + 0.02:
                    phase, m['n'], m['hist'] = 'release', 0, []
        if phase == 'release':
            desired, grab = hand.copy(), RELEASE
            if (m['n'] >= 6 and grip > 0.9) or m['n'] > 30:  # open enough to clear the cube (0.9 ~ 60 mm)
                env.new_place_target()
                phase, m['n'], m['fails'] = 'retreat', 0, 0
        if phase == 'retreat':
            desired, grab = np.array([hand[0], hand[1], TRAVEL_Z]), RELEASE
            if hand[2] > LIFT_Z:
                phase, m['sub'], m['n'] = 'pick', 'above', 0
        if phase == 'wander':  # hand to random in-view points above the bins, one after another
            if m['wp'] is None or np.linalg.norm(hand - m['wp']) < 0.02 or m['n'] > 60:
                rng = env.np_random
                for _ in range(50):
                    wp = rng.uniform((-0.20, 0.55, 0.10), (0.20, 0.80, 0.22))
                    if env.in_view(wp):
                        break
                m['wp'], m['n'] = wp, 0
            desired, grab = m['wp'], RELEASE
        m['phase'] = phase
        return np.concatenate([(desired - hand) * P_GAIN, [grab]])


HOOK_DZ = DRAWER_DZ  # drawer-open: the hand above the handle point, as RM75DrawerOpenPolicy (the script: 0.02 below)


class RM75DrawerThreePartPolicy(DrawerThreePartPolicy):
    """envs.metaworld's DrawerThreePartPolicy (drawer-open) for the RM75, its oracle with two changes:
    * the hand's height at the handle bar: HOOK_DZ above the handle point instead of 2 cm below (metaworld's hook
      point): the EG2-4C2's 3.35 cm pads straddle the bar at its height, as the Sawyer's 9 cm pads do from 2 cm below;
      from below they touch only one side of the bar and slip off while pulling;
    * the hand target is the RM75 controller's (env._mw._ee_target) where the Sawyer version reads its mocap.
    Everything else (targets, speeds, offsets, waypoints, no retries) is the Sawyer version's."""

    @staticmethod
    def _oracle(m, env, info):
        hand = np.asarray(info['hand_pos'], dtype=np.float64)
        h = np.asarray(info['obj_pos'], dtype=np.float64) + np.array([0.0, 0.0, HOOK_DZ])  # hook point: the pads straddle the bar
        target = np.asarray(info['place_target'], dtype=np.float64)
        opening = float(info['drawer_open'])
        t = int(info['step_idx'])
        if m['last'] is not None and t != m['last'] + 1:  # back from random steps: start over
            m['phase'], m['n'], m['hist'] = 'go', 0, []
        m['last'] = t
        m['hist'] = (m['hist'] + [opening])[-2:]

        def act(desired):
            return np.concatenate([(desired - hand) * SAWYER_P_GAIN, [-1.0]])

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
        if phase == 'move':  # y through the hand target (the hand lags it under load)
            desired = np.array([h[0], hand[1], h[2]])
            d = m['y_end'] - float(env._mw._ee_target[1])
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
