"""envs.metaworld.ThreePartPolicy (bin-picking) for the RM75: the plan (backbone / branch / random), the noise and the
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

from ..policy import LIFT_Z, PLACE_Z, TRAVEL_Z, ThreePartPolicy

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
