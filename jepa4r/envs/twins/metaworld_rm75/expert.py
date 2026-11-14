"""metaworld 3.0's SawyerBinPickingV3Policy -- the scripted Sawyer expert whose outputs are CortexBench's bin-picking
demos -- on the RM75, run the way it runs on the Sawyer: every action is a function of the current observation only
(metaworld's 39-D observation, which also carries the previous step's hand and gripper, obs[18:22]), gain p 25, and the
move not clipped: metaworld's move() returns p * (target - hand), the env clips it to [-1, 1] when it executes it, and
the demos record the script's output (bench/cortexbench/make_rm75_demos.py).

The script's rules (pos_cube = cube + 3 cm, its y pulled into [0.675, 0.725]; bin = (0.12, 0.7, 0.02)):
  target: xy > 2 cm from pos_cube -> 15 cm above it | z > 1 cm from it -> pos_cube | xy > 2 cm from the bin -> up to
          z 0.15 first, then (bin xy, 0.18) | else the bin;
  grab:   0.6 when within 2 cm of pos_cube in xy and z, else -1 (open).
Changed for the RM75 / EG2-4C2 (the script's numbers are the Sawyer's: a large, fast arm, ~400 N gripper, 9 cm pads):
  grasp height  cube + 4.3 cm, not + 3 cm: the RM75's hand point sits 4.28 cm above a held cube, the Sawyer's 3.97 cm;
                the script's 1 cm tolerance around + 3 cm fits only the Sawyer;
  bottom        close only within 5 mm of the grasp height, not 2 cm: the EG2-4C2's 3.35 cm pads closing earlier catch
                only the top of the cube;
  firm          lift only once the pads squeeze the cube (gripper reading obs[3] < FIRM): the 20 N EG2-4C2 closes more
                slowly and drops a cube lifted before it has closed;
  hold          while closed on the cube (obs[3] < HOLDING) the cube counts as held within 4 cm in xy and 2 cm in z, not
                2 cm / 1 cm: the 20 N grip lets the cube slide between the pads during the carry;
  wall          a cube within ~2 cm of a start-bin y wall (y outside WALL_Y) is not pulled towards the bin's middle (the
                Sawyer's long pads tolerate that pull); the fingers come down pre-opened to 45.3 mm (grab PRESHAPE)
                instead of fully open, the hand placed so that the wall-side finger (21.2 mm thick outside its pad)
                clears the wall's inner face by 0.3 mm. A cube away from the walls keeps the script's pull (1 cm at
                most there);
  retry         if the hand is blocked above the grasp height at such a cube (the cube lies closer to the wall than the
                finger is thick, down to 20.2 mm): close on it (that nudges the cube off the wall), then open, go back
                up and come down again.
Success, as CortexBench counts it (info['success'] at step 500):
  * metaworld's own layouts (robot.make + seed(s) + reset(), seeds 12345 .. 13344): 1000 / 1000, as the Sawyer with the
    unchanged script on the same seeds; the unchanged rules on the RM75: 49.7 %;
  * the layouts of the CortexBench env the demos and evaluations use (bench/cortexbench/rm75_env.py: MWTaskEnv's reset
    with seed s, which maps a seed to a different layout; seeds 12345 .. 12504): 159 / 160, seed 12404 fails.
"""

from __future__ import annotations

import numpy as np

P = 25.0
BIN = np.array([0.12, 0.7, 0.02])
GRASP_DZ = 0.043  # hand above the cube's centre when grasping (the script: 0.03)
WALL_Y = (0.665, 0.735)  # start bin: a cube outside this y range is within ~2 cm of a y wall
BIN_WALLS = (0.610, 0.790)  # the start bin's inner y wall faces
FINGER = 0.0212  # EG2-4C2 finger thickness outside the pad (collision geometry)
OPEN = 0.0662  # EG2-4C2 opening, grab <= 0: 66.2 mm; grab c > 0: 66.2 * (1 - c) mm (measured)
PRESHAPE = 0.315  # grab command -> 45.3 mm between the pads (0.25 -> 49.7, 0.30 -> 46.3, 0.35 -> 43.0; measured)
CLEARANCE = 0.0003  # m between the wall-side finger and the wall
# hand from the wall face: finger + clearance + half the opening
WALL_OFF = FINGER + CLEARANCE + 0.5 * OPEN * (1.0 - PRESHAPE)
HOLDING = 0.665  # obs[3] below: closed on something (a held cube reads 0.60-0.65; the 45.3 mm preshape 0.685)
FIRM = 0.615  # obs[3] below: the pads squeeze the cube
HOLD_WIN, HOLD_ZTOL = 0.04, 0.02  # held cube: xy window / z tolerance (the script: 0.02 / 0.01)


def _near_wall(cube):
    return not (WALL_Y[0] <= cube[1] <= WALL_Y[1])


def _pos_cube(cube):
    p = cube + np.array([0.0, 0.0, GRASP_DZ])
    if not _near_wall(cube):
        p[1] = max(0.675, min(p[1], 0.725))  # the script's pull towards the middle
    elif cube[1] < 0.7:  # no pull; the wall-side finger just clears the wall
        p[1] = max(p[1], BIN_WALLS[0] + WALL_OFF)
    else:
        p[1] = min(p[1], BIN_WALLS[1] - WALL_OFF)
    return p


def _desired_pos(hand, grip, pos_cube):
    win = HOLD_WIN if grip < HOLDING else 0.02
    if np.linalg.norm(hand[:2] - pos_cube[:2]) > win:
        return pos_cube + np.array([0.0, 0.0, 0.15])
    ztol = HOLD_ZTOL if grip < HOLDING else 0.01
    if abs(hand[2] - pos_cube[2]) > ztol or grip > FIRM:
        return pos_cube
    if np.linalg.norm(hand[:2] - BIN[:2]) > 0.02:
        if hand[2] < 0.15:
            return hand + np.array([0.0, 0.0, 0.1])
        return np.array([*BIN[:2], 0.18])
    return BIN


def _grab_effort(hand, grip, cube, pos_cube):
    dz = abs(hand[2] - pos_cube[2])
    win = HOLD_WIN if grip < HOLDING else 0.02
    if np.linalg.norm(hand[:2] - pos_cube[:2]) <= win and dz <= win and (dz <= 0.005 or grip < HOLDING):
        return 0.6
    return PRESHAPE if _near_wall(cube) else -1.0


class RM75BinPickingPolicy:
    """get_action(obs) -> the 4-D action (move not clipped). Stateless: reset() does nothing."""

    def reset(self):
        pass

    def get_action(self, o):
        hand, grip, cube = o[:3], o[3], o[4:7]
        pos_cube = _pos_cube(cube)
        target = _desired_pos(hand, grip, pos_cube)
        grab = _grab_effort(hand, grip, cube, pos_cube)
        if cube[0] < 0.0 and cube[2] < 0.04 and np.linalg.norm(hand[:2] - pos_cube[:2]) <= 0.02:
            high = hand[2] - pos_cube[2]
            if 0.012 < high < 0.045:  # blocked above the grasp height at the start bin
                # metaworld's observation carries the previous step's hand / gripper (obs[18:21], obs[21])
                stuck = abs(o[20] - hand[2]) < 0.0003
                closing = grip < o[21] - 0.001
                if grip >= HOLDING:
                    if stuck:  # on the cube's edge / the wall's top with the fingers open: close on it (nudges it)
                        target, grab = pos_cube, 0.6
                elif closing:  # still closing: go on until the fingers stop
                    target, grab = pos_cube, 0.6
                else:  # closed (stopped) or opening again, up here and not holding the cube: open, go back up, retry
                    target, grab = pos_cube + np.array([0.0, 0.0, 0.15]), -1.0
        a = np.zeros(4)
        a[:3] = P * (target - hand)  # not clipped: the env clips
        a[3] = grab
        return a


DRAWER_DZ = 0.028  # drawer-open: the hand this far above the handle point (the script: 0.02 below)


class RM75DrawerOpenPolicy:
    """metaworld 3.0's SawyerDrawerOpenV3Policy (the scripted Sawyer expert of CortexBench's drawer-open demos) on the
    RM75 with one change, stateless as the script: the hand's height at the handle, DRAWER_DZ above the handle point
    instead of 2 cm below. The EG2-4C2's 3.35 cm pads then straddle the bar at its height, as the Sawyer's 9 cm pads do
    from 2 cm below; from below they touch only one side of the bar, slide off it while pulling, and the drawer, let go,
    comes back off its end stop (metaworld's success needs it held there). Everything else is the script's: p 4 to 30 cm
    above the handle (beyond 6 cm in xy), p 4 down to it (beyond 4 cm in z), else p 50 towards 6 cm behind it, fingers
    open; the move not clipped (the env clips it). Success, as CortexBench counts it (info['success'] at step 500),
    seeds 12345 .. 13344: 998 / 1000 on the layouts of the CortexBench env the demos and evaluations use
    (bench/cortexbench/rm75_env.py), 994 / 1000 on metaworld's own; the unchanged script on the RM75 70 / 200 and 68 /
    200 (each of these episodes succeeded on the way, then the drawer came back off its stop), the Sawyer with it 200 /
    200."""

    def reset(self):
        pass

    def get_action(self, o):
        hand = o[:3]
        drw = o[4:7] + np.array([0.0, 0.0, DRAWER_DZ])
        if np.linalg.norm(hand[:2] - drw[:2]) > 0.06:
            to, p = drw + np.array([0.0, 0.0, 0.3]), 4.0
        elif abs(hand[2] - drw[2]) > 0.04:
            to, p = drw, 4.0
        else:
            to, p = drw + np.array([0.0, -0.06, 0.0]), 50.0
        a = np.zeros(4)
        a[:3] = p * (to - hand)  # not clipped: the env clips
        a[3] = -1.0
        return a


ASSEMBLY_FIRM = 0.585  # assembly: obs[3] below: the pads squeeze the handle (closed on it the reading is 0.573)
ASSEMBLY_HOLDING = 0.63  # assembly: obs[3] below: closed on something


class RM75AssemblyPolicy:
    """metaworld 3.0's SawyerAssemblyV3Policy (the scripted Sawyer expert of CortexBench's assembly demos) on the RM75
    with two changes, stateless as the script:
      bottom  the fingers close only at the grasp height, within 5 mm of handle + 3 cm (or while closed on something);
              the script closes them from 12 cm above it on, and the EG2-4C2's closed 3.35 cm pads then land on top of
              the handle instead of around it;
      firm    the hand stays down at the handle until the pads squeeze it (obs[3] < ASSEMBLY_FIRM); the script lifts as
              soon as the hand is within 5 cm of the handle, and the slow 20 N EG2-4C2 has not closed by then.
    Everything else is the script's: grasp point handle - 2 cm in x, + 3 cm up; lift to the peg's plane (peg + (0.12, 0,
    0.14)), over it, then down 20 cm onto the peg; grab 0.6; p 10, the move not clipped (the env clips it).
    Success, as CortexBench counts it (info['success'] at step 500), seeds 12345 .. 13344: 1000 / 1000 on the layouts
    of the CortexBench env the demos and evaluations use, 1000 / 1000 on metaworld's own; the unchanged script on the
    RM75 0 / 200 on both, the Sawyer with it 200 / 200."""

    def reset(self):
        pass

    def get_action(self, o):
        hand, grip = o[:3], o[3]
        wrench = o[4:7] + np.array([-0.02, 0.0, 0.0])
        peg = o[-3:] + np.array([0.12, 0.0, 0.14])
        if np.linalg.norm(hand[:2] - wrench[:2]) > 0.02:
            to = wrench + np.array([0.0, 0.0, 0.1])
        elif np.linalg.norm(hand[:2] - peg[:2]) <= 0.02:
            to = peg + np.array([0.0, 0.0, -0.2])
        elif abs(hand[2] - wrench[2]) > 0.05 or grip > ASSEMBLY_FIRM:
            to = wrench + np.array([0.0, 0.0, 0.03])
        elif abs(hand[2] - peg[2]) > 0.04:
            to = np.array([hand[0], hand[1], peg[2]])
        else:
            to = peg
        close = (
            np.linalg.norm(hand[:2] - wrench[:2]) <= 0.02
            and abs(hand[2] - wrench[2]) <= 0.12
            and (abs(hand[2] - (wrench[2] + 0.03)) <= 0.005 or grip < ASSEMBLY_HOLDING)
        )
        a = np.zeros(4)
        a[:3] = 10.0 * (to - hand)  # not clipped: the env clips
        a[3] = 0.6 if close else 0.0
        return a
