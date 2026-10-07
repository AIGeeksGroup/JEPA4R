"""The experts of CortexBench's TriFinger demos (move_cube, reach_cube), reproduced. Their code is not released
(not in eai-vc's trifinger_vc, which ships the demos and the control utilities the experts are built from); each demo
records what the expert did (assets/data/trifinger-demos/<task>/*/downsample.pth: 'ft_pos_des' = its fingertip plan,
'delta_ftpos' = the plan's increments = the BC actions; move also 'ft_pos_targets_per_mode' = its three targets), the
same for all 125 demos of a task:
  * targets from trifinger_vc.control.cube_utils: contact = each finger's contact point (get_cp_params assigns each
    finger a side face by the fingers' base directions; get_cp_pos_wf_from_cp_params with ft_radius =
    finger_utils.get_ft_radius('trifinger_meta') = 7 mm); pre-contact = the same point 2 cm further out.
  * move (MoveExpert), every target from the cube pose observed when its phase starts: the plan goes piecewise linearly
    in time from the fingertips' start to pre-contact over 0-1.0 s, to contact over 1.0-1.5 s, to contact + (goal -
    cube position) over 1.5-4.0 s; 10 steps.
  * reach (ReachExpert, the cube fixed): finger 0 alone, to its pre-contact point over 0-1.5 s, to its contact point
    over 1.5-2.0 s; 5 steps.
  An env step (0.4 s) moves the plan by its increment over the step.
Episodes as in the demos: the envs' fixed initial joint angles (-0.08, 1.15, -1.5) x 3, the cube from
move_cube.sample_goal(-1), move's goal from move_cube.sample_goal(1) (position only).
scripts/checks/check_tf_expert.py replays the 125 move demos in CortexBench's MoveCubeEnv: the reproduced plans are
within 0.13 cm of the demos', the rollouts' fingertips / cube within 0.14 / 0.09 cm (median of a demo's largest
error)."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TF_VC = os.path.normpath(os.path.join(HERE, '..', '..', 'third_party', 'eai-vc', 'cortexbench', 'trifinger_vc', 'src'))
if TF_VC not in sys.path:  # trifinger_vc (CortexBench's TriFinger package; not installed in lewm_tf)
    sys.path.insert(0, TF_VC)

import numpy as np  # noqa: E402
import trifinger_vc.control.cube_utils as c_utils  # noqa: E402
from trifinger_vc.control.finger_utils import get_ft_radius  # noqa: E402

FINGER_TYPE = 'trifinger_meta'
FT_RADIUS = get_ft_radius(FINGER_TYPE)
PRE_OUT = 0.02                     # pre-contact: 2 cm outside the contact point
KNOTS = (0.0, 1.0, 1.5, 4.0)       # s: start, pre-contact reached, contact reached, carry target reached
REACH_KNOTS = (0.0, 1.5, 2.0)      # s: start, pre-contact reached, contact reached (finger 0)
STEP_S = 0.4                       # one env step (100 sim steps of 4 ms)
N_STEPS = 10                       # 4.0 s
REACH_STEPS = 5                    # 2.0 s


def contact_points(cube_pos, cube_quat, extra=0.0) -> np.ndarray:
    pose = {'position': np.asarray(cube_pos, dtype=np.float64), 'orientation': np.asarray(cube_quat, dtype=np.float64)}
    cp = c_utils.get_cp_params(pose, FINGER_TYPE)
    return np.concatenate(c_utils.get_cp_pos_wf_from_cp_params(cp, pose, ft_radius=FT_RADIUS + extra))


class MoveExpert:
    """Call reset(ft_pos, cube_pos, cube_quat, goal_pos) at t = 0, then act(k, cube_pos, cube_quat) before step k
    (k = 0..9) with the cube pose observed then; returns the step's fingertip displacement (9,), metres."""

    def reset(self, ft_pos, cube_pos, cube_quat, goal_pos):
        # copies: callers may pass views of buffers they overwrite later (stable-worldmodel's EnvPool infos)
        self.goal = np.array(goal_pos, dtype=np.float64)
        self.pts = [np.array(ft_pos, dtype=np.float64), contact_points(cube_pos, cube_quat, PRE_OUT), None, None]

    def _plan(self, t):
        pts = np.stack(self.pts[:1 + sum(p is not None for p in self.pts[1:])])
        return np.array([np.interp(t, KNOTS[:len(pts)], pts[:, j]) for j in range(9)])

    def act(self, k, cube_pos, cube_quat):
        t0, t1 = k * STEP_S, (k + 1) * STEP_S
        if self.pts[2] is None and t1 > KNOTS[1]:   # contact phase starts in this step: its target from the cube now
            self.pts[2] = contact_points(cube_pos, cube_quat)
        if self.pts[3] is None and t1 > KNOTS[2]:   # carry phase starts in this step
            self.pts[3] = contact_points(cube_pos, cube_quat) + np.tile(self.goal - np.asarray(cube_pos), 3)
        return self._plan(min(t1, KNOTS[-1])) - self._plan(min(t0, KNOTS[-1]))


class ReachExpert:
    """reset(ft_pos, cube_pos, cube_quat) at t = 0, then act(k) before step k (k = 0..4): finger 0's displacement in
    the first 3 entries, 0 for the other two fingers (9,), metres."""

    def reset(self, ft_pos, cube_pos, cube_quat):
        ft0 = np.array(ft_pos, dtype=np.float64)[:3]  # a copy (see MoveExpert.reset)
        self.pts = np.stack([ft0, contact_points(cube_pos, cube_quat, PRE_OUT)[:3], contact_points(cube_pos, cube_quat)[:3]])

    def _plan(self, t):
        return np.array([np.interp(t, REACH_KNOTS, self.pts[:, j]) for j in range(3)])

    def act(self, k, *_):
        a = np.zeros(9)
        a[:3] = self._plan(min((k + 1) * STEP_S, REACH_KNOTS[-1])) - self._plan(min(k * STEP_S, REACH_KNOTS[-1]))
        return a
