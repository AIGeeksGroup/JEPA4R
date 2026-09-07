"""Minimal native PickCube settling phase on the original OGB cube oracle.

The original approach, grasp, transport, release and retreat remain upstream.
Only a short noisy hold is inserted when a grasped cube reaches the goal.
The outer, unchanged ExpertPolicy still adds its official Gaussian noise to
every action, including these settling actions; failed attempts are retained.
"""

from __future__ import annotations

import numpy as np

from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle


class PickCubeHoldOracle(CubeMarkovOracle):
    """Allow PickCube's native static-at-goal condition before release."""

    def __init__(self, *args, hold_steps=30, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(hold_steps, bool) or int(hold_steps) != hold_steps or hold_steps < 1:
            raise ValueError("hold_steps must be a positive integer")
        self.hold_steps = int(hold_steps)
        self._hold_remaining = 0
        self._hold_completed = False

    def reset(self, ob, info):
        super().reset(ob, info)
        self._hold_remaining = 0
        self._hold_completed = False

    def select_action(self, ob, info):
        block = info["privileged/target_block"]
        block_pos = np.asarray(info[f"privileged/block_{block}_pos"])
        goal_error = np.asarray(info["privileged/target_block_pos"]) - block_pos
        distance = float(np.linalg.norm(goal_error))
        grasped = bool(np.asarray(info["proprio/gripper_contact"]).reshape(-1)[0] > 0.5)
        if not self._hold_completed and self._hold_remaining == 0 and grasped and distance <= 0.025:
            self._hold_remaining = self.hold_steps

        # A disturbed or dropped grasp returns to the original recovery logic.
        # It may enter the hold again on reaching the goal with a secure grasp.
        if self._hold_remaining and (not grasped or distance > 0.08):
            self._hold_remaining = 0

        if not self._hold_remaining:
            return super().select_action(ob, info)

        self.print_phase("6b: Hold at PickCube target")
        action = np.zeros(5)
        # OGB's 0.05 m action scale makes this a 0.25 proportional correction
        # per step. Do not apply the moving oracle's minimum-norm boost here:
        # at a 1 mm error that boost requests a large movement, never settling.
        action[:3] = goal_error * 5.0
        action[4] = 1.0
        self._hold_remaining -= 1
        self._hold_completed = self._hold_remaining == 0
        self._step += 1
        if self._step >= self._max_step:
            self._done = True
        return np.clip(action, -1.0, 1.0)
