"""MetaWorld assembly (a CortexBench task): put the nut (a ring with a handle, body 'RoundNut') over the peg.

* Layout: metaworld 3.0 ``assembly-v3-goal-observable`` (the same XML as CortexBench's v2 fork); every reset places the
  peg anew (x -0.1..0.1, y 0.75..0.85); the nut starts at the same place every episode (ring centre (0, 0.6)).
* Place targets: 'peg' (the ring over the peg: the assembly) or a spot on the table, given as the ring centre's xy.
  With the nut on the table the next target is the peg with probability 0.7, else another table spot; with the nut on
  the peg it is a table spot (lift it off again). So a backbone episode chains assemblies and disassemblies
  (AssemblyThreePartPolicy below).
* info adds the ring centre, the peg top, the target (xy; z = 1 for the peg) and whether the ring is on the peg;
  obj_pos is metaworld's object position, the handle site 'RoundNut-8'.
* Evaluation: set_state also restores the peg, a fixture the reset moves (model data, not qpos).
* Backbone controller (AssemblyThreePartPolicy): the nut is grasped by its handle as metaworld's SawyerAssemblyV3Policy /
  SawyerDisassembleV3Policy do (hand 2 cm behind the handle site, fingers closed at 0.6; lifting starts once the
  gripper reading shows them closed), and carried to the target. A carry always starts with a straight lift to
  LIFT_Z, so a nut on the peg comes off it before moving sideways; during the carry the hand moves so that the ring
  centre (not the hand) reaches the target, which keeps the ring aligned with the peg whatever the nut's yaw. A
  placement that gets stuck (the ring resting on the peg top) is released and the nut picked again; a dropped nut
  is picked again. Branch and random steps: policy.ThreePartPolicy's.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .env import MWTaskEnv
from .policy import GRAB, P_GAIN, RELEASE, ThreePartPolicy

TABLE = ((-0.22, 0.56), (0.08, 0.70))  # ring-centre spots for table placements (the handle, +0.13 m in x, stays in view)
PEG_CLEARANCE = 0.13                   # m between a table spot and the peg (ring outer radius ~0.066)
GRASP_OFF = np.array([-0.02, 0.0, 0.045])  # hand relative to the handle site when holding it (metaworld: -0.02 in x)
APPROACH_DZ = 0.10
LIFT_Z = 0.20      # straight up to here before any sideways carry: the ring (~0.06 below the hand) clears the peg top (0.10)
TRAVEL_Z = 0.22
PLACE_Z = 0.05     # lowest hand target (mocap_low z); the nut stops the hand at ~0.065 on the table


class MWAssemblyEnv(MWTaskEnv):
    TASK = 'assembly'
    HAND_INIT = ((-0.20, 0.55, 0.15), (0.20, 0.80, 0.24))  # above the peg top (0.10)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._peg_body = self._model.body('peg').id
        self._target = ('peg', np.zeros(2))

    def peg_top(self):
        return np.asarray(self._mw._target_pos, dtype=np.float64).copy()

    def ring_pos(self):
        return self._data.body('RoundNut').xpos.copy()

    def on_peg(self) -> bool:
        ring, peg = self.ring_pos(), self.peg_top()
        return bool(np.linalg.norm(ring[:2] - peg[:2]) < 0.02 and ring[2] < 0.05)

    def _draw_target(self, rng):
        if not self.on_peg() and rng.uniform() < 0.7:
            return ('peg', self.peg_top()[:2])
        peg, ring = self.peg_top()[:2], self.ring_pos()[:2]
        (x0, y0), (x1, y1) = TABLE
        for _ in range(100):
            p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
            if np.linalg.norm(p - peg) >= PEG_CLEARANCE and np.linalg.norm(p - ring) >= 0.05:
                return ('table', p)
        return ('table', p)

    def new_place_target(self):
        """Called by the policy when a placement is done: the next target."""
        self._target = self._draw_target(self.np_random)

    def _reset_target(self, rng, obs):
        # metaworld's reset runs reset_model twice and reset_model moves the peg (model data) after its last forward
        # pass, so the reset row's frame and point cloud would show the peg where the first call put it: one more
        # forward pass, and the reset observation redone (both stacked frames the current one), as MWDrawerEnv
        mujoco.mj_forward(self._model, self._data)
        cur = self._mw._get_curr_obs_combined_no_goal()
        self._mw._prev_obs = cur.copy()
        obs[:18], obs[18:36] = cur, cur
        self._target = self._draw_target(rng)

    def _task_info(self):
        kind, xy = self._target
        return {
            'ring_pos': self.ring_pos(),
            'peg_pos': self.peg_top(),
            'place_target': np.array([xy[0], xy[1], 1.0 if kind == 'peg' else 0.0]),  # z = 1: the peg
            'on_peg': np.int64(self.on_peg()),
        }

    def set_state(self, qpos, qvel, mocap_pos=None, mocap_quat=None, peg_pos=None):
        if peg_pos is not None:  # the peg is a fixture placed at reset: model data, not qpos
            top = np.asarray(peg_pos, dtype=np.float64)
            self._mw._target_pos = top.copy()
            self._model.body_pos[self._peg_body] = top - np.array([0.0, 0.0, 0.05])
            self._model.site('pegTop').pos = top
        super().set_state(qpos, qvel, mocap_pos, mocap_quat)


class AssemblyThreePartPolicy(ThreePartPolicy):
    LOW_Z = PLACE_Z

    @staticmethod
    def _new_mem():
        return {'phase': 'pick', 'n': 0, 'zs': []}

    @staticmethod
    def _oracle(m, env, info):
        hand = np.asarray(info['hand_pos'], dtype=np.float64)
        site = np.asarray(info['obj_pos'], dtype=np.float64)
        ring = np.asarray(info['ring_pos'], dtype=np.float64)
        target = np.asarray(info['place_target'], dtype=np.float64)
        grasp = site + GRASP_OFF
        grab = RELEASE
        phase = m['phase']

        if phase == 'carry' and np.linalg.norm(hand - grasp) > 0.05:  # dropped it: pick again
            phase, m['n'] = 'pick', 0
        if phase == 'pick':
            if np.linalg.norm(hand[:2] - grasp[:2]) > 0.015:
                desired = grasp + np.array([0.0, 0.0, APPROACH_DZ])
            elif abs(hand[2] - grasp[2]) > 0.01:
                desired = grasp
                grab = GRAB if abs(hand[2] - grasp[2]) <= 0.02 else RELEASE
            else:
                desired, grab = grasp, GRAB
                m['n'] += 1
                if (m['n'] >= 6 and info['gripper'] <= 0.7) or m['n'] >= 25:  # fingers closed (reading ~0.5 on the handle): lift
                    phase, m['n'], m['zs'] = 'carry', 0, []
        if phase == 'carry':
            grab = GRAB
            err = target[:2] - ring[:2]  # move the hand so that the ring centre reaches the target
            if np.linalg.norm(err) > 0.012:
                desired = hand + np.array([0.0, 0.0, 0.1]) if hand[2] < LIFT_Z else np.array([*(hand[:2] + err), TRAVEL_Z])
                m['zs'] = []
            else:
                desired = np.array([*(hand[:2] + err), PLACE_Z])
                m['zs'].append(hand[2])
                stuck = len(m['zs']) >= 12 and m['zs'][-12] - hand[2] < 0.002  # e.g. the ring rests on the peg top
                if hand[2] < PLACE_Z + 0.02 or stuck:
                    phase, m['n'] = 'release', 0
        if phase == 'release':
            desired, grab = hand.copy(), RELEASE
            m['n'] += 1
            if m['n'] >= 6:
                env.new_place_target()
                phase, m['n'] = 'retreat', 0
        if phase == 'retreat':
            desired, grab = np.array([hand[0], hand[1], TRAVEL_Z]), RELEASE
            if hand[2] > LIFT_Z - 0.02:
                phase = 'pick'
        m['phase'] = phase
        return np.concatenate([(desired - hand) * P_GAIN, [grab]])
