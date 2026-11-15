"""MetaWorld hammer (a CortexBench task): pick the hammer up and drive the nail into the box with it.

* Layout: metaworld 3.0 ``hammer-v3-goal-observable``; every reset places the hammer anew (x -0.1..0.1, y 0.4..0.5,
  handle along x, head at +x). The nail box never moves (0.24, 0.85); its nail is a damped slide joint (0 .. 0.1 m in,
  no spring): it stays where it is left, so an episode drives it in about once (in steps: each hit drives it to a deeper
  target depth). The hammer's claw can catch the nail head when the hand backs off after a hit and pull the nail out
  again (in about 3 % of the full collection's episodes, by 1 cm in the median).
* Targets: 'grasp' (pick the hammer up), 'hit' (drive the nail to a target depth, then hold the hammer against it for a
  while, as CortexBench's demos do after success), 'wp' (carry the hammer to a point in the air), 'place' (put it down
  on the table where resets place it, release; a 'grasp' follows). After a grasp / hit / waypoint the next target is
  drawn among hit (0.6), wp (0.3) and place (0.1).
* info adds the nail depth and the target (x, y, z, kind 0 grasp / 1 hit / 2 wp / 3 place, value = hit depth).
  obj_pos is metaworld's object position: the hammer (handle centre).
* Evaluation: the hammer and the nail are joints (qpos): the base set_state restores them.
* Backbone controller (HammerThreePartPolicy): metaworld's SawyerHammerV3Policy (grasp 4 cm behind the handle centre,
  grab effort 0.8, hit pose hand (0.05, y, 0.16), push along +y), with the target as a parameter.
"""

from __future__ import annotations

import numpy as np

from .env import MWTaskEnv
from .policy import P_GAIN, ThreePartPolicy

NAIL_Q = 16  # qpos index of NailSlideJoint
NAIL_MAX = 0.10
HIT_X, HIT_Z = 0.05, 0.16  # hand pose that lines the head up with the nail (metaworld's policy)
HIT_Y0, HIT_Y1 = 0.55, 0.715  # push from here; the hand stops here at full depth (head against the box)
CARRY_Z = 0.16
LIFTED_Z = 0.12  # the grasp is done once the hand is this high (the next target goes on from there)
WAYPOINTS = ((-0.18, 0.42, 0.12), (0.08, 0.56, 0.26))  # hand, holding the hammer (head 20 cm further in +x)
PLACES = ((-0.10, 0.40), (0.10, 0.52))  # where the hammer is put down: the reset region
HAND_INIT = ((-0.20, 0.40, 0.12), (0.20, 0.50, 0.26))
GRASP_OFF = np.array([-0.04, 0.0, -0.01])  # metaworld's grasp point relative to the handle centre
GRASP_Z = 0.05  # lowest the hand gets over the lying hammer (metaworld's policy closes from about here)
GRAB = 0.8
HOLD = (20, 70)  # steps the hammer is held against the nail after a hit
KINDS = ("grasp", "hit", "wp", "place")
P_HIT, P_WP = 0.60, 0.30  # next target while holding the hammer: hit / waypoint / place (the rest)


class MWHammerEnv(MWTaskEnv):
    TASK = "hammer"
    HAND_INIT = HAND_INIT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target = ("grasp", 0.0, np.zeros(3))

    def nail_depth(self) -> float:
        return float(self._data.qpos[NAIL_Q])

    def _draw_target(self, rng, prev: str):
        if prev == "place":
            return ("grasp", 0.0, np.zeros(3))
        u = rng.uniform()
        if u < P_HIT:
            d = self.nail_depth()
            depth = NAIL_MAX if d > NAIL_MAX - 0.03 else float(rng.uniform(d + 0.02, NAIL_MAX))
            return ("hit", depth, np.array([HIT_X, HIT_Y1, HIT_Z]))
        if u < P_HIT + P_WP:
            for _ in range(100):
                p = rng.uniform(WAYPOINTS[0], WAYPOINTS[1])
                if self.in_view(p) and self.in_view(p + np.array([0.20, 0.0, -0.04])):
                    return ("wp", 0.0, p)
            return ("wp", 0.0, p)
        (x0, y0), (x1, y1) = PLACES
        return ("place", 0.0, np.array([rng.uniform(x0, x1), rng.uniform(y0, y1), 0.0]))

    def new_place_target(self):
        self._target = self._draw_target(self.np_random, self._target[0])

    def grasp_again(self):
        """The policy lost the hammer: pick it up again."""
        self._target = ("grasp", 0.0, np.zeros(3))

    def _reset_target(self, rng, obs):
        self._target = ("grasp", 0.0, np.zeros(3))

    def _task_info(self):
        kind, value, xyz = self._target
        return {
            "nail_depth": np.float64(self.nail_depth()),
            "place_target": np.array([*xyz, float(KINDS.index(kind)), value]),
        }


class HammerThreePartPolicy(ThreePartPolicy):
    @staticmethod
    def _new_mem():
        return {"phase": "approach", "n": 0, "last": None, "hold": 0, "fails": 0}

    @staticmethod
    def _oracle(m, env, info):
        hand = np.asarray(info["hand_pos"], dtype=np.float64)
        hammer = np.asarray(info["obj_pos"], dtype=np.float64)
        target = np.asarray(info["place_target"], dtype=np.float64)
        kind = KINDS[int(round(target[3]))]
        t = int(info["step_idx"])
        if m["last"] is not None and t != m["last"] + 1:  # back from random steps: start this target over
            m["phase"], m["n"] = ("approach" if kind == "grasp" else "go"), 0
        m["last"] = t
        m["n"] += 1

        def act(desired, grab):
            return np.concatenate([(desired - hand) * P_GAIN, [grab]])

        def done():
            env.new_place_target()
            nk = KINDS[int(round(env._task_info()["place_target"][3]))]
            m["phase"], m["n"] = ("approach" if nk == "grasp" else "go"), 0

        # the grasp point, turned with the hammer (a knocked hammer lies at some yaw)
        q = np.asarray(info["obj_quat"], dtype=np.float64)
        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        c, s = np.cos(yaw), np.sin(yaw)
        g = hammer + np.array([c * GRASP_OFF[0], s * GRASP_OFF[0], GRASP_OFF[2]])
        held = np.linalg.norm(hand[:2] - g[:2]) < 0.05 and hammer[2] > 0.05

        # dropped it: pick it up again
        if kind != "grasp" and m["phase"] not in ("lower", "release") and not held and m["n"] > 3:
            env.grasp_again()
            kind, m["phase"], m["n"] = "grasp", "approach", 0

        if kind == "grasp":
            phase = m["phase"]
            if phase == "approach":  # above the grasp point, fingers open
                if hand[2] < GRASP_Z + 0.04 and np.linalg.norm(hand[:2] - g[:2]) > 0.03:
                    return act(np.array([hand[0], hand[1], GRASP_Z + 0.08]), -1.0)
                if np.linalg.norm(hand[:2] - g[:2]) > 0.015:
                    return act(np.array([g[0], g[1], GRASP_Z + 0.07]), -1.0)
                m["phase"], m["n"] = "down", 0
            gz = np.array([g[0], g[1], max(g[2], GRASP_Z)])
            if m["phase"] == "down":
                m.setdefault("zs", []).append(hand[2])
                stalled = len(m["zs"]) > 4 and m["zs"][-5] - hand[2] < 0.002
                if hand[2] - gz[2] > 0.008 and not stalled and m["n"] < 25:
                    return act(gz, -1.0 if hand[2] - gz[2] > 0.03 else GRAB)
                m["phase"], m["n"], m["zs"] = "close", 0, []
            if m["phase"] == "close":
                if m["n"] < 6:
                    return act(gz, GRAB)
                m["phase"], m["n"] = "lift", 0
            if m["phase"] == "lift":
                if hand[2] < LIFTED_Z and m["n"] < 25:
                    return act(np.array([hand[0], hand[1], CARRY_Z]), GRAB)
                if hammer[2] > 0.06:  # it came up with the hand
                    m["fails"] = 0
                    done()
                    return act(np.array([hand[0], hand[1], CARRY_Z]), GRAB)
                m["fails"] += 1
                m["phase"], m["n"] = "approach", 0
                return act(np.array([hand[0], hand[1], CARRY_Z]), -1.0)
            return act(hand, -1.0)

        if kind == "hit":
            phase = m["phase"]
            if phase == "go":  # in front of the nail (from further in, back off first), then push
                pre = (
                    np.array([HIT_X, HIT_Y0, HIT_Z])
                    if hand[1] < HIT_Y0 + 0.02
                    else np.array([hand[0], hand[1] - 0.03, HIT_Z])
                )
                if np.linalg.norm(hand - pre) > 0.02 and m["n"] < 60:
                    return act(pre, GRAB)
                m["phase"], m["n"], m["ys"] = "push", 0, []
            if m["phase"] == "push":  # along +y until the nail is at the target depth, or the head stops (at the box)
                m["ys"].append(hand[1])
                stuck = len(m["ys"]) > 6 and m["ys"][-1] - m["ys"][-6] < 0.002
                if env.nail_depth() < target[4] - 0.003 and hand[1] < HIT_Y1 and not stuck and m["n"] < 50:
                    return act(np.array([HIT_X, hand[1] + 0.02, HIT_Z]), GRAB)
                m["phase"], m["n"], m["hold"] = "hold", 0, int(env.np_random.integers(HOLD[0], HOLD[1] + 1))
            if m["phase"] == "hold":
                if m["n"] < m["hold"]:
                    return act(np.array([HIT_X, hand[1], HIT_Z]), GRAB)
                m["phase"], m["n"] = "back", 0
            if m["phase"] == "back":
                if hand[1] > HIT_Y0 + 0.01 and m["n"] < 30:
                    return act(np.array([HIT_X, HIT_Y0, HIT_Z]), GRAB)
                done()
            return act(hand, GRAB)

        if kind == "wp":
            wp = target[:3]
            if np.linalg.norm(hand - wp) < 0.03 or m["n"] > 60:
                done()
            return act(wp, GRAB)

        # place: over the spot, down until the hammer rests on the table, open, up
        spot = target[:3]
        phase = m["phase"]
        if phase == "go":
            if np.linalg.norm(hand[:2] - spot[:2]) > 0.02 and m["n"] < 60:
                return act(np.array([spot[0], spot[1], CARRY_Z]), GRAB)
            m["phase"], m["n"] = "lower", 0
        if m["phase"] == "lower":
            if hammer[2] > 0.03 and m["n"] < 30:
                return act(np.array([spot[0], spot[1], hand[2] - 0.03]), GRAB)
            m["phase"], m["n"] = "release", 0
        if m["phase"] == "release":
            if m["n"] < 6:
                return act(hand, -1.0)
            if hand[2] < hammer[2] + 0.10 and m["n"] < 30:
                return act(np.array([hand[0], hand[1], hammer[2] + 0.14]), -1.0)
            done()
        return act(hand, -1.0)
