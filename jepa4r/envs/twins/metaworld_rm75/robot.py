# Moved from cortexbench@9331494:lewm_pcd/envs/metaworld/rm75/robot.py (verbatim)
"""MetaWorld 3.0 task envs with the RM75-B + EG2-4C2 (scene.py). make(task) returns the task's own metaworld env class
with RM75Mixin in front, so MetaWorld's object resets, observations, rewards and success checks run unchanged; only the
robot layer is replaced:

* action (unchanged protocol): clipped to [-1, 1] (as CortexBench's metaworld fork); a[:3] * 1/100 moves the hand
  target (MetaWorld moves its mocap), clipped to the task's mocap box; a[3] is the gripper command. Each env step the
  target goes through damped-least-squares IK (7 arm joints, joint limits, tool pointing down with the fingers closing
  along world y, as the Sawyer's) into the 7 position servos, then 5 physics steps of 2.5 ms (frame_skip 5, as
  MetaWorld). A target the arm cannot reach (IK residual > 1 cm) is pulled back to where the IK got. A new target is
  set at most LEASH (3 cm) from the hand, so a blocked or lagging hand does not let it run on (MetaWorld's clips only
  to the mocap box), as a drive stops at a large following error instead of winding up; during the step the hand can
  fall further behind, so the recorded target can be more than 3 cm from the recorded hand.
* gripper: EG2-4C2 position command 255 * clip(a[3], 0, 1) (0 = open; MetaWorld's Sawyer: finger targets +-a[3]).
* observation: MetaWorld's 39-D vector; hand = body 'hand' (4.5 cm above the fingertips, as the Sawyer's), obs[3] =
  pad gap / 66.2 mm (1 open .. 0 closed; the Sawyer's is claw distance / 0.1 m).
* hand reset: the arm is put at the IK solution for the task's hand_init_pos (MetaWorld drags its mocap there for 50
  steps), gripper open.
* MetaWorld addresses the main object's qpos from index 9 (after the Sawyer's 9 joints); the robot's joints come last
  here, so _set_obj_xyz / _set_hammer_xyz use the object's own address.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .scene import build_model

GAP_OPEN = 0.0662
# tool down, fingers closing along world y: gripper x -> +y, gripper z (approach) -> -z
R_T = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
HOME = np.array([0.0, 0.2725, 0.0, 1.4912, 0.0, 1.3779, 0.0])  # the cube env's tool-vertical posture
VLIM = np.radians([180, 180, 225, 225, 225, 225, 225])  # datasheet joint speeds (scripts/checks/verify_rm75.py)
LEASH = 0.03  # m: a new hand target is set at most this far from the hand
W_ROT = 0.3               # yaw (the fingers' closing direction, about the vertical)
W_TILT = 0.03              # tilt of the tool from vertical: cheap, so the tool stays vertical wherever that reaches
                           # (all 6 targets met exactly) and leans only as much as a position needs
TILT_MAX = np.radians(30)  # beyond this the tilt costs as much as the yaw


def _rot_err(Rt, R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (Rt @ R.T).reshape(-1))
    q = q if q[0] >= 0 else -q
    return q[1:] / (np.linalg.norm(q[1:]) + 1e-12) * 2 * np.arccos(np.clip(q[0], -1, 1))


class RM75Mixin:
    # ------------------------------------------------------------------ model
    def _initialize_simulation(self):
        model = build_model(self.model_name)
        model.vis.global_.offwidth, model.vis.global_.offheight = self.width, self.height
        data = mujoco.MjData(model)
        m = model
        self._arm_j = np.array([m.joint(f'rm75/joint_{i}').id for i in range(1, 8)])
        self._arm_q = m.jnt_qposadr[self._arm_j]
        self._arm_v = m.jnt_dofadr[self._arm_j]
        self._lo, self._hi = m.jnt_range[self._arm_j, 0], m.jnt_range[self._arm_j, 1]
        self._arm_act = np.array([m.actuator(f'rm75/joint_{i}').id for i in range(1, 8)])
        self._grip_act = m.actuator('rm75/g/fingers_actuator').id
        self._kv_kp = -m.actuator_biasprm[self._arm_act, 2] / m.actuator_gainprm[self._arm_act, 0]
        self._grip_j = np.array([j for j in range(m.njnt) if m.joint(j).name.startswith('rm75/g/')])
        robot = {j for j in range(m.njnt) if m.joint(j).name.startswith('rm75/')}
        obj_j = [j for j in range(m.njnt) if j not in robot]
        self._q0 = int(m.jnt_qposadr[obj_j[0]])  # MetaWorld's index 9
        self._v0 = int(m.jnt_dofadr[obj_j[0]])
        self._hand_site = m.site('rm75_hand').id
        self._pads = (m.geom('leftpad_geom').id, m.geom('rightpad_geom').id)
        self._ik_model, self._ik_d = m, mujoco.MjData(m)
        self.leash = LEASH
        self._q_ik = None
        self._R_T = R_T
        self._ik_err = 0.0
        self._q_seed = self._solve_seed()
        return model, data

    # ------------------------------------------------------------------ IK
    def _ik(self, target, q0, iters=30, rt=None, w_tilt=W_TILT):
        """Damped least squares on the hand point: position, the yaw about the vertical (weight W_ROT) and the tilt from
        vertical (weight w_tilt, W_ROT beyond TILT_MAX); the rotation error is in the world frame, so its z component is
        the yaw and its x, y components the tilt."""
        m, d = self._ik_model, self._ik_d
        q = np.array(q0, dtype=np.float64)
        rt = self._R_T if rt is None else rt
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[self._arm_q] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            ep = target - d.site_xpos[self._hand_site]
            R = d.site_xmat[self._hand_site].reshape(3, 3)
            er = _rot_err(rt, R)
            if np.linalg.norm(ep) < 3e-4 and np.linalg.norm(er) < 0.005:
                break
            tilt = np.arccos(np.clip(-R[2, 2], -1.0, 1.0))
            w = np.array([w_tilt if tilt < TILT_MAX else W_ROT] * 2 + [W_ROT])
            mujoco.mj_jacSite(m, d, jp, jr, self._hand_site)
            J = np.vstack([jp[:, self._arm_v], w[:, None] * jr[:, self._arm_v]])
            e = np.r_[ep, w * er]
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 0.02 ** 2 * np.eye(6), e), self._lo, self._hi)
        d.qpos[self._arm_q] = q
        mujoco.mj_kinematics(m, d)
        return q, d.site_xpos[self._hand_site].copy(), d.site_xmat[self._hand_site].reshape(3, 3).copy()

    def _solve_seed(self):
        """An arm posture with the hand at (0, 0.6, 0.2), tool down: the IK start for every hand reset (from the cube
        env's tool-vertical posture, else the best of a few random starts)."""
        target = np.array([0.0, 0.6, 0.2])
        rng = np.random.default_rng(0)
        best = None
        for q0 in [HOME] + [rng.uniform(self._lo, self._hi) for _ in range(20)]:
            q, p, R = self._ik(target, q0, iters=400, w_tilt=W_ROT)
            err = np.linalg.norm(p - target) + 0.1 * np.linalg.norm(_rot_err(self._R_T, R))
            if best is None or err < best[0]:
                best = (err, q)
            if err < 1e-3:
                break
        return best[1]

    # ------------------------------------------------------------------ robot layer used by metaworld
    def reset_mocap_welds(self):
        pass

    def _pad_gap(self):
        (l, r), d = self._pads, self.data
        x = self.model.geom_size[l][0]
        lx = d.geom_xpos[l] + d.geom_xmat[l].reshape(3, 3)[:, 0] * x
        rx = d.geom_xpos[r] - d.geom_xmat[r].reshape(3, 3)[:, 0] * x
        return float(np.linalg.norm(rx - lx))

    def _get_curr_obs_combined_no_goal(self):
        obs = super()._get_curr_obs_combined_no_goal()
        obs[3] = np.clip(self._pad_gap() / GAP_OPEN, 0.0, 1.0)
        return obs

    def step(self, action):
        # CortexBench's metaworld (v2 fork) clips the action in step(); metaworld 3.0 does not
        return super().step(np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0))

    def set_xyz_action(self, action):
        t = np.clip(self._ee_target + np.asarray(action) * self.action_scale, self.mocap_low, self.mocap_high)
        hand = self.data.site_xpos[self._hand_site]
        d = t - hand
        n = float(np.linalg.norm(d))
        self._ee_target = hand + d * (self.leash / n) if n > self.leash else t

    def do_simulation(self, ctrl, n_frames):
        n, dt = int(n_frames), self.model.opt.timestep
        q, reached, _ = self._ik(self._ee_target, self._q_ik)
        self._ik_err = float(np.linalg.norm(reached - self._ee_target))
        if self._ik_err > 0.01:  # out of reach: the target stays where the arm can get
            self._ee_target = reached
        q_prev = self._q_ik
        q = q_prev + np.clip(q - q_prev, -VLIM * n * dt, VLIM * n * dt)
        self._q_ik = q
        lead = self._kv_kp * (q - q_prev) / (n * dt)
        self.data.ctrl[self._grip_act] = 255.0 * float(np.clip(ctrl[0], 0.0, 1.0))
        for i in range(n):
            self.data.ctrl[self._arm_act] = q_prev + (q - q_prev) * (i + 1) / n + lead
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def _reset_hand(self, steps: int = 50):
        q, reached, _ = self._ik(np.asarray(self.hand_init_pos, dtype=np.float64), self._q_seed, iters=400)
        qpos, qvel = self.data.qpos.copy(), self.data.qvel.copy()
        qpos[self._arm_q], qvel[self._arm_v] = q, 0.0
        for j in self._grip_j:
            qpos[self.model.jnt_qposadr[j]] = 0.0
            qvel[self.model.jnt_dofadr[j]] = 0.0
        self.set_state(qpos, qvel)
        self._q_ik, self._ee_target = q, reached
        self.data.ctrl[self._arm_act] = q
        self.data.ctrl[self._grip_act] = 0.0
        mujoco.mj_step(self.model, self.data, nstep=10)  # let the servos take the load
        mujoco.mj_forward(self.model, self.data)
        self.init_tcp = self.tcp_center

    def _set_obj_xyz(self, pos):
        qpos, qvel = self.data.qpos.flat.copy(), self.data.qvel.flat.copy()
        if np.ndim(pos) == 0 or np.size(pos) == 1:  # button-press-topdown: the button's slide joint
            qpos[self._q0], qvel[self._v0] = float(np.ravel(pos)[0]), 0.0
        else:
            qpos[self._q0:self._q0 + 3] = np.asarray(pos)[:3]
            qvel[self._v0:self._v0 + 6] = 0.0
        self.set_state(qpos, qvel)

    def _set_hammer_xyz(self, pos):
        self._set_obj_xyz(pos)


_CLASSES = {}


def make(task: str, seed: int | None = None, **kwargs):
    """The metaworld <task>-v3-goal-observable env with the RM75, as CortexBench builds its envs: a new layout every
    reset (_freeze_rand_vec False), drawn from the reset seed."""
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as ENVS
    base = ENVS[f'{task}-v3-goal-observable'].__mro__[1]  # the task class under metaworld's goal-observable wrapper
    if task not in _CLASSES:
        _CLASSES[task] = type(f'RM75_{base.__name__}', (RM75Mixin, base), {})
    env = _CLASSES[task](**kwargs)
    # what metaworld's goal-observable wrapper does in its __init__ (env_dict._create_observable_goal_envs)
    env._partially_observable = False
    env._freeze_rand_vec = False
    del env.sawyer_observation_space
    env._set_task_called = True
    env.seeded_rand_vec = True
    env.seed(0 if seed is None else seed)
    env.reset()
    return env
