# Moved from cortexbench@9331494:lewm_pcd/envs/ogb_scene_expert.py (verbatim except: lines wrapped at 120 columns)
"""Expert for OGBench scene's five evaluation tasks (the CortexBench demos, bench/cortexbench/make_scene_demos.py):
OGBench's four per-object oracles (the Markov ones, ogb_scene_env.make_oracles, OGBench's collection parameters) chained
by the task's goal. OGBench ships per-object oracles only, no expert for a whole task. The task env:
ogb_scene_env.make_env(mode='task', max_episode_steps=HORIZON).

Each step looks at the current state and the goal and picks the subtask to work on (closed loop: a subtask that did not
work out is picked again):
  1. the goal has the cube in the drawer and it is not there yet: drawer not open (qpos > -0.12, OGBench's condition for
     putting the cube in) -> unlock it (press button 0), then pull it open; open -> put the cube in (target = OGBench's
     _drawer_center);
  2. the drawer more than 0.04 off its goal (OGBench's success threshold): locked -> press button 0 to unlock, else
     push / pull it to the goal;
  3. the window likewise (button 1);
  4. the cube's goal on the table, more than 0.04 off: move the cube;
  5. a button state differing from the goal: press that button;
  6. everything matches: no motion (an all-zero action).
A subtask counts as done when the oracle has reached its target and lifted the hand to the oracle's own height (what the
oracle does next is moving the hand to a random end point; that part is skipped and the next subtask picked), or when
the oracle reports done itself (its step limit).
The task mode's info (mode='task') lacks the target fields the oracles read; they are filled in for the current subtask
(the drawer / window handle targets from the current handle position and the opening difference, at the same place as
OGBench's target sites).
"""
from __future__ import annotations

import os

os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np

from . import ogb_scene_env as sd

TASK_NAMES = (
    'task1_open',
    'task2_unlock_and_lock',
    'task3_rearrange_medium',
    'task4_put_in_drawer',
    'task5_rearrange_hard',
)
HORIZON = 750           # OGBench scene-v0's max_episode_steps
TOL = 0.04              # OGBench's success threshold (cube position, drawer, window)
DRAWER_IS_OPEN = -0.12  # OGBench set_new_target: the cube goes into the drawer only when qpos < -0.12
LIFT = {'cube': 0.16, 'button': 0.16, 'drawer': 0.18, 'window': 0.28}  # the oracles' above_threshold


class SceneTaskExpert:
    def __init__(self, env):
        self.env = env
        self.u = env.unwrapped
        self.oracles = sd.make_oracles(env)
        self.cur = None

    def reset(self):
        g = self.u.cur_task_info['goal']
        self.goal = dict(
            cube=np.asarray(g['block_xyzs'][0], np.float64),
            buttons=np.asarray(g['button_states']).astype(int),
            drawer=float(g['drawer_pos']),
            window=float(g['window_pos']),
        )
        self.goal_in_drawer = bool(self.goal['cube'][1] < -0.3)
        self.cur = None
        self.log = []

    def _state(self):
        d = self.u._data
        return dict(cube=d.joint('object_joint_0').qpos[:3].copy(), in_drawer=sd.cube_in_drawer(self.u),
                    buttons=self.u._cur_button_states.astype(int).copy(), drawer=float(d.joint('drawer_slide').qpos[0]),
                    window=float(d.joint('window_slide').qpos[0]))

    def plan(self):
        s, g = self._state(), self.goal
        if self.goal_in_drawer and not s['in_drawer']:
            if s['drawer'] > DRAWER_IS_OPEN:
                return ('button', 0) if s['buttons'][0] == 0 else ('drawer', sd.DRAWER_OPEN)
            return ('cube', self.u._drawer_center.copy())
        if abs(s['drawer'] - g['drawer']) > TOL:
            return ('button', 0) if s['buttons'][0] == 0 else ('drawer', g['drawer'])
        if abs(s['window'] - g['window']) > TOL:
            return ('button', 1) if s['buttons'][1] == 0 else ('window', g['window'])
        if not self.goal_in_drawer and np.linalg.norm(s['cube'] - g['cube']) > TOL:
            return ('cube', g['cube'].copy())
        for i in (0, 1):
            if s['buttons'][i] != g['buttons'][i]:
                return ('button', i)
        return None

    def _info(self, info):
        """Fill in the current subtask's target fields (the ones the oracles read from info in collection mode)."""
        kind, tgt = self.cur
        info = dict(info)
        if kind == 'cube':
            info['privileged/target_block'] = 0
            info['privileged/target_block_pos'] = np.asarray(tgt, np.float64)
            info['privileged/target_block_yaw'] = np.array([0.0])
        elif kind == 'button':
            info['privileged/target_button'] = tgt
            info['privileged/target_button_state'] = self.button_target
            info['privileged/target_button_top_pos'] = self.u._data.site_xpos[self.u._button_site_ids[tgt]].copy()
        elif kind == 'drawer':
            q = float(info['privileged/drawer_pos'][0])
            info['privileged/target_drawer_handle_pos'] = info['privileged/drawer_handle_pos'] - np.array(
                [0.0, tgt - q, 0.0]
            )
        elif kind == 'window':
            q = float(info['privileged/window_pos'][0])
            info['privileged/target_window_handle_pos'] = info['privileged/window_handle_pos'] + np.array(
                [tgt - q, 0.0, 0.0]
            )
        return info

    def _achieved_and_lifted(self, info):
        kind, tgt = self.cur
        eff = info['proprio/effector_pos']
        if eff[2] <= LIFT[kind]:
            return False
        if kind == 'cube':
            return bool(np.linalg.norm(np.asarray(tgt) - info['privileged/block_0_pos']) <= 0.02
                        and info['proprio/gripper_contact'][0] < 0.1)
        if kind == 'button':
            return int(info[f'privileged/button_{tgt}_state']) == self.button_target
        if kind == 'drawer':
            return bool(
                np.linalg.norm(info['privileged/target_drawer_handle_pos'] - info['privileged/drawer_handle_pos'])
                <= 0.01
            )
        return bool(
            np.linalg.norm(info['privileged/target_window_handle_pos'] - info['privileged/window_handle_pos']) <= 0.01
        )

    def act(self, ob, t=None):
        info = self.u.compute_ob_info()
        switch = self.cur is None
        if not switch:
            full = self._info(info)
            switch = self.oracles[self.cur[0]].done or self._achieved_and_lifted(full)
        if switch:
            self.cur = self.plan()
            if self.cur is None:
                return np.zeros(5)
            if self.cur[0] == 'button':
                self.button_target = 1 - int(self.u._cur_button_states[self.cur[1]])
            self.log.append((t, self.cur[0], self.cur[1] if self.cur[0] == 'button' else None))
            self.oracles[self.cur[0]].reset(ob, self._info(info))
        return np.asarray(self.oracles[self.cur[0]].select_action(ob, self._info(info)), np.float64)
