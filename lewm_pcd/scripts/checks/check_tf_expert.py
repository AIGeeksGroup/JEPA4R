"""envs/trifinger/expert.py's experts against CortexBench's demos of a task (move or reach, 125 each): per demo, the
expert's actions next to the demo's (delta_ftpos, the BC targets) when fed the demo's own cube poses, then a rollout in
the task's CortexBench env (MoveCubeEnv / CubeReachEnv with sim_nn.Task's settings, cameras off) from the demo's start
(and goal): fingertip and cube trajectories against the demo's; move: the final scaled success against the demo's. One
task per process (CortexBench's env code uses pybullet client 0 in places, e.g. close()). From the CortexBench
TriFinger demo home (bench/cortexbench/run_tf_grid.sh):
    cd $STABLEWM_HOME/datasets/cortexbench/trifinger && PYTHONPATH=<lewm_pcd>:<lewm_pcd>/third_party/stable-worldmodel \
        $PY <lewm_pcd>/scripts/checks/check_tf_expert.py move|reach"""
import json
import os
import sys

import numpy as np
import torch

from envs.trifinger.expert import MoveExpert, ReachExpert, N_STEPS, REACH_STEPS  # also puts trifinger_vc on sys.path
from trifinger_simulation.trifinger_platform import ObjectType
from trifinger_vc.trifinger_envs.action import ActionType

TASK = sys.argv[1]
JSON = {'move': 'feb_demos_dtrain-110_train-0-100_dtest-110_test-100-125_scale-100_dts-0p4.json',
        'reach': 'demos_dtrain-110_train-0-100_dtest-110_test-100-125_scale-100_dts-0p4.json'}[TASK]
info = json.load(open(f'assets/bc_demos/{JSON}'))
demos = [torch.load(os.path.join('assets/data/trifinger-demos', s['path'], 'downsample.pth'), weights_only=False)
         for s in info['train_demo_stats'] + info['test_demo_stats']]
common = dict(action_type=ActionType.TORQUE, step_size=100, visualization=False, enable_cameras=False,
              finger_type='trifinger_meta', time_step=0.004, camera_delay_steps=0, object_type=ObjectType.COLORED_CUBE,
              enable_shadows=False, camera_view='default', arena_color='default', visual_observation=False,
              run_rl_policy=False)
if TASK == 'move':
    from trifinger_vc.trifinger_envs.gym_cube_env import MoveCubeEnv
    env = MoveCubeEnv(goal_pose=None, goal_visualization=False, no_collisions=False, **common)
    ex, n = MoveExpert(), N_STEPS
else:
    from trifinger_vc.trifinger_envs.cube_reach import CubeReachEnv
    env = CubeReachEnv(**common)
    ex, n = ReachExpert(), REACH_STEPS


def expert_reset(ft, cube_pos, cube_quat, d):
    if TASK == 'move':
        ex.reset(ft, cube_pos, cube_quat, d['o_pos_des'][0] / 100)
    else:
        ex.reset(ft, cube_pos, cube_quat)


act_err, ft_err, cube_err, succ, succ_demo = [], [], [], [], []
for d in demos:
    # 1. actions on the demo's own cube poses
    expert_reset(d['ft_pos_cur'][0] / 100, d['o_pos_cur'][0] / 100, d['o_ori_cur'][0], d)
    a = np.stack([ex.act(k, d['o_pos_cur'][k] / 100, d['o_ori_cur'][k]) for k in range(n)]) * 100
    act_err.append(np.abs(a - d['delta_ftpos'][:n]).max())
    # 2. rollout in the task's env from the demo's start (and goal)
    init = {'position': d['o_pos_cur'][0] / 100, 'orientation': d['o_ori_cur'][0]}
    if TASK == 'move':
        obs = env.reset(goal_pose_dict={'position': d['o_pos_des'][0] / 100, 'orientation': d['o_ori_des'][0]},
                        init_pose_dict=init, init_robot_position=d['robot_pos'][0])
    else:
        obs = env.reset(init_pose_dict=init, init_robot_position=d['robot_pos'][0])
    expert_reset(obs['ft_pos_cur'], obs['object_position'], obs['object_orientation'], d)
    ft, cube = [obs['ft_pos_cur']], [obs['object_position']]
    for k in range(n):
        a = ex.act(k, obs['object_position'], obs['object_orientation'])
        obs, _, _, _ = env.step(a if TASK == 'move' else a[:3])
        ft.append(obs['ft_pos_cur']); cube.append(obs['object_position'])
    ft, cube = np.stack(ft) * 100, np.stack(cube) * 100
    ft_err.append(np.linalg.norm((ft - d['ft_pos_cur']).reshape(-1, 3, 3), axis=-1).max())
    cube_err.append(np.linalg.norm(cube - d['o_pos_cur'], axis=-1).max())
    if TASK == 'move':
        start = np.linalg.norm(d['o_pos_cur'][0] - d['o_pos_des'][0])
        succ.append(max(0.0, 1 - np.linalg.norm(cube[-1] - d['o_pos_des'][0]) / start))
        succ_demo.append(max(0.0, 1 - np.linalg.norm(d['o_pos_cur'][-1] - d['o_pos_des'][0]) / start))
q = lambda v: f'median {np.median(v):.3f}, 90% {np.percentile(v, 90):.3f}, max {np.max(v):.3f}'
print(f'{TASK}: {len(demos)} demos')
print(f'actions on the demo cube poses vs delta_ftpos (cm, max over a demo): {q(act_err)}')
print(f'rollout vs demo, fingertips (cm, max over a demo): {q(ft_err)}')
print(f'rollout vs demo, cube (cm, max over a demo): {q(cube_err)}')
if TASK == 'move':
    print(f'final scaled success: reproduced expert {100 * np.mean(succ):.1f}, demos {100 * np.mean(succ_demo):.1f} '
          f'(per demo |difference| {q(100 * np.abs(np.array(succ) - np.array(succ_demo)))})')
