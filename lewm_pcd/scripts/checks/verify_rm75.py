"""Acceptance check for the RM75-B + EG2-4C2 OGBench cube env (envs/cube_rm75_env.py): one command, PASS/FAIL per
claim, measured vs expected. Every reference is outside the env code: the vendor URDFs (loaded with MuJoCo's own URDF
importer from $STABLEWM_HOME/vendor/realman/, RealMan's rm_models), the vendor dimension drawing and datasheets, the
original UR5e env, and the UR5e dataset (section E runs only when both point-cloud datasets exist).

    source env.sh; CUDA_VISIBLE_DEVICES= python scripts/checks/verify_rm75.py      # ~5 min on CPU, report -> $STABLEWM_HOME/logs/verify_rm75.txt
"""
import os
import re
import sys
from pathlib import Path

import h5py
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle  # noqa: E402
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv  # noqa: E402

from envs.cube_rm75_env import ARM_XML, GRIPPER_XML, RM75CubeEnv  # noqa: E402

HOME = Path(os.environ.get('STABLEWM_HOME', str(REPO.parent / 'lewm_pcd_data')))
VENDOR = HOME / 'vendor' / 'realman'


def import_urdf(urdf, pkg, meshdir):
    """The vendor URDF as MuJoCo's own URDF importer reads it (package:// stripped, meshes from the vendor folder)."""
    txt = urdf.read_text().replace(f'package://{pkg}/meshes/', '')
    txt = re.sub(r'<robot\s+name="([^"]+)">',
                 lambda m: m.group(0) + f'<mujoco><compiler meshdir="{meshdir}" balanceinertia="true" discardvisual="false" fusestatic="false"/></mujoco>',
                 txt, count=1)
    return mujoco.MjModel.from_xml_string(txt)


ARM_URDF = VENDOR / 'RM75-B/urdf/RM75-B/urdf/RM75-B.urdf'
GRIPPER_URDF = VENDOR / 'EG2-4C2/urdf/EG2-4C2.urdf'
for _p in (ARM_URDF, GRIPPER_URDF):
    if not _p.exists():
        sys.exit(f'missing vendor file {_p}: put the RM75-B and EG2-4C2 folders of RealMan rm_models (urdf + meshes) under {VENDOR}')
REPORT = HOME / 'logs' / 'verify_rm75.txt'
REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text('')

RESULTS = []


def check(section, name, ok, measured, expected):
    tag = 'INFO' if ok is None else ('PASS' if ok else 'FAIL')
    RESULTS.append((tag, section, name, measured, expected))
    line = f'[{tag}] {section} | {name}: {measured}   (expected: {expected})'
    print(line, flush=True)
    with REPORT.open('a') as f:  # line by line, so a crash keeps what passed
        f.write(line + '\n')


# ---------------------------------------------------------------- A: arm model vs vendor URDF / drawing / datasheet
vend = import_urdf(ARM_URDF, 'RM75-B', VENDOR / 'RM75-B/urdf/RM75-B/meshes')
mine = mujoco.MjModel.from_xml_path(str(ARM_XML))
links = [f'link_{i}' for i in range(1, 8)]
d = np.abs(vend.jnt_range - mine.jnt_range).max()
check('A arm', 'joint ranges = vendor URDF', d < 1e-9, f'max diff {d:.1e} rad', '0')
deg = np.degrees(mine.jnt_range[:, 1])
sheet = np.array([178, 130, 178, 135, 178, 128, 360])
check('A arm', 'joint ranges = datasheet (+-178/130/178/135/178/128/360 deg)', np.allclose(deg, sheet, atol=0.5),
      f'+-{np.round(deg, 1).tolist()} deg', f'+-{sheet.tolist()} deg')
for field, unit in [('body_mass', 'kg'), ('body_ipos', 'm')]:
    d = np.abs(getattr(vend, field) - getattr(mine, field)).max()
    check('A arm', f'link {field[5:]} = vendor URDF', d < 1e-9, f'max diff {d:.1e} {unit}', '0')
rel = (np.abs(vend.body_inertia - mine.body_inertia) / np.maximum(np.abs(vend.body_inertia), 1e-12))[1:].max()
check('A arm', 'link inertia = vendor URDF (principal moments, relative)', rel < 1e-3, f'max relative diff {rel:.1e}',
      '< 0.1 % (both are MuJoCo eigen-decompositions of the same URDF numbers)')
check('A arm', 'total arm mass', abs(mine.body_mass.sum() - 7.776) < 1e-3, f'{mine.body_mass.sum():.3f} kg', '7.776 kg (URDF), datasheet 7.8 kg')
urdf = ARM_URDF.read_text()
eff = [float(x) for x in re.findall(r'effort="([0-9.]+)"', urdf)]
fr = mine.actuator_forcerange[:, 1].tolist()
check('A arm', 'actuator torque limits = URDF effort limits', np.allclose(eff, fr), f'{fr} Nm', f'{eff} Nm')
dv, dm = mujoco.MjData(vend), mujoco.MjData(mine)
rng = np.random.default_rng(0)
perr = rerr = 0.0
for _ in range(2000):
    q = rng.uniform(*mine.jnt_range.T)
    dv.qpos[:], dm.qpos[:] = q, q
    mujoco.mj_kinematics(vend, dv)
    mujoco.mj_kinematics(mine, dm)
    for n in links:
        perr = max(perr, np.linalg.norm(dv.body(n).xpos - dm.body(n).xpos))
        rerr = max(rerr, np.abs(dv.body(n).xmat - dm.body(n).xmat).max())
check('A arm', 'forward kinematics = vendor URDF (2000 random joint vectors, all 7 links)', perr < 5e-5 and rerr < 1e-4,
      f'max {perr * 1e3:.4f} mm, rotation {rerr:.1e}', '< 0.05 mm (the URDF writes pi/2 as 1.5708)')
dm.qpos[:] = 0
mujoco.mj_kinematics(mine, dm)
h = [round(dm.body(n).xpos[2] * 1000, 2) for n in ['link_1', 'link_3', 'link_5', 'link_7']]
drawing = [240.5, 496.5, 706.5, 850.5]
check('A arm', 'joint heights at zero pose = vendor drawing (240.5 + 256 + 210 + 144 mm)', np.allclose(h, drawing, atol=0.01),
      f'{h} mm', f'{drawing} mm')

# ---------------------------------------------------------------- B: gripper model vs vendor URDF / datasheet
vg = import_urdf(GRIPPER_URDF, 'EG2-4C2', VENDOR / 'EG2-4C2/meshes')
mg = mujoco.MjModel.from_xml_path(str(GRIPPER_XML))
mimic = {'4C2_Joint1': 1, '4C2_Joint5': 1, '4C2_Joint2': 1, '4C2_Joint3': -1, '4C2_Joint4': -1, '4C2_Joint6': 1}
pairs = {'4C2_Link1': 'left_driver', '4C2_Link5': 'left_pad', '4C2_Link2': 'left_follower',
         '4C2_Link4': 'right_driver', '4C2_Link6': 'right_pad', '4C2_Link3': 'right_follower'}
dvg, dmg = mujoco.MjData(vg), mujoco.MjData(mg)
perr = rerr = 0.0
for th in np.linspace(0.02, 0.82, 17):
    for jn, s in mimic.items():
        dvg.qpos[vg.joint(jn).qposadr[0]] = s * th
    dmg.qpos[:] = 0.82 - th  # every hinge of the MJCF counts 0 = open
    mujoco.mj_kinematics(vg, dvg)
    mujoco.mj_kinematics(mg, dmg)
    base = dmg.body('base').xpos  # the MJCF has the 11.8 mm adapter plate below the vendor base frame
    for vn, mn in pairs.items():
        perr = max(perr, np.linalg.norm(dvg.body(vn).xpos - (dmg.body(mn).xpos - base)))
        rerr = max(rerr, np.abs(dvg.body(vn).xmat - dmg.body(mn).xmat).max())
check('B gripper', 'linkage kinematics = vendor URDF + its mimic tags (17 openings, 6 links)', perr < 1e-5 and rerr < 1e-5,
      f'max {perr * 1e3:.5f} mm, rotation {rerr:.1e}', '0')


def gap(q):
    dmg.qpos[:] = q
    mujoco.mj_kinematics(mg, dmg)
    lp, rp = mg.geom('left_pad1').id, mg.geom('right_pad1').id
    lx = dmg.geom_xpos[lp] + dmg.geom_xmat[lp].reshape(3, 3)[:, 0] * mg.geom_size[lp][0]
    rx = dmg.geom_xpos[rp] - dmg.geom_xmat[rp].reshape(3, 3)[:, 0] * mg.geom_size[rp][0]
    return (rx - lx)[0]


check('B gripper', 'max opening between the finger pads', 60e-3 < gap(0.0) < 72e-3, f'{gap(0.0) * 1e3:.1f} mm',
      'datasheet stroke 70 mm (the vendor finger meshes give 66)')
check('B gripper', 'closed', gap(0.8) < 3e-3, f'{gap(0.8) * 1e3:.1f} mm', '< 3 mm')
check('B gripper', 'mass', abs(mg.body_mass.sum() - 0.271) < 0.005, f'{mg.body_mass.sum() * 1e3:.0f} g', '231 g datasheet + 40 g adapter plate')
spec = mujoco.MjSpec.from_file(str(GRIPPER_XML))
blk = spec.worldbody.add_body(name='block', pos=[0, 0, 0.0118 + 0.131])
blk.add_geom(name='block', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02])
mb = spec.compile()
mb.opt.gravity[:] = 0
db = mujoco.MjData(mb)
db.ctrl[0] = 255
for _ in range(1500):
    mujoco.mj_step(mb, db)
F, f6 = 0.0, np.zeros(6)
for i in range(db.ncon):
    c = db.contact[i]
    if {mb.geom(c.geom1).name, mb.geom(c.geom2).name} == {'block', 'right_pad1'}:
        mujoco.mj_contactForce(mb, db, i, f6)
        F += f6[0]
check('B gripper', 'grip force on a 40 mm cube (steady state)', 15 <= F <= 25, f'{F:.1f} N', 'datasheet max 20 N')

# ---------------------------------------------------------------- C: interface = original UR5e env
KW = dict(env_type='single', ob_type='states', mode='data_collection', visualize_info=False, terminate_at_goal=False,
          width=64, height=64)
U, R = CubeEnv(**KW), RM75CubeEnv(**KW)
ou, iu = U.reset(seed=0)
orr, ir = R.reset(seed=0)
A, B = U.action_space, R.action_space
check('C interface', 'action space', A.shape == B.shape and np.allclose(A.low, B.low) and np.allclose(A.high, B.high),
      f'{B.shape} in [{B.low.min()}, {B.high.max()}]', f'{A.shape} in [{A.low.min()}, {A.high.max()}] (UR5e)')
check('C interface', 'action scaling (5 cm / 5 cm / 5 cm / 0.3 rad / 1 per step)', np.allclose(U.action_high, R.action_high),
      R.action_high.tolist(), U.action_high.tolist())
check('C interface', 'control / physics timestep', (U._control_timestep, U._physics_timestep) == (R._control_timestep, R._physics_timestep),
      f'{R._control_timestep} / {R._physics_timestep} s', f'{U._control_timestep} / {U._physics_timestep} s')
_, _, _, _, su = U.step(np.zeros(5))
_, _, _, _, sr = R.step(np.zeros(5))
check('C interface', 'reset-info and step-info keys', set(iu) == set(ir) and set(su) == set(sr), f'{len(ir)} / {len(sr)} keys',
      f'the same {len(iu)} / {len(su)} keys as UR5e')
allowed = {'proprio/joint_pos', 'proprio/joint_vel', 'qpos', 'qvel', 'prev_qpos', 'prev_qvel', 'control', 'target'}
diff = sorted(k for k in su if np.shape(su[k]) != np.shape(sr[k]))
check('C interface', 'entries whose shape differs', set(diff) <= allowed, diff, 'only joint-count ones (7 arm joints, other gripper)')
check('C interface', 'state observation', (ou.shape, orr.shape) == ((28,), (30,)), f'{ou.shape} -> {orr.shape}', '(28,) -> (30,): +1 joint pos, +1 joint vel')
check('C interface', 'cameras (names, positions, orientations)', U.cameras == R.cameras, list(R.cameras), 'identical to UR5e')
check('C interface', 'variation-space entries', U.variation_space.names() == R.variation_space.names(), f'{len(R.variation_space.names())} entries', 'identical to UR5e')


def grip(env):
    env.reset(seed=1)
    for _ in range(40):
        *_, info = env.step(np.array([0, 0, 0, 0, 1.0]))
    closed = info['proprio/gripper_opening'][0]
    for _ in range(40):
        *_, info = env.step(np.array([0, 0, 0, 0, -1.0]))
    return closed, info['proprio/gripper_opening'][0]


gu, gr = grip(U), grip(R)
check('C interface', 'gripper_opening after 2 s closing / opening', gr[0] > 0.95 and gr[1] < 0.05,
      f'{gr[0]:.2f} / {gr[1]:.2f}', f'~1 / ~0 (UR5e {gu[0]:.2f} / {gu[1]:.2f})')

# ---------------------------------------------------------------- D: behaviour under the collection policy (plan oracle)


def rollouts(env, episodes, seed):
    np.random.seed(seed)
    ok, lifted, viol, selfc, vmax = [], [], 0, 0, 0
    for ep in range(episodes):
        ob, info = env.reset(seed=seed + ep)
        m, dd = env._model, env._data
        robot = np.array([m.body(b).name.startswith('ur5e/') for b in range(m.nbody)])
        lo, hi = m.jnt_range[env._arm_joint_ids].T
        oracle = CubePlanOracle(env=env, noise=0.1, noise_smoothing=0.5)
        oracle.reset(ob, info)
        maxz = 0.0
        for _ in range(200):
            ob, _, _, _, info = env.step(np.clip(oracle.select_action(ob, info), -1, 1))
            q = info['proprio/joint_pos']
            viol += int(np.any((q < lo - 1e-3) | (q > hi + 1e-3)))
            vmax = np.maximum(vmax, np.abs(info['proprio/joint_vel']))
            selfc += int(any(robot[m.geom_bodyid[c.geom1]] and robot[m.geom_bodyid[c.geom2]] for c in dd.contact[: dd.ncon]))
            maxz = max(maxz, info['privileged/block_0_pos'][2])
            if oracle.done:
                ok.append(bool(env._success))
                lifted.append(maxz > 0.035)
                maxz = 0.0
                ob, info = env.set_new_target(p_stack=0.0)
                oracle.reset(ob, info)
    return np.mean(ok), np.mean(lifted), len(ok), viol, selfc, vmax


ur = rollouts(U, 50, 5000)
rm = rollouts(R, 50, 5000)
check('D behaviour', 'plan-oracle subtask success (50 episodes)', rm[0] >= 0.95, f'RM75 {rm[0] * 100:.1f} % of {rm[2]}',
      f'>= 95 % (UR5e {ur[0] * 100:.1f} % of {ur[2]})')
check('D behaviour', 'cube lifted', rm[1] >= 0.95, f'{rm[1] * 100:.1f} %', f'>= 95 % (UR5e {ur[1] * 100:.1f} %)')
check('D behaviour', 'arm joint-limit violations', rm[3] == 0, f'{rm[3]} steps', '0')
vlim = np.radians([180, 180, 225, 225, 225, 225, 225])
check('D behaviour', 'peak joint speeds <= datasheet (180 deg/s J1-2, 225 deg/s J3-7)', np.all(rm[5] <= vlim),
      f'{np.round(np.degrees(rm[5])).astype(int).tolist()} deg/s', f'<= {np.degrees(vlim).astype(int).tolist()}')
check('D behaviour', 'robot self-collisions', rm[4] == 0, f'{rm[4]} steps', f'0 (UR5e {ur[4]})')

# ---------------------------------------------------------------- E: datasets / point clouds (collected with the unchanged pipeline; needs both datasets)
DS = HOME / 'datasets' / 'ogbench'
RM75_DS, UR5E_DS = DS / 'cube_rm75_single_expert_pcd.h5', DS / 'cube_single_expert_pcd.h5'
if not (RM75_DS.exists() and UR5E_DS.exists()):
    check('E data', 'datasets', None, f'skipped: need {RM75_DS.name} and {UR5E_DS.name}', 'both point-cloud datasets present')
    fr_ = fu_ = None
else:
    fr_, fu_ = h5py.File(RM75_DS, 'r'), h5py.File(UR5E_DS, 'r')
    check('E data', 'dataset columns', set(fr_) == set(fu_), f'{len(fr_)} columns', 'the same names as the UR5e dataset')
    allowed = {'observation', 'target', 'qpos', 'qvel', 'prev_qpos', 'prev_qvel', 'control', 'proprio_joint_pos', 'proprio_joint_vel'}
    diff = sorted(k for k in fr_ if fr_[k].shape[1:] != fu_[k].shape[1:])
    check('E data', 'columns whose shape differs', set(diff) <= allowed, diff, 'only joint-count ones')


    def pcd_stats(f, base_xy):
        xyz, cube_pos = f['pcd_xyz'][::997], f['privileged_block_0_pos'][::997]
        on_cube = np.all(np.abs(xyz - cube_pos[:, None, :]) < 0.022, axis=-1)  # 40 mm cube + 2 mm
        base = (np.hypot(xyz[..., 0] - base_xy[0], xyz[..., 1] - base_xy[1]) < 0.075) & (xyz[..., 2] < 0.28)
        return on_cube.mean(), base.mean(), xyz[..., 0].min()


    cr, br, xr = pcd_stats(fr_, (0.15, 0.0))
    cu, bu, xu = pcd_stats(fu_, (0.0, 0.0))
    check('E data', 'share of points on the cube (pipeline quota 25 %, fewer when the cube is occluded)', abs(cr - cu) < 0.05,
          f'RM75 {cr * 100:.1f} %', f'about UR5e {cu * 100:.1f} %')
    check('E data', 'share of points on the static robot base', br < 0.01, f'RM75 {br * 100:.2f} %', f'~0 (UR5e {bu * 100:.2f} %)')
    check('E data', 'crop x_min', xr >= 0.2099, f'{xr:.3f} m', '0.21 m (UR5e data 0.10)')

n_pass = sum(r[0] == 'PASS' for r in RESULTS)
n_fail = sum(r[0] == 'FAIL' for r in RESULTS)
print(f'\n{n_pass} PASS, {n_fail} FAIL')
with REPORT.open('a') as f:
    f.write(f'\n{n_pass} PASS, {n_fail} FAIL\n')
