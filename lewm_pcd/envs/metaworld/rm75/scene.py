"""MuJoCo model of a MetaWorld 3.0 task with the RM75-B + EG2-4C2 in place of the Sawyer.

* The task XML is loaded as is; the Sawyer (body 'base' and its subtree), its mocap body, the mocap weld and the two
  finger actuators are removed. Everything else (table, walls, objects, fixtures, lights) is unchanged.
* The arm and gripper are the repo's vendor-based MJCFs (envs/descriptions, the models of envs/cube_rm75_env.py),
  attached with prefixes rm75/ and rm75/g/. The base stands on a 20 cm pedestal on the table's +x side, facing -x
  (MOUNT_POS, MOUNT_YAW): from there the hand positions of metaworld's scripted Sawyer experts in the five CortexBench
  tasks are reachable with the tool pointing down (96-100 % of them); closer to the table (x 0.30) the arm cannot fold
  enough for assembly's peg on the +x side.
* MetaWorld looks the robot up by name; the gripper gets those names, placed so that MetaWorld's conventions hold:
  sites rightEndEffector / leftEndEffector at the two fingertips (MetaWorld's tcp_center = their midpoint), body
  'hand' HAND_ABOVE_TIP above the fingertips (MetaWorld's hand position = the Sawyer 'hand' body, 4.5 cm above its
  fingertips), bodies leftpad / rightpad at the pad centres, leftclaw / rightclaw on the pads, and the pad collision
  geoms renamed leftpad_geom / rightpad_geom (MetaWorld's grasp check). The scripted experts' offsets are relative to
  the hand position, so the fingertips go where the Sawyer's went.
* Gravity compensation for the robot (the arm's through its servos, within their torque limits).
* Physics options: MetaWorld's (dt 2.5 ms, Newton, elliptic cone) plus the gripper model's impratio 10 and the arm
  model's implicitfast integrator (its position servos have velocity damping).
"""
from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

DESC_DIR = Path(__file__).resolve().parents[2] / 'descriptions'  # envs/descriptions
ARM_XML = DESC_DIR / 'realman_rm75b' / 'rm75b.xml'
GRIPPER_XML = DESC_DIR / 'inspire_eg2_4c2' / 'eg2_4c2.xml'
MOUNT_POS = (0.35, 0.65, 0.20)  # m: the base on a pedestal on the table's +x side ...
MOUNT_YAW = np.pi               # ... facing -x
PEDESTAL_R = 0.07
HAND_ABOVE_TIP = 0.045
# fingertip in the pad bodies' frames: pad geom centre (+-0.0157, 0.006, 0.01175) + its half-length 0.01675 along z
TIP = {'left': (0.0157, 0.006, 0.0285), 'right': (-0.0157, 0.006, 0.0285)}
PAD = {'left': (0.0157, 0.006, 0.01175), 'right': (-0.0157, 0.006, 0.01175)}
# 'hand' in the gripper's 'base' body frame: fingertips 0.1495 m out (at a 40 mm opening; 0.144..0.150 over the stroke)
HAND_IN_BASE = (0.0, 0.0, 0.1495 - HAND_ABOVE_TIP)


def _absolute_meshes(spec: mujoco.MjSpec, xml: Path) -> None:
    d = xml.parent / (spec.meshdir or '')
    for m in spec.meshes:
        if m.file and not os.path.isabs(m.file):
            m.file = str(d / m.file)
    spec.meshdir = ''


def robot_spec() -> mujoco.MjSpec:
    arm = mujoco.MjSpec.from_file(str(ARM_XML))
    grip = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    _absolute_meshes(arm, ARM_XML)
    _absolute_meshes(grip, GRIPPER_XML)
    _set_options(arm)
    arm.attach(grip, prefix='g/', site=arm.site('attachment_site'))
    return arm


def _set_options(spec: mujoco.MjSpec) -> None:
    """The options the combined model uses (build_spec), set on a parent before an attach so the children's (the arm's
    implicitfast, the gripper's elliptic cone + impratio 10) do not conflict (MuJoCo logs every conflict to a file)."""
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0


def build_spec(task_xml: str) -> mujoco.MjSpec:
    pos = np.asarray(MOUNT_POS, dtype=np.float64)
    spec = mujoco.MjSpec.from_file(task_xml)
    for a in list(spec.actuators):          # the Sawyer's two finger actuators (the tasks have no others)
        spec.delete(a)
    for e in list(spec.equalities):         # the mocap weld
        if 'mocap' in (e.name1, e.name2):
            spec.delete(e)
    spec.delete(spec.body('base'))          # the Sawyer: pedestal, arm, hand, fingers
    spec.delete(spec.body('mocap'))

    spec.worldbody.add_geom(name='rm75_pedestal', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                            size=[PEDESTAL_R, pos[2] / 2, 0], pos=[pos[0], pos[1], pos[2] / 2], rgba=[0.35, 0.35, 0.38, 1])
    frame = spec.worldbody.add_frame(pos=pos, quat=np.array([np.cos(MOUNT_YAW / 2), 0, 0, np.sin(MOUNT_YAW / 2)]))
    _set_options(spec)
    spec.attach(robot_spec(), prefix='rm75/', frame=frame)

    for side in ('left', 'right'):
        pad = spec.body(f'rm75/g/{side}_pad')
        pad.add_site(name=f'{side}EndEffector', pos=TIP[side], size=[0.004, 0, 0], rgba=[1, 0, 0, 0], group=5)
        pad.add_body(name=f'{side}pad', pos=PAD[side])
        pad.add_body(name=f'{side}claw', pos=PAD[side])
        spec.geom(f'rm75/g/{side}_pad1').name = f'{side}pad_geom'
    hand = spec.body('rm75/g/base').add_body(name='hand', pos=HAND_IN_BASE)
    hand.add_site(name='rm75_hand', size=[0.004, 0, 0], rgba=[0, 1, 0, 0], group=5)

    # gravity compensation through the arm's servos (actgravcomp: it counts against their torque limits), as the
    # arm's drive does; the gripper's small links get it as a passive force
    for b in spec.bodies:
        if b.name.startswith('rm75/') or b.name in ('hand', 'leftpad', 'rightpad', 'leftclaw', 'rightclaw'):
            b.gravcomp = 1.0
    for j in spec.joints:
        if j.name.startswith('rm75/joint_'):
            j.actgravcomp = True

    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.impratio = 10.0
    return spec


def build_model(task_xml: str) -> mujoco.MjModel:
    return build_spec(task_xml).compile()
