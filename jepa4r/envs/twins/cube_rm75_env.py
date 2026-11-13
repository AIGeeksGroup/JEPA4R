# Moved from cortexbench@9331494:lewm_pcd/envs/cube_rm75_env.py (verbatim except: CubePCDEnv imported from
# jepa4r.envs.ogbench.cube_pcd_env; lines wrapped at 120 columns)
"""OGBench cube on the lab robot: RealMan RM75-B arm + Inspire EG2-4C2 gripper instead of UR5e + Robotiq 2F-85.

Only OGBench's robot layer (``ManipSpaceEnv``) is replaced; the task layers above it are reused unchanged:

    RM75CubeEnv(CubeEnv, RM75ManipSpaceEnv)          'swm/OGBCubeRM75-v0'     = swm/OGBCube-v0, new robot
    RM75CubePCDEnv(CubePCDEnv, RM75ManipSpaceEnv)    'swm/OGBCubeRM75PCD-v0'  = swm/OGBCubePCD-v0, new robot

Changed, all in RM75ManipSpaceEnv:
  * robot MJCFs (descriptions/, built on OGBench's ur5e.xml / 2f85.xml templates from the vendor URDFs in RealMan's
    rm_models: RM75-B, EG2-4C2 and the B-series two-finger adapter plate). The MJCF namespaces keep OGBench's slot
    names ``ur5e`` / ``robotiq`` and the gripper keeps the names OGBench looks up (site pinch, joint
    right_driver_joint, body right_pad, actuator fingers_actuator, materials), so every ManipSpaceEnv / CubeEnv
    method that finds the robot by name runs as is;
  * robot base position and the four workspace boxes. The RM75 reaches 610 mm (UR5e 850 mm); with the gripper
    pointing down it covers OGBench's boxes scaled by ``workspace_scale`` about (x 0.425, y 0, table), with the
    base at ``arm_base_pos``. The xy centre does not move, so the cameras and the point-cloud normaliser stay
    valid. The point-cloud crop box does not: its x_min has to follow the base (0.10 -> 0.21, ``CROP_BOX``; set in
    the RM-75 collection / eval configs). Any other (arm_base_pos, workspace_scale) pair has to be re-checked
    (reachability of the far corners, plan-oracle success: scripts/checks/verify_rm75.py section D);
  * 7-joint home posture, IK model and seven PD gains. The IK is OGBench's DiffIKController, unchanged: with 7
    joints its damped-least-squares step is the minimum-norm one, so the elbow swivel is left free (its qpos0
    null-space option, gain 1 on every joint, drives the RM75 into its joint limits and is not used).
The task layer's own UR5e-workspace constants -- the default cube start / goal position of the variation space and
the task-mode cube positions -- are moved into the new workspace with the same scaling (scale_cube_task).
Unchanged: 5-D relative end-effector action and its scaling, 20 Hz control / 500 Hz physics, observation and info
keys, gripper-opening convention (0 open .. 1 closed), cameras, variation space ranges, and the plan oracle that
every collection script uses. Caveat: ``proprio/gripper_contact`` is |pad force| / 50 N; the EG2-4C2 grips with
at most ~20 N (datasheet), so a grasp reads ~0.42 instead of the Robotiq's ~1.0, and swm's markov oracle (grasped =
contact > 0.5; ExpertPolicy's default policy_type) stalls on this robot.
Acceptance checks against the vendor URDFs / datasheets / the UR5e env: scripts/checks/verify_rm75.py.
"""

from pathlib import Path

import mujoco
import numpy as np
from dm_control import mjcf
from ogbench.manipspace import controllers, lie, mjcf_utils
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

from jepa4r.envs.ogbench.cube_pcd_env import CubePCDEnv

DESC_DIR = Path(__file__).resolve().parent / 'descriptions'
ARM_XML = DESC_DIR / 'realman_rm75b' / 'rm75b.xml'
GRIPPER_XML = DESC_DIR / 'inspire_eg2_4c2' / 'eg2_4c2.xml'

ARM_BASE_POS = (0.15, 0.0, 0.0)  # base_link origin (bottom of the base) on the table, behind the workspace
WORKSPACE_SCALE = 0.5  # OGBench boxes about SCALE_CENTER; largest copy the RM75 reaches with the tool down
SCALE_CENTER = np.array([0.425, 0.0, 0.02])  # workspace centre on the table (cube centre height)
# = envs.pcd_wrapper.DEFAULT_CROP_BOX with x_min past the RM75 base (which stands at x 0.09..0.21); the RM-75
# collection / eval configs pass it to PointCloudWrapper
CROP_BOX = ((0.21, -0.50, -0.02), (0.75, 0.50, 0.60))


class RM75ManipSpaceEnv(ManipSpaceEnv):
    """``ManipSpaceEnv`` with the RealMan RM75-B + Inspire EG2-4C2 (see the module docstring)."""

    def __init__(self, *args, arm_base_pos=ARM_BASE_POS, workspace_scale=WORKSPACE_SCALE, **kwargs):
        super().__init__(*args, **kwargs)

        # Robot constants (ManipSpaceEnv.__init__ has just set the UR5e ones).
        self._arm_base_pos = np.asarray(arm_base_pos, dtype=np.float64)
        # Elbow up, joints 1/3/5/7 at zero, tool vertical, pinch at (0.425, 0, 0.16).
        self._home_qpos = np.asarray([0.0, 0.2725, 0.0, 1.4912, 0.0, 1.3779, 0.0])
        self._workspace_scale = workspace_scale
        c = SCALE_CENTER
        for name in ['_workspace_bounds', '_arm_sampling_bounds', '_object_sampling_bounds', '_target_sampling_bounds']:
            box = getattr(self, name)
            setattr(self, name, c[: box.shape[1]] + workspace_scale * (box - c[: box.shape[1]]))

        # Inverse kinematics on the bare arm, placed as in the scene.
        ik_mjcf = self._arm_mjcf()
        xml_str = mjcf_utils.to_string(ik_mjcf)
        assets = mjcf_utils.get_assets(ik_mjcf)
        ik_model = mujoco.MjModel.from_xml_string(xml_str, assets)

        self._ik = controllers.DiffIKController(model=ik_model, sites=['attachment_site'])

    def _arm_mjcf(self):
        arm_mjcf = mjcf.from_path(ARM_XML.as_posix(), escape_separators=True)
        arm_mjcf.find('body', 'base_link').pos = self._arm_base_pos
        return arm_mjcf

    def build_mjcf_model(self):
        # == ManipSpaceEnv.build_mjcf_model with the RM75-B / EG2-4C2 files.
        # Set scene.
        arena_mjcf = mjcf.from_path((self._desc_dir / 'floor_wall.xml').as_posix())
        arena_mjcf.model = 'ur5e_arena'

        arena_mjcf.statistic.center = (0.3, 0, 0.15)
        arena_mjcf.statistic.extent = 0.7
        getattr(arena_mjcf.visual, 'global').elevation = -20
        getattr(arena_mjcf.visual, 'global').azimuth = 180
        arena_mjcf.statistic.meansize = 0.04
        arena_mjcf.visual.map.znear = 0.1
        arena_mjcf.visual.map.zfar = 10.0

        # Add the RM75-B arm (OGBench slot name 'ur5e').
        ur5e_mjcf = self._arm_mjcf()
        ur5e_mjcf.model = 'ur5e'

        for light in ur5e_mjcf.find_all('light'):
            light.remove()
            del light

        # Attach the EG2-4C2 gripper (slot name 'robotiq') to the flange.
        gripper_mjcf = mjcf.from_path(GRIPPER_XML.as_posix(), escape_separators=True)
        gripper_mjcf.model = 'robotiq'
        mjcf_utils.attach(ur5e_mjcf, gripper_mjcf, 'attachment_site')

        # Attach the arm to the scene.
        mjcf_utils.attach(arena_mjcf, ur5e_mjcf)

        self.add_objects(arena_mjcf)

        # Cache joint and actuator elements.
        self._arm_jnts = mjcf_utils.safe_find_all(
            ur5e_mjcf,
            'joint',
            exclude_attachments=True,
        )
        self._arm_acts = mjcf_utils.safe_find_all(
            ur5e_mjcf,
            'actuator',
            exclude_attachments=True,
        )
        self._gripper_jnts = mjcf_utils.safe_find_all(gripper_mjcf, 'joint', exclude_attachments=True)
        self._gripper_acts = mjcf_utils.safe_find_all(gripper_mjcf, 'actuator', exclude_attachments=True)

        if self._ob_type == 'pixels':
            # Adjust colors for pixel-based tasks.
            arena_mjcf.find('material', 'ur5e/robotiq/black').rgba = self._colors['purple']
            arena_mjcf.find('material', 'ur5e/robotiq/pad_gray').rgba = self._colors['purple']
            if self._pixel_transparent_arm:
                arena_mjcf.find('material', 'ur5e/robotiq/metal').rgba[3] = 0.1
                arena_mjcf.find('material', 'ur5e/robotiq/silicone').rgba[3] = 0.1
                arena_mjcf.find('material', 'ur5e/robotiq/gray').rgba[3] = 0.1
                arena_mjcf.find('material', 'ur5e/robotiq/black').rgba[3] = 0.1
                arena_mjcf.find('material', 'ur5e/robotiq/pad_gray').rgba[3] = 0.5
                arena_mjcf.find('material', 'ur5e/black').rgba[3] = 0.1
                arena_mjcf.find('material', 'ur5e/jointgray').rgba[3] = 0.1
                arena_mjcf.find('material', 'ur5e/linkgray').rgba[3] = 0.1
                arena_mjcf.find('material', 'ur5e/lightblue').rgba[3] = 0.1

        # Add bounding boxes to visualize the workspace and object sampling bounds.
        mjcf_utils.add_bounding_box_site(
            arena_mjcf.worldbody,
            lower=np.asarray((*self._target_sampling_bounds[0], 0.02)),
            upper=np.asarray((*self._target_sampling_bounds[1], 0.02)),
            rgba=(0.6, 0.3, 0.3, 0.2),
            group=4,
            name='object_bounds',
        )
        mjcf_utils.add_bounding_box_site(
            arena_mjcf.worldbody,
            lower=np.asarray(self._arm_sampling_bounds[0]),
            upper=np.asarray(self._arm_sampling_bounds[1]),
            rgba=(0.3, 0.6, 0.3, 0.2),
            group=4,
            name='arm_bounds',
        )

        return arena_mjcf

    def post_compilation(self):
        # == ManipSpaceEnv.post_compilation with seven PD gains instead of six.
        # Arm joint and actuator IDs.
        arm_joint_names = [j.full_identifier for j in self._arm_jnts]
        self._arm_joint_ids = np.asarray([self._model.joint(name).id for name in arm_joint_names])
        actuator_names = [a.full_identifier for a in self._arm_acts]
        self._arm_actuator_ids = np.asarray([self._model.actuator(name).id for name in actuator_names])
        gripper_actuator_names = [a.full_identifier for a in self._gripper_acts]
        self._gripper_actuator_ids = np.asarray([self._model.actuator(name).id for name in gripper_actuator_names])
        self._gripper_opening_joint_id = self._model.joint('ur5e/robotiq/right_driver_joint').id

        # Modify PD gains.
        self._model.actuator_gainprm[self._arm_actuator_ids, 0] = np.asarray([4500, 4500, 4500, 4500, 2000, 2000, 500])
        self._model.actuator_gainprm[self._arm_actuator_ids, 2] = np.asarray([-450, -450, -450, -450, -200, -200, -50])
        self._model.actuator_biasprm[self._arm_actuator_ids, 1] = -np.asarray([4500, 4500, 4500, 4500, 2000, 2000, 500])

        # Site IDs.
        self._pinch_site_id = self._model.site('ur5e/robotiq/pinch').id
        self._attach_site_id = self._model.site('ur5e/attachment_site').id

        pinch_pose = lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_matrix(self._data.site_xmat[self._pinch_site_id].reshape(3, 3)),
            translation=self._data.site_xpos[self._pinch_site_id],
        )
        attach_pose = lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_matrix(self._data.site_xmat[self._attach_site_id].reshape(3, 3)),
            translation=self._data.site_xpos[self._attach_site_id],
        )
        self._T_pa = pinch_pose.inverse() @ attach_pose

        self.post_compilation_objects()


def scale_cube_task(env):
    """Scale CubeEnv's own UR5e-workspace constants like the boxes (xy about SCALE_CENTER; heights kept): the
    default cube start / goal position of the variation space, which would otherwise lie outside the scaled
    sampling box, and the task-mode cube positions. Call at the end of an RM75 cube env's __init__."""
    s, c = env._workspace_scale, SCALE_CENTER[:2]
    for key in ['start_position', 'goal_position']:
        box = env.variation_space['cube'][key]
        value = c + s * (np.asarray(box.init_value, dtype=np.float64) - c)
        box._init_value, box._value = value, value.copy()  # swm Box has no public setter for its initial value
    for task in getattr(env, 'task_infos', None) or []:
        for key in ['init_xyzs', 'goal_xyzs']:
            xyz = np.array(task[key], dtype=np.float64)
            xyz[:, :2] = c + s * (xyz[:, :2] - c)
            task[key] = xyz


class RM75CubeEnv(CubeEnv, RM75ManipSpaceEnv):
    """``swm/OGBCube-v0`` on the RM75-B + EG2-4C2."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scale_cube_task(self)


class RM75CubePCDEnv(CubePCDEnv, RM75ManipSpaceEnv):
    """``swm/OGBCubePCD-v0`` (cube + the point-cloud cameras) on the RM75-B + EG2-4C2. Collect and evaluate with
    ``CROP_BOX`` (the RM-75 configs do)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scale_cube_task(self)


__all__ = [
    'RM75ManipSpaceEnv',
    'RM75CubeEnv',
    'RM75CubePCDEnv',
    'scale_cube_task',
    'CROP_BOX',
    'ARM_XML',
    'GRIPPER_XML',
]
