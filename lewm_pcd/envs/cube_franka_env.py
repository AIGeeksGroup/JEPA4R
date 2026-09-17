"""OGBench Cube with a real Franka Emika Panda and the original 5-D action.

This is a parallel robot adapter.  The Cube task, cameras, point-cloud wrapper,
collectors, LeWM model and CEM code stay unchanged.  The external action remains
``[dx, dy, dz, dyaw, dgripper]``; positive ``dgripper`` closes the hand, matching
the original OGBench convention.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from dm_control import mjcf
from ogbench.manipspace import controllers, lie, mjcf_utils
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
from stable_worldmodel.envs.utils import perturb_camera_angle

from .cube_pcd_env import CubePCDEnv


DESC_DIR = Path(__file__).resolve().parent / "descriptions"
PANDA_XML = DESC_DIR / "franka_emika_panda" / "panda.xml"
PANDA_NOHAND_XML = DESC_DIR / "franka_emika_panda" / "panda_nohand.xml"

ARM_BASE_POS = (0.0, 0.0, 0.0)
HOME_QPOS = np.array(
    [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398],
    dtype=np.float64,
)
CROP_BOX = ((0.10, -0.50, -0.02), (0.75, 0.50, 0.60))

FINGER_MAX_OPENING = 0.04
FINGER_GRASP_OPENING = 0.002
FINGER_PAD_SLIDE_FRICTION = 2.0
GRIPPER_FORCE_LIMIT = 70.0
GRIPPER_CTRL_OPEN = 255.0
PINCH_POS_IN_HAND = (0.0, 0.0, 0.10)
PINCH_POS_IN_LINK7 = (0.0, 0.0, 0.207)
PINCH_QUAT_IN_LINK7 = (0.9238795, 0.0, 0.0, -0.3826834)


class FrankaManipSpaceEnv(ManipSpaceEnv):
    """Replace only OGBench's UR5e/Robotiq robot layer with Panda."""

    def __init__(self, *args, arm_base_pos=ARM_BASE_POS, **kwargs):
        super().__init__(*args, **kwargs)
        self._arm_base_pos = np.asarray(arm_base_pos, dtype=np.float64)
        self._home_qpos = HOME_QPOS.copy()

        ik_mjcf = mjcf.from_path(
            PANDA_NOHAND_XML.as_posix(), escape_separators=True
        )
        ik_mjcf.find("body", "link0").pos = self._arm_base_pos
        ik_mjcf.find("body", "link7").add(
            "site",
            name="pinch",
            pos=PINCH_POS_IN_LINK7,
            quat=PINCH_QUAT_IN_LINK7,
        )
        ik_model = mujoco.MjModel.from_xml_string(
            mjcf_utils.to_string(ik_mjcf), mjcf_utils.get_assets(ik_mjcf)
        )
        self._ik = controllers.DiffIKController(model=ik_model, sites=["pinch"])

    def build_mjcf_model(self):
        arena_mjcf = mjcf.from_path(
            (self._desc_dir / "floor_wall.xml").as_posix()
        )
        arena_mjcf.model = "franka_arena"
        arena_mjcf.statistic.center = (0.3, 0.0, 0.15)
        arena_mjcf.statistic.extent = 0.7
        getattr(arena_mjcf.visual, "global").elevation = -20
        getattr(arena_mjcf.visual, "global").azimuth = 180
        arena_mjcf.statistic.meansize = 0.04
        arena_mjcf.visual.map.znear = 0.1
        arena_mjcf.visual.map.zfar = 10.0

        franka_mjcf = mjcf.from_path(PANDA_XML.as_posix(), escape_separators=True)
        franka_mjcf.model = "franka"
        franka_mjcf.find("body", "link0").pos = self._arm_base_pos
        franka_mjcf.find("body", "hand").add(
            "site", name="pinch", pos=PINCH_POS_IN_HAND
        )
        for light in franka_mjcf.find_all("light"):
            light.remove()
        for key in franka_mjcf.find_all("key"):
            key.remove()

        self._arm_jnts = [
            mjcf_utils.safe_find(franka_mjcf, "joint", f"joint{index}")
            for index in range(1, 8)
        ]
        self._arm_acts = [
            mjcf_utils.safe_find(franka_mjcf, "actuator", f"actuator{index}")
            for index in range(1, 8)
        ]
        self._gripper_jnts = [
            mjcf_utils.safe_find(franka_mjcf, "joint", name)
            for name in ("finger_joint1", "finger_joint2")
        ]
        self._gripper_acts = [
            mjcf_utils.safe_find(franka_mjcf, "actuator", "actuator8")
        ]

        if self._ob_type == "pixels" and self._pixel_transparent_arm:
            for name in (
                "white",
                "off_white",
                "black",
                "green",
                "light_blue",
                "dark_grey",
            ):
                material = franka_mjcf.find("material", name)
                if material is not None:
                    material.rgba[3] = 0.1

        mjcf_utils.attach(arena_mjcf, franka_mjcf)
        self.add_objects(arena_mjcf)

        mjcf_utils.add_bounding_box_site(
            arena_mjcf.worldbody,
            lower=np.asarray((*self._target_sampling_bounds[0], 0.02)),
            upper=np.asarray((*self._target_sampling_bounds[1], 0.02)),
            rgba=(0.6, 0.3, 0.3, 0.2),
            group=4,
            name="object_bounds",
        )
        mjcf_utils.add_bounding_box_site(
            arena_mjcf.worldbody,
            lower=np.asarray(self._arm_sampling_bounds[0]),
            upper=np.asarray(self._arm_sampling_bounds[1]),
            rgba=(0.3, 0.6, 0.3, 0.2),
            group=4,
            name="arm_bounds",
        )
        return arena_mjcf

    def post_compilation(self):
        arm_joint_names = [joint.full_identifier for joint in self._arm_jnts]
        self._arm_joint_ids = np.asarray(
            [self._model.joint(name).id for name in arm_joint_names]
        )
        arm_actuator_names = [act.full_identifier for act in self._arm_acts]
        self._arm_actuator_ids = np.asarray(
            [self._model.actuator(name).id for name in arm_actuator_names]
        )
        gripper_actuator_names = [
            act.full_identifier for act in self._gripper_acts
        ]
        self._gripper_actuator_ids = np.asarray(
            [self._model.actuator(name).id for name in gripper_actuator_names]
        )
        gripper_joint_names = [joint.full_identifier for joint in self._gripper_jnts]
        self._gripper_qpos_ids = np.asarray(
            [self._model.joint(name).qposadr[0] for name in gripper_joint_names]
        )
        self._gripper_dof_ids = np.asarray(
            [self._model.joint(name).dofadr[0] for name in gripper_joint_names]
        )
        self._gripper_opening_joint_id = self._gripper_qpos_ids[0]
        self._model.actuator_forcerange[self._gripper_actuator_ids] = (
            -GRIPPER_FORCE_LIMIT,
            GRIPPER_FORCE_LIMIT,
        )

        for geom_id in range(self._model.ngeom):
            body_name = self._model.body(
                int(self._model.geom_bodyid[geom_id])
            ).name or ""
            is_finger_pad = (
                body_name in {"franka/left_finger", "franka/right_finger"}
                and self._model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
            )
            if is_finger_pad:
                self._model.geom_friction[geom_id, 0] = FINGER_PAD_SLIDE_FRICTION
                self._model.geom_priority[geom_id] = 1
                self._model.geom_solref[geom_id] = (0.004, 1.0)
                self._model.geom_solimp[geom_id] = (
                    0.95,
                    0.99,
                    0.001,
                    0.5,
                    2.0,
                )

        self._pinch_site_id = self._model.site("franka/pinch").id
        self._T_pa = lie.SE3.identity()
        self.post_compilation_objects()

    def _pinch_yaw(self) -> float:
        return lie.SO3.from_matrix(
            self._data.site_xmat[self._pinch_site_id].copy().reshape(3, 3)
        ).compute_yaw_radians()

    def initialize_franka_gripper(self):
        # Start demonstrations open.  The Panda mechanical zero has no gap;
        # keep the OGBench closure range [0, 1] over the useful 4 cm -> 2 cm
        # span so closure=1 does not eject a 4 cm cube.
        self._data.qpos[self._arm_joint_ids] = np.clip(
            self._data.qpos[self._arm_joint_ids],
            self._model.jnt_range[self._arm_joint_ids, 0],
            self._model.jnt_range[self._arm_joint_ids, 1],
        )
        self._data.qpos[self._gripper_qpos_ids] = FINGER_MAX_OPENING
        self._data.qvel[self._gripper_dof_ids] = 0.0
        mujoco.mj_forward(self._model, self._data)

    def set_control(self, action):
        action = self.unnormalize_action(action)
        a_pos, a_yaw, a_gripper = action[:3], action[3], action[4]

        effector_pos = self._data.site_xpos[self._pinch_site_id].copy()
        effector_yaw = self._pinch_yaw()
        physical_opening = np.clip(
            np.mean(self._data.qpos[self._gripper_qpos_ids])
            / FINGER_MAX_OPENING,
            0.0,
            1.0,
        )

        target_translation = effector_pos + a_pos
        np.clip(
            target_translation,
            *self._workspace_bounds,
            out=target_translation,
        )
        target_yaw = np.clip(effector_yaw + a_yaw, -np.pi, np.pi)
        target_orientation = (
            lie.SO3.from_z_radians(target_yaw) @ self._effector_down_rotation
        )
        # OGBench's fifth action is closure (positive closes); Panda's actuator
        # target is physical opening (zero closes, 255 opens).
        current_closure = np.clip(
            (FINGER_MAX_OPENING - physical_opening * FINGER_MAX_OPENING)
            / (FINGER_MAX_OPENING - FINGER_GRASP_OPENING),
            0.0,
            1.0,
        )
        target_closure = np.clip(current_closure + a_gripper, 0.0, 1.0)
        target_opening = (
            FINGER_MAX_OPENING - target_closure *
            (FINGER_MAX_OPENING - FINGER_GRASP_OPENING)
        ) / FINGER_MAX_OPENING

        self._target_effector_pose = lie.SE3.from_rotation_and_translation(
            rotation=target_orientation,
            translation=target_translation,
        )
        qpos_target = self._ik.solve(
            pos=target_translation,
            quat=target_orientation.wxyz,
            curr_qpos=self._data.qpos[self._arm_joint_ids],
        )
        qpos_target = np.clip(
            qpos_target,
            self._model.jnt_range[self._arm_joint_ids, 0],
            self._model.jnt_range[self._arm_joint_ids, 1],
        )
        self._data.ctrl[self._arm_actuator_ids] = qpos_target
        self._data.ctrl[self._gripper_actuator_ids] = (
            GRIPPER_CTRL_OPEN * target_opening
        )

    def compute_ob_info(self):
        ob_info = {
            "proprio/joint_pos": self._data.qpos[
                self._arm_joint_ids
            ].copy(),
            "proprio/joint_vel": self._data.qvel[
                self._arm_joint_ids
            ].copy(),
            "proprio/effector_pos": self._data.site_xpos[
                self._pinch_site_id
            ].copy(),
            "proprio/effector_yaw": np.array([self._pinch_yaw()]),
            "proprio/gripper_opening": np.array(
                [
                    np.clip(
                        (
                            FINGER_MAX_OPENING
                            - np.mean(self._data.qpos[self._gripper_qpos_ids])
                        )
                        / (FINGER_MAX_OPENING - FINGER_GRASP_OPENING),
                        0.0,
                        1.0,
                    )
                ]
            ),
            "proprio/gripper_vel": np.array(
                [
                    -np.mean(self._data.qvel[self._gripper_dof_ids])
                    / (FINGER_MAX_OPENING - FINGER_GRASP_OPENING)
                ]
            ),
            "proprio/gripper_contact": np.array(
                [
                    np.clip(
                        np.linalg.norm(
                            self._data.body("franka/right_finger").cfrc_ext
                        )
                        / 50.0,
                        0.0,
                        1.0,
                    )
                ]
            ),
        }
        self.add_object_info(ob_info)
        ob_info.update(
            {
                "prev_qpos": self._prev_qpos.copy(),
                "prev_qvel": self._prev_qvel.copy(),
                "qpos": self._data.qpos.copy(),
                "qvel": self._data.qvel.copy(),
                "control": self._data.ctrl.copy(),
                "time": np.array([self._data.time]),
            }
        )
        return ob_info

    def modify_cube_mjcf_model(self, mjcf_model):
        """CubeEnv domain randomization with explicit Panda material names."""
        grid_texture = mjcf_model.find("texture", "grid")
        floor_value = self.variation_space["floor"]["color"].value
        texture_changed = grid_texture.rgb1 is None or not np.allclose(
            grid_texture.rgb1, floor_value[0]
        )
        texture_changed = texture_changed or grid_texture.rgb2 is None or not np.allclose(
            grid_texture.rgb2, floor_value[1]
        )
        grid_texture.rgb1 = floor_value[0]
        grid_texture.rgb2 = floor_value[1]

        agent_color = self.variation_space["agent"]["color"].value
        agent_color_changed = False
        for name in (
            "franka/white",
            "franka/off_white",
            "franka/black",
            "franka/green",
            "franka/light_blue",
            "franka/dark_grey",
        ):
            material = mjcf_model.find("material", name)
            if material is not None:
                agent_color_changed |= not np.allclose(
                    material.rgba[:3], agent_color
                )
                material.rgba[:3] = agent_color

        size_changed = False
        for index in range(self._num_cubes):
            desired_size = self.variation_space["cube"]["size"].value[
                index
            ] * np.ones(3, dtype=np.float32)
            for body_name in (f"object_{index}", f"object_target_{index}"):
                body = mjcf_model.find("body", body_name)
                if body is None:
                    continue
                for geom in body.find_all("geom"):
                    if geom.size is None or not np.allclose(
                        geom.size, desired_size
                    ):
                        size_changed = True
                    geom.size = desired_size

        camera_angle_changed = False
        cameras_to_vary = (
            ["front_pixels", "side_pixels"]
            if self._multiview
            else ["front_pixels"]
        )
        for index, camera_name in enumerate(cameras_to_vary):
            camera = mjcf_model.find("camera", camera_name)
            camera.xyaxes = perturb_camera_angle(
                self.cameras[camera_name]["xyaxes"],
                self.variation_space["camera"]["angle_delta"].value[index],
            )
            camera_angle_changed |= not np.allclose(
                camera.xyaxes, self.cameras[camera_name]["xyaxes"]
            )

        light = mjcf_model.find("light", "global")
        desired_diffuse = self.variation_space["light"]["intensity"].value[
            0
        ] * np.ones(3, dtype=np.float32)
        light_changed = light.diffuse is None or not np.allclose(
            light.diffuse, desired_diffuse
        )
        light.diffuse = desired_diffuse

        if any(
            (
                size_changed,
                light_changed,
                texture_changed,
                camera_angle_changed,
                agent_color_changed,
            )
        ):
            self.mark_dirty()
        return mjcf_model


class FrankaCubeEnv(CubeEnv, FrankaManipSpaceEnv):
    """``swm/OGBCube-v0`` using the Franka Emika Panda."""

    def initialize_arm(self):
        super().initialize_arm()
        self.initialize_franka_gripper()

    def modify_mjcf_model(self, mjcf_model):
        return self.modify_cube_mjcf_model(mjcf_model)


class FrankaCubePCDEnv(CubePCDEnv, FrankaManipSpaceEnv):
    """The Franka Cube environment with the existing PCD camera layout."""

    def initialize_arm(self):
        super().initialize_arm()
        self.initialize_franka_gripper()

    def modify_mjcf_model(self, mjcf_model):
        return self.modify_cube_mjcf_model(mjcf_model)


__all__ = [
    "FrankaManipSpaceEnv",
    "FrankaCubeEnv",
    "FrankaCubePCDEnv",
    "PANDA_XML",
    "PANDA_NOHAND_XML",
    "ARM_BASE_POS",
    "CROP_BOX",
]
