# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas import modify_rigid_body_properties
from isaaclab.sim.spawners.materials import spawn_rigid_body_material

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat
import isaaclab.utils.math as math_utils


from isaaclab.controllers import DifferentialIKController

from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

from . import factory_control, factory_utils
from .factory_env import FactoryEnv
from .factory_ik_joint_control_env_cfg import FactoryIKJointControlEnvCfg


class FactoryIKJointControlEnv(FactoryEnv):
    """Factory Env, but instead of using Factory Torque Control, Isaac Lab differentialbe IK controller is used.

    The generate_ctrl_signals() method does not compute joint torques.generate_ctrl_signal, but
    computes the target joint positions directily based on the crtl_target_fingertip pose with Isaac Lab IK controller.

    The rest of the Env is the same as the TacEx factory env.

    Args:
        FactoryEnv (_type_): _description_

    Returns:
        _type_: _description_
    """

    cfg: FactoryIKJointControlEnvCfg

    def __init__(
        self, cfg: FactoryIKJointControlEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # --- For IK actions ---

        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )

        body_ids, _ = self._robot.find_bodies("TCP")
        # save only the first body index
        self._ee_idx = body_ids[0]

        # For a fixed base robot, the frame index is one less than the body index.
        # This is because the root body is not included in the returned Jacobians.
        self._jacobi_ee_idx = self._ee_idx - 1

        # ee offset w.r.t TCP -> TCP is defined so that z-axis shows down. In our case here we want z to show upwards
        self._ee_pos_offset = torch.tensor(
            self.cfg.ee_pos_offset, device=self.device
        ).repeat(self.num_envs, 1)
        self._ee_rot_offset = torch.tensor(
            self.cfg.ee_rot_offset, device=self.device
        ).repeat(self.num_envs, 1)
        # ---

    def generate_ctrl_signals(
        self,
        ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat,
        ctrl_target_gripper_dof_pos,
    ):
        """Get Jacobian. Set Franka DOF position targets (fingers) or DOF torques (arm)."""
        # self.joint_torque, self.applied_wrench = factory_control.compute_dof_torque(
        #     cfg=self.cfg,
        #     dof_pos=self.joint_pos,
        #     dof_vel=self.joint_vel,
        #     fingertip_midpoint_pos=self.fingertip_midpoint_pos,
        #     fingertip_midpoint_quat=self.fingertip_midpoint_quat,
        #     fingertip_midpoint_linvel=self.fingertip_midpoint_linvel,
        #     fingertip_midpoint_angvel=self.fingertip_midpoint_angvel,
        #     jacobian=self.fingertip_midpoint_jacobian,
        #     arm_mass_matrix=self.arm_mass_matrix,
        #     ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
        #     ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
        #     task_prop_gains=self.task_prop_gains,
        #     task_deriv_gains=self.task_deriv_gains,
        #     device=self.device,
        #     dead_zone_thresholds=self.dead_zone_thresholds,
        # )

        # # set target for gripper joints to use physx's PD controller
        # self.ctrl_target_joint_pos[:, 7:9] = ctrl_target_gripper_dof_pos
        # self.joint_torque[:, 7:9] = 0.0

        # self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        # self._robot.set_joint_effort_target(self.joint_torque)

        # --- instead of factory torque control, use Isaac IK control

        target_pose = torch.cat(
            (ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat),
            dim=1,
        )

        # obtain ee positions and orientation w.r.t root (=base) frame
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        # set command into controller
        self._ik_controller.set_command(target_pose, ee_pos_curr_b, ee_quat_curr_b)

        joint_pos = self._robot.data.joint_pos[:]

        # compute desired joint positions
        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(
                ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos
            )
        else:
            joint_pos_des = joint_pos.clone()

        self.ctrl_target_joint_pos[:, 7:9] = ctrl_target_gripper_dof_pos
        self.ctrl_target_joint_pos[:, :7] = joint_pos_des[:, :7]
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

    def set_pos_inverse_kinematics(
        self,
        ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat,
        env_ids,
    ):
        """Set robot joint position using DLS IK."""
        ik_time = 0.0
        while ik_time < 0.25:
            # Compute error to target.
            pos_error, axis_angle_error = factory_control.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos[
                    env_ids
                ],
                ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat[
                    env_ids
                ],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            # delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # # Solve DLS problem.
            # delta_dof_pos = factory_control.get_delta_dof_pos(
            #     delta_pose=delta_hand_pose,
            #     ik_method="dls",
            #     jacobian=self.fingertip_midpoint_jacobian[env_ids],
            #     device=self.device,
            # )
            # self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            # self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            # ---
            target_pose = torch.cat(
                (
                    ctrl_target_fingertip_midpoint_pos,
                    ctrl_target_fingertip_midpoint_quat,
                ),
                dim=1,
            )

            # obtain ee positions and orientation w.r.t root (=base) frame
            ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
            # set command into controller
            self._ik_controller.set_command(target_pose, ee_pos_curr_b, ee_quat_curr_b)

            joint_pos = self._robot.data.joint_pos[:]

            # compute desired joint positions
            if ee_pos_curr_b.norm() != 0:
                jacobian = self._compute_frame_jacobian()
                joint_pos_des = self._ik_controller.compute(
                    ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos
                )
            else:
                joint_pos_des = joint_pos.clone()
            self.joint_pos[env_ids, 0:7] = joint_pos_des[env_ids, :7]
            # ---

            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            # Update dof state.
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            # Simulate and update tensors.
            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def _set_franka_to_default_pose(self, joints, env_ids):
        """Return Franka to its default joint position."""
        gripper_width = self.cfg_task.held_asset_cfg.diameter / 2 * 1.25
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7:] = gripper_width  # MIMIC
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(
            self.ctrl_target_joint_pos[env_ids], env_ids=env_ids
        )
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        # self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    # -- Utils for doing IK
    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._robot.root_physx_view.get_jacobians()[:, self._jacobi_ee_idx, :, :]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot = self._robot.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the ee pose in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_idx]
        ee_quat_w = self._robot.data.body_quat_w[:, self._ee_idx]

        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w

        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

        # apply ee offset
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b, self._ee_pos_offset, self._ee_rot_offset
        )

        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # read the parent jacobian
        jacobian = self.jacobian_b
        # account for the offset
        if self.cfg.ee_pos_offset is not None:
            # Modify the jacobian to account for the offset
            # -- translational part
            # v_link = v_ee + w_ee x r_link_ee = v_J_ee * q + w_J_ee * q x r_link_ee
            #        = (v_J_ee + w_J_ee x r_link_ee ) * q
            #        = (v_J_ee - r_link_ee_[x] @ w_J_ee) * q
            jacobian[:, 0:3, :] += torch.bmm(
                -math_utils.skew_symmetric_matrix(self._ee_pos_offset),
                jacobian[:, 3:, :],
            )
            # -- rotational part
            # w_link = R_link_ee @ w_ee
            jacobian[:, 3:, :] = torch.bmm(
                math_utils.matrix_from_quat(self._ee_rot_offset), jacobian[:, 3:, :]
            )

        return jacobian
