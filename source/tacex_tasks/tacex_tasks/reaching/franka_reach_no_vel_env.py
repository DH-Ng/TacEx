# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)

import os
import time

import math
import torch
import numpy as np

import omni
from isaacsim.core.api.objects import VisualCuboid

from isaacsim.core.prims import XFormPrim

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg, RenderCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    quat_error_magnitude,
    quat_mul,
    sample_uniform,
    wrap_to_pi,
    combine_frame_transforms,
    quat_unique,
)
from isaaclab.utils.noise import (
    GaussianNoiseCfg,
    NoiseModelCfg,
    UniformNoiseCfg,
    gaussian_noise,
)

# from tacex import GelSightSensor
# from tacex.simulation_approaches.gpu_taxim import TaximSimulatorCfg

# # from tactile_sim import GsMiniSensorCfg, GsMiniSensor
# from tacex_assets import TACEX_ASSETS_DATA_DIR
# from tacex_assets.sensors.gelsight_mini.generic_gsmini_cfg import GeneralGelSightMiniCfg

# from tacex_tasks.utils import DirectLiveVisualizer
import isaaclab.envs.mdp as mdp

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
from tacex_assets import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG

@configclass
class ReachEnvCfg(DirectRLEnvCfg):
    decimation = 1

    debug_vis = True

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=1, replicate_physics=False
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 30,
        render_interval=decimation,
        physx=PhysxCfg(
            enable_ccd=True,  # needed for more stable ball_rolling
            # bounce_threshold_velocity=10000,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=5.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
        use_fabric=True,
        render=RenderCfg(enable_translucency=True),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=10, env_spacing=1.5, replicate_physics=False
    )

    # use robot with stiff PD control for better IK tracking
    isaac_robot_sim: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/RobotSim",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
        ),
    )

    ik_controller_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=True, ik_method="dls"
    )

    ee_pos_offset = [0.0, 0.0, 0.0]  # (0.0, 0.0, 0.1034)  #
    ee_rot_offset = (1.0, 0.0, 0.0, 0.0)  # [0.0, 1.0, 0.0, 0.0]

    # MARK: reward cfg
    reward_cfg = {
        "ee_pos_tracking": {"weight": -0.2},
        "ee_pos_fine_tracking": {"weight": 0.1, "std": 0.1},
        "ee_orient_tracking": {"weight": -0.1},
        "action_penalty": {"weight": -0.0001},
    }

    # env
    episode_length_s = 8.3333 * 2
    action_space = (
        6  # we use relative task_space actions: (dx, dy, dz, droll, dpitch, dyaw)
    )
    # observation_space = {
    #     "joint_pos_rel": 9,
    #     "pose_command": 7,
    #     "prev_actions": 6,
    # }
    observation_space = 22
    state_space = 0

    action_scale = 0.5
    joint_vel_scale = 0.1
    # x_bounds = (0.2, 0.8)
    # y_bounds = (-0.4, 0.4)
    x_bounds = (0.0, 1.0)
    y_bounds = (-1.5, 1.5)

    min_height_threshold = 0.01

    default_goal_pos = [0.45, 0.0, 0.3]
    default_goal_orient = [1.0, 0.0, 0.0, 0.0]  # x=180 deg, y=0, z=0
    ranges = mdp.UniformPoseCommandCfg.Ranges(
        pos_x=(0.35, 0.65),
        pos_y=(-0.2, 0.2),
        pos_z=(0.15, 0.5),
        roll=(0.0, 0.0),
        pitch=(math.pi, math.pi),
        yaw=(-3.14, 3.14),
    )


class ReachEnv(DirectRLEnv):
    """RL env in which the robot has to reach a goal position."""

    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: ReachEnvCfg

    def __init__(self, cfg: ReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = 1 / 30.0  # 1 / 120.0

        self._robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0]
        self._robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1]
        self._robot_dof_speed_scales = (
            torch.ones_like(self._robot_dof_lower_limits) * cfg.joint_vel_scale
        )

        # --- For IK actions ---

        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )

        body_ids, _ = self._robot.find_bodies("panda_hand")
        # save only the first body index
        self._body_idx = body_ids[0]

        # For a fixed base robot, the frame index is one less than the body index.
        # This is because the root body is not included in the returned Jacobians.
        self._jacobi_body_idx = self._body_idx - 1

        # ee offset w.r.t TCP -> TCP is defined so that z-axis shows down. In our case here we want z to show upwards
        self._ee_pos_offset = torch.tensor(
            self.cfg.ee_pos_offset, device=self.device
        ).repeat(self.num_envs, 1)
        self._ee_rot_offset = torch.tensor(
            self.cfg.ee_rot_offset, device=self.device
        ).repeat(self.num_envs, 1)
        # ---

        # create auxiliary variables for computing applied action, observations and rewards
        self.processed_actions = torch.zeros(
            (self.num_envs, self._ik_controller.action_dim), device=self.device
        )
        self.prev_actions = torch.zeros_like(self.actions)

        self._goal_pos_b = torch.zeros((self.num_envs, 3), device=self.device)
        self._goal_pos_b[:] = torch.tensor(
            self.cfg.default_goal_pos, device=self.device
        )

        self._goal_orient = torch.zeros((self.num_envs, 4), device=self.device)
        self._goal_orient[:] = torch.tensor(
            self.cfg.default_goal_orient, device=self.device
        )
        self._time_out = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        self.object_ee_distance = torch.zeros((self.num_envs, 1), device=self.device)
        self.ee_goal_distance = torch.zeros((self.num_envs, 1), device=self.device)
        self.obj_goal_distance = torch.zeros((self.num_envs, 1), device=self.device)

        # Used to move the real franka robot
        self.joint_pos_des = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device
        )

        # self.goal_prim_view = None

        # create buffers for commands
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

    def _setup_scene(self):
        """Visualizes the current robot state in Isaac Sim."""
        self._robot = Articulation(self.cfg.isaac_robot_sim)
        self.scene.articulations["robot_sim"] = self._robot

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)

        # Ground-plane
        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
            spawn=sim_utils.GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        ground.spawn.func(
            ground.prim_path,
            ground.spawn,
            translation=ground.init_state.pos,
            orientation=ground.init_state.rot,
        )

        # add light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        ee_frame_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/RobotSim/panda_link0",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/RobotSim/panda_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=self.cfg.ee_pos_offset, rot=self.cfg.ee_rot_offset
                    ),
                ),
            ],
        )
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        current_command_visualizer_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/goal_pose"
        )
        current_command_visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        self.goal_pose_visualizer = VisualizationMarkers(current_command_visualizer_cfg)

    # MARK: pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = actions
        # preprocess actions and use them for IK
        self.processed_actions[:, :] = (actions * self.cfg.action_scale).clamp(
            -1.0, 1.0
        )

        # obtain ee positions and orientation w.r.t root (=base) frame
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        # set command into controller
        self._ik_controller.set_command(
            self.processed_actions, ee_pos_curr_b, ee_quat_curr_b
        )

    def _apply_action(self):
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:]

        # compute the delta in joint-space
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(
                ee_pos_curr, ee_quat_curr, jacobian, joint_pos
            )
        else:
            joint_pos_des = joint_pos.clone()

        self.joint_pos_des = joint_pos_des
        self._robot.set_joint_position_target(self.joint_pos_des)

    # post-physics step calls

    # MARK: dones
    def _get_dones(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()

        # out_of_bounds_x = (ee_pos_curr_b[:, 0] < self.cfg.x_bounds[0]) | (
        #     ee_pos_curr_b[:, 0] > self.cfg.x_bounds[1]
        # )
        # out_of_bounds_y = (ee_pos_curr_b[:, 1] < self.cfg.y_bounds[0]) | (
        #     ee_pos_curr_b[:, 1] > self.cfg.y_bounds[1]
        # )

        # min_height = ee_pos_curr_b[:, 2] < self.cfg.min_height_threshold

        # reset_cond = out_of_bounds_x | out_of_bounds_y | min_height
        terminated = torch.zeros(self.num_envs)

        self._time_out = (
            self.episode_length_buf >= self.max_episode_length - 1
        )  # episode length limit

        return terminated, self._time_out

    # MARK: reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
        # self.goal_prim_view = XFormPrim(prim_paths_expr="/Goal", name="Goal", usd=True)

        # reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # reset buffers
        super()._reset_idx(env_ids)

        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(
            euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
        )
        # make sure the quaternion has real part as positive
        self.pose_command_b[env_ids, 3:] = quat_unique(quat)

        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = (
            combine_frame_transforms(
                self._robot.data.root_pos_w,
                self._robot.data.root_quat_w,
                self.pose_command_b[:, :3],
                self.pose_command_b[:, 3:],
            )
        )

        self.goal_pose_visualizer.visualize(
            translations=self.pose_command_w[:, :3],
            orientations=self.pose_command_w[:, 3:],
        )

        # self._goal_pos_b[env_ids] = self.pose_command_w[:, :3]
        # self._goal_orient[env_ids] = self.pose_command_w[:, 3:]

        # self.goal_prim_view.set_world_poses(self._goal_pos_b, self._goal_orient)

        # initial target - default position
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        self.joint_pos_des[env_ids] = joint_pos

        self.prev_actions[env_ids] = 0.0

    # MARK: rewards
    def _get_rewards(self) -> torch.Tensor:
        # Position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:

        # extract the asset (to enable type hinting)
        asset: RigidObject = self._robot

        # positions, orientations = self.goal_prim_view.get_world_poses()
        # self._goal_pos_b = positions - self.scene.env_origins
        # self._goal_orient = orientations

        # pose_command = torch.cat((self._goal_pos_b, self._goal_orient), dim=1)
        pose_command = self.pose_command_b

        # obtain the desired and current ee positions
        des_pos_b = pose_command[:, :3]
        # des_pos_w, _ = combine_frame_transforms(
        #     asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
        # )
        # curr_pos_w = asset.data.body_pos_w[:, self._body_idx]

        # apply ee offset
        ee_frame_pos_b, ee_frame_orient_b = self._compute_frame_pose()

        # Penalize tracking of the position error using L2-norm.
        # distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
        distance = torch.norm(ee_frame_pos_b - des_pos_b, dim=1)
        tracking_rew = distance

        # Reward tracking of the position using the tanh kernel.
        fine_tracking_rew = 1 - torch.tanh(
            distance / self.cfg.reward_cfg["ee_pos_fine_tracking"]["std"]
        )

        # Penalize tracking orientation error using shortest path.
        des_quat_b = pose_command[:, 3:7]
        # des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
        # curr_quat_w = asset.data.body_quat_w[:, self._body_idx]  # type: ignore
        # orient_tracking_rew = quat_error_magnitude(curr_quat_w, des_quat_w)
        orient_tracking_rew = quat_error_magnitude(ee_frame_orient_b, des_quat_b)

        # Penalize the rate of change of the actions using L2 squared kernel.
        action_rate_rew = torch.sum(
            torch.square(self.actions - self.prev_actions), dim=1
        )

        full_reward = (
            tracking_rew * self.cfg.reward_cfg["ee_pos_tracking"]["weight"]
            + fine_tracking_rew * self.cfg.reward_cfg["ee_pos_fine_tracking"]["weight"]
            + orient_tracking_rew * self.cfg.reward_cfg["ee_orient_tracking"]["weight"]
            + action_rate_rew * self.cfg.reward_cfg["action_penalty"]["weight"]
        )

        return full_reward

    # MARK: observations
    def _get_observations(self) -> dict:
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        # joint_vel = self._robot.data.joint_vel - self._robot.data.default_joint_vel

        # positions, orientations = self.goal_prim_view.get_world_poses()
        # self._goal_pos_b = positions - self.scene.env_origins
        # self._goal_orient = orientations
        # pose_command = torch.cat((self._goal_pos_b, self._goal_orient), dim=1)
        pose_command = self.pose_command_b

        obs = {
            "joint_pos_rel": joint_pos,
            # "joint_vel_rel": joint_vel,
            "pose_command": pose_command,
            "prev_actions": self.prev_actions,
        }
        obs = torch.cat([v for v in obs.values()], dim=1).float()

        return {"policy": obs}

    """
    Helper Functions for IK control (from task_space_actions.py of IsaacLab).
    """

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._robot.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, :
        ]

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
        ee_pos_w = self._robot.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_quat_w[:, self._body_idx]

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
