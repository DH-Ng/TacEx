# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
import isaaclab.utils.math as lab_math
import isaaclab.utils.math as math_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBase, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    combine_frame_transforms,
    euler_xyz_from_quat,
    sample_uniform,
    subtract_frame_transforms,
    wrap_to_pi,
)

#  from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
# from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg


from isaaclab.markers import POSITION_GOAL_MARKER_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class CustomEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: DirectRLEnvCfg, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class BallRollingEnvCfg(DirectRLEnvCfg):

    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.5, 1, 0.5)
    viewer.env_idx = 50
    viewer.origin_type = "robot"

    debug_vis = True

    ui_window_class_type = CustomEnvWindow

    decimation = 1
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,  # 1 / 120, #0.001
        render_interval=decimation,
        # device="cpu",
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
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=2.0, replicate_physics=True)

    # use robot with stiff PD control for better IK tracking
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/isaaclab/data_storage/TacEx/assets/robots_with_gelsight_mini/assets/standard_isaac_models/franka_single_gsmini_rigid.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,  # True instead of False
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # joint_pos={
            #     "panda_joint1": 0.0,
            #     "panda_joint2": -0.569,
            #     "panda_joint3": 0.0,
            #     "panda_joint4": -2.810,
            #     "panda_joint5": 0.0,
            #     "panda_joint6": 3.037,
            #     "panda_joint7": 0.741,
            # },
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": -2.46,
                "panda_joint5": 0.0,
                "panda_joint6": 2.5,
                "panda_joint7": 0.741,
            },
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=400,  # instead of 80.0,
                damping=80,  # instead of 4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=400,  # instead of 80.0,
                damping=80,  # instead 4.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=True, ik_method="dls")

    # rigid body ball
    ball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/rigid_ball",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/isaaclab/data_storage/TacEx/examples/assets/ball_wood.usd",  # TODO replace with relative path
            # scale=(2, 1, 0.6),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.03)),
    )

    # sensors
    # gsmini = GsMiniSensorCfg(prim_path="/World/envs/env_0/Robot/gelsight_mini_case")

    # MARK: reward configuration
    reaching_penalty = {"weight": -0.2}
    reaching_reward_tanh = {"std": 0.2, "weight": 0.4}
    at_obj_reward = {"weight": 15, "minimal_distance": 0.005}
    tracking_reward = {"w": 0.01, "v": 1, "alpha": 1e-5, "minimal_distance": 0.005}
    success_reward = {
        "weight": 10,
        "threshold": 0.0025,
    }  # we count it as a sucess when dist obj <-> goal is less than the threshold
    height_penalty = {
        "weight": -1,
        "min_height": 0.0085,
    }  # ball has diameter of 1cm, plate 0.5 cm -> 0.005m + 0.0025m = 0.0075m is above the ball
    orient_penalty = {"weight": -0.5}

    # reward scales
    action_rate_penalty_scale = -1e-4
    joint_vel_penalty_scale = -1e-4
    curriculum_steps = 2500

    # env
    episode_length_s = 8.3333 * 2  # 500 timesteps
    action_space = 3  # (dx, dy, dz)
    observation_space = 12  # 3 for rel ee pos, 2 for orient (roll, pitch, yaw), 2 for goal (x,y) and 2 for obj (x,y,z) pos, 5 for actions
    state_space = 0

    ball_radius = 0.01
    x_bounds = (0.2, 0.75)
    y_bounds = (-0.4, 0.4)
    too_far_away_threshold = 0.5


class BallRollingNoRotEnv(DirectRLEnv):
    """RL env in which the robot has to push/roll a ball to a goal position.

    This base env uses (absolute) joint positions.
    Absolute joint pos and vel are used for the observations.
    """

    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: BallRollingEnvCfg

    def __init__(self, cfg: BallRollingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        # for computing the tracking reward
        self.init_goal_distances = torch.zeros(self.num_envs, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # make height of goal pos fixed
        self._desired_pos_w[:, 2] = 0.00125

        #### Stuff for IK ##################################################
        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        # Obtain the frame index of the end-effector
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        # save only the first body index
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

        # For a fixed base robot, the frame index is one less than the body index.
        # This is because the root body is not included in the returned Jacobians.
        self._jacobi_body_idx = self._body_idx - 1
        # self._jacobi_joint_ids = self._joint_ids # we take every joint

        # ee offset w.r.t panda hand -> based on the asset
        self._offset_pos = torch.tensor([0.0, 0.0, 0.131], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        ####################################################################

        # create auxiliary variables for computing applied action, observations and rewards
        self.processed_actions = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

        # ik controller for moving ee to ball position after reset -> not relative mode!
        reset_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self._reset_ik_controller = DifferentialIKController(reset_ik_cfg, num_envs=self.num_envs, device=self.device)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.object = RigidObject(self.cfg.ball)
        self.scene.rigid_objects["object"] = self.object

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        ee_frame_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.131),  # 0.1034
                    ),
                ),
            ],
        )

        # sensors
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        # self.gsmini = GsMiniSensor(self.cfg.gsmini)
        # self.scene.sensors["gsmini"] = self.gsmini

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
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )

        # plate
        plate = RigidObjectCfg(
            prim_path="/World/envs/env_.*/plate",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0]),
            spawn=sim_utils.UsdFileCfg(
                usd_path="/workspace/isaaclab/data_storage/TacEx/examples/assets/plate.usd",
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    kinematic_enabled=True,
                ),
            ),
        )
        plate.spawn.func(
            plate.prim_path, plate.spawn, translation=plate.init_state.pos, orientation=ground.init_state.rot
        )

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # MARK: pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions[:] = actions.clamp(-1, 1)
        #! preprocess the action and turn it into IK action
        self.processed_actions[:, :] = self.actions

        # obtain ee positions and orientation w.r.t root (=base) frame
        self.ee_pos_curr_b, self.ee_quat_curr_b = self._compute_frame_pose()
        # set command into controller
        self._ik_controller.set_command(self.processed_actions, self.ee_pos_curr_b, self.ee_quat_curr_b)

    def _apply_action(self):
        # obtain quantities from simulation
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]
        # compute the delta in joint-space
        if self.ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        self._robot.set_joint_position_target(joint_pos_des)

        # pass

    # post-physics step calls

    # MARK: dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:  # which environment is done
        obj_pos = self.object.data.root_link_pos_w - self.scene.env_origins
        out_of_bounds_x = (obj_pos[:, 0] < 0.2) | (obj_pos[:, 0] > 0.75)
        out_of_bounds_y = (obj_pos[:, 1] < -0.4) | (obj_pos[:, 1] > 0.4)

        obj_goal_distance = torch.norm(
            self._desired_pos_w[:, :2] - self.scene.env_origins[:, :2] - obj_pos[:, :2], dim=1
        )
        obj_too_far_away = obj_goal_distance > 1

        ee_frame_pos = (
            self._ee_frame.data.target_pos_w[..., 0, :] - self.scene.env_origins
        )  # end-effector positions in world frame: (num_envs, 3)
        ee_too_far_away = torch.norm(obj_pos - ee_frame_pos, dim=1) > self.cfg.too_far_away_threshold

        reset_cond = out_of_bounds_x | out_of_bounds_y | obj_too_far_away | ee_too_far_away

        time_out = self.episode_length_buf >= self.max_episode_length - 1  # episode length limit

        return reset_cond, time_out

    # MARK: rewards
    def _get_rewards(self) -> torch.Tensor:
        # - Reward the agent for reaching the object using tanh-kernel.
        obj_pos = self.object.data.root_link_state_w[:, :3]
        # for compensating that obj_pos is based on the center of the ball
        obj_pos[:, 2] += 0.005  # ball has diameter of 1cm -> r=0.005m, plate height (above ground)=0.0025
        ee_frame_pos = self._ee_frame.data.target_pos_w[
            ..., 0, :
        ]  # end-effector positions in world frame: (num_envs, 3)

        # Distance of the end-effector to the object: (num_envs,)
        object_ee_distance = torch.norm(obj_pos - ee_frame_pos, dim=1)
        reaching_penalty = self.cfg.reaching_penalty["weight"] * torch.square(object_ee_distance)
        # use tanh-kernel
        object_ee_distance_tanh = 1 - torch.tanh(object_ee_distance / self.cfg.reaching_reward_tanh["std"])
        # for giving agent incentive to touch the obj
        at_obj_reward = (object_ee_distance < self.cfg.at_obj_reward["minimal_distance"]) * self.cfg.at_obj_reward[
            "weight"
        ]

        # distance between obj and goal: (num_envs,)
        obj_goal_distance = torch.norm(self._desired_pos_w[:, :2] - self.object.data.root_link_state_w[:, :2], dim=1)

        d = torch.square(obj_goal_distance)  # make it for fine pos change
        tracking_goal = -(
            self.cfg.tracking_reward["w"] * d
            + self.cfg.tracking_reward["v"] * torch.log(d + self.cfg.tracking_reward["alpha"])
            - 1
        )
        # only apply when ee is at object (with this our tracking goal always needs to be positive, otherwise reaching part wont work anymore)
        tracking_goal = (object_ee_distance < self.cfg.tracking_reward["minimal_distance"]) * tracking_goal

        success_reward = (obj_goal_distance < self.cfg.success_reward["threshold"]) * self.cfg.success_reward["weight"]

        # height penalty -> distance to the ground
        height_penalty = (ee_frame_pos[:, 2] < self.cfg.height_penalty["min_height"]) * self.cfg.height_penalty[
            "weight"
        ]

        # # penalize when ee orient is to big
        # ee_frame_orient = euler_xyz_from_quat(self._ee_frame.data.target_quat_source[..., 0, :])
        # x = wrap_to_pi(ee_frame_orient[0]-math.pi) # panda hand asset has rotation from (180,0,-45) -> we substract 180 for defining the rotation limits
        # y = wrap_to_pi(ee_frame_orient[1])
        # orient_penalty = (
        #     (torch.abs(x) > math.pi/8)
        #     | (torch.abs(y) > math.pi/8)
        # ) * self.cfg.orient_penalty["weight"]

        # Penalize the rate of change of the actions using L2 squared kernel.
        action_rate_penalty = torch.sum(torch.square(self.actions - self.prev_actions), dim=1)

        # Penalize joint velocities on the articulation using L2 squared kernel.
        joint_vel_penalty = torch.sum(torch.square(self._robot.data.joint_vel[:, :]), dim=1)

        rewards = (
            +reaching_penalty
            + self.cfg.reaching_reward_tanh["weight"] * object_ee_distance_tanh
            + at_obj_reward
            + tracking_goal
            + success_reward
            # + orient_penalty
            + height_penalty
            + self.cfg.action_rate_penalty_scale * action_rate_penalty
            + self.cfg.joint_vel_penalty_scale * joint_vel_penalty
        )

        self.extras["log"] = {
            "reaching_penalty": reaching_penalty.float().mean(),
            "reaching_reward_tanh": (self.cfg.reaching_reward_tanh["weight"] * object_ee_distance_tanh).mean(),
            "at_obj_reward": at_obj_reward.float().mean(),
            "success": success_reward.float().mean(),
            "tracking_goal_reward": tracking_goal.float().mean(),
            # penalties for nice looking behavior
            # "orientation_penalty": orient_penalty.float().mean(),
            "height_penalty": height_penalty.float().mean(),
            "action_rate_penalty": (self.cfg.action_rate_penalty_scale * action_rate_penalty).mean(),
            "joint_vel_penalty": (self.cfg.joint_vel_penalty_scale * joint_vel_penalty).mean(),
            # task metrics
            "Metric/num_ee_at_obj": torch.sum(object_ee_distance < self.cfg.tracking_reward["minimal_distance"]),
            "Metric/ee_obj_error": object_ee_distance.mean(),
            "Metric/obj_goal_error": obj_goal_distance.mean(),
        }
        return rewards

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        obj_pos = self.object.data.default_root_state[env_ids]
        obj_pos[:, :2] += sample_uniform(-0.1, 0.1, (len(env_ids), 2), self.device)
        obj_pos[:, :3] += self.scene.env_origins[env_ids]
        self.object.write_root_state_to_sim(obj_pos, env_ids=env_ids)

        # reset robot state
        joint_pos = (
            self._robot.data.default_joint_pos[env_ids]
            # + sample_uniform(
            #     -0.125,
            #     0.125,
            #     (len(env_ids), self._robot.num_joints),
            #     self.device,
            #     )
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # # set joint pos, so that ee is at the ball
        # des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device) + torch.tensor([0,0,0,0,1,0,0],device=self.device)
        # des_ee_pose[env_ids, :] = obj_pos[:, :7].clone()
        # des_ee_pose[env_ids, :3] -= self.scene.env_origins[env_ids]
        # des_ee_pose[env_ids, 2] += 2*self.cfg.ball_radius

        # # set command into controller
        # self._reset_ik_controller.set_command(des_ee_pose)

        # set commands: random position
        self._desired_pos_w[env_ids, :2] = (
            self.object.data.default_root_state[env_ids][:, :2] + self.scene.env_origins[env_ids][:, :2]
        )
        self._desired_pos_w[env_ids, :2] += sample_uniform(-0.1, 0.1, (len(env_ids), 2), self.device)
        # self._desired_pos_w[env_ids, :2] += self.scene.env_origins[env_ids, :2]

        # reset actions
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._ik_controller.reset(env_ids)

    # MARK: observations
    def _get_observations(self) -> dict:
        """The position of the object in the robot's root frame."""

        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        # ee_frame_orient = euler_xyz_from_quat(ee_quat_curr_b)
        # x = wrap_to_pi(ee_frame_orient[0]).unsqueeze(1) # add dimension for concatenating with other observations
        # y = wrap_to_pi(ee_frame_orient[1]).unsqueeze(1)

        # obj position in the robots root frame
        object_pos_w = self.object.data.root_link_pos_w[:, :3]
        object_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], object_pos_w
        )

        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._desired_pos_w
        )

        obs = torch.cat(
            (
                ee_pos_curr_b,
                # x,
                # y,
                object_pos_b[:, :2],
                desired_pos_b[:, :2],  # we only care about x and y
                self.actions,
            ),
            dim=-1,
        )

        # self.gsmini.update_gui_windows()
        return {"policy": obs}

    ####
    ## Helper Functions
    ####

    ################################# For IK
    #  From task_space_actions.py
    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot = self._robot.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        # account for the offset
        # if self.cfg.body_offset is not None:
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
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
        # if self.cfg.body_offset is not None:
        # Modify the jacobian to account for the offset
        # -- translational part
        # v_link = v_ee + w_ee x r_link_ee = v_J_ee * q + w_J_ee * q x r_link_ee
        #        = (v_J_ee + w_J_ee x r_link_ee ) * q
        #        = (v_J_ee - r_link_ee_[x] @ w_J_ee) * q
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        # -- rotational part
        # w_link = R_link_ee @ w_ee
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])

        return jacobian

    ###################################

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.01, 0.01, 0.01)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)

            # if not hasattr(self, "ik_des_pose_visualizer"):
            #     marker_cfg = FRAME_MARKER_CFG.copy()
            #     marker_cfg.markers["frame"].scale = (0.025, 0.025, 0.025)
            #     marker_cfg.prim_path = "/Visuals/Command/ik_des_pose"
            #     self.ik_des_pose_visualizer = VisualizationMarkers(marker_cfg)
            # # set their visibility to true
            # self.ik_des_pose_visualizer.set_visibility(True)

        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            # if hasattr(self, "ik_des_pose_visualizer"):
            #     self.ik_des_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

        # ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        # self.ik_des_pose_visualizer.visualize(
        #     translations=ee_pos_curr + self.scene.env_origins,#self._ik_controller.ee_pos_des[:, :3] - self.scene.env_origins,
        #     orientations=ee_quat_curr
        #     )
