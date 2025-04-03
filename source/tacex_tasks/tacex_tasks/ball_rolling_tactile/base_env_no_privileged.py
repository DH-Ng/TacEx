# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import math
import numpy as np

from isaacsim.core.api.simulation_context import SimulationContext

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBase, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, combine_frame_transforms, subtract_frame_transforms, euler_xyz_from_quat, wrap_to_pi
import isaaclab.utils.math as lab_math
#  from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
# from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg


from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers

from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.envs import ViewerCfg

from isaaclab.markers import POSITION_GOAL_MARKER_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
import isaaclab.utils.math as math_utils
from isaaclab.utils.noise import GaussianNoiseCfg, UniformNoiseCfg, NoiseModelCfg



# from tactile_sim import GsMiniSensorCfg, GsMiniSensor
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_single_adapter_rigid import FRANKA_PANDA_ARM_GSMINI_SINGLE_ADAPTER_HIGH_PD_CFG
from tacex_assets.sensors.gelsight_mini.gelsight_mini_cfg import GelSightMiniCfg

from tacex_tasks.utils import DirectLiveVisualizer

from tacex import GelSightSensor
from tacex.simulation_approaches.fots import FOTSMarkerSimulator, FOTSMarkerSimulatorCfg


from .base_env import BallRollingEnv, BallRollingEnvCfg


class CustomEnvWindow(BaseEnvWindow):
    """Window manager for the RL environment."""

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
class BallRollingEnvNoPrivilegedCfg(BallRollingEnvCfg):

    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1, -0.5, 0.1)
    viewer.lookat = (-19.4, 18.2, -1.1)

    viewer.origin_type = "env"
    viewer.env_idx = 0
    viewer.resolution = (1280, 720)   
    
    debug_vis = True

    ui_window_class_type = CustomEnvWindow

    decimation = 1
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, #0.001
        render_interval=decimation,
        #device="cpu",
        physx=PhysxCfg(
            enable_ccd=True, # needed for more stable ball_rolling
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=1, replicate_physics=True)

    # use robot with stiff PD control for better IK tracking
    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_SINGLE_ADAPTER_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.44,
                "panda_joint3": 0.0,
                "panda_joint4": -2.369,
                "panda_joint5": 0.0,
                "panda_joint6": 2.82,
                "panda_joint7": 0.741,
            },
            # joint_pos={
            #     "panda_joint1": 1.7708,
            #     "panda_joint2": -1.4164,
            #     "panda_joint3": -1.8159,
            #     "panda_joint4": -2.2501,
            #     "panda_joint5": -1.6057,
            #     "panda_joint6": 1.8573,
            #     "panda_joint7": 1.6513,
            # },
        ),
    )
# tensor([[ 1.7708, -1.4164, -1.8159, -2.2501, -1.6057,  1.8573,  1.6513]],

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")

    # use an proper ik solver for computing desired ee pose after resets
    ik_solver_cfg = {
        "urdf_path": f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Single_Adapter/physx_rigid_gelpad.urdf",
        "ee_link_name": "panda_hand", #gelsight_mini_gelpad
        "max_iterations": 75,
        "num_retries": 1,
        "learning_rate": 0.2
    }

    # rigid body ball
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path= "/World/envs/env_.*/rigid_ball",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/ball_wood.usd", 
            #scale=(2, 1, 0.6),
            rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=120,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.01)),
    )

    # sensors
    gsmini = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
        sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix = "/Camera",
            update_period= 0,
            resolution = (64,64), #(120, 160),
            data_types = ["depth"],
            clipping_range = (0.024, 0.034),
        ),
        device = "cuda",
        tactile_img_res = (480, 640),
        debug_vis=True, # for being able to see sensor output in the gui
        # update Taxim cfg
        optical_sim_cfg=None,

        # update FOTS cfg
        marker_motion_sim_cfg=None,
        # marker_motion_sim_cfg=FOTSMarkerSimulatorCfg(
        #     lamb = [0.00125,0.00021,0.00038],
        #     pyramid_kernel_size = [51, 21, 11, 5], #[11, 11, 11, 11, 11, 5], 
        #     kernel_size = 5,
        #     marker_params = FOTSMarkerSimulatorCfg.MarkerParams(
        #         num_markers_col=25, #11,
        #         num_markers_row=20, #9,
        #         x0=26,
        #         y0=15,
        #         dx=29,
        #         dy=26
        #     ),
        #     tactile_img_res = (480, 640),
        #     device = "cuda",
        #     frame_transformer_cfg = FrameTransformerCfg(
        #         prim_path="/World/envs/env_.*/Robot/gelsight_mini_gelpad", #"/World/envs/env_.*/Robot/gelsight_mini_case",
        #         source_frame_offset=OffsetCfg(),
        #         target_frames=[
        #             FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/rigid_ball")
        #         ],
        #         debug_vis=False,
        #     )
        # ),
        data_types=["height_map"], #marker_motion
    )

    # noise models
    action_noise_model = NoiseModelCfg(
        noise_cfg = UniformNoiseCfg(n_min=-0.001, n_max=0.001, operation="add")
    )
    # observation_noise_model = 

    #MARK: reward configuration
    reaching_penalty = {"weight": -0.2}
    reaching_reward_tanh = {"std": 0.2, "weight": 0.4}
    at_obj_reward = {"weight": 2.5, "minimal_distance": 0.004}
    tracking_reward = {"weight":1.0, "w": 1, "v": 1, "alpha":1e-5, "minimal_distance": 0.004}
    off_the_ground_penalty = {"weight": -15, "max_height": 0.025}
    success_reward = {"weight": 10.0, "threshold": 0.005} # 0.0025 we count it as a sucess when dist obj <-> goal is less than the threshold
    height_penalty = {"min_weight": -0.5, "max_weight": -0.5, "min_height": 0.006, "max_height": 0.012}  # ball has diameter of 1cm, plate 0.5 cm -> 0.005m + 0.0025m = 0.0075m is above the ball
    orient_penalty = {"weight": -0.5}

    # curriculum settings
    curriculum_steps = [8.5e6] # after this amount of common_steps (= total steps), we make the task more difficult
    
    # extra reward scales
    action_rate_penalty_scale = [-1e-4, -1e-2] # give list for curriculum learning (-1e2 after common_step_count > currciculum_steps)
    joint_vel_penalty_scale = [-1e-4, -1e-2] 

    obj_pos_randomization_range = [[-0.125, 0.125], [-0.2, 0.2]]

    # env
    episode_length_s = 8.3333/2 # 1000/2 timesteps (dt = 1/120 -> 8.3333/(1/120) = 1000)
    action_space = 6 # we use relative task_space actions: (dx, dy, dz, droll, dpitch) -> dyaw is ommitted
    observation_space = {
        "proprio_obs": 14, #16, # 3 for ee pos, 2 for orient (roll, pitch), 2 for init goal-pos (x,y), 5 for actions
        "vision_obs": [64,64,1], # from tactile sensor
    }
    # observation_space = 14
    state_space = 0

    ball_radius = 0.005 # don't change, because rewards are tuned for this size 

    x_bounds = (0.2, 0.75)
    y_bounds = (-0.375, 0.375)
    too_far_away_threshold = 0.02 #0.125 #0.2 #0.15
    min_height_threshold = 0.002

class BallRollingEnvNoPrivileged(BallRollingEnv):
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

    cfg: BallRollingEnvNoPrivilegedCfg

    def __init__(self, cfg: BallRollingEnvNoPrivilegedCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # for training curriculum 
        self.curriculum_phase_id = 0

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        
        # create auxiliary variables for computing applied action, observations and rewards
        self.processed_actions = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

        self._goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device) 
        # make height of goal pos fixed
        self._goal_pos_w[:, 2] = 0.00125

        # add plots
        self.live_vis = DirectLiveVisualizer(self.cfg.debug_vis, self.num_envs, self._window, visualizer_name="Actions")
        self.live_vis.terms["actions"] = self.actions
        self.live_vis.create_visualizer()

        self.live_vis_obs = DirectLiveVisualizer(self.cfg.debug_vis, self.num_envs, self._window, visualizer_name="Observations")
        self.live_vis_obs.terms["proprio"] = self._get_observations()["policy"]["proprio_obs"]
        self.live_vis_obs.terms["vision"] = self._get_observations()["policy"]["vision_obs"]
        self.live_vis_obs.create_visualizer()

        self.metric_vis = DirectLiveVisualizer(self.cfg.debug_vis, self.num_envs, self._window, visualizer_name="Metrics")
        self.metric_vis.terms["ee_height"] = torch.zeros((self.num_envs,1))
        self.metric_vis.terms["ee_distance"] = torch.zeros((self.num_envs,1))
        self.metric_vis.create_visualizer()

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
                        pos=(0.0, 0.0, 0.131), #0.1034
                    ),
                ),
            ],
        )
        
        # sensors
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        self.gsmini = GelSightSensor(self.cfg.gsmini)
        self.scene.sensors["gsmini"] = self.gsmini

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
            orientation=ground.init_state.rot
        )

        # plate
        plate = RigidObjectCfg(
            prim_path="/World/envs/env_.*/plate",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0]),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/plate.usd",
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    kinematic_enabled=True
                )
            )
        )
        plate.spawn.func(
            plate.prim_path,
            plate.spawn,
            translation=plate.init_state.pos,
            orientation=ground.init_state.rot
        )

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    #MARK: rewards
    def _get_rewards(self) -> torch.Tensor:        
        #- Reward the agent for reaching the object using tanh-kernel.
        obj_pos = self.object.data.root_link_pos_w
        # for compensating that obj_pos is based on the center of the ball
        obj_pos[:, 2] += 0.005  # ball has diameter of 1cm -> r=0.005m, plate height (above ground)=0.0025
        ee_frame_pos = self._ee_frame.data.target_pos_w[..., 0, :] # end-effector positions in world frame: (num_envs, 3)
                
        # Distance of the end-effector to the object: (num_envs,)
        object_ee_distance = torch.norm(obj_pos - ee_frame_pos, dim=1) 
        # for giving agent incentive to touch the obj
        at_obj_reward = torch.where(object_ee_distance < self.cfg.at_obj_reward["minimal_distance"], self.cfg.at_obj_reward["weight"], 0)

        #- reaching part, but now its ee to goal (but only if ee touches ball) because current_ball pos is not part of the observations
        goal_ee_distance = torch.norm(self._goal_pos_w[:, :2] - ee_frame_pos[:, :2], dim=1)
        goal_reaching_penalty = torch.square(goal_ee_distance)
        goal_reaching_penalty = torch.where(object_ee_distance < self.cfg.tracking_reward["minimal_distance"], goal_reaching_penalty, 1.25*goal_reaching_penalty)
        goal_reaching_penalty *= self.cfg.reaching_penalty["weight"]

        # use tanh-kernel for additional reward
        goal_ee_distance_tanh = torch.where(
            object_ee_distance < self.cfg.at_obj_reward["minimal_distance"], 
            1 - torch.tanh(goal_ee_distance / self.cfg.reaching_reward_tanh["std"]),
            0.0
        )
        goal_ee_distance_tanh *= self.cfg.reaching_reward_tanh["weight"]
        
        # add penalty if ball goes flying
        off_the_ground = torch.where(obj_pos[:, 2] > self.cfg.off_the_ground_penalty["max_height"], self.cfg.off_the_ground_penalty["weight"], 0.0)

        # distance between obj and goal: (num_envs,)
        obj_goal_distance = torch.norm(self._goal_pos_w[:, :2] - obj_pos[:, :2], dim=1)
        tracking_goal = -(
            self.cfg.tracking_reward["w"]*obj_goal_distance
            + self.cfg.tracking_reward["v"]*torch.log(obj_goal_distance + self.cfg.tracking_reward["alpha"])
        ) 
        # only apply when ee is at object (with this our tracking goal always needs to be positive, otherwise reaching part wont work anymore)
        tracking_goal = (object_ee_distance < self.cfg.tracking_reward["minimal_distance"]) * tracking_goal
        tracking_goal *= self.cfg.tracking_reward["weight"]

        # penalize ee being too close to the ground
        height_penalty = torch.where(
            (ee_frame_pos[:, 2] < self.cfg.height_penalty["min_height"]), 
            self.cfg.height_penalty["min_weight"]*(self.cfg.height_penalty["min_height"]-ee_frame_pos[:, 2])*10, 
            0.0
        )
        # penalize ee being too high
        height_penalty = torch.where(
            (ee_frame_pos[:, 2] > self.cfg.height_penalty["max_height"]), 
            self.cfg.height_penalty["max_weight"]*(ee_frame_pos[:, 2]-self.cfg.height_penalty["max_height"])*10, 
            height_penalty
        )

        # penalize when ee orient is too big
        ee_frame_orient = euler_xyz_from_quat(self._ee_frame.data.target_quat_source[..., 0, :])
        x = wrap_to_pi(ee_frame_orient[0]-math.pi) # our panda hand asset has rotation from (180,0,-45) -> we substract 180 for defining the rotation limits
        y = wrap_to_pi(ee_frame_orient[1])
        orient_penalty = (
            (torch.abs(x) > math.pi/8)
            | (torch.abs(y) > math.pi/8)
        ) * self.cfg.orient_penalty["weight"]

        success_reward = torch.where(
            obj_goal_distance < self.cfg.success_reward["threshold"], 
            self.cfg.success_reward["weight"], 
            0.0
        )
        # only apply success_reward when ee is at the ball
        success_reward = (object_ee_distance < self.cfg.tracking_reward["minimal_distance"]) * success_reward
        
        # Penalize the rate of change of the actions using L2 squared kernel.
        action_rate_penalty = torch.sum(torch.square(self.actions - self.prev_actions), dim=1)
        # Penalize joint velocities on the articulation using L2 squared kernel.
        joint_vel_penalty = torch.sum(torch.square(self._robot.data.joint_vel[:, :]), dim=1)
        
        # curriculum: for more stable movement
        #if self.common_step_counter > self.cfg.curriculum_steps[self.curriculum_phase_id-1]:
        if self.common_step_counter > self.cfg.curriculum_steps[self.curriculum_phase_id-1]:
            self.curriculum_phase_id = 1

        rewards = (
            # + goal_reaching_penalty
            # + goal_ee_distance_tanh
            + at_obj_reward
            + off_the_ground
            # + tracking_goal
            # + success_reward
            + orient_penalty
            + height_penalty
            # + self.cfg.action_rate_penalty_scale[self.curriculum_phase_id] * action_rate_penalty
            + self.cfg.joint_vel_penalty_scale[self.curriculum_phase_id] * joint_vel_penalty
        )
        
        self.extras["log"] = {
            "goal_reaching_penalty": goal_reaching_penalty.float().mean(),
            "goal_ee_distance_tanh": goal_ee_distance_tanh.mean(),
            "at_obj_reward": at_obj_reward.float().mean(),
            "off_the_ground_penalty": off_the_ground.float().mean(),
            # "tracking_goal": tracking_goal.float().mean(),
            # "success_reward": success_reward.float().mean(),
            # penalties for nice looking behavior
            "orientation_penalty": orient_penalty.float().mean(),
            "height_penalty": height_penalty.mean(),
            "action_rate_penalty": (self.cfg.action_rate_penalty_scale[self.curriculum_phase_id] * action_rate_penalty).mean(), 
            "joint_vel_penalty": (self.cfg.joint_vel_penalty_scale[self.curriculum_phase_id] * joint_vel_penalty).mean(),
            # task metrics
            "Metric/ee_obj_error": object_ee_distance.mean(),
            "Metric/obj_goal_error": obj_goal_distance.mean()
        }
        ee_height = ee_frame_pos[:, 2]
        self.metric_vis.terms["ee_height"] = ee_height.reshape(-1,1)
        self.metric_vis.terms["ee_distance"] = object_ee_distance.reshape(-1,1)
        return rewards

    #MARK: reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # spawn obj at random position
        obj_pos = self.object.data.default_root_state[env_ids] 
        obj_pos[:, :3] += self.scene.env_origins[env_ids]
        # obj_pos[:, :2] += sample_uniform(
        #     self.cfg.obj_pos_randomization_range[self.curriculum_phase_id][0], 
        #     self.cfg.obj_pos_randomization_range[self.curriculum_phase_id][1],
        #     (len(env_ids), 2), 
        #     self.device
        # )
        self.object.write_root_state_to_sim(obj_pos, env_ids=env_ids)

        # reset robot state
        # joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos = (
            self._robot.data.default_joint_pos[env_ids]
            # + sample_uniform(
            #     -0.001,
            #     0.001,
            #     (len(env_ids), self._robot.num_joints),
            #     self.device,
            # )
        )
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # set commands: random target position 
        self._goal_pos_w[env_ids, :2] = self.object.data.default_root_state[env_ids, :2] + self.scene.env_origins[env_ids, :2] + sample_uniform(
            self.cfg.obj_pos_randomization_range[self.curriculum_phase_id][0], 
            self.cfg.obj_pos_randomization_range[self.curriculum_phase_id][1],
            (len(env_ids), 2), 
            self.device
        )

        # reset actions
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        # self._ik_controller.reset(env_ids)

        # reset sensors
        self.gsmini.reset(env_ids=env_ids)

    #MARK: observations
    def _get_observations(self) -> dict:
        """The position of the object in the robot's root frame."""

        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        ee_frame_orient = euler_xyz_from_quat(ee_quat_curr_b)
        x = wrap_to_pi(ee_frame_orient[0]).unsqueeze(1) # add dimension for concatenating with other observations
        y = wrap_to_pi(ee_frame_orient[1]).unsqueeze(1) 
        z = wrap_to_pi(ee_frame_orient[2]).unsqueeze(1) 
        # # obj position in the robots root frame
        # object_pos_b, _ = subtract_frame_transforms(
        #     self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self.object.data.root_link_pos_w[:, :3]
        # )
        goal_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._goal_pos_w
        )
        proprio_obs = torch.cat(
            (
                ee_pos_curr_b,
                x,
                y,
                z,
                # object_pos_b[:, :2],
                goal_pos_b[:, :2],  
                self.actions
            ),
            dim=-1,
        )
        vision_obs = self.gsmini._data.output["height_map"]
        
        # normalize images
        normalized = vision_obs.view(vision_obs.size(0), -1)
        normalized -= normalized.min(1, keepdim=True)[0]
        normalized /= normalized.max(1, keepdim=True)[0]
        normalized = (normalized*255).type(dtype=torch.int)
        vision_obs = normalized.reshape((self.num_envs,64,64,1)) # add a channel to the depth image for debug_vis
        
        obs = {
            "proprio_obs": proprio_obs,
            "vision_obs": vision_obs
        }
        
        # obs = proprio_obs
        # change goal_pos for env with long enough episodes
        # env_ids = ((self.episode_length_buf +1) % int(self.max_episode_length/2) == 0).nonzero(as_tuple=False).squeeze(-1)
        # if len(env_ids) > 0:
        #     # set commands: random target position 
        #     self._goal_pos_w[env_ids, :2] = self.object.data.default_root_state[env_ids, :2] + self.scene.env_origins[env_ids, :2] + sample_uniform(
        #         self.cfg.obj_pos_randomization_range[self.curriculum_phase_id][0], 
        #         self.cfg.obj_pos_randomization_range[self.curriculum_phase_id][1],
        #         (len(env_ids), 2), 
        #         self.device
        #     )

        # self.live_vis.terms["actions"][:] = self.actions[:]
        self.live_vis_obs.terms["proprio"] = proprio_obs
        self.live_vis_obs.terms["vision"] = vision_obs
        return {"policy": obs}
    