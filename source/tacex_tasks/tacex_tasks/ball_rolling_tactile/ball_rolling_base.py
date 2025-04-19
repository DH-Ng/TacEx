# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math

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
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
import isaaclab.utils.math as math_utils
from isaaclab.utils.noise import GaussianNoiseCfg, UniformNoiseCfg, NoiseModelCfg

# from tactile_sim import GsMiniSensorCfg, GsMiniSensor
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_single_adapter_rigid import FRANKA_PANDA_ARM_GSMINI_SINGLE_ADAPTER_HIGH_PD_CFG
from tacex_assets.sensors.gelsight_mini.gelsight_mini_cfg import GelSightMiniCfg

from tacex import GelSightSensor
from tacex.simulation_approaches.fots import FOTSMarkerSimulator, FOTSMarkerSimulatorCfg

from tacex_tasks.utils import DirectLiveVisualizer


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
class BallRollingEnvCfg(DirectRLEnvCfg):

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
        dt=1 / 60, #0.001
        render_interval=decimation,
        #device="cpu",
        physx=PhysxCfg(
            enable_ccd=True, # needed for more stable ball_rolling
            # bounce_threshold_velocity=10000,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.2,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=1, replicate_physics=True)

    # use robot with stiff PD control for better IK tracking
    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_SINGLE_ADAPTER_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": -1.6945,
                "panda_joint2": -1.56,
                "panda_joint3": 1.778,
                "panda_joint4": -2.29,
                "panda_joint5": 1.71,
                "panda_joint6": 1.79,
                "panda_joint7": 1.59,
            },
            # joint_pos={
            #     "panda_joint1": 0.0,
            #     "panda_joint2": 0.44,
            #     "panda_joint3": 0.0,
            #     "panda_joint4": -2.38,
            #     "panda_joint5": 0.0,
            #     "panda_joint6": 2.82,
            #     "panda_joint7": 0.741,
            # },
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

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")

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
                    pos=(0.0, 0.0, 0.131), # 0ffset from panda hand frame origin to gelpad top
                    rot=(0.0, 0.0, 1.0, 0.0)
                    #rot=(0, 0.92388, -0.38268, 0) # our panda hand asset has rotation from (180,0,-45) -> we substract 180 for defining the rotation limits
                ),
            ),
        ],
    )
    
    # rigid body ball
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path= "/World/envs/env_.*/rigid_ball",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/ball_wood.usd", 
            #scale=(2, 1, 0.6),
            rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=60,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.0051+0.0025)),
    )

    # sensors
    gsmini = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
        sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix = "/Camera",
            update_period= 0,
            resolution = (32,32), #(120, 160),
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
    at_obj_reward = {"weight": 1.25, "minimal_distance": 0.005}
    off_the_ground_penalty = {"weight": -15, "max_height": 0.025}
    # ball has diameter of 1cm, plate height = 0.5 cm -> 0.01m + 0.0025m = 0.0125m is above the ball
    height_reward = {"weight": 0.5, "w": 10.0, "v": 0.3, "alpha": 0.00067, "target_height_cm": 1.25} # 0.5cm = gelpad height 
    orient_reward = {"weight": 0.75}
    # for solving the task
    ee_goal_tracking = {"weight": -0.005, "std": 0.0798}
    # ee_goal_tracking = {"weight": 0.75, "w": 0.3276, "v": 0.1672, "alpha": 0.00117}
    obj_goal_tracking = {"weight": 0.85, "w": 0.0482, "v": 0.7870, "alpha": 0.0083}
    # obj_goal_tracking = {"weight": 0.85, "w": 0.1717, "v": 0.3133, "alpha": 0.01825}
    # obj_goal_tracking = {"std": 0.0798, "weight": -0.001}
    obj_goal_fine_tracking = {"weight": 3.25, "std": 0.6661} #0.0322 0.2672
    obj_goal_super_fine_tracking = {"weight": 6.75, "std": 0.9363}
    success_reward = {"weight": 10.0, "threshold": 0.005} # 0.0025 we count it as a sucess when dist obj <-> goal is less than the threshold
    too_far_penalty = {"weight": -10.0}

    # extra reward scales
    action_rate_penalty_scale = [-1e-4, -1e-2] # give list for curriculum learning (-1e2 after common_step_count > currciculum_steps)
    joint_vel_penalty_scale = [-1e-4, -1e-2] 

    # curriculum settings
    curriculum_steps = [8.5e6] # after this amount of common_steps (= total steps), we make the task more difficult
    
    obj_pos_randomization_range = [-0.1, 0.1]

    # env
    num_goals = 2 # how many goal positions per episode
    episode_length_s = 8.3333 * num_goals # 1000 timesteps per goal (dt = 1/120 -> 8.3333/(1/120) = 1000)
    action_space = 6 # we use relative task_space actions: (dx, dy, dz, droll, dpitch) -> dyaw is ommitted
    observation_space = {
        "proprio_obs": 14, #16, # 3 for ee pos, 2 for orient (roll, pitch), 2 for init goal-pos (x,y), 5 for actions
        "vision_obs": [32,32,1], # from tactile sensor
    }
    # observation_space = 14
    state_space = 0
    action_scale = 0.05 # [cm]

    ball_radius = 0.005 # don't change, because rewards are tuned for this ball size 

    x_bounds = (0.2, 0.75)
    y_bounds = (-0.375, 0.375)
    too_far_away_threshold = 0.015 
    min_height_threshold = 0.002


class BallRollingEnv(DirectRLEnv):
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

        # for training curriculum 
        self.curriculum_phase_id = 0

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        #### Stuff for IK actions ##################################################
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
        self._offset_rot = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1) 
        # self._offset_rot = torch.tensor([0, 0.92388, -0.38268, 0], device=self.device).repeat(self.num_envs, 1) 

        ####################################################################

        # create auxiliary variables for computing applied action, observations and rewards
        self.processed_actions = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

        self._goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device) 
        # make height of goal pos fixed
        self._goal_pos_w[:, 2] = self.cfg.ball_radius*2 + 0.0025 # plate height above ground = 0.0025
        self.success_env = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        if self.cfg.debug_vis:
            # add plots
            self.visualizers = {
                "Actions": DirectLiveVisualizer(self.cfg.debug_vis, self.num_envs, self._window, visualizer_name="Actions"),
                "Observations": DirectLiveVisualizer(self.cfg.debug_vis, self.num_envs, self._window, visualizer_name="Observations"),
                "Rewards": DirectLiveVisualizer(self.cfg.debug_vis, self.num_envs, self._window, visualizer_name="Rewards"),
                "Metrics": DirectLiveVisualizer(self.cfg.debug_vis, self.num_envs, self._window, visualizer_name="Metrics"),
            }
            self.visualizers["Actions"].terms["actions"] = self.actions

            self.visualizers["Observations"].terms["ee_pos"] = torch.zeros((self.num_envs,3))
            self.visualizers["Observations"].terms["ee_rot"] = torch.zeros((self.num_envs,3))
            self.visualizers["Observations"].terms["goal"] = torch.zeros((self.num_envs,2))
            self.visualizers["Observations"].terms["sensor_output"] = self._get_observations()["policy"]["vision_obs"]

            self.visualizers["Rewards"].terms["rewards"] = torch.zeros((self.num_envs, 9))

            self.visualizers["Metrics"].terms["ee_height"] = torch.zeros((self.num_envs,1))
            self.visualizers["Metrics"].terms["ee_goal_distance"] = torch.zeros((self.num_envs,1))
            self.visualizers["Metrics"].terms["obj_ee_distance"] = torch.zeros((self.num_envs,1))
            self.visualizers["Metrics"].terms["obj_goal_distance"] = torch.zeros((self.num_envs,1))


            for vis in self.visualizers.values():
                vis.create_visualizer()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.object = RigidObject(self.cfg.ball)
        self.scene.rigid_objects["object"] = self.object

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        # sensors
        self._ee_frame = FrameTransformer(self.cfg.ee_frame_cfg)
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

    #MARK: pre-physics step calls
        
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions[:] = actions.clamp(-1.0, 1.0) 
        #! preprocess the action and turn it into IK action
        # self.processed_actions[:, :5] = self.actions
        # # fixed z rotation
        # self.processed_actions[:, 5] = 0 # dont change the z rotation
        self.processed_actions[:, :] = self.actions*self.cfg.action_scale

        # obtain ee positions and orientation w.r.t root (=base) frame
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        # set command into controller
        self._ik_controller.set_command(self.processed_actions, ee_pos_curr_b, ee_quat_curr_b)
        
    def _apply_action(self):
        # obtain quantities from simulation
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]
        # compute the delta in joint-space
        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        self._robot.set_joint_position_target(joint_pos_des)

        # pass

    # post-physics step calls    

    #MARK: dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]: # which environment is done
        obj_pos = self.object.data.root_link_pos_w - self.scene.env_origins
        out_of_bounds_x = (obj_pos[:, 0] < self.cfg.x_bounds[0]) | (obj_pos[:, 0] > self.cfg.x_bounds[1])
        out_of_bounds_y = (obj_pos[:, 1] < self.cfg.y_bounds[0]) | (obj_pos[:, 1] > self.cfg.y_bounds[1])

        obj_goal_distance = torch.norm(self._goal_pos_w[:, :2] - self.scene.env_origins[:, :2] - obj_pos[:,:2], dim=1)
        obj_too_far_away = obj_goal_distance > 0.75
            
        ee_frame_pos = self._ee_frame.data.target_pos_w[..., 0, :] - self.scene.env_origins # end-effector positions in world frame: (num_envs, 3)
        ee_too_far_away = torch.norm(obj_pos - ee_frame_pos, dim=1) > self.cfg.too_far_away_threshold

        # reset when ee orient is too large
        ee_frame_orient = euler_xyz_from_quat(self._ee_frame.data.target_quat_source[..., 0, :])
        x = wrap_to_pi(ee_frame_orient[0]) 
        y = wrap_to_pi(ee_frame_orient[1])
        orient_cond = (
            (torch.abs(x) > math.pi/4)
            | (torch.abs(y) > math.pi/4)
        )

        min_height = ee_frame_pos[:, 2] < self.cfg.min_height_threshold
                
        reset_cond = (
            out_of_bounds_x
            | out_of_bounds_y
            | obj_too_far_away
            | ee_too_far_away
            | orient_cond
            | min_height
        )

        # #! new goal position, if sucess
        # success_env = (obj_goal_distance < self.cfg.success_reward["threshold"]).nonzero(as_tuple=False).squeeze(-1)
        # if len(success_env) > 0:
        #     self._goal_pos_w[success_env, :2] = self.object.data.default_root_state[success_env, :2] + self.scene.env_origins[success_env, :2] + sample_uniform(
        #         self.cfg.obj_pos_randomization_range[0], 
        #         self.cfg.obj_pos_randomization_range[1],
        #         (len(success_env), 2), 
        #         self.device
        #     )
        #     # reset episode length
        #     self.episode_length_buf[success_env] = 0

        time_out = self.episode_length_buf >= self.max_episode_length - 1 # episode length limit

        return reset_cond, time_out
    #MARK: reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # reset to intial positions, if not successful
        #env_ids = (self.reset_buf & torch.logical_not(self.success_env)).nonzero(as_tuple=False).squeeze(-1)
        #env_ids = env_ids
        
        # spawn obj at initial position
        obj_pos = self.object.data.default_root_state[env_ids] 
        obj_pos[:, :2] += sample_uniform(
            -0.0005, 
            0.0005,
            (len(env_ids), 2), 
            self.device
        )
        obj_pos[:, :3] += self.scene.env_origins[env_ids]
        self.object.write_root_state_to_sim(obj_pos, env_ids=env_ids)

        # reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        # randomize joints 3 and 4 a little bit
        # joint_pos[:, 2:4] += sample_uniform(-0.0015, 0.0015, (len(env_ids), 2), self.device) 
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # set commands: random target position 
        self._goal_pos_w[env_ids, :2] = self.object.data.default_root_state[env_ids, :2] + self.scene.env_origins[env_ids, :2] + sample_uniform(
            self.cfg.obj_pos_randomization_range[0], 
            self.cfg.obj_pos_randomization_range[1],
            (len(env_ids), 2), 
            self.device
        )

        self.prev_actions[env_ids] = 0.0
        # reset sensors
        self.gsmini.reset(env_ids=env_ids)    
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
        at_obj_reward = torch.where(
            object_ee_distance <= self.cfg.at_obj_reward["minimal_distance"], 
            self.cfg.at_obj_reward["weight"], 
            0.0
        )
        too_far_penalty = torch.where(
            object_ee_distance > self.cfg.too_far_away_threshold*0.85, 
            self.cfg.too_far_penalty["weight"], 
            0.0
        )
        # penalty for preventing the ball from jumping
        off_the_ground_penalty = torch.where(
            obj_pos[:, 2] > self.cfg.off_the_ground_penalty["max_height"], 
            self.cfg.off_the_ground_penalty["weight"], 
            0.0
        )

        height_diff = self.cfg.height_reward["target_height_cm"] - ee_frame_pos[:, 2]*100
        height_reward = -(
            self.cfg.height_reward["w"]*height_diff**2
            + self.cfg.height_reward["v"]*torch.log(height_diff**2 + self.cfg.height_reward["alpha"])
        ).clamp(-1,1)
        # penalize ee being too close to ground
        height_reward = torch.where(
            (ee_frame_pos[:, 2] <= self.cfg.min_height_threshold), 
            height_reward-10, 
            height_reward
        )
        height_reward *= self.cfg.height_reward["weight"]
        # penalize when ee orient is too big
        ee_frame_orient = euler_xyz_from_quat(self._ee_frame.data.target_quat_source[..., 0, :])
        x = wrap_to_pi(ee_frame_orient[0])
        y = wrap_to_pi(ee_frame_orient[1])
        orient_reward = torch.where(
            (torch.abs(x) < math.pi/8) 
            | (torch.abs(y) < math.pi/8),
            0.25*self.cfg.orient_reward["weight"],
            0.0
        )
        orient_reward = torch.where(
            (torch.abs(x) < math.pi/10) 
            | (torch.abs(y) < math.pi/10),
            1.0*self.cfg.orient_reward["weight"],
            orient_reward
        )

        ee_goal_distance = torch.norm(ee_frame_pos - self._goal_pos_w, dim=1)
        # ee_goal_tracking_reward = torch.square(ee_goal_distance*100)
        ee_goal_tracking_reward = (ee_goal_distance*100)**2 # [cm]
        # ee_goal_tracking_reward = -(
        #     self.cfg.ee_goal_tracking["w"]*ee_goal_distance**2
        #     + self.cfg.ee_goal_tracking["v"]*torch.log(ee_goal_distance**2 + self.cfg.ee_goal_tracking["alpha"])
        # )#.clamp(-1.5,1.5)
        ee_goal_tracking_reward *= self.cfg.ee_goal_tracking["weight"]
        
        # distance between obj and goal: (num_envs,)
        obj_goal_distance = torch.norm(obj_pos[:, :2] - self._goal_pos_w[:, :2], dim=1)
        # obj_goal_tracking_reward = (obj_goal_distance*100)**3
        obj_goal_tracking_reward = -(
            self.cfg.obj_goal_tracking["w"]*(obj_goal_distance*10)**2 #[dm]
            + self.cfg.obj_goal_tracking["v"]*torch.log((obj_goal_distance*10)**2 + self.cfg.obj_goal_tracking["alpha"])
        ).clamp(-1.5,1.5)
        obj_goal_tracking_reward *= self.cfg.obj_goal_tracking["weight"]
        
        # obj_goal_fine_tracking_reward = torch.where(
        #     object_ee_distance < self.cfg.at_obj_reward["minimal_distance"], 
        #     1 - torch.tanh((obj_goal_distance*10) / self.cfg.obj_goal_fine_tracking["std"]), #[dm]
        #     0.0
        # )
        obj_goal_fine_tracking_reward = 1 - torch.tanh((obj_goal_distance*10) / self.cfg.obj_goal_fine_tracking["std"]) #[dm]
        obj_goal_fine_tracking_reward *= self.cfg.obj_goal_fine_tracking["weight"]

        obj_goal_super_fine_tracking_reward = 1 - torch.tanh((obj_goal_distance*100) / self.cfg.obj_goal_super_fine_tracking["std"])**2 #[cm]
        obj_goal_super_fine_tracking_reward *= self.cfg.obj_goal_super_fine_tracking["weight"]

        success_reward = torch.where(
            (obj_goal_distance < self.cfg.success_reward["threshold"]) 
            & (object_ee_distance < self.cfg.at_obj_reward["minimal_distance"]), 
            1.0*self.cfg.success_reward["weight"], 
            0.0
        )

        # Penalize the rate of change of the actions using L2 squared kernel.
        # action_rate_penalty = torch.sum(torch.square(self.actions), dim=1)
        action_rate_penalty = torch.sum(torch.square(self.actions - self.prev_actions), dim=1)
        # Penalize joint velocities on the articulation using L2 squared kernel.
        joint_vel_penalty = torch.sum(torch.square(self._robot.data.joint_vel[:, :]), dim=1)
        
        rewards = (
            + at_obj_reward
            + off_the_ground_penalty
            + height_reward
            + orient_reward
            + ee_goal_tracking_reward
            + obj_goal_tracking_reward
            + obj_goal_fine_tracking_reward
            + obj_goal_super_fine_tracking_reward
            + success_reward
            + too_far_penalty
            + self.cfg.action_rate_penalty_scale[self.curriculum_phase_id] * action_rate_penalty
            + self.cfg.joint_vel_penalty_scale[self.curriculum_phase_id] * joint_vel_penalty
        )
        
        self.extras["log"] = {
            "at_obj_reward": at_obj_reward.float().mean(),
            "off_the_ground_penalty": off_the_ground_penalty.float().mean(),
            "height_reward": height_reward.mean(),
            "orient_reward": orient_reward.float().mean(),
            "ee_goal_tracking_reward": ee_goal_tracking_reward.float().mean(),
            "obj_goal_tracking_reward": obj_goal_tracking_reward.float().mean(),
            "obj_goal_fine_tracking_reward": obj_goal_fine_tracking_reward.float().mean(),
            "obj_goal_super_fine_tracking_reward": obj_goal_super_fine_tracking_reward.float().mean(),
            "success_reward": success_reward.float().mean(),
            # penalties for nice looking behavior
            "action_rate_penalty": (self.cfg.action_rate_penalty_scale[self.curriculum_phase_id] * action_rate_penalty).mean(), 
            "joint_vel_penalty": (self.cfg.joint_vel_penalty_scale[self.curriculum_phase_id] * joint_vel_penalty).mean(),
            # task metrics
            "Metric/ee_obj_error": object_ee_distance.mean(),
            "Metric/ee_goal_error": ee_goal_distance.mean(),
            "Metric/obj_goal_error": obj_goal_distance.mean()
        }

        if self.cfg.debug_vis:
            self.visualizers["Rewards"].terms["rewards"][:, 0] = at_obj_reward
            self.visualizers["Rewards"].terms["rewards"][:, 1] = height_reward
            self.visualizers["Rewards"].terms["rewards"][:, 2] = orient_reward
            self.visualizers["Rewards"].terms["rewards"][:, 3] = ee_goal_tracking_reward
            self.visualizers["Rewards"].terms["rewards"][:, 4] = obj_goal_tracking_reward
            self.visualizers["Rewards"].terms["rewards"][:, 5] = obj_goal_fine_tracking_reward
            self.visualizers["Rewards"].terms["rewards"][:, 6] = obj_goal_super_fine_tracking_reward
            self.visualizers["Rewards"].terms["rewards"][:, 7] = success_reward
            self.visualizers["Rewards"].terms["rewards"][:, -1] = rewards

            self.visualizers["Metrics"].terms["ee_height"]  = ee_frame_pos[:, 2].reshape(-1,1)
            self.visualizers["Metrics"].terms["ee_goal_distance"] = ee_goal_distance
            self.visualizers["Metrics"].terms["obj_ee_distance"] = object_ee_distance.reshape(-1,1)
            self.visualizers["Metrics"].terms["obj_goal_distance"] = obj_goal_distance
        return rewards

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
                goal_pos_b[:, :2],  
                self.actions
            ),
            dim=-1,
        )
        vision_obs = self.gsmini._data.output["height_map"]
        
        # normalize depth images
        normalized = vision_obs.view(vision_obs.size(0), -1)
        normalized -= 24.0
        normalized /= 29.0
        normalized = (normalized*255).type(dtype=torch.uint8)
        vision_obs = normalized.reshape((self.num_envs, 32, 32, 1)) # add a channel to the depth image for debug_vis

        obs = {
            "proprio_obs": proprio_obs,
            "vision_obs": vision_obs
        }
        
        # obs = proprio_obs
        # change goal_pos for env with long enough episodes
        env_ids = ((self.episode_length_buf + 1) % int(self.max_episode_length/self.cfg.num_goals) == 0).nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            # set commands: random target position 
            self._goal_pos_w[env_ids, :2] = self.object.data.default_root_state[env_ids, :2] + self.scene.env_origins[env_ids, :2] + sample_uniform(
                self.cfg.obj_pos_randomization_range[0], 
                self.cfg.obj_pos_randomization_range[1],
                (len(env_ids), 2), 
                self.device
            )

        # self.visualizers["Actions"].terms["actions"][:] = self.actions[:]
        if self.cfg.debug_vis:
            self.visualizers["Observations"].terms["ee_pos"] = ee_pos_curr_b[:, :3]
            self.visualizers["Observations"].terms["ee_rot"][:, :1] = x
            self.visualizers["Observations"].terms["ee_rot"][:, 1:2] = y
            self.visualizers["Observations"].terms["ee_rot"][:, 2:3] = z
            self.visualizers["Observations"].terms["goal"] = goal_pos_b[:, :2]
            self.visualizers["Observations"].terms["sensor_output"] = vision_obs.clone()
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
        #if self.cfg.body_offset is not None:
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
        #if self.cfg.body_offset is not None:
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
                marker_cfg = VisualizationMarkersCfg(
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=self.cfg.success_reward["threshold"],
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity= 0.5),
                        ),
                    }
                )
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
        translations = self._goal_pos_w.clone()
        translations[:, 2] = self.cfg.ball_radius + 0.0025
        self.goal_pos_visualizer.visualize(translations=translations)

        # ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        # self.ik_des_pose_visualizer.visualize(
        #     translations=ee_pos_curr + self.scene.env_origins,#self._ik_controller.ee_pos_des[:, :3] - self.scene.env_origins, 
        #     orientations=ee_quat_curr
        #     )
    
