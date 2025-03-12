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

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
import isaaclab.utils.math as math_utils

import pytorch_kinematics as pk

# from tactile_sim import GsMiniSensorCfg, GsMiniSensor
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_single_adapter_rigid import FRANKA_PANDA_ARM_GSMINI_SINGLE_ADAPTER_HIGH_PD_CFG

from .base_env import BallRollingEnv, BallRollingEnvCfg
from .height_map_env import BallRollingHeightMapEnv, BallRollingHeightMapEnvCfg

@configclass
class BallRollingIKResetEnvCfg(BallRollingHeightMapEnvCfg):
# class BallRollingIKResetEnvCfg(BallRollingEnvCfg):
    # use an proper ik solver for computing desired ee pose after resets
    ik_solver_cfg = {
        "urdf_path": f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Single_Adapter/physx_rigid_gelpad.urdf",
        "ee_link_name": "panda_hand", #gelsight_mini_gelpad
        "max_iterations": 100,
        "num_retries": 1,
        "learning_rate": 0.2
    }
    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")

    #MARK: reward configuration
    reaching_penalty = {"weight": -0.2} 
    reaching_reward_tanh = {"std": 0.2, "weight": 0.4}
    at_obj_reward = {"weight": 1, "minimal_distance": 0.01}
    off_the_ground_penalty = {"weight": -1, "max_height": 0.025}
    tracking_reward = {"weight":0.3, "w": 1, "v": 1, "alpha":1e-5, "minimal_distance": 0.01}
    # fine_tracking_reward = {"weight":0.01, "std": 0.23, "minimal_distance": 0.005}
    success_reward = {"weight": 10, "threshold": 0.005} # 0.0025 we count it as a sucess when dist obj <-> goal is less than the threshold
    height_penalty = {"weight": -0.1, "min_height": 0.008}  # ball has diameter of 1cm, plate 0.5 cm -> 0.005m + 0.0025m = 0.0075m is above the ball
    orient_penalty = {"weight": -0.1}

    episode_length_s = 8.3333/2
    too_far_away_threshold = 0.25

# class BallRollingIKResetEnv(BallRollingEnv):  
class BallRollingIKResetEnv(BallRollingHeightMapEnv):
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

    cfg: BallRollingIKResetEnvCfg

    def __init__(self, cfg: BallRollingIKResetEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        #### IK Solver ##########################
        ik_chain = pk.build_chain_from_urdf(open(self.cfg.ik_solver_cfg["urdf_path"], mode="rb").read())
        #ik_chain.print_tree()
        # extract a specific serial chain such for inverse kinematics
        ik_chain = pk.SerialChain(ik_chain, self.cfg.ik_solver_cfg["ee_link_name"])
        #ik_chain.print_tree()
        ik_chain = ik_chain.to(dtype=torch.float32, device=self.device)
        
        # get robot joint limits
        #ik_chain_lim = torch.tensor(ik_chain.get_joint_limits(), device=self.device)
        ik_chain_lim = torch.stack((self.robot_dof_lower_limits, self.robot_dof_upper_limits))

        # ik_chain_lim = self.robot_dof_lower_limits, self.robot_dof_upper_limits
        # create the IK object
        # see the constructor for more options and their explanations, such as convergence tolerances
        self.ik_solver = pk.PseudoInverseIK(
            ik_chain, 
            pos_tolerance=0.0001,
            rot_tolerance=0.001,
            max_iterations=self.cfg.ik_solver_cfg["max_iterations"], 
            num_retries=self.cfg.ik_solver_cfg["num_retries"],
            joint_limits=ik_chain_lim.T,
            early_stopping_any_converged=True,
            early_stopping_no_improvement="any",#"all", None
            debug=False,
            lr=self.cfg.ik_solver_cfg["learning_rate"]
        )
        self.des_reset_ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # self.des_reset_ee_rot = lab_math.matrix_from_quat(torch.tensor([0,0,1,0],device=self.device).repeat(self.num_envs, 1))
        self.des_reset_ee_rot = torch.tensor(
            [[1,0,0],
            [0,-1,0],
            [0,0,-1]], device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1, 1)
        
    #MARK: pre-physics step calls    
    # same as base_env
    #uncomment if you only want to check behavior of IK solver for reset
    # def _apply_action(self):
    #     pass

    #MARK: dones
    # def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     # same as base env
    
    #MARK: rewards
    def _get_rewards(self) -> torch.Tensor:        
        #- Reward the agent for reaching the object using tanh-kernel.
        obj_pos = self.object.data.root_link_state_w[:, :3]
        # for compensating that obj_pos is based on the center of the ball
        obj_pos[:,2] += 0.005  # ball has diameter of 1cm -> r=0.005m, plate height (above ground)=0.0025
        ee_frame_pos = self._ee_frame.data.target_pos_w[..., 0, :] # end-effector positions in world frame: (num_envs, 3)
        
        # Distance of the end-effector to the object: (num_envs,)
        object_ee_distance = torch.norm(obj_pos - ee_frame_pos, dim=1) 
        reaching_penalty = self.cfg.reaching_penalty["weight"]*torch.square(object_ee_distance)
        # use tanh-kernel
        object_ee_distance_tanh = 1 - torch.tanh(object_ee_distance / self.cfg.reaching_reward_tanh["std"])
        # for giving agent incentive to touch the obj
        at_obj_reward = (object_ee_distance < self.cfg.at_obj_reward["minimal_distance"]) * self.cfg.at_obj_reward["weight"]

        # add big penalty if ball goes flying
        off_the_ground = torch.where(obj_pos[:, 2] > self.cfg.off_the_ground_penalty["max_height"], self.cfg.off_the_ground_penalty["weight"], 0)

        # distance between obj and goal: (num_envs,)
        obj_goal_distance = torch.norm(self._desired_pos_w[:, :2] - self.object.data.root_link_state_w[:, :2], dim=1)
        tracking_goal = -(
            self.cfg.tracking_reward["w"]*obj_goal_distance
            + self.cfg.tracking_reward["v"]*torch.log(obj_goal_distance + self.cfg.tracking_reward["alpha"])
        ) 
        # only apply when ee is at object (with this our tracking goal always needs to be positive, otherwise reaching part wont work anymore)
        tracking_goal = (object_ee_distance < self.cfg.tracking_reward["minimal_distance"]) * tracking_goal
        tracking_goal *= self.cfg.tracking_reward["weight"]

        # additional reward, when object is close to the goal
        # fine_tracking_reward = 1 - torch.tanh(object_ee_distance / self.cfg.fine_tracking_reward["std"])
        # fine_tracking_reward = (object_ee_distance < self.cfg.fine_tracking_reward["minimal_distance"]) * fine_tracking_reward 
        # fine_tracking_reward *= self.cfg.fine_tracking_reward["weight"]

        #height penalty -> distance to the ground
        height_penalty = (ee_frame_pos[:, 2] < self.cfg.height_penalty["min_height"]) * self.cfg.height_penalty["weight"]
        
        # penalize when ee orient is to big
        ee_frame_orient = euler_xyz_from_quat(self._ee_frame.data.target_quat_source[..., 0, :])
        x = wrap_to_pi(ee_frame_orient[0]-math.pi) # our panda hand asset has rotation from (180,0,-45) -> we substract 180 for defining the rotation limits
        y = wrap_to_pi(ee_frame_orient[1])
        orient_penalty = (
            (torch.abs(x) > math.pi/8)
            | (torch.abs(y) > math.pi/8)
        ) * self.cfg.orient_penalty["weight"]
        
        success_reward = (obj_goal_distance < self.cfg.success_reward["threshold"]) * self.cfg.success_reward["weight"]
        
        # Penalize the rate of change of the actions using L2 squared kernel.
        action_rate_penalty = torch.sum(torch.square(self.actions - self.prev_actions), dim=1)
        # Penalize joint velocities on the articulation using L2 squared kernel.
        joint_vel_penalty = torch.sum(torch.square(self._robot.data.joint_vel[:, :]), dim=1)
        
        # curriculum: for more stable movement
        #if self.common_step_counter > self.cfg.curriculum_steps[self.curriculum_phase_id-1]:
        if self.common_step_counter > self.cfg.curriculum_steps[self.curriculum_phase_id-1]:
            self.curriculum_phase_id = 1

        rewards = (
            + reaching_penalty
            + self.cfg.reaching_reward_tanh["weight"] * object_ee_distance_tanh
            + at_obj_reward
            + off_the_ground
            + tracking_goal
            # + fine_tracking_reward
            + success_reward
            + orient_penalty
            + height_penalty
            + self.cfg.action_rate_penalty_scale[self.curriculum_phase_id] * action_rate_penalty
            + self.cfg.joint_vel_penalty_scale[self.curriculum_phase_id] * joint_vel_penalty
        )
        
        self.extras["log"] = {
            "reaching_penalty": reaching_penalty.float().mean(),
            "reaching_reward_tanh": (self.cfg.reaching_reward_tanh["weight"] * object_ee_distance_tanh).mean(),
            "at_obj_reward": at_obj_reward.float().mean(),
            "off_the_ground_penalty": off_the_ground.float().mean(),
            "tracking_goal": tracking_goal.float().mean(),
            # "fine_tracking_reward": fine_tracking_reward.float().mean(),
            "success_reward": success_reward.float().mean(),
            # penalties for nice looking behavior
            "orientation_penalty": orient_penalty.float().mean(),
            "height_penalty": height_penalty.mean(),
            "action_rate_penalty": (self.cfg.action_rate_penalty_scale[self.curriculum_phase_id] * action_rate_penalty).mean(), 
            "joint_vel_penalty": (self.cfg.joint_vel_penalty_scale[self.curriculum_phase_id] * joint_vel_penalty).mean(),
            # task metrics
            "Metric/num_ee_at_obj": torch.sum(object_ee_distance < self.cfg.tracking_reward["minimal_distance"]),
            "Metric/ee_obj_error": object_ee_distance.mean(),
            "Metric/obj_goal_error": obj_goal_distance.mean()
        }
        return rewards

    #MARK: reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # spawn obj at random position
        obj_pos = self.object.data.default_root_state[env_ids] 
        obj_pos[:, :3] += self.scene.env_origins[env_ids]
        obj_pos[:, :2] += sample_uniform(
            self.cfg.obj_pos_randomization_range[self.curriculum_phase_id][0], 
            self.cfg.obj_pos_randomization_range[self.curriculum_phase_id][1],
            (len(env_ids), 2), 
            self.device
        )
        self.object.write_root_state_to_sim(obj_pos, env_ids=env_ids)

        # compute desired ee pose so that ee is at the ball after reset

        # make sure that ee pose is in robot frame
        self.des_reset_ee_pos[env_ids, :] = obj_pos[:, :3].clone() - self.scene.env_origins[env_ids]
        # add offset between gelsight mini case frame (which is at the bottom of the sensor) to the gelpad
        self.des_reset_ee_pos[env_ids, 2] += 0.134 # cant set it too close to the ball, otherwise "teleporting" robot there is gonna kick ball away   
        # convert desired pos into transformation matrix
        goal_poses = pk.Transform3d(
            pos=self.des_reset_ee_pos[env_ids], 
            rot=self.des_reset_ee_rot[env_ids],
            device=self.device
        )
        # solve via IK for desired joint pos 
        sol = self.ik_solver.solve(goal_poses)
        # indices = torch.argmin(sol.err_pos, dim=1)
        # best_sol_currently = sol.solutions[torch.arange(indices.size(0)), indices]
        # print(sol.err_pos[torch.arange(indices.size(0)), indices])

        # write the computed IK values into the joint state of the robot
        #joint_pos = torch.clamp(best_sol_currently, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_pos = sol.solutions[:,0]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # set commands: random target position 
        self._desired_pos_w[env_ids, :2] = obj_pos[:, :2].clone() #+ self.scene.env_origins[env_ids][:,:2]
        self._desired_pos_w[env_ids, :2] += sample_uniform(
            self.cfg.obj_pos_randomization_range[self.curriculum_phase_id][0], 
            self.cfg.obj_pos_randomization_range[self.curriculum_phase_id][1],
            (len(env_ids), 2), 
            self.device
        )

        # reset actions
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._ik_controller.reset(env_ids)

        
    
####
## Helper Functions
####
    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                # marker_cfg.markers["cuboid"].size = (0.01, 0.01, 0.01)
                marker_cfg.markers["cuboid"].size = (2*self.cfg.success_reward["threshold"], 2*self.cfg.success_reward["threshold"], 0.01)
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

        # # ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        # self.ik_des_pose_visualizer.visualize(
        #     translations=self.des_reset_ee_pos[:,:3] + self.scene.env_origins,#self._ik_controller.ee_pos_des[:, :3] - self.scene.env_origins, 
        #     orientations=torch.tensor([0,1,0,0],device=self.device).repeat(self.num_envs, 1)
        # )