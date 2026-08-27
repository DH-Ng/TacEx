# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply

from isaaclab.controllers import DifferentialIKController

from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

from tacex import GelSightSensor

from . import factory_control, factory_utils
from .factory_tactile_env_cfg import FactoryTactileEnvCfg


from .factory_ik_joint_control_env import FactoryIKJointControlEnv

from .feature_extractor_tactile_rgb_images import TactileRGBFeatureExtractor


class FactoryTactileEnv(FactoryIKJointControlEnv):
    cfg: FactoryTactileEnvCfg

    def __init__(
        self, cfg: FactoryTactileEnvCfg, render_mode: str | None = None, **kwargs
    ):
        """Factory tasks with GelSight Mini output for the policy observations.

        Uses the IK control defined in FactoryIKJointControlEnv.
        We follow the shadow_hand_vision_env feature-extractor implementation to train
        a feature extractor that uses the tactile rgb images from left and right GelSight Mini.

        The feature extractor regresses keypoint positions of the held asset.
        Specifically, 3 keypoints: at the top of the asset, in the middle and the bottom.

        Args:
            cfg (FactoryTactileEnvCfg): _description_
            render_mode (str | None, optional): _description_. Defaults to None.
        """

        # # Update number of obs/states
        # cfg.observation_space = sum([cfg.obs_dim_cfg[obs] for obs in cfg.obs_order])
        # cfg.state_space = sum([cfg.state_dim_cfg[state] for state in cfg.state_order])

        super().__init__(cfg, render_mode, **kwargs)

        # Feature extractor to extract 3D position of keypoints of the held asset from tactile RGB images
        self.feature_extractor = TactileRGBFeatureExtractor(
            self.cfg.tactile_rgb_feature_extractor, self.device, f"{self.cfg.log_dir}/feature_extractor"
        )

        # keypoints buffer
        self.gt_keypoints = torch.ones(
            self.num_envs, 3, 3, dtype=torch.float32, device=self.device
        )

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(),
            translation=(0.0, 0.0, -1.05),
        )

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        )
        cfg.func(
            "/World/envs/env_.*/Table",
            cfg,
            translation=(0.55, 0.0, 0.0),
            orientation=(0.70711, 0.0, 0.0, 0.70711),
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # GelSight Mini's
        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.scene.sensors["gsmini_left"] = self.gsmini_left

        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_right"] = self.gsmini_right

    def _compute_image_observations(self):
        # generate ground truth keypoints for held-asset
        self.compute_keypoints(
            held_asset_pose=torch.cat((self.held_pos, self.held_quat), dim=1),
            size=(
                self.cfg_task.held_asset_cfg.diameter,
                self.cfg_task.held_asset_cfg.diameter,
                self.cfg_task.held_asset_cfg.height,
            ),
            out=self.gt_keypoints,
        )

        # train CNN to regress on keypoint positions
        pose_loss, embeddings = self.feature_extractor.step(
            self.gsmini_left.data.output["tactile_rgb"],
            self.gsmini_right.data.output["tactile_rgb"],
            self.gt_keypoints.view(-1, 9),
        )

        self.embeddings = embeddings.clone().detach()

        # log pose loss from CNN training
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["feature_ext_pose_loss"] = pose_loss

        return self.embeddings

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        feature_extractor_embeddings = self._compute_image_observations()
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        obs_dict["tactile_rgb_features"] = feature_extractor_embeddings

        state_dict["tactile_rgb_features"] = feature_extractor_embeddings
        state_dict["gt_keypoints"] = self.gt_keypoints.view(-1, 9)

        obs_tensors = factory_utils.collapse_obs_dict(
            obs_dict, self.cfg.obs_order + ["prev_actions"]
        )
        state_tensors = factory_utils.collapse_obs_dict(
            state_dict, self.cfg.state_order + ["prev_actions"]
        )
        return {"policy": obs_tensors, "critic": state_tensors}

    def _reset_buffers(self, env_ids):
        """Reset buffers."""
        self.ep_succeeded[env_ids] = 0
        self.ep_success_times[env_ids] = 0

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Get successful and failed envs at current timestep
        check_rot = self.cfg_task.name == "nut_thread"
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )

        rew_dict, rew_scales = self._get_factory_rew_dict(curr_successes)

        rew_buf = torch.zeros_like(rew_dict["kp_coarse"])
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self.prev_actions = self.actions.clone()

        self._log_factory_metrics(rew_dict, curr_successes)
        return rew_buf

    def _get_factory_rew_dict(self, curr_successes):
        """Compute reward terms at current timestep."""
        rew_dict, rew_scales = {}, {}

        # Compute pos of keypoints on held asset, and fixed asset in world frame
        held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
            self.held_pos,
            self.held_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )
        target_held_base_pos, target_held_base_quat = (
            factory_utils.get_target_held_base_pose(
                self.fixed_pos,
                self.fixed_quat,
                self.cfg_task.name,
                self.cfg_task.fixed_asset_cfg,
                self.num_envs,
                self.device,
            )
        )

        keypoints_held = torch.zeros(
            (self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device
        )
        keypoints_fixed = torch.zeros(
            (self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device
        )
        offsets = factory_utils.get_keypoint_offsets(
            self.cfg_task.num_keypoints, self.device
        )
        keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        for idx, keypoint_offset in enumerate(keypoint_offsets):
            keypoints_held[:, idx] = torch_utils.tf_combine(
                held_base_quat,
                held_base_pos,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
                .unsqueeze(0)
                .repeat(self.num_envs, 1),
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
            keypoints_fixed[:, idx] = torch_utils.tf_combine(
                target_held_base_quat,
                target_held_base_pos,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
                .unsqueeze(0)
                .repeat(self.num_envs, 1),
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
        keypoint_dist = torch.norm(keypoints_held - keypoints_fixed, p=2, dim=-1).mean(
            -1
        )

        a0, b0 = self.cfg_task.keypoint_coef_baseline
        a1, b1 = self.cfg_task.keypoint_coef_coarse
        a2, b2 = self.cfg_task.keypoint_coef_fine
        # Action penalties.
        action_penalty_ee = torch.norm(self.actions, p=2)
        action_grad_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
        curr_engaged = self._get_curr_successes(
            success_threshold=self.cfg_task.engage_threshold, check_rot=False
        )

        # Penalize ee being too close to fixed asset based on rel. height
        ee_fixed_asset_rel_height = (self.fingertip_midpoint_pos - self.fixed_pos_obs_frame)[:, 2]
        too_close = torch.where(
            ee_fixed_asset_rel_height < self.cfg_task.too_close_penalty_threshold,
            1.0,
            0.0,
        )
    
        rew_dict = {
            "kp_baseline": factory_utils.squashing_fn(keypoint_dist, a0, b0),
            "kp_coarse": factory_utils.squashing_fn(keypoint_dist, a1, b1),
            "kp_fine": factory_utils.squashing_fn(keypoint_dist, a2, b2),
            "action_penalty_ee": action_penalty_ee,
            "action_grad_penalty": action_grad_penalty,
            "curr_engaged": curr_engaged.float(),
            "curr_success": curr_successes.float(),
            "too_close_penalty": too_close.float(),
        }
        rew_scales = {
            "kp_baseline": 1.0,
            "kp_coarse": 1.0,
            "kp_fine": 1.0,
            "action_penalty_ee": -self.cfg_task.action_penalty_ee_scale,
            "action_grad_penalty": -self.cfg_task.action_grad_penalty_scale,
            "curr_engaged": 1.0,
            "curr_success": 1.0,
            "too_close_penalty": -self.cfg_task.too_close_penalty_scale,
        }
        return rew_dict, rew_scales

    def _reset_idx(self, env_ids):
        """We assume all envs will always be reset at the same time."""
        super()._reset_idx(env_ids)

        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(
            joints=self.cfg.ctrl.reset_joints, env_ids=env_ids
        )
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

    def compute_keypoints(
        self,
        held_asset_pose: torch.Tensor,
        num_keypoints: int = 3,
        size: tuple[float, float, float] = (0.007986, 0.007986, 0.05),
        out: torch.Tensor | None = None,
    ):
        """Computes positions of 3 keypoints of the held asset. 

        The positions are relative to the finger middle point.

        Assumes that xform of held asset is at its center.
        Args:
            held_asset_pose: (Local) position and orientation of the center of the held asset. Shape is (N, 7)
            num_keypoints: Number of keypoints to compute. Default = 3
            size: Length of X, Y, Z dimensions of held asset. Defaults to the PEG dimensions = [0.007986, 0.007986, 0.05]
            out: Buffer to store keypoints. If None, a new buffer will be created. Shape: (N, num_keypoints, 3)
        """
        num_envs = held_asset_pose.shape[0]
        if out is None:
            out = torch.ones(
                num_envs,
                num_keypoints,
                3,
                dtype=torch.float32,
                device=held_asset_pose.device,
            )
        else:
            out[:] = 1.0

        half_asset_height = size[2] / 2.0

        position = held_asset_pose[:, :3]
        local_axis_offset = torch.zeros_like(position)
        local_axis_offset[:, 2] = half_asset_height
        quat = held_asset_pose[:, 3:]

        world_axis_offset = quat_apply(quat, local_axis_offset)

        top_position = position + world_axis_offset
        bottom_position = position - world_axis_offset

        # local env 3D positions of keypoints
        out[:, 0] = top_position
        out[:, 1] = position
        out[:, 2] = bottom_position

        # Relative position to the finger midpoint
        out -= self.fingertip_midpoint_pos.unsqueeze(1)

        return out
