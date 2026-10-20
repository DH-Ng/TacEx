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
from isaaclab.sim.schemas import modify_rigid_body_properties
from isaaclab.sim.spawners.materials import spawn_rigid_body_material
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply

from isaaclab.controllers import DifferentialIKController

from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from tacex import GelSightSensor

from .. import factory_utils
from ..factory_ik_joint_control_env import FactoryIKJointControlEnv

# from ..factory_env import FactoryEnv

from .factory_tactile_env_cfg import FactoryTactileEnvCfg
from .feature_extractor_tactile_rgb_images import TactileRGBFeatureExtractor


class FactoryTactileEnv(FactoryIKJointControlEnv):
    cfg: FactoryTactileEnvCfg

    def __init__(self, cfg: FactoryTactileEnvCfg, render_mode: str | None = None, **kwargs):
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

        super().__init__(cfg, render_mode, **kwargs)

        # Feature extractor to extract 3D position of keypoints of the held asset from tactile RGB images
        self.feature_extractor = TactileRGBFeatureExtractor(
            self.cfg.tactile_rgb_feature_extractor,
            self.device,
            f"{self.cfg.log_dir}/feature_extractor",
        )

        # keypoints buffer
        self.gt_keypoints = torch.ones(self.num_envs, 3, 3, dtype=torch.float32, device=self.device)

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(),
            translation=(0.0, 0.0, -1.05),
        )

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table",
            cfg,
            translation=(0.55, 0.0, 0.0),
            orientation=(0.70711, 0.0, 0.0, 0.70711),
        )

        self._robot = Articulation(self.cfg.robot)

        # Modify gelpad physics
        gelpad_left_rigid_body_path = "/World/envs/env_0/Robot/gsmini_gelpad_left"
        gelpad_right_rigid_body_path = "/World/envs/env_0/Robot/gsmini_gelpad_right"
        modify_rigid_body_properties(
            prim_path=gelpad_left_rigid_body_path,
            cfg=self.cfg.gelpad_rigidbody_properties,
        )

        modify_rigid_body_properties(
            prim_path=gelpad_right_rigid_body_path,
            cfg=self.cfg.gelpad_rigidbody_properties,
        )
        # change physics material
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.9,
            dynamic_friction=0.7,
            restitution=0.0,
            compliant_contact_stiffness=10.0,
            compliant_contact_damping=5.0,
        )
        spawn_rigid_body_material(
            prim_path="/World/gelpad_material",
            cfg=material_cfg,
        )

        sim_utils.bind_physics_material(
            prim_path=gelpad_left_rigid_body_path,
            material_path="/World/gelpad_material",
        )

        sim_utils.bind_physics_material(
            prim_path=gelpad_right_rigid_body_path,
            material_path="/World/gelpad_material",
        )

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
            asset_size=(
                self.cfg_task.held_asset_cfg.diameter,
                self.cfg_task.held_asset_cfg.diameter,
                self.cfg_task.held_asset_cfg.height,
            ),
            max_rel_pos=self.cfg.gsmini_left.gelpad_dimensions.as_tuple,
            out=self.gt_keypoints,
        )

        # train CNN to regress on keypoint positions
        pose_loss, embeddings = self.feature_extractor.step(
            self.gsmini_left.data.output["tactile_rgb"],
            self.gsmini_right.data.output["tactile_rgb"],
            self.gt_keypoints.view(-1, 9),
        )

        self.feature_extractor_embeddings = embeddings.clone().detach()

        # log pose loss from CNN training
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["feature_ext_pose_loss"] = pose_loss

        return self.feature_extractor_embeddings

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        feature_extractor_embeddings = self._compute_image_observations()
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        obs_dict["tactile_rgb_features"] = feature_extractor_embeddings

        state_dict["tactile_rgb_features"] = feature_extractor_embeddings
        state_dict["gt_keypoints"] = self.gt_keypoints.view(-1, 9)

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

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
        target_held_base_pos, target_held_base_quat = factory_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )

        keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        keypoints_fixed = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        offsets = factory_utils.get_keypoint_offsets(self.cfg_task.num_keypoints, self.device)
        keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        for idx, keypoint_offset in enumerate(keypoint_offsets):
            keypoints_held[:, idx] = torch_utils.tf_combine(
                held_base_quat,
                held_base_pos,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
            keypoints_fixed[:, idx] = torch_utils.tf_combine(
                target_held_base_quat,
                target_held_base_pos,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
        keypoint_dist = torch.norm(keypoints_held - keypoints_fixed, p=2, dim=-1).mean(-1)

        a0, b0 = self.cfg_task.keypoint_coef_baseline
        a1, b1 = self.cfg_task.keypoint_coef_coarse
        a2, b2 = self.cfg_task.keypoint_coef_fine
        # Action penalties.
        action_penalty_ee = torch.norm(self.actions, p=2)
        action_grad_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
        curr_engaged = self._get_curr_successes(success_threshold=self.cfg_task.engage_threshold, check_rot=False)

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

    def compute_keypoints(
        self,
        held_asset_pose: torch.Tensor,
        num_keypoints: int = 3,
        asset_size: tuple[float, float, float] = (0.007986, 0.007986, 0.05),
        max_rel_pos: tuple[float, float, float] = (1.0, 1.0, 1.0),
        out: torch.Tensor | None = None,
    ):
        """Compute keypoint positions for the held asset.

        Keypoint positions are expressed relative to the middle point of the
        finger. The asset transform is assumed to be located at the center of
        the asset.

        Args:
            held_asset_pose: Local position and orientation of the asset center,
                with shape ``(N, 7)``. The pose contains three position values (units are in [m])
                followed by four orientation values (quaternion).
            num_keypoints: Number of keypoints to compute. Defaults to 3.
            asset_size: Asset dimensions along the X, Y, and Z axes, in meters.
                Defaults to ``(0.007986, 0.007986, 0.05)``.
            max_rel_pos: Maximum allowed relative position along each axis (x,y,z).
                Relative positions exceeding these limits are clamped to the
                fixed value (-1, -1, -1). (x,y) correspond to gelpad (width, height) and z to gelpad depth.
            out: Optional output buffer with shape ``(N, num_keypoints, 3)``.
                If provided, the result is written into this tensor. Otherwise,
                a new tensor is allocated.

        Returns:
            A tensor containing the keypoint positions, with shape
            ``(N, num_keypoints, 3)``. If ``out`` is provided, the returned
            tensor is the same object as ``out``.
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

        half_asset_height = asset_size[2] / 2.0

        position = held_asset_pose[:, :3]
        axis_offset_local = torch.zeros_like(position)
        axis_offset_local[:, 2] = half_asset_height
        quat = held_asset_pose[:, 3:]

        axis_offset_world = quat_apply(quat, axis_offset_local)

        # # Code assumes that XForm is at the center of the held object.
        # top_position = position + axis_offset_world
        # center_position = position
        # bottom_position = position - axis_offset_world

        # Code assumes that XForm is at the bottom of the held object.
        top_position = position + 2 * axis_offset_world
        center_position = position + axis_offset_world
        bottom_position = position

        # local env 3D positions of keypoints
        out[:, 0] = top_position
        out[:, 1] = center_position
        out[:, 2] = bottom_position

        # Relative position to the finger midpoint
        out -= self.fingertip_midpoint_pos.unsqueeze(1)

        # Check if keypoint is in sensor area. If not, set value (-1,-1,-1) for the keypoint
        max_pos = torch.as_tensor(
            max_rel_pos,
            dtype=out.dtype,
            device=out.device,
        )
        exceeds_limit = (out.abs() > max_pos).any(dim=-1, keepdim=True)

        # Replace invalid keypoints with a fixed value.
        invalid_value = torch.ones(
            3,
            dtype=out.dtype,
            device=out.device,
        ) * (-1.0)

        out = torch.where(
            exceeds_limit,
            invalid_value,
            out,
        )

        return out

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)

        # Spawn visualization for current regressed keypoints
        self.keypoint_1_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/World/Visuals/key_point_1",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.004,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.0, 1.0),
                            opacity=0.1,
                        ),
                    )
                },
            )
        )

        self.keypoint_2_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/World/Visuals/key_point_2",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.004,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.0, 1.0),
                            opacity=0.1,
                        ),
                    )
                },
            )
        )

        self.keypoint_3_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/World/Visuals/key_point_3",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.004,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.0, 1.0),
                            opacity=0.1,
                        ),
                    )
                },
            )
        )

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)

        if not hasattr(self, "fingertip_midpoint_pos"):
            return

        # Update the visualizations of keypoints
        if hasattr(self, "feature_extractor_embeddings"):
            # feature_extractor embeddings are the flattened keypoints of each env.

            # So we need to reshape the tensor
            keypoints_pos = self.feature_extractor_embeddings[:].reshape(self.num_envs, 3, 3)

            # Top Position of held asset
            keypoint_1_pos = keypoints_pos[:, 0]
            estimated_held_asset_pos = self.fingertip_midpoint_pos[:, :3] + keypoint_1_pos
            estimated_held_asset_pos_w = estimated_held_asset_pos + self.scene.env_origins
            self.keypoint_1_marker.visualize(translations=estimated_held_asset_pos_w)

            # Center Position
            keypoint_2_pos = keypoints_pos[:, 1]
            estimated_held_asset_pos = self.fingertip_midpoint_pos[:, :3] + keypoint_2_pos
            estimated_held_asset_pos_w = estimated_held_asset_pos + self.scene.env_origins
            self.keypoint_2_marker.visualize(translations=estimated_held_asset_pos_w)

            # Bottom Position
            keypoint_3_pos = keypoints_pos[:, 2]
            estimated_held_asset_pos = self.fingertip_midpoint_pos[:, :3] + keypoint_3_pos
            estimated_held_asset_pos_w = estimated_held_asset_pos + self.scene.env_origins
            self.keypoint_3_marker.visualize(translations=estimated_held_asset_pos_w)
