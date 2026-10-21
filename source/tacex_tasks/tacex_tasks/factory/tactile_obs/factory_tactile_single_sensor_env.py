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

from tacex import GelSightSensor

from .. import factory_utils
from .factory_tactile_env import FactoryTactileEnv

from .factory_tactile_single_sensor_env_cfg import FactoryTactileSingleEnvCfg
from .feature_extractor_tactile_rgb_images import TactileRGBFeatureExtractor


class FactoryTactileSingleEnv(FactoryTactileEnv):
    cfg: FactoryTactileSingleEnvCfg

    def __init__(
        self, cfg: FactoryTactileSingleEnvCfg, render_mode: str | None = None, **kwargs
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

        super().__init__(cfg, render_mode, **kwargs)


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

        self._robot = self.cfg.robot.class_type(self.cfg.robot)

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

        # Only use the right GelSight Mini
        self.gsmini_right = self.cfg.gsmini_right.class_type(self.cfg.gsmini_right)
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
            max_rel_pos=self.cfg.gsmini_right.gelpad_dimensions.as_tuple,
            out=self.gt_keypoints,
        )

        # train CNN to regress on keypoint positions
        pose_loss, embeddings = self.feature_extractor.step(
            None,
            self.gsmini_right.data.output["tactile_rgb"],
            self.gt_keypoints.view(-1, 9),
        )

        self.feature_extractor_embeddings = embeddings.clone().detach()

        # log pose loss from CNN training
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["feature_ext_pose_loss"] = pose_loss

        return self.feature_extractor_embeddings