from dataclasses import MISSING
import datetime
import json
import numpy as np
import psutil
import time
import torch
import traceback
from pathlib import Path

import carb
import pynvml

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from tacex import GelSightSensor

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets import TACEX_SENSORS_DATA_DIR

from tacex_assets.robots.franka.franka_gsmini_single_rigid import (
    FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_RIGID_CFG,
)
from tacex_assets.sensors.gelsight_mini import (
    GELSIGHT_MINI_TAXIM_CFG,
    GELSIGHT_MINI_TAXIM_FOTS_CFG,
    GELSIGHT_MINI_TAXIM_FEM_CFG,
)

from tacex.simulation_approaches.fots import FOTSMarkerSimulatorCfg
from tacex.simulation_approaches.gpu_taxim import TaximSimulatorCfg
from tacex.simulation_approaches.fem_based import ManiSkillSimulatorCfg

TAXIM_CFG = GELSIGHT_MINI_TAXIM_CFG.replace(
    prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
    sensor_camera_cfg=GELSIGHT_MINI_TAXIM_CFG.SensorCameraCfg(
        prim_name="Camera",
        update_period=0,
        resolution=(320, 240),
        data_types=["depth"],
        clipping_range=(0.024, 0.029),
    ),
    update_period=0.01,
    optical_sim_cfg=TaximSimulatorCfg(
        calib_folder_path=f"{TACEX_SENSORS_DATA_DIR}/GelSight_Mini/calibs/taxim/640x480",
        gelpad_height=GELSIGHT_MINI_TAXIM_CFG.gelpad_dimensions.height,
        gelpad_to_camera_min_distance=0.024,
        with_shadow=False,
        tactile_img_res=(320, 240),
        device="cuda",
    ),
    marker_motion_sim_cfg=None,
    compute_indentation_depth_class="optical_sim",
    device="cuda",
    data_types=["tactile_rgb"],
)

TAXIM_FOTS_CFG = GELSIGHT_MINI_TAXIM_FOTS_CFG.replace(
    prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
    sensor_camera_cfg=GELSIGHT_MINI_TAXIM_FOTS_CFG.SensorCameraCfg(
        prim_name="Camera",
        update_period=0,
        resolution=(32, 24),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    ),
    device="cuda",
    debug_vis=True,
    optical_sim_cfg=TaximSimulatorCfg(
        calib_folder_path=f"{TACEX_SENSORS_DATA_DIR}/GelSight_Mini/calibs/taxim/640x480",
        gelpad_height=GELSIGHT_MINI_TAXIM_FOTS_CFG.gelpad_dimensions.height,
        gelpad_to_camera_min_distance=0.024,
        with_shadow=False,
        tactile_img_res=(320, 240),
        device="cuda",
    ),
    marker_motion_sim_cfg=FOTSMarkerSimulatorCfg(
        lamb=[0.00125, 0.00021, 0.00038],
        pyramid_kernel_size=[51, 21, 11, 5],  # [11, 11, 11, 11, 11, 5],
        kernel_size=5,
        marker_params=FOTSMarkerSimulatorCfg.MarkerParams(
            num_markers_col=11, num_markers_row=9, num_markers=99, x0=15, y0=26, dx=26, dy=29
        ),
        tactile_img_res=(320, 240),
        device="cuda",
        frame_transformer_cfg=FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_mini_gelpad",  # "/World/envs/env_.*/Robot/gelsight_mini_case",
            # you have to make sure that the asset frame center is correct, otherwise wrong shear/twist motions
            source_frame_offset=OffsetCfg(
                # rot=(0.0, 0.92388, -0.38268, 0.0) # values for the robot used here
            ),
            target_frames=[FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/ball")],
            debug_vis=True,
            visualizer_cfg=FRAME_MARKER_CFG.replace(
                # keep existing markers, only scale them down
                markers={
                    **FRAME_MARKER_CFG.markers,
                    "frame": FRAME_MARKER_CFG.markers["frame"].replace(scale=(0.01, 0.01, 0.01)),
                },
                prim_path="/Visuals/FrameTransformerForFots",
            ),
        ),
    ),
    data_types=["marker_motion", "tactile_rgb"],
)

TAXIM_FEM_CFG = GELSIGHT_MINI_TAXIM_FEM_CFG.replace(
    prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
    sensor_camera_cfg=GELSIGHT_MINI_TAXIM_FEM_CFG.SensorCameraCfg(
        prim_name="Camera",
        update_period=0,
        resolution=(32, 24),
        data_types=["depth"],
        clipping_range=(0.024, 0.029),
    ),
    update_period=0.01,
    optical_sim_cfg=TaximSimulatorCfg(
        calib_folder_path=f"{TACEX_SENSORS_DATA_DIR}/GelSight_Mini/calibs/taxim/640x480",
        gelpad_height=GELSIGHT_MINI_TAXIM_FEM_CFG.gelpad_dimensions.height,
        gelpad_to_camera_min_distance=0.024,
        with_shadow=False,
        tactile_img_res=(320, 240),
        device="cuda",
    ),
    marker_motion_sim_cfg=ManiSkillSimulatorCfg(
        marker_interval_range=(2.0625, 2.0625),
        marker_rotation_range=0.0,
        marker_translation_range=(0.0, 0.0),
        marker_pos_shift_range=(0.0, 0.0),
        marker_random_noise=0.0,
        marker_lose_tracking_probability=0.0,
        normalize=False,
        marker_flow_size=128,
        camera_params=(
            340,
            325,
            160,
            125,
            0.0,
        ),
        tactile_img_res=(320, 240),
    ),
    compute_indentation_depth_class="optical_sim",
    device="cuda",
    data_types=["tactile_rgb", "marker_motion"],
)
