# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from tacex_assets import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG
from tacex_assets.sensors.gelsight_mini import GELSIGHT_MINI_TAXIM_CFG

from .factory_tasks_cfg import ASSET_DIR, FactoryTask, GearMesh, NutThread, PegInsert
from .factory_ik_joint_control_env_cfg import FactoryIKJointControlEnvCfg, CtrlCfg

from .feature_extractor_tactile_rgb_images import TactileRGBFeatureExtractorCfg

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "tactile_rgb_features": 9,  # use tactile rgb feature extractor to regress 3 keypoints of held asset, which contain rel. position information
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
    "tactile_rgb_features": 9,
    "gt_keypoints": 9,
}


@configclass
class ObsRandCfg:
    fixed_asset_pos = [0.001, 0.001, 0.001]


@configclass
class FactoryTactileEnvCfg(FactoryIKJointControlEnvCfg):
    # num_*: will be overwritten to correspond to obs_order, state_order.
    observation_space = (
        21 + 9
    )  # state observation + vision CNN embedding (3 keypoints with 3 points = 9 values)
    state_space = (
        72 + 9 + 9
    )  # asymetric states + vision CNN embedding + groundtruth of the CNN embedding (= the keypoints)
    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "tactile_rgb_features",
    ]
    obs_dim_cfg = OBS_DIM_CFG

    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        "tactile_rgb_features",
        "gt_keypoints",
    ]
    state_dim_cfg = STATE_DIM_CFG

    task_name: str = "peg_insert"  # peg_insert, gear_mesh, nut_thread
    task: FactoryTask = FactoryTask()
    obs_rand: ObsRandCfg = ObsRandCfg()
    ctrl: CtrlCfg = CtrlCfg()

    episode_length_s = 10.0  # Probably need to override.

    # GelSight Mini Sensors
    gsmini_left = GELSIGHT_MINI_TAXIM_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
        sensor_camera_cfg=GELSIGHT_MINI_TAXIM_CFG.SensorCameraCfg(
            prim_name="Camera",
            update_period=0,
            resolution=(32, 32),
            data_types=["depth"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=True,  # for rendering sensor output in the gui
        # update Taxim cfg
        marker_motion_sim_cfg=None,
        data_types=["tactile_rgb"],  # marker_motion
    )
    # settings for optical sim
    gsmini_left.optical_sim_cfg = gsmini_left.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(32, 32),
    )
    gsmini_right = gsmini_left.copy().replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
    )

    tactile_rgb_feature_extractor = TactileRGBFeatureExtractorCfg(
        write_image_to_file=False,
        save_step_frequency=int(
            episode_length_s / (1 / 120)
        ),  # save after each episode
        load_checkpoint=True,
    )


@configclass
class FactoryTaskPegInsertTactileCfg(FactoryTactileEnvCfg):
    task_name = "peg_insert"
    task = PegInsert()
    episode_length_s = 10.0


@configclass
class FactoryTaskGearMeshTactileCfg(FactoryTactileEnvCfg):
    task_name = "gear_mesh"
    task = GearMesh()
    episode_length_s = 20.0


@configclass
class FactoryTaskNutThreadTactileCfg(FactoryTactileEnvCfg):
    task_name = "nut_thread"
    task = NutThread()

    episode_length_s = 30.0


# --- Play scripts ---
@configclass
class FactoryTaskPegInsertTactilePlayCfg(FactoryTactileEnvCfg):
    task_name = "peg_insert"
    task = PegInsert()
    episode_length_s = 10.0

    tactile_rgb_feature_extractor = TactileRGBFeatureExtractorCfg(
        train=False,
        load_checkpoint=True,
        write_image_to_file=False,
    )
