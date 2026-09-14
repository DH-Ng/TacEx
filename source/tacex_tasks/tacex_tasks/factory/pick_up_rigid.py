from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Control Franka, which is equipped with two GelSight Mini sensors, by moving the Frame in the GUI"
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
parser.add_argument(
    "--sys", type=bool, default=True, help="Whether to track system utilization."
)
parser.add_argument(
    "--debug_vis",
    default=True,
    action="store_true",
    help="Whether to render tactile images in the# append AppLauncher cli args",
)
AppLauncher.add_app_launcher_args(parser)
# parse the arguments

args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import traceback
from contextlib import suppress

import carb
import omni.ui
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.prims import XFormPrim
import isaacsim.core.utils.torch as torch_utils

with suppress(ImportError):
    # isaacsim.gui is not available when running in headless mode.
    import isaacsim.gui.components.ui_utils as ui_utils

import pynvml

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
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.schemas import modify_rigid_body_properties
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from tacex import GelSightSensor

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG
from tacex_assets.sensors.gelsight_mini import GELSIGHT_MINI_TAXIM_CFG

from tacex_tasks.factory.factory_tasks_cfg import (
    FactoryTask,
    GearMesh,
    NutThread,
    PegInsert,
)


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

        # track the current object for the gripper (not the robot itself)
        self.current_object = "ball"  # ball is the default object
        self.objects = ["cube", "cylinder", "ball"]
        self.gripper_actions = ["Reach", "Lift", "Reach + Lift"]
        self.current_gripper_action = "Reach"
        # Start with open fingers
        self.left_finger_pos = 0.04
        self.right_finger_pos = 0.04

        # flags for simulation code
        self.reset = False

        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)

        with self.ui_window_elements["main_vstack"]:
            self._build_control_frame()
            # collapse some frames which we don't need
            self.ui_window_elements["debug_frame"].collapsed = True
            self.ui_window_elements["sim_frame"].collapsed = True

    def _build_control_frame(self):
        self.ui_window_elements["action_frame"] = omni.ui.CollapsableFrame(
            title="Gripping Demo Script",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=False,
            style=ui_utils.get_style(),
            horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with self.ui_window_elements["action_frame"]:
            self.ui_window_elements["action_vstack"] = omni.ui.VStack(
                spacing=5, height=50
            )
            with self.ui_window_elements["action_vstack"]:
                self.ui_window_elements[
                    "left_finger_pos"
                ] = ui_utils.combo_floatfield_slider_builder(
                    label="Left Finger Position",
                    default_val=self.left_finger_pos,  # open per default -> its open at joint pos 0.04
                    min=0.0,
                    max=0.04,
                    step=0.001,
                    tooltip="Specifies the position of the left finger of the franka.",
                )[
                    0
                ]  # we just want to access the value model and not the floatslider
                self.ui_window_elements["right_finger_pos"] = (
                    ui_utils.combo_floatfield_slider_builder(
                        label="Right Finger Position",
                        default_val=self.left_finger_pos,
                        min=0.0,
                        max=0.04,
                        step=0.001,
                        tooltip="Specifies the position of the right finger of the franka.",
                    )[0]
                )

                # objects_dropdown_cfg = {
                #     "label": "Objects",
                #     "type": "dropdown",
                #     "default_val": 0,
                #     "items": self.objects,
                #     "tooltip": "Select an action for the gripper",
                #     "on_clicked_fn": None,
                # }
                # self.ui_window_elements["object_dropdown"] = ui_utils.dropdown_builder(**objects_dropdown_cfg)

                # gripper_action_dropdown_cfg = {
                #     "label": "Gripper Action",
                #     "type": "dropdown",
                #     "default_val": 0,
                #     "items": self.gripper_actions,
                #     "tooltip": "Select an action for the gripper",
                #     "on_clicked_fn": None,
                # }
                # self.ui_window_elements["action_dropdown"] = ui_utils.dropdown_builder(**gripper_action_dropdown_cfg)

                # self.ui_window_elements["action_button"] = ui_utils.btn_builder(
                #     type="button",
                #     text="Apply Action",
                #     tooltip="Sends the above selected action to the robot.",
                #     on_clicked_fn=self._apply_gripper_action
                # )

                self.ui_window_elements["reset_button"] = ui_utils.btn_builder(
                    type="button",
                    text="Reset Env",
                    tooltip="Resets the environment, i.e. the objects are spawned back at their initial position.",
                    on_clicked_fn=self._reset_env,
                )

    ###
    # Functions for ui elements
    ###
    def _apply_gripper_action(self):
        # print("Applying the action")
        self.new_action = True

        current_obj_idx = (
            self.ui_window_elements["object_dropdown"]
            .get_item_value_model()
            .get_value_as_int()
        )  # dropdown options are returned as numbers -> index in list
        self.current_object = self.objects[current_obj_idx]

        current_action_idx = (
            self.ui_window_elements["action_dropdown"]
            .get_item_value_model()
            .get_value_as_int()
        )  # dropdown options are returned as numbers -> index in list
        self.current_gripper_action = self.gripper_actions[current_action_idx]

    def _reset_env(self):
        self.reset = True

        # Start with open fingers
        self.left_finger_pos = 0.04
        self.right_finger_pos = 0.04


@configclass
class PickUpEnvCfg(DirectRLEnvCfg):
    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.9, 1.4, 0.3)
    viewer.lookat = (-1.5, -1.9, -1.1)

    debug_vis = True

    ui_window_class_type = CustomEnvWindow

    decimation = 1
    # simulation
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1.0 / 120.0,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            enable_ccd=True,
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=8,
            min_position_iteration_count=192,
            min_velocity_iteration_count=8,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.5,
        replicate_physics=True,
        lazy_sensor_update=True,  # only update sensors when they are accessed
    )

    # plate for different friction setting
    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ground_plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/plate.usd",
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

    task = NutThread()

    # Use same robot cfg as we do in factory env
    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            },
            # joint_pos={
            #     "panda_joint1": 0.0,
            #     "panda_joint2": -0.569,
            #     "panda_joint3": 0.0,
            #     "panda_joint4": -2.810,
            #     "panda_joint5": 0.0,
            #     "panda_joint6": 3.037,
            #     "panda_joint7": 0.741,
            #     "panda_finger_joint.*": 0.04,
            # },
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                velocity_limit_sim=2.175,
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                velocity_limit_sim=2.61,
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=40.0,
                velocity_limit_sim=0.02,
                stiffness=2e2,  # important that stiffness isn't too high, otherwise held object is going to tunnel through the gelpads
                damping=1e1,
                friction=0.1,
                armature=0.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    # Adjust physics properties of robot
    robot.spawn.replace(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  # or true?
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
        ),
    )
    gelpad_rigidbody_properties = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,  # or true?
        max_depenetration_velocity=5.0,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=3666.0,
        enable_gyroscopic_forces=True,
        solver_position_iteration_count=192,
        solver_velocity_iteration_count=1,
        max_contact_impulse=1e32,
    )
    gsmini_left = GELSIGHT_MINI_TAXIM_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
        sensor_camera_cfg=GELSIGHT_MINI_TAXIM_CFG.SensorCameraCfg(
            prim_name="Camera",
            update_period=0,
            resolution=(32, 24),
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
        tactile_img_res=(32, 24),
    )
    gsmini_right = gsmini_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
    )

    ik_controller_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={
            "lambda_val": 0.1,  # same value as factory control -> higher than the default value
        },
    )
    ee_pos_offset = (0.0, 0.0, 0.0)  # (0.0, 0.0, 0.13768)
    ee_rot_offset = (1.0, 0.0, 0.0, 0.0)

    # some filler values, needed for DirectRLEnv
    episode_length_s = 0
    action_space = 0
    observation_space = 0
    state_space = 0


class PickUpEnv(DirectRLEnv):
    cfg: PickUpEnvCfg

    def __init__(self, cfg: PickUpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- for IK ---
        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        body_ids, _ = self._robot.find_bodies("TCP")
        # save only the first body index
        self._ee_idx = body_ids[0]

        # For a fixed base robot, the frame index is one less than the body index.
        # This is because the root body is not included in the returned Jacobians.
        self._jacobi_ee_idx = self._ee_idx - 1

        # Index of fingers -> first id is left, second id is right finger
        self._finger_joint_ids, self._finger_joint_names = self._robot.find_joints(
            ["panda_finger.*"]
        )

        # ee offset w.r.t TCP -> TCP is defined so that z-axis shows down. In our case here we want z to show upwards
        self._ee_pos_offset = torch.tensor(
            self.cfg.ee_pos_offset, device=self.device
        ).repeat(self.num_envs, 1)
        self._ee_rot_offset = torch.tensor(
            self.cfg.ee_rot_offset, device=self.device
        ).repeat(self.num_envs, 1)
        # ---

        # create buffer to store actions (= ik_commands)
        self.ik_commands = torch.zeros(
            (self.num_envs, self._ik_controller.action_dim), device=self.device
        )
        # self.ik_commands[:, 3:] = torch.tensor([0,1,0,0],device=self.device)

        self.step_count = 0

        self.goal_prim_view = None

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

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

        if self.cfg.task.name == "nut_thread":
            # rotate -90deg along z-axis
            initial_rot_deg = self.cfg.task.held_asset_rot_init
            rot_yaw_euler = torch.tensor(
                [0.0, 0.0, initial_rot_deg * np.pi / 180.0], device=self.device
            ).repeat(self.num_envs, 1)
            new_rot_quat = torch_utils.quat_from_euler_xyz(
                roll=rot_yaw_euler[:, 0],
                pitch=rot_yaw_euler[:, 1],
                yaw=rot_yaw_euler[:, 2],
            )
            # Update the init orientation of the asset
            self.cfg.task.held_asset.init_state = (
                self.cfg.task.held_asset.init_state.replace(rot=new_rot_quat[0, :])
            )

        self._held_asset = Articulation(self.cfg.task.held_asset)
        self.scene.articulations["held_asset"] = self._held_asset

        # also spawn the fixed asset to manually test how well this works
        # -> I guess we could use this to collect demonstrations?
        self._fixed_asset = Articulation(self.cfg.task.fixed_asset)
        self.scene.articulations["fixed_asset"] = self._fixed_asset

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
                        pos=(0.0, 0.0, 0.11841),
                    ),
                ),
            ],
        )

        # sensors
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.scene.sensors["gsmini_left"] = self.gsmini_left

        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_right"] = self.gsmini_right

        RigidObject(self.cfg.plate)

        # Spawn Default objects manually

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

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # For setting ee goal pose
        VisualCuboid(
            prim_path="/Goal",
            size=0.01,
            position=np.array([0.5, 0.0, 0.035]),
            orientation=np.array([0, 1, 0, 0]),
            color=np.array([255.0, 0.0, 0.0]),
        )

    # MARK: pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self._ik_controller.set_command(self.ik_commands)

    def _apply_action(self):
        # obtain quantities from simulation
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]

        # compute the delta in joint-space
        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(
                ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos
            )
        else:
            joint_pos_des = joint_pos.clone()

        # set finger position -> only have 1 robot
        joint_pos_des[0, self._finger_joint_ids[0]] = self._window.ui_window_elements[
            "left_finger_pos"
        ].get_value_as_float()
        joint_pos_des[0, self._finger_joint_ids[1]] = self._window.ui_window_elements[
            "right_finger_pos"
        ].get_value_as_float()

        self._robot.set_joint_position_target(joint_pos_des)

        self.step_count += 1

    # post-physics step calls

    # MARK: dones
    def _get_dones(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # which environment is done
        pass

    # MARK: rewards
    def _get_rewards(self) -> torch.Tensor:
        pass

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, :3] += self.scene.env_origins[env_ids]
        held_state[:, 0] = 0.5
        held_state[:, 1] = 0.0
        held_state[:, 2] = 0.035
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_state_to_sim(held_state, env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 0] = 0.5
        fixed_state[:, 1] = 0.0
        fixed_state[:, 2] = 0.005
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(
            fixed_state[:, 7:], env_ids=env_ids
        )
        self._fixed_asset.reset()

        # reset robot state
        joint_pos = (
            self._robot.data.default_joint_pos[env_ids]
            # + sample_uniform(
            #     -0.125,
            #     0.125,
            #     (len(env_ids), self._robot.num_joints),
            #     self.device,
            # )
        )
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # MARK: observations
    def _get_observations(self) -> dict:
        pass

    """
    Helper Functions for IK control (from task_space_actions.py of IsaacLab).
    """

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._robot.root_physx_view.get_jacobians()[:, self._jacobi_ee_idx, :, :]

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
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._ee_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._ee_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )
        # account for the offset
        # if self.cfg.body_offset is not None:
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
        # if self.cfg.body_offset is not None:
        # Modify the jacobian to account for the offset
        # -- translational part
        # v_link = v_ee + w_ee x r_link_ee = v_J_ee * q + w_J_ee * q x r_link_ee
        #        = (v_J_ee + w_J_ee x r_link_ee ) * q
        #        = (v_J_ee - r_link_ee_[x] @ w_J_ee) * q
        jacobian[:, 0:3, :] += torch.bmm(
            -math_utils.skew_symmetric_matrix(self._ee_pos_offset), jacobian[:, 3:, :]
        )
        # -- rotational part
        # w_link = R_link_ee @ w_ee
        jacobian[:, 3:, :] = torch.bmm(
            math_utils.matrix_from_quat(self._ee_rot_offset), jacobian[:, 3:, :]
        )

        return jacobian


def run_simulator(env: PickUpEnv):
    """Runs the simulation loop."""

    print(f"Starting simulation with {env.num_envs} envs")
    env.reset()

    env.goal_prim_view = XFormPrim(prim_paths_expr="/Goal", name="Goal", usd=True)

    # Simulation loop
    while simulation_app.is_running():
        if env._window.reset:  # hacky way of getting the ui information, but it works
            print("-" * 80)
            print("[INFO]: Resetting environment...")
            # toggle flags
            env._window.reset = False
            env._window.new_action = False
            env.reset()

            # let the gripper be open
            # finger_joint_pos = torch.tensor([[0.04, 0.04]], device=env.device)
            # make sure that the ui value is consistent
            env._window.ui_window_elements["left_finger_pos"].set_value(0.04)
            env._window.ui_window_elements["right_finger_pos"].set_value(0.04)

        # perform physics step
        env._pre_physics_step(None)
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)

        positions, orientations = env.goal_prim_view.get_world_poses()
        env.ik_commands[:, :3] = positions - env.scene.env_origins
        env.ik_commands[:, 3:] = orientations

        # update isaac buffers() -> also updates sensors
        env.scene.update(dt=env.physics_dt)
        # render scene for cameras (used by sensor)
        env.sim.render()

    env.close()

    pynvml.nvmlShutdown()


def main():
    """Main function."""
    # Define simulation env
    env_cfg = PickUpEnvCfg()
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    env_cfg.gsmini_left.debug_vis = args_cli.debug_vis

    experiment = PickUpEnv(env_cfg)

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(env=experiment)


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim apply
        simulation_app.close()
