from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Control Franka, which is equipped with two GelSight Mini sensors, by moving the Frame in the GUI"
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--sys", type=bool, default=True, help="Whether to track system utilization.")
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

with suppress(ImportError):
    # isaacsim.gui is not available when running in headless mode.
    import isaacsim.gui.components.ui_utils as ui_utils

import pynvml

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from tacex import GelSightSensor, TacExUipcRobot, TacExUipcRobotCfg

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_gripper_uipc import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_UIPC_CFG
from tacex_assets.sensors.gelsight_mini import GELSIGHT_MINI_TAXIM_CFG

from tacex_uipc.utils import TetMeshCfg
from tacex_uipc import (
    UipcIsaacAttachmentsCfg,
    UipcDeformableObject,
    UipcDeformableObjectCfg,
    UipcRLEnv,
    UipcSimCfg,
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
        self.left_finger_pos = 0.0
        self.right_finger_pos = 0.0

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
            self.ui_window_elements["action_vstack"] = omni.ui.VStack(spacing=5, height=50)
            with self.ui_window_elements["action_vstack"]:
                self.ui_window_elements["left_finger_pos"] = ui_utils.combo_floatfield_slider_builder(
                    label="Left Finger Position",
                    default_val=0.0,  # open per default -> its open at joint pos 0
                    min=0.0,
                    max=0.04,
                    step=0.001,
                    tooltip="Specifies the position of the left finger of the franka.",
                )[0]  # we just want to access the value model and not the floatslider
                self.ui_window_elements["right_finger_pos"] = ui_utils.combo_floatfield_slider_builder(
                    label="Right Finger Position",
                    default_val=0.0,
                    min=0.0,
                    max=0.04,
                    step=0.001,
                    tooltip="Specifies the position of the right finger of the franka.",
                )[0]

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
            self.ui_window_elements["object_dropdown"].get_item_value_model().get_value_as_int()
        )  # dropdown options are returned as numbers -> index in list
        self.current_object = self.objects[current_obj_idx]

        current_action_idx = (
            self.ui_window_elements["action_dropdown"].get_item_value_model().get_value_as_int()
        )  # dropdown options are returned as numbers -> index in list
        self.current_gripper_action = self.gripper_actions[current_action_idx]

    def _reset_env(self):
        self.reset = True


# TODO Need to fix script, since newest tacex_uipc version
@configclass
class PickUpEnvCfg(DirectRLEnvCfg):
    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.9, 1.4, 0.3)
    viewer.lookat = (-1.5, -1.9, -1.1)

    # viewer.origin_type = "env"
    # viewer.env_idx = 50

    debug_vis = True

    ui_window_class_type = CustomEnvWindow

    decimation = 1
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
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

    uipc_sim = UipcSimCfg(
        # logger_level="Info"
        ground_height=0.0025,
        contact=UipcSimCfg.Contact(d_hat=0.0001),
        debug_vis=True,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.5,
        replicate_physics=True,
        lazy_sensor_update=True,  # only update sensors when they are accessed
    )

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

    # light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # plate
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

    mesh_cfg = TetMeshCfg(
        stop_quality=8,
        max_its=100,
        edge_length_r=1 / 5,
        # epsilon_r=0.01
    )
    ball = UipcDeformableObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.035]),  # rot=(0.72,-0.3,0.42,-0.45)
        spawn=sim_utils.UsdFileCfg(
            scale=(2.5, 2.5, 2.5),
            # usd_path="/workspace/tacex/source/tacex_assets/tacex_assets/data/Sensors/GelSight_Mini/Gelpad_low_res.usd",
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/ball_wood.usd",
        ),
        mesh_cfg=mesh_cfg,
        constitution_cfg=UipcDeformableObjectCfg.StableNeoHookeanCfg(youngs_modulus=0.0005),
    )

    # gelpads simulated as uipc mesh; they belong to the robot and are created by it
    gelpad_mesh_cfg = TetMeshCfg(
        stop_quality=8,
        max_its=100,
        edge_length_r=1 / 15,
        # epsilon_r=0.01
    )
    robot: TacExUipcRobotCfg = TacExUipcRobotCfg.from_articulation_cfg(
        FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_UIPC_CFG,
        ik_body_name="TCP",
        ik_joint_names=["panda_joint.*"],
        prim_path="/World/envs/env_.*/Robot",
        uipc_objects={
            "gelpad_left": UipcDeformableObjectCfg(
                prim_path="gsmini_gelpad_left",
                mesh_cfg=gelpad_mesh_cfg,
                constitution_cfg=UipcDeformableObjectCfg.StableNeoHookeanCfg(),
                constraint_cfg=UipcIsaacAttachmentsCfg(
                    constraint_strength_ratio=100.0,
                    body_name="gelsight_mini_case_left",
                ),
            ),
            "gelpad_right": UipcDeformableObjectCfg(
                prim_path="gsmini_gelpad_right",
                mesh_cfg=gelpad_mesh_cfg,
                constitution_cfg=UipcDeformableObjectCfg.StableNeoHookeanCfg(),
                constraint_cfg=UipcIsaacAttachmentsCfg(
                    constraint_strength_ratio=100.0,
                    body_name="gelsight_mini_case_right",
                ),
            ),
        },
    )

    gsmini_left = GELSIGHT_MINI_TAXIM_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
        sensor_camera_cfg=GELSIGHT_MINI_TAXIM_CFG.SensorCameraCfg(
            prim_name="Camera",
            update_period=0,
            resolution=(32, 24),  # 4:3 like the real image; tactile_img_res sets the output shape
            data_types=["depth"],
            clipping_range_m=(0.024, 0.034),
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

    gsmini_right = GELSIGHT_MINI_TAXIM_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
        sensor_camera_cfg=GELSIGHT_MINI_TAXIM_CFG.SensorCameraCfg(
            prim_name="Camera",
            update_period=0,
            resolution=(32, 24),  # 4:3 like the real image; tactile_img_res sets the output shape
            data_types=["depth"],
            clipping_range_m=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=True,  # for rendering sensor output in the gui
        # update Taxim cfg
        marker_motion_sim_cfg=None,
        data_types=["tactile_rgb"],  # marker_motion
    )
    # settings for optical sim
    gsmini_right.optical_sim_cfg = gsmini_left.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(32, 32),
    )

    obj_pos_randomization_range = [-0.15, 0.15]

    # some filler values, needed for DirectRLEnv
    episode_length_s = 0
    action_space = 0
    observation_space = 0
    state_space = 0


class PickUpEnv(UipcRLEnv):
    cfg: PickUpEnvCfg

    def __init__(self, cfg: PickUpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Index of fingers -> first id is left, second id is right finger
        self._finger_joint_ids, self._finger_joint_names = self._robot.find_joints(["panda_finger.*"])

        # create buffer to store actions (= ik_commands)
        self._env0_ids = torch.tensor([0], device=self.device)  # only have 1 robot
        self.ik_commands = torch.zeros((self.num_envs, 7), device=self.device)
        # self.ik_commands[:, 3:] = torch.tensor([0,1,0,0],device=self.device)

        self.step_count = 0

        self.goal_prim_view = None

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = TacExUipcRobot(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

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
                    prim_path="/World/envs/env_.*/Robot/TCP",
                    name="end_effector",
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

        # Spawn AssetBase objects manually
        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )

        VisualCuboid(
            prim_path="/Goal",
            size=0.01,
            position=np.array([0.5, 0.0, 0.15]),
            orientation=np.array([0, 1, 0, 0]),
            color=np.array([255.0, 0.0, 0.0]),
        )

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # --- UIPC simulation setup ---

        # gelpads are created by the robot (`TacExUipcRobotCfg.uipc_objects`)
        self.object = UipcDeformableObject(self.cfg.ball)

    # MARK: pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self._robot.set_ee_pose_target(self.ik_commands[:, :3], self.ik_commands[:, 3:])

    def _apply_action(self):
        # set finger position -> only have 1 robot
        finger_pos = torch.tensor(
            [
                [
                    self._window.ui_window_elements["left_finger_pos"].get_value_as_float(),
                    self._window.ui_window_elements["right_finger_pos"].get_value_as_float(),
                ]
            ],
            device=self.device,
        )
        self._robot.set_joint_position_target(finger_pos, joint_ids=self._finger_joint_ids, env_ids=self._env0_ids)

        self.step_count += 1

    # post-physics step calls

    # MARK: dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:  # which environment is done
        pass

    # MARK: rewards
    def _get_rewards(self) -> torch.Tensor:
        pass

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

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

        # reset uipc objects #todo implement that properly
        # self._uipc_gelpad_left.write_vertex_positions_to_sim(vertex_positions=self._uipc_gelpad_left.init_vertex_pos)
        # self._uipc_gelpad_right.write_vertex_positions_to_sim(vertex_positions=self._uipc_gelpad_right.init_vertex_pos)
        self.object.write_nodal_pos_to_sim(self.object.data.default_nodal_state_w[env_ids, :, :3])

    # MARK: observations
    def _get_observations(self) -> dict:
        pass


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
            env._window.ui_window_elements["left_finger_pos"].set_value(0.0)
            env._window.ui_window_elements["right_finger_pos"].set_value(0.0)

        # perform physics step
        env._pre_physics_step(None)
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.uipc_sim.update_render_meshes()
        env.sim.render()

        positions, orientations = env.goal_prim_view.get_world_poses()
        env.ik_commands[:, :3] = positions - env.scene.env_origins
        env.ik_commands[:, 3:] = orientations

        env.scene.update(dt=env.physics_dt)

    env.close()

    pynvml.nvmlShutdown()


def main():
    """Main function."""
    # Define simulation env
    env_cfg = PickUpEnvCfg()
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
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
