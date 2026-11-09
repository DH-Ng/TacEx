"""Collecting tactile data of the shapes from https://danfergo.github.io/gelsight-simulation/.

Use
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Control Franka, which is equipped with one GelSight Mini Sensor, by moving the Frame in the GUI"
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--debug_vis",
    default=True,
    action="store_true",
    help="If tactile images should be rendered inside the GUI.",
)
AppLauncher.add_app_launcher_args(parser)
# parse the arguments

args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import traceback
from contextlib import suppress
from pathlib import Path

import omni.ui

with suppress(ImportError):
    # isaacsim.gui is not available when running in headless mode.
    import isaacsim.gui.components.ui_utils as ui_utils

import numpy as np
import torch

import carb
import pynvml
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.prims import XFormPrim

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
from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg
from isaaclab.utils import configclass

from tacex import GelSightSensor
from tacex.simulation_approaches.pix2pix import Pix2PixSimulatorCfg

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_single_rigid import (
    FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_RIGID_CFG,
)
from tacex_assets.sensors.gelsight_mini import GELSIGHT_MINI_PIX2PIX_CFG


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

        self.object_names = list(create_shapes_cfg().keys())
        self.current_object_name = self.object_names[0]

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
            title="Shape Touching Demo Script",
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
                objects_dropdown_cfg = {
                    "label": "Objects",
                    "type": "dropdown",
                    "default_val": 0,
                    "items": self.object_names,
                    "tooltip": "Select an action for the gripper",
                    "on_clicked_fn": self._set_main_obj,
                }
                self.ui_window_elements["object_dropdown"] = ui_utils.dropdown_builder(**objects_dropdown_cfg)

                self.ui_window_elements["reset_button"] = ui_utils.btn_builder(
                    type="button",
                    text="Reset Env",
                    tooltip="Resets the environment, i.e. the objects are spawned back at their initial position.",
                    on_clicked_fn=self._reset_env,
                )

    def get_current_object_name(self):
        current_obj_idx = (
            self.ui_window_elements["object_dropdown"].get_item_value_model().get_value_as_int()
        )  # dropdown options are returned as numbers -> index in list
        self.current_object_name = self.object_names[current_obj_idx]
        return self.current_object_name

    def _set_main_obj(self, value):
        print("Set new main obj ", value)
        # get pose of old obj and of new current obj
        old_obj: RigidObject = self.env.scene.rigid_objects[self.current_object_name]

        obj_name = self.get_current_object_name()
        new_obj: RigidObject = self.env.scene.rigid_objects[obj_name]
        new_pose = new_obj.data.root_pose_w

        # swap position of old and new obj
        old_obj.write_root_pose_to_sim(new_pose)
        new_obj.write_root_pose_to_sim(self.env.main_pose)

    def _reset_env(self):
        self.env.reset()


def create_shapes_cfg() -> dict[str, RigidObjectCfg]:
    """Creates RigidObjectCfg's for each usd file in the `{TACEX_ASSETS_DATA_DIR}/Props/tactile_test_shapes/` directory.

    The objects are spawned on a line along the y axis.
    Returns:
        shapes: dict of names and corresponding RigidObjectCfg's
    """
    shapes = {}
    usd_files_path = sorted(list(Path(f"{TACEX_ASSETS_DATA_DIR}/Props/tactile_test_shapes/").glob("*.usd")))

    for i, file_path in enumerate(usd_files_path):
        file_name = file_path.stem

        if i == 0:
            pos = [0.5, 0.0, 0.02]
        else:
            pos = [
                0.5 - 0.25,
                -0.025 * len(usd_files_path) / 2 + 0.025 * i,
                0.02,
            ]

        shapes[file_name] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/{file_name}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(file_path),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    kinematic_enabled=True,
                    disable_gravity=False,
                ),
            ),
        )

    return shapes


@configclass
class ShapeTouchEnvCfg(DirectRLEnvCfg):
    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    # viewer.eye = (0.6, 0.1, 0.025)
    # viewer.lookat = (-1.2, -2.0, -0.2)
    viewer.eye = (0.55, -0.06, 0.025)
    viewer.lookat = (-4.8, 6.0, -0.2)

    debug_vis = True

    ui_window_class_type = CustomEnvWindow

    decimation = 1
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            enable_ccd=True,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=5.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
        render=RenderCfg(enable_translucency=True),
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

    robot: ArticulationCfg = FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    shapes: dict = create_shapes_cfg()

    gsmini = GELSIGHT_MINI_PIX2PIX_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
        sensor_camera_cfg=GELSIGHT_MINI_PIX2PIX_CFG.SensorCameraCfg(
            prim_name="Camera",
            resolution=(320, 240),
            clipping_range_m=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=True,  # for being able to see sensor output in the gui
        data_types=["tactile_rgb", "height_map", "camera_depth"],
    )

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ee_pos_offset = [0.0, 0.0, 0.0]
    ee_rot_offset = [0.0, 1.0, 0.0, 0.0]

    main_pose = [0.5, 0.0, 0.02, 1, 0, 0, 0]

    # some filler values, needed for DirectRLEnv class
    episode_length_s = 0
    action_space = 0
    observation_space = 0
    state_space = 0


class ShapeTouchEnv(DirectRLEnv):
    cfg: ShapeTouchEnvCfg

    def __init__(self, cfg: ShapeTouchEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- for IK ---
        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        # Obtain the frame index of the end-effector
        body_ids, _ = self._robot.find_bodies("TCP")
        # only save the first body index
        self._body_tcp_idx = body_ids[0]

        # For a fixed base robot, the frame index is one less than the body index.
        # This is because the root body is not included in the returned Jacobians.
        self._jacobi_body_idx = self._body_tcp_idx - 1

        # ee offset w.r.t TCP -> TCP is defined so that z-axis shows down. In our case here we want z to show upwards
        self._ee_pos_offset = torch.tensor(self.cfg.ee_pos_offset, device=self.device).repeat(self.num_envs, 1)
        self._ee_rot_offset = torch.tensor(self.cfg.ee_rot_offset, device=self.device).repeat(self.num_envs, 1)
        # ---

        # create buffer to store actions (= ik_commands)
        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)

        self.step_count = 0

        self.goal_prim_view = None

        self.main_pose = torch.tensor([self.cfg.main_pose], device=self.device)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        for obj_name, cfg in self.cfg.shapes.items():
            self.scene.rigid_objects[obj_name] = RigidObject(cfg)

        self.object = list(self.scene.rigid_objects.values())[0]

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        ee_frame_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/panda_link0",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/TCP",
                    name="end_effector",
                    offset=OffsetCfg(pos=self.cfg.ee_pos_offset, rot=self.cfg.ee_rot_offset),
                ),
            ],
        )

        # sensors
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        self.gsmini = GelSightSensor(self.cfg.gsmini)
        self.scene.sensors["gsmini"] = self.gsmini

        # Spawn AssetBase objects manually
        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path,
            ground.spawn,
            translation=ground.init_state.pos,
            orientation=ground.init_state.rot,
        )

        # goal marker
        VisualCuboid(
            prim_path="/Goal",
            size=0.01,
            position=np.array([0.5, 0.0, 0.021]),
            orientation=np.array([1, 0, 0, 0]),
            visible=False,
        )

        # just for visualizen what the main prim is
        VisualCuboid(
            prim_path="/Visuals/main_area",
            size=0.02,
            position=np.array([0.5, 0.0, -0.005]),
            color=np.array([255.0, 0.0, 0.0]),
        )

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # MARK: pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self._ik_controller.set_command(self.ik_commands)

    def _apply_action(self):
        # obtain quantities from simulation
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]
        ee_pose_w = self._robot.data.body_pose_w[:, self._body_tcp_idx]
        root_pose_w = self._robot.data.root_pose_w
        joint_pos = self._robot.data.joint_pos[:, :]

        # compute ee frame in root frame
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )

        # apply ee offset
        ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pos_b, ee_quat_b, self._ee_pos_offset, self._ee_rot_offset
        )

        # compute the joint commands
        joint_pos_des = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

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

        # set goal pos
        if self.goal_prim_view is not None:
            # move goal back to main area
            goal_pos = self.start_pose[:, :3]
            goal_pos[:, 2] += 0.001

            goal_orient = torch.tensor([[1, 0, 0, 0]], device=self.device)
            self.goal_prim_view.set_world_poses(positions=goal_pos, orientations=goal_orient)

        # reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]

        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # MARK: observations
    def _get_observations(self) -> dict:
        pass


def run_simulator(env: ShapeTouchEnv):
    """Runs the simulation loop."""
    # for convenience, we directly turn on debug_vis
    if env.cfg.gsmini.debug_vis:
        for data_type in env.cfg.gsmini.data_types:
            env.gsmini._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)

    # for saving tactile images
    # timestamp = f"{datetime.datetime.now():%Y-%m-%d-%H_%M_%S}"
    # output_dir = Path(__file__).parent.resolve()
    # file_name = str(output_dir) + f"/envs_{env.num_envs}_{timestamp}.txt"

    print(f"Starting simulation with {env.num_envs} envs")

    env.reset()
    env.goal_prim_view = XFormPrim(prim_paths_expr="/Goal", name="Goal", usd=True)

    # Simulation loop
    while simulation_app.is_running():
        positions, orientations = env.goal_prim_view.get_world_poses()
        env.ik_commands[:, :3] = positions - env.scene.env_origins
        env.ik_commands[:, 3:] = orientations

        # perform physics step
        env._pre_physics_step(None)
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)

        # env.ik_commands = env.start_pose

        # update isaac buffers() -> also updates sensors
        env.scene.update(dt=env.physics_dt)
        # render scene for cameras (used by sensor)
        env.sim.render()

    env.close()

    pynvml.nvmlShutdown()


def main():
    """Main function."""
    # Define simulation env
    env_cfg = ShapeTouchEnvCfg()
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.gsmini.debug_vis = args_cli.debug_vis

    experiment = ShapeTouchEnv(env_cfg)

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
