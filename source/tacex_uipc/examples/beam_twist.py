from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Control Franka, which is equipped with one GelSight Mini Sensor, by moving the Frame in the GUI"
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

import carb
import pynvml
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.prims import XFormPrim

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from tacex import GelSightSensor

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_gripper_uipc import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_UIPC_CFG
from tacex_assets.sensors.gelsight_mini import GELSIGHT_MINI_TAXIM_CFG

from tacex_uipc import (
    UipcIsaacAttachmentsCfg,
    UipcRLEnv,
    UipcSimCfg,
)
from tacex_uipc.objects import UipcDeformableObject, UipcDeformableObjectCfg
from tacex_uipc.utils import TetMeshCfg


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
class BeamTwistEnvCfg(DirectRLEnvCfg):
    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.9, 1.4, 0.3)
    viewer.lookat = (-1.5, -1.9, -1.1)

    debug_vis = True

    ui_window_class_type = CustomEnvWindow

    decimation = 1
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            enable_ccd=True,  # for more stable ball_rolling
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.0025)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/plate.usd",
            scale=(0.05, 0.05, 10),  # shape should be a cuboid
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # -- Configs for GsMinis
    gsmini_left = GELSIGHT_MINI_TAXIM_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
        sensor_camera_cfg=GELSIGHT_MINI_TAXIM_CFG.SensorCameraCfg(
            prim_name="Camera",
            update_period=0,
            resolution=(32, 32),
            data_types=["depth"],
            clipping_range_m=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=True,  # for rendering sensor output in the gui
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
            resolution=(32, 32),
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

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")

    ball_radius = 0.005

    obj_pos_randomization_range = [-0.15, 0.15]

    # some filler values, needed for DirectRLEnv
    episode_length_s = 0
    action_space = 0
    observation_space = 0
    state_space = 0

    # -- Confgis for UIPC simulation
    uipc_sim = UipcSimCfg(
        # logger_level="Info"
        ground_height=0.0025,
        contact=UipcSimCfg.Contact(d_hat=0.0001, default_friction_ratio=2.5, default_contact_resistance=25.0),
        debug_vis=False,
    )

    # simulate the gelpads as uipc mesh
    gel_mesh_cfg = TetMeshCfg(
        stop_quality=8,
        max_its=100,
        edge_length_r=1 / 15,
        # epsilon_r=0.01
    )
    gelpad_left_cfg = UipcDeformableObjectCfg(
        prim_path="/World/envs/env_.*/Robot/gelpad_left",
        mesh_cfg=gel_mesh_cfg,
        constitution_cfg=UipcDeformableObjectCfg.StableNeoHookeanCfg(youngs_modulus=1),
        constraint_cfg=UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=100.0,
            body_name="gelsight_mini_case_left",
            # debug_vis=debug_vis,
            compute_attachment_data=True,
            isaaclab_rigid_body_prim_path="/World/envs/env_.*/Robot",
        ),
    )
    gelpad_right_cfg = UipcDeformableObjectCfg(
        prim_path="/World/envs/env_.*/Robot/gelpad_right",
        mesh_cfg=gel_mesh_cfg,
        constitution_cfg=UipcDeformableObjectCfg.StableNeoHookeanCfg(youngs_modulus=1),
        constraint_cfg=UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=100.0,
            body_name="gelsight_mini_case_right",
            # debug_vis=debug_vis,
            compute_attachment_data=True,
            isaaclab_rigid_body_prim_path="/World/envs/env_.*/Robot",
        ),
    )

    beam = UipcDeformableObjectCfg(
        prim_path="/World/envs/env_.*/beam",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.005]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/beam.usd",
        ),
        mesh_cfg=TetMeshCfg(
            stop_quality=8,
            max_its=250,
            edge_length_r=1 / 20,
            epsilon_r=1e-3,
        ),
        usd_mesh_prim_name="beam",
        constitution_cfg=UipcDeformableObjectCfg.StableNeoHookeanCfg(youngs_modulus=0.005),
        constraint_cfg=UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=100.0,
            debug_vis=False,
            compute_attachment_data=True,
            isaaclab_rigid_body_prim_path="/World/envs/env_.*/ground_plate",
        ),
        debug_vis=True,
        debug_deformation_vis=True,
    )


class BeamTwist(UipcRLEnv):
    cfg: BeamTwistEnvCfg

    def __init__(self, cfg: BeamTwistEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- for IK ---
        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        # Obtain the frame index of the end-effector
        body_ids, _ = self._robot.find_bodies("panda_hand")
        # save only the first body index
        self._body_tcp_idx = body_ids[0]

        # Index of fingers -> first id is left, second id is right finger
        self._finger_joint_ids, self._finger_joint_names = self._robot.find_joints(["panda_finger.*"])

        # For a fixed base robot, the frame index is one less than the body index.
        # This is because the root body is not included in the returned Jacobians.
        self._jacobi_body_idx = self._body_tcp_idx - 1

        # ee offset w.r.t panda hand -> based on the asset
        self._ee_pos_offset = torch.tensor([0.0, 0.0, 0.131], device=self.device).repeat(self.num_envs, 1)
        self._ee_rot_offset = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # create buffer to store actions (= ik_commands)
        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)

        # ---
        self.step_count = 0

        self.goal_prim_view = None

        self.set_debug_vis(self.cfg.debug_vis)

        self._left_finger_pos = 0.04
        self._right_finger_pos = 0.04

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
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
                    prim_path="/World/envs/env_.*/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.131),
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

        plate = RigidObject(self.cfg.plate)

        # Spawn AssetBase objects manually
        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )

        VisualCuboid(
            prim_path="/Goal",
            size=0.01,
            position=np.array([0.5, 0.0, 0.25]),
            orientation=np.array([0, 0.0, -1.0, 0.0]),
            color=np.array([255.0, 0.0, 0.0]),
        )

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # --- UIPC simulation setup ---
        # gelpad simulated via uipc
        self._uipc_gelpad_left = UipcDeformableObject(self.cfg.gelpad_left_cfg, self.uipc_sim)
        self._uipc_gelpad_right = UipcDeformableObject(self.cfg.gelpad_right_cfg, self.uipc_sim)

        # set rigid object for attachment-constraint
        self._uipc_gelpad_left.constraint.isaaclab_rigid_object = self.scene.articulations["robot"]
        self._uipc_gelpad_right.constraint.isaaclab_rigid_object = self.scene.articulations["robot"]

        self.beam = UipcDeformableObject(self.cfg.beam, self.uipc_sim)

        self.beam.constraint.isaaclab_rigid_object = plate

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

        # set finger position -> only have 1 robot
        joint_pos_des[:, self._finger_joint_ids[0]] = self._left_finger_pos
        joint_pos_des[:, self._finger_joint_ids[1]] = self._right_finger_pos

        self._robot.set_joint_position_target(joint_pos_des)

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
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # reset uipc objects
        self._uipc_gelpad_left.write_nodal_pos_to_sim(self._uipc_gelpad_left.data.default_nodal_state_w[:, :, :3])
        self._uipc_gelpad_right.write_nodal_pos_to_sim(self._uipc_gelpad_right.data.default_nodal_state_w[:, :, :3])
        self.beam.write_nodal_pos_to_sim(self.beam.data.default_nodal_state_w[:, :, :3])

        self.step_count = 0
        self._left_finger_pos = 0.04
        self._right_finger_pos = 0.04

        if self.goal_prim_view is not None:
            self.goal_prim_view.set_world_poses(
                positions=torch.tensor([0.5, 0.0, 0.25], device=self.device).repeat(self.num_envs, 1),
                orientations=torch.tensor([0, 0.0, -1.0, 0.0], device=self.device).repeat(self.num_envs, 1),
            )

    # MARK: observations
    def _get_observations(self) -> dict:
        pass


def run_simulator(env: BeamTwist):
    """Runs the simulation loop."""

    print(f"Starting simulation with {env.num_envs} envs")
    env.reset()

    env.goal_prim_view = XFormPrim(prim_paths_expr="/Goal", name="Goal", usd=True)

    # Simulation loop
    while simulation_app.is_running():
        env._pre_physics_step(None)
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.uipc_sim.update_render_meshes()
        env.scene.update(dt=env.physics_dt)

        # render scene
        env.sim.render()

        if env.step_count == 50:
            # move gripper down
            print("Moving ee down.")
            env.goal_prim_view.set_world_poses(
                positions=torch.tensor([0.5, 0.0, 0.15], device=env.device).repeat(env.num_envs, 1)
            )

        if env.step_count == 70:
            # close gripper
            env._left_finger_pos = 0.0
            env._right_finger_pos = 0.0

        if env.step_count == 100:
            # twist
            print("Twisting Beam!!!")
            env.goal_prim_view.set_world_poses(
                orientations=torch.tensor([0, -1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1),
            )

        if env.step_count == 200:
            # move gripper up
            print("Moving ee up.")
            env.goal_prim_view.set_world_poses(
                positions=torch.tensor([0.5, 0.0, 0.25], device=env.device).repeat(env.num_envs, 1)
            )

        if env.step_count == 500:
            print("Reset")
            print("")
            # open gripper
            env._left_finger_pos = 0.04
            env._right_finger_pos = 0.04

            for _ in range(50):
                env._pre_physics_step(None)
                env._apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.uipc_sim.update_render_meshes()
                env.scene.update(dt=env.physics_dt)
                env.sim.render()

            env.reset()

        positions, orientations = env.goal_prim_view.get_world_poses()
        env.ik_commands[:, :3] = positions - env.scene.env_origins
        env.ik_commands[:, 3:] = orientations
    env.close()

    pynvml.nvmlShutdown()


def main():
    """Main function."""
    # Define simulation env
    env_cfg = BeamTwistEnvCfg()
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.gsmini_left.debug_vis = args_cli.debug_vis
    env_cfg.gsmini_right.debug_vis = args_cli.debug_vis

    experiment = BeamTwist(env_cfg)

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
