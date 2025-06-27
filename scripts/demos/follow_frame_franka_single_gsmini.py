from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Control Franka, which is equipped with one GelSight Mini Sensor, by moving the Frame in the GUI")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--sys", type=bool, default=True, help="Whether to track system utilization.")
parser.add_argument("--debug_vis", default=True, action="store_true", help="Whether to render tactile images in the# append AppLauncher cli args")
AppLauncher.add_app_launcher_args(parser)
# parse the arguments

args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import traceback
import carb

import torch
import numpy as np

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from isaacsim.core.prims import XFormPrim
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

from tacex_assets.robots.franka.franka_gsmini_single_adapter_rigid import FRANKA_PANDA_ARM_GSMINI_SINGLE_ADAPTER_HIGH_PD_CFG
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.sensors.gelsight_mini.gelsight_mini_cfg import GelSightMiniCfg

from tacex import GelSightSensor, GelSightSensorCfg
from tacex.simulation_approaches.gpu_taxim import TaximSimulatorCfg

import time
import psutil
import pynvml
import datetime
from pathlib import Path
import json
import platform

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
class BallRollingEnvCfg(DirectRLEnvCfg):

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
        dt=1/60, 
        render_interval=decimation,
        physx=PhysxCfg(
            enable_ccd=True, # needed for more stable ball_rolling
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
        lazy_sensor_update=True, # only update sensors when they are accessed
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/plate.usd",
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                kinematic_enabled=True
            )
        )
    )

    ball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.015]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/ball_wood.usd",
            #scale=(0.8, 0.8, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                kinematic_enabled=False,
                disable_gravity=False,
            )
        )
    )

    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_SINGLE_ADAPTER_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        # init_state=ArticulationCfg.InitialStateCfg(
        #     # joint_pos={
        #     #     "panda_joint1": 0.0,
        #     #     "panda_joint2": 0.43,
        #     #     "panda_joint3": 0.0,
        #     #     "panda_joint4": -2.37,
        #     #     "panda_joint5": 0.0,
        #     #     "panda_joint6": 2.79,
        #     #     "panda_joint7": 0.741,
        #     # },
        # ),
    )
    gsmini = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
        sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix = "/Camera",
            update_period= 0,
            resolution = (32,32), #(120, 160),
            data_types = ["depth"],
            clipping_range = (0.024, 0.034),
        ),
        device = "cuda",
        debug_vis=True, # for rendering sensor output in the gui
        # update Taxim cfg
        marker_motion_sim_cfg=None,
        data_types=["tactile_rgb"], #marker_motion
    )
    # settings for optical sim
    gsmini.optical_sim_cfg = gsmini.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(32, 32),
    )

    # frame for setting goal position
    frame = AssetBaseCfg(
        prim_path="/World/envs/env_.*/goal_frame",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.15], rot=[0, 1, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.025, 0.025, 0.025),
        ),
    )    

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")

    ball_radius = 0.005

    obj_pos_randomization_range = [-0.15, 0.15]

    # some filler values, needed for DirectRLEnv     
    episode_length_s = 0
    action_space = 0 
    observation_space = 0 
    state_space = 0

class BallRollingEnv(DirectRLEnv):
    cfg: BallRollingEnvCfg

    def __init__(self, cfg: BallRollingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        #### Stuff for IK ##################################################
        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        # Obtain the frame index of the end-effector
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        # save only the first body index
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        
        # For a fixed base robot, the frame index is one less than the body index.
        # This is because the root body is not included in the returned Jacobians.
        self._jacobi_body_idx = self._body_idx - 1
        # self._jacobi_joint_ids = self._joint_ids # we take every joint
        
        # ee offset w.r.t panda hand -> based on the asset
        self._offset_pos = torch.tensor([0.0, 0.0, 0.131], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)  
        ####################################################################

        # create buffer to store actions (= ik_commands)
        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        # self.ik_commands[:, 3:] = torch.tensor([0,1,0,0],device=self.device) 

        self.step_count = 0

        self.frame_prim_view = None

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
        
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.object = RigidObject(self.cfg.ball)
        self.scene.rigid_objects["object"] = self.object

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
                        pos=(0.0, 0.0, 0.131), #0.1034
                    ),
                ),
            ],
        )
        
        # sensors
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        self.gsmini = GelSightSensor(self.cfg.gsmini)
        self.scene.sensors["gsmini"] = self.gsmini

        plate = RigidObject(self.cfg.plate)

        # Spawn AssetBase objects manually
        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path,
            ground.spawn,
            translation=ground.init_state.pos,
            orientation=ground.init_state.rot
        )

        frame = self.cfg.frame
        frame.spawn.func(
            frame.prim_path,
            frame.spawn,
            translation=frame.init_state.pos,
            orientation=frame.init_state.rot
        )

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    #MARK: pre-physics step calls
        
    def _pre_physics_step(self, actions: torch.Tensor):
        # update movement pattern according to the ball position
        ball_pos = self.object.data.root_link_pos_w - self.scene.env_origins

        self._ik_controller.set_command(self.ik_commands)

    def _apply_action(self):
        # obtain quantities from simulation
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]
        
        # compute the delta in joint-space
        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        self._robot.set_joint_position_target(joint_pos_des)

        self.step_count += 1

    # post-physics step calls    

    #MARK: dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]: # which environment is done
        pass
    
    #MARK: rewards
    def _get_rewards(self) -> torch.Tensor:        
        pass
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # spawn robot at random position
        obj_pos = self.object.data.default_root_state[env_ids] 
        obj_pos[:, :3] += self.scene.env_origins[env_ids]
        # obj_pos[:, :2] += sample_uniform(
        #     self.cfg.obj_pos_randomization_range[0], 
        #     self.cfg.obj_pos_randomization_range[1],
        #     (len(env_ids), 2), 
        #     self.device
        # )
        self.object.write_root_state_to_sim(obj_pos, env_ids=env_ids)

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
        
    #MARK: observations
    def _get_observations(self) -> dict:
        pass
        
    ####
    ## Helper Functions
    ####

    ################################# For IK
    #  From task_space_actions.py
    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]

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
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        # account for the offset
        #if self.cfg.body_offset is not None:
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
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
        #if self.cfg.body_offset is not None:
        # Modify the jacobian to account for the offset
        # -- translational part
        # v_link = v_ee + w_ee x r_link_ee = v_J_ee * q + w_J_ee * q x r_link_ee
        #        = (v_J_ee + w_J_ee x r_link_ee ) * q
        #        = (v_J_ee - r_link_ee_[x] @ w_J_ee) * q
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        # -- rotational part
        # w_link = R_link_ee @ w_ee
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])

        return jacobian
    
"""
System diagnosis
-> adapted from benchmark_cameras.py script of IsaacLab
"""
def _get_utilization_percentages(reset: bool = False, max_values: list[float] = [0.0, 0.0, 0.0, 0.0]) -> list[float]:
    """Get the maximum CPU, RAM, GPU utilization (processing), and
    GPU memory usage percentages since the last time reset was true."""
    if reset:
        max_values[:] = [0, 0, 0, 0]  # Reset the max values
    
    # # CPU utilization
    #cpu_usage = psutil.cpu_percent(interval=0.1) # blocking slows down Isaac Sim a lot
    cpu_usage = psutil.cpu_percent(interval=None)
    max_values[0] = max(max_values[0], cpu_usage)

    # # RAM utilization
    memory_info = psutil.virtual_memory()
    ram_usage = memory_info.percent
    max_values[1] = max(max_values[1], ram_usage)

    #GPU utilization using pynvml
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # GPU Utilization
            gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_processing_utilization_percent = gpu_utilization.gpu  # GPU core utilization
            max_values[2] = max(max_values[2], gpu_processing_utilization_percent)

            # GPU Memory Usage
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_total = memory_info.total
            gpu_memory_used = memory_info.used
            gpu_memory_utilization_percent = (gpu_memory_used / gpu_memory_total) * 100
            max_values[3] = max(max_values[3], gpu_memory_utilization_percent)
    else:
        gpu_processing_utilization_percent = None
        gpu_memory_utilization_percent = None
    
    return max_values


def run_simulator(env: BallRollingEnv):
    """Runs the simulation loop."""

    #! for time measurements
    timestamp = "{:%Y-%m-%d-%H_%M_%S}".format(datetime.datetime.now())
    output_dir = Path(__file__).parent.resolve()
    file_name = str(output_dir) + f"/envs_{env.num_envs}_{timestamp}.txt"

    frame_times_physics = []
    frame_times_tactile = []

    save_after_num_resets = 3
    current_num_resets = 0
    # GPU utilization using pynvml
    if torch.cuda.is_available():
        pynvml.nvmlInit()
    system_utilization_analytics = _get_utilization_percentages(reset=False)

    print(f"Starting simulation with {env.num_envs} envs")

    total_sim_time = time.time()

    env.reset()

    env.frame_prim_view = XFormPrim(prim_paths_expr=env.cfg.frame.prim_path, name=f"{env.cfg.frame.prim_path}", usd=True)
        
    # Simulation loop
    while simulation_app.is_running():

        # perform physics step
        physics_start = time.time()
        env._pre_physics_step(None)
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        physics_end = time.time()
        ###

        positions, orientations = env.frame_prim_view.get_world_poses() 
        env.ik_commands[:, :3] = positions - env.scene.env_origins
        env.ik_commands[:, 3:] = orientations

        # update isaac buffers() -> also updates sensors
        env.scene.update(dt=env.physics_dt)
        # render scene for cameras (used by sensor)
        env.sim.render()

        ### update sensors again to measure tactile sim time separately
        tactile_sim_start = time.time()
        env.gsmini.update(dt=env.physics_dt, force_recompute=True)
        # tactile_rgb = env.gsmini.data.output["tactile_rgb"]
        tactile_sim_end = time.time()
        ###

        #- add frame times, if sensor was in contact
        # #todo currently assumed that every env was in contact. Need to filter out the envs where no contact
        # contact_idx, = torch.where(env.gsmini._indentation_depth > 0)
        # if contact_idx.shape[0] != 0:
        #     frame_times_physics.append(1000 * (physics_end - physics_start))
        #     frame_times_tactile.append(1000 * (tactile_sim_end - tactile_sim_start))
        
        # print("Current total amount of 'in-contact' frames per env: ", len(frame_times_physics))
        # print("Total sim time currently: {:8.4f}ms".format(time.time()-total_sim_time))
        # print("Avg physics_sim time per env:    {:8.4f}ms".format(np.mean(np.array(frame_times_physics)/env.num_envs)))
        # print("Avg tactile_sim time per env:    {:8.4f}ms".format(np.mean(np.array(frame_times_tactile)/env.num_envs)))
        # system_utilization_analytics = _get_utilization_percentages(reset=False)
        # print(
        #     f"| CPU:{system_utilization_analytics[0]}% | "
        #     f"RAM:{system_utilization_analytics[1]}% | "
        #     f"GPU Compute:{system_utilization_analytics[2]}% | "
        #     f"GPU Memory: {system_utilization_analytics[3]:.2f}% |"
        # )
        # print("")

    env.close()
    
    pynvml.nvmlShutdown() 

def main():
    """Main function."""
    # Define simulation env
    env_cfg = BallRollingEnvCfg()
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.gsmini.debug_vis = args_cli.debug_vis 

    experiment = BallRollingEnv(env_cfg)

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(env = experiment)

if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim appy
        simulation_app.close()
