from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Ball rolling experiment with a Franka, which is equipped with one GelSight Mini Sensor.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments

args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import traceback

import carb

import omni
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.managers import SceneEntityCfg

from isaaclab.utils import configclass
from isaaclab.utils.math import transform_points, sample_uniform
import isaaclab.utils.math as orbit_math

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
import isaaclab.utils.math as isaac_lab_math

from pxr import UsdGeom, Usd, Sdf, PhysxSchema, UsdPhysics, Gf, UsdShade, Vt
from isaacsim.util.debug_draw import _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()

from isaacsim.core.prims import XFormPrim

###! 
import isaacsim.core.utils.bounds as bounds_utils
from isaacsim.core.utils.collisions import ray_cast
#from isaacsim.core.utils.bounds import create_bbox_cache, compute_aabb
from omni.physx import get_physx_cooking_interface, get_physx_interface, get_physx_scene_query_interface

import numpy as np
import torch
import warp as wp
import cv2
import time


from tacex_assets.robots.franka.franka_gsmini_single_adapter_rigid import FRANKA_PANDA_ARM_GSMINI_SINGLE_ADAPTER_HIGH_PD_CFG
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.sensors.gelsight_mini.gelsight_mini_cfg import GelSightMiniCfg

from tacex import GelSightSensor

@configclass
class BallRollingSceneCfg(InteractiveSceneCfg):
    """Configuration for the Ball rolling scene"""
    
    # Ground-plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]), #pos=[0, 0, -1.05]
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # plate
    plate = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/plate",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0]),
        spawn=sim_utils.UsdFileCfg(usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/plate.usd"),
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.65, 0.0, 0.015]),
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

    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_SINGLE_ADAPTER_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")         
    
    gsmini = GelSightMiniCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gelsight_mini_case",
        sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix = "/Camera",
            update_period= 0,
            resolution = (480, 640),
            data_types = ["depth"],
            clipping_range = (0.024, 0.034),
        ),
        tactile_img_res = (480, 640),
        debug_vis=True # for being able to see sensor output in the gui
    )

def movement_pattern(ball_position, sim_device, ball_radius=0.01, gel_length=20.75/1000, gel_width=25.25/1000, gel_height=4.25/1000):
    """ Computes the goal positions for the ee of the Franka arm, so that the ball is being rolled around.
        The goal positions are computed based on the current position of the ball and the gel dimensions.

        The method returns a list with 10 tensors of the shape (num_envs x 7).
        Each of these 10 (10, because our pattern consists of 9 actions) tensors stands for one action (e.g. "move above the ball").
        
    Args:
        ball_pose (_type_): torch tensor containing the ball position for each environment.
        ball_radius (float, optional): The radius of the ball. Defaults to 0.01.
        gel_length (_type_, optional): Length of the gel. Defaults to 20.75/1000.
        gel_width (_type_, optional): Width of the gel. Defaults to 25.25/1000.
        gel_height (_type_, optional): The height of the gel. Defaults to 4.25/1000.
    """
    z_offset = ball_radius - gel_height/4 # offset of the ee -> ball should be inside the gel
                                          # goal ee position in z is then: z coo. of ball center + z_offset
    
    #! need to define the pose in the robots base frame -> i.e. frame of robots base link, in our case it aligns with the base of the local env
    # set the rotation of ee to always be (0,1,0,0)
    rot = torch.zeros((ball_position.shape[0], 4), device=sim_device) + torch.tensor([0,1,0,0],device=sim_device) 

    ee_pose = torch.cat((ball_position.clone(), rot), 1) 
    above_ball = ee_pose + torch.tensor([0,0,ball_radius + 3/2*gel_height,0,0,0,0],device=sim_device) 
    on_top = ee_pose + torch.tensor([0,0,z_offset,0,0,0,0],device=sim_device)
    backward = ee_pose + torch.tensor([-gel_length/2,0,z_offset,0,0,0,0],device=sim_device)
    forward = ee_pose + torch.tensor([gel_length,0,z_offset,0,0,0,0],device=sim_device)
    left = ee_pose + torch.tensor([0,gel_width/2,z_offset,0,0,0,0],device=sim_device)
    right = ee_pose + torch.tensor([0,-gel_width,z_offset,0,0,0,0],device=sim_device)
    back_to_center = ee_pose + torch.tensor([0,+gel_width/2,z_offset,0,0,0,0],device=sim_device)
    pose_for_reset = ee_pose + torch.tensor([0,0,ball_radius + 4*gel_height,0,0,0,0],device=sim_device) 

    ee_goals = [
        above_ball, # first, place ee above ball 
        on_top, # then ee on top of the ball, so that there is contact
        on_top,
        backward, # move ee backwards
        forward,    
        on_top, # back to the center
        left, # move ee to the left
        right,
        back_to_center, # back to the center by going a little bit more to the left (observed that this results in prettier rolling motion)
        pose_for_reset # move ee over ball again, to prevent that the ball gets thrown around when the scene is resetted (not sure why this happens tho, I think cause the ball spawns directly where the ee is)
    ]
    return ee_goals

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    #Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    frame_marker_cfg.markers["frame"].visible = False
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the end effector of the franka arm
    ball_position = scene["ball"].data.root_pos_w[:, :3] - scene.env_origins
    ee_goals = movement_pattern(ball_position, sim.device, ball_radius=0.005)

    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=scene["robot"].device) # hmm, I cannot rely on IntelliSense anymore, when I use interactive scense, right?
    ik_commands[:] = ee_goals[current_goal_idx]

    # specify robot-specific parameters
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand", "gelsight_mini_case", "gelsight_mini_gelpad"]) # body = robot end-effector  #gelsight_mini_case #gelsight_mini_gel_pad
    # resolving scene entities
    robot_entity_cfg.resolve(scene)
    
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index.
    # This is because the root body is not included in the returned Jacobians.
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1

    # gelpad rigid body
    case_idx = robot_entity_cfg.body_ids[1] - 1
    #gelpad_idx = robot_entity_cfg.body_ids[2] - 1

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0 #! test
    
    #! for time measurements
    frame_times_physics = []
    frame_times_tactile = []
    num_resets = 0
    file_name = "rigid_ball_rolling_tactile_perf.txt"

    goal_change_num_step = 50
    print(f"Starting simulation with {scene.num_envs} envs")
    print("Number of steps till reset: ", len(ee_goals)*goal_change_num_step*2)
    # Simulation loop
    while simulation_app.is_running():
        # reset at the beginning of the simulation and after doing pattern 2 times
        if count % (len(ee_goals)*goal_change_num_step*2) == 0:   # reset after 900 steps, cause every 50 steps we change action -> pattern once = 450 steps
            print(f"[INFO]: Reset number {num_resets}...")

            stage = omni.usd.get_context().get_stage() 
            # reset time
            count = 0
            # print("Robot bodies ", scene["robot"].body_names)
            # reset robot joint state 
            joint_pos = scene["robot"].data.default_joint_pos.clone()
            joint_vel = scene["robot"].data.default_joint_vel.clone()
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()

            scene["gsmini"].reset()

            # reset ball states 
            ball_root_state = scene["ball"].data.default_root_state.clone()
            ball_root_state[:, :3] += scene.env_origins #? why do I need to add the env_origins here, but when I compute the desired ee pos, I need to subtract them?
            scene["ball"].write_root_state_to_sim(ball_root_state)

            # reset soft gel
            # deformableView

            # reset goal
            current_goal_idx = 0
            #print("Changed goal to: ", current_goal_idx)
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)

            if len(frame_times_physics) != 0:
                print("num_envs: ", scene.num_envs)
                print("Total amount of 'in-contact' frames per env: ", len(frame_times_physics))
                print("avg physics_sim time per env:    {:8.4f}ms".format(np.mean(np.array(frame_times_physics)/scene.num_envs)))
                print("avg tactile_sim time per env:    {:8.4f}ms".format(np.mean(np.array(frame_times_tactile)/scene.num_envs)))
                print("")

                #! write down the average simulation times
                if num_resets==4:
                    print("writing performance data into ", file_name)

                    with open(file_name, "a+") as f:
                        f.write(f"num_envs: {scene.num_envs} \n")
                        f.write(f"Total amount of 'in-contact' frames per env (ran pattern {num_resets} times): {len(frame_times_physics)}\n")
                        f.write("avg physics_sim time for one frame per env:    {:8.4f}ms \n".format(np.mean(np.array(frame_times_physics)/scene.num_envs)))
                        f.write("avg tactile_sim time for one frame per env:    {:8.4f}ms \n".format(np.mean(np.array(frame_times_tactile)/scene.num_envs)))
                        f.write("\n")
                    frame_times_physics = []
                    frame_times_tactile = []
                    break
            num_resets += 1

        elif count % goal_change_num_step == 0:
            # update movement pattern according to the ball position
            ball_position = scene["ball"].data.root_pos_w[:, :3] - scene.env_origins
            ee_goals = movement_pattern(ball_position, sim.device, ball_radius=0.005) #movement_pattern(ball_position, sim.device, ball_radius=0.0039)
    
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)

            ik_commands[:] = ee_goals[current_goal_idx]
            # add some randomization 
            ik_commands[:, :2] += sample_uniform(
                -0.005, 
                0.005,
                (scene.num_envs, 2), 
                sim.device
            )
            diff_ik_controller.set_command(ik_commands)
        else:
            # obtain quantities from simulation for IK controller 
            jacobian = scene["robot"].root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_offset = torch.tensor([0, 0, 0.13, 0, 0, 0, 0], device=sim.device).repeat(scene.num_envs, 1)
            ee_pose_w = scene["robot"].data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7] - ee_offset  # body_ids[0] is the ee
            root_pose_w = scene["robot"].data.root_link_state_w[:, 0:7]
            joint_pos = scene["robot"].data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            # # apply rotation 
            # if current_goal_idx == 2:
            #     joint_pos_des[:, 6] = -joint_pos_des[:, 6] # hand 
            # if current_goal_idx == 3:
            #     joint_pos_des[:, 6] = -joint_pos_des[:, 6] 

        # apply actions
        scene["robot"].set_joint_position_target(joint_pos_des, robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()

        physics_start = time.time()

        # # # perform physics step
        sim.step(render=False)
        # update isaac buffers()
        scene.update(sim_dt)

        physics_end = time.time()
        # print("physics_sim time:    {:8.4f}ms".format(1000 * (physics_end - physics_start)))

        count += 1
        # update sensors
        tactile_sim_start = time.time()
        gsmini = scene["gsmini"]
        gsmini.update(dt=sim_dt)
        
        # if gsmini._marker_sim:
        #     case_pos = scene["robot"].data.body_state_w[:, robot_entity_cfg.body_ids[1], :3] + 0.024
        #     case_rot = scene["robot"].data.body_state_w[:, robot_entity_cfg.body_ids[1], 3:7] 
        #     object_pos_b, object_rot_b = subtract_frame_transforms(
        #         case_pos,  case_rot, scene["ball"].data.root_pos_w[:, :3].clone(), scene["ball"].data.root_quat_w
        #     )

        #     rot_x, rot_y, rot_z = isaac_lab_math.euler_xyz_from_quat(object_rot_b)
        #     rot_z = isaac_lab_math.wrap_to_pi(rot_z)
        #     gsmini.theta = rot_z
        #     # retrieve tactile data -> triggers tactile sim
        #     marker_motion = gsmini.data.output["marker_motion"]
        # else:
        tactile_rgb = gsmini.data.output["tactile_rgb"]

        tactile_sim_end = time.time()
        # print("tactile_sim time:    {:8.4f}ms".format(1000 * (tactile_sim_end - tactile_sim_start)))

        #- add frame times, if sensor was in contact
        # contact_idx, = torch.where(gsmini.press_depth > 0)
        # if contact_idx.shape[0] != 0:
        #     frame_times_physics.append(1000 * (physics_end - physics_start))
        #     frame_times_tactile.append(1000 * (tactile_sim_end - tactile_sim_start))
        #     # # draw trajectory -> you cannot really see anything significant
        #     # points = ee_pose_w[:, 0:3].cpu().numpy()
        #     # draw.draw_points(points, [(255,0,0,0.5)]*points.shape[0], [15]*points.shape[0])
        # gsmini.update_gui_windows()
        sim.render()

        # update buffers()
        # scene.update(sim_dt)

        # obtain quantities from simulation for markers
        ee_offset = torch.tensor([0, 0, 0.13, 0, 0, 0, 0], device=sim.device).repeat(scene.num_envs, 1)
        ee_pose_w = scene["robot"].data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7] - ee_offset
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = BallRollingSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.05, replicate_physics=False)
    scene = InteractiveScene(scene_cfg) 
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


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

