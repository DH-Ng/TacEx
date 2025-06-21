"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    isaaclab -p ./examples/falling_cubes.py 
"""


"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Test scene for GIPC.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# simulation_app.set_setting("/app/useFabricSceneDelegate", True)
#simulation_app.set_setting("/app/usdrt/scene_delegate/enableProxyCubes", False)
#simulation_app.set_setting("/app/usdrt/scene_delegate/geometryStreaming/enabled", False)
#simulation_app.set_setting("/omnihydra/parallelHydraSprimSync", False)

"""Rest everything follows."""
import pathlib

import omni

from isaacsim.core.prims import XFormPrim
import isaacsim.core.utils.prims as prims_utils
from isaacsim.util.debug_draw import _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.timer import Timer

from pxr import UsdGeom, Usd, Sdf, PhysxSchema, UsdPhysics, Gf, UsdShade, Vt

import numpy as np
import random
import warp as wp

# import vtk
from uipc.core import Engine, World, Scene, SceneIO

from tacex_uipc import UipcSim, UipcSimCfg, UipcObject, UipcObjectCfg
from tacex_uipc.utils import TetMeshCfg

def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_dome = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_dome.func("/World/lightDome", cfg_light_dome, translation=(1, 0, 10))

    # create a new xform prim for all objects to be spawned under
    prims_utils.define_prim("/World/Objects", "Xform")
    

def change_mat_color(stage, shader_prim_path, color):
    # source: https://forums.developer.nvidia.com/t/randomize-materials-and-textures-based-on-a-probability-extract-path-to-material-and-texture-from-usd/270188/4
    shader_prim = stage.GetPrimAtPath(shader_prim_path)
    if not shader_prim.GetAttribute("inputs:diffuse_color_constant").IsValid():
        shader_prim.CreateAttribute("inputs:diffuse_color_constant", Sdf.ValueTypeNames.Color3f, custom=True).Set((0.0, 0.0, 0.0))

    if not shader_prim.GetAttribute("inputs:diffuse_tint").IsValid():
        shader_prim.CreateAttribute("inputs:diffuse_tint", Sdf.ValueTypeNames.Color3f, custom=True).Set((0.0, 0.0, 0.0))

    # Set the diffuse color to the input color
    shader_prim.GetAttribute('inputs:diffuse_color_constant').Set(color)
    shader_prim.GetAttribute('inputs:diffuse_tint').Set(color)
     
def _usd_set_xform(xform, pos: tuple, rot: tuple, scale: tuple):
    from pxr import UsdGeom, Gf

    xform = UsdGeom.Xform(xform)

    xform_ops = xform.GetOrderedXformOps()

    if pos is not None:
        xform_ops[0].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    if rot is not None:
        xform_ops[1].Set(Gf.Quatd(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2])))
    if scale is not None:
        xform_ops[2].Set(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])))

# Probably useful when we need to update multiple bodies -> look into the warp render code
# def update_body_transforms(self, body_q):
#     from pxr import UsdGeom, Sdf
#     if isinstance(body_q, wp.array):
#         body_q = body_q.numpy()
#     with Sdf.ChangeBlock():
#         for b in range(self.model.body_count):
#             node_name = self.body_names[b]
#             node = UsdGeom.Xform(self.stage.GetPrimAtPath(self.root.GetPath().AppendChild(node_name)))
#             # unpack rigid transform
#             X_sb = wp.transform_expand(body_q[b])
#             _usd_set_xform(node, X_sb.p, X_sb.q, (1.0, 1.0, 1.0), self.time)
       
def main():
    """Main function."""

    # Initialize the simulation context
    # render_cfg = sim_utils.RenderCfg(rendering_mode=)
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # Design scene by spawning assets
    design_scene()

    # Initialize uipc sim
    uipc_cfg = UipcSimCfg(
        # logger_level="Info",
        contact=UipcSimCfg.Contact(
            # enable=False,
            d_hat=0.01,
        )
    )
    uipc_sim = UipcSim(uipc_cfg)

    mesh_cfg = TetMeshCfg(
        stop_quality=8,
        max_its=100,
        edge_length_r=0.1,
        # epsilon_r=0.01
    )
    print("Mesh cfg ", mesh_cfg)

    # spawn uipc cube
    #tet_cube_asset_path = "/workspace/tacex/source/tacex_assets/tacex_assets/data/Sensors/GelSight_Mini/Gelpad_low_res.usd"
    tet_cube_asset_path = pathlib.Path(__file__).parent.resolve() / "assets" / "cube.usd"
    cube_cfg = UipcObjectCfg(
        prim_path="/World/Objects/Cube0",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 2.25]), #rot=(0.72,-0.3,0.42,-0.45)
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(tet_cube_asset_path),
            # scale=(0.1, 0.1, 0.1)
        ),
        # mesh_cfg=mesh_cfg,
        constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg() #UipcObjectCfg.AffineBodyConstitutionCfg() #
    )
    cube = UipcObject(cube_cfg, uipc_sim)

    # tet_ball_asset_path = pathlib.Path(__file__).parent.resolve() / "assets" / "ball.usd"
    # ball_cfg = UipcObjectCfg(
    #     prim_path="/World/Objects/ball",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 1.0]), #rot=(0.72,-0.3,0.42,-0.45)
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(tet_ball_asset_path),
    #         scale=(1.0, 1.0, 1.0)
    #     ),
    #     mesh_cfg=mesh_cfg,
    #     constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg()
    # )
    # ball = UipcObject(ball_cfg, uipc_sim)

    tet_cube_asset_path = pathlib.Path(__file__).parent.resolve() / "assets" / "cube.usd"

    num_cubes = 4 #30
    cubes = []
    for i in range(num_cubes):
        if i % 2 == 0:
            constitution_type = UipcObjectCfg.AffineBodyConstitutionCfg()
        else:
            constitution_type = UipcObjectCfg.StableNeoHookeanCfg()
        # might lead to intersections due to random pos
        # pos = (random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(2.5, 6.0))
        pos = (0, 0, 3.0 + 0.3*i)
        rot = (random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
        cube_cfg = UipcObjectCfg(
            prim_path=f"/World/Objects/Cube{i+1}",
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot), #rot=(0.72,-0.3,0.42,-0.45)
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(tet_cube_asset_path),
                scale=(0.15, 0.15, 0.15)
            ),
            constitution_cfg=constitution_type
        )
        cubeX = UipcObject(cube_cfg, uipc_sim)
        cubes.append(cubeX)

    rot = (random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
    cube_cfg = UipcObjectCfg(
        prim_path="/World/Objects/CubeTop",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 3.65 + 0.3*num_cubes],rot=rot),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(tet_cube_asset_path),
            scale=(1.0, 1.0, 1.0)
        ),
        # mesh_cfg=mesh_cfg,
        constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg()#UipcObjectCfg.StableNeoHookeanCfg()
    )
    cube_top = UipcObject(cube_cfg, uipc_sim)

    # Play the simulator
    sim.reset()
    
    # only after Isaac Sim got resetted (= objects init), otherwise wold init is false
    # because _initialize_impl() of the object is called in the sim.reset() method
    # and setup_scene() relies on objects being _intialized_impl()
    uipc_sim.setup_sim()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    step = 1
    start_uipc_test = False

    total_uipc_sim_time = 0.0
    total_uipc_render_time = 0.0
    # Simulate physics
    while simulation_app.is_running():

        # perform step
        sim.step()

        if start_uipc_test:
            # draw the old points
            # draw.clear_points()
            # points = np.array(gipc_sim.sim.get_vertices())
            # draw.draw_points(points, [(255,0,255,0.5)]*points.shape[0], [30]*points.shape[0])
            print("")
            print("====================================================================================")
            print("====================================================================================")
            print("Step number ", step)
            ## if you want to replay frames which were saved via world.dump()-> tet meshes need to be the same as when the frames were saved!
            # if(uipc_sim.world.recover(uipc_sim.world.frame() + 1)):
            #     uipc_sim.replay_frame(uipc_sim.world.frame() + 1)
            # else:
            #     uipc_sim.step()
            with Timer("[INFO]: Time taken for uipc sim step.", name="uipc_step"):
                uipc_sim.step()
                # uipc_sim.save_current_world_state()
            # gipc_sim.render_tet_surface_wireframes(clean_up_first=True)
            with Timer("[INFO]: Time taken for updating the render meshes.", name="render_update"):
                uipc_sim.update_render_meshes()
                #sim.forward()
                # sim._update_fabric(0.0, 0.0)
                # render the updated meshes
                sim.render()
            # get time reports
            # uipc_sim.get_time_report()
            total_uipc_sim_time += Timer.get_timer_info("uipc_step")
            total_uipc_render_time += Timer.get_timer_info("render_update")

            step += 1      
        
        # start UIPC sim after pausing and playing the sim
        if sim.is_playing() is False:
            start_uipc_test = True
            print("Start uipc simulation by pressing Play")

        if step % 250 == 0: #500
            print("")
            print("====================================================================================")
            print("====================================================================================")
            print("Reset simulation")
            if start_uipc_test:
                print("systems offsets ", uipc_sim._system_vertex_offsets)
                cube.write_vertex_positions_to_sim(vertex_positions=cube.init_vertex_pos)
                cube_top.write_vertex_positions_to_sim(vertex_positions=cube_top.init_vertex_pos)

                small_cube_id = random.randint(0, num_cubes-1)
                cubes[small_cube_id].write_vertex_positions_to_sim(vertex_positions=cubes[small_cube_id].init_vertex_pos)

                uipc_sim.reset()
                # start_uipc_test = False
                # draw.clear_points()
                # draw.clear_lines()
                sim.render()
            avg_uipc_step_time = total_uipc_sim_time/step
            print(f"Sim step for uipc took in avg {avg_uipc_step_time} per frame.")
            
            avg_uipc_render_time = total_uipc_render_time/step
            print(f"Render update for uipc took in avg {avg_uipc_render_time} per frame.")
            print("====================================================================================")

            step = 1
          
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()