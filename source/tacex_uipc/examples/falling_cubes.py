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

simulation_app.set_setting("/app/useFabricSceneDelegate", True)
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

from pxr import UsdGeom, Usd, Sdf, PhysxSchema, UsdPhysics, Gf, UsdShade, Vt

import numpy as np
import random
import warp as wp

# import vtk
from uipc.core import Engine, World, Scene, SceneIO

from tacex_uipc import UipcSim, UipcSimCfg, UipcObject, UipcObjectCfg

def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_dome = sim_utils.DomeLightCfg(
        intensity=1000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_dome.func("/World/lightDome", cfg_light_dome, translation=(1, 0, 10))

    # create a new xform prim for all objects to be spawned under
    prims_utils.define_prim("/World/Objects", "Xform")
    
    # cube_big_cfg = sim_utils.UsdFileCfg(usd_path="/workspace/isaaclab/data_storage/TacEx/examples/assets/cube_big.usd")

    # cube_big_cfg.func(f"/World/Objects/Big_cube", cube_big_cfg, translation=(0.0, 0, 0.5 + 0.05 + (amount_cubes/10))) # make sure that big cube is at the top
    # #cube_big_cfg.func(f"/World/Objects/Big_cube_x", cube_big_cfg, translation=(1, 0, 0.5 + 0.05 + (amount_cubes/10))) # make sure that big cube is at the top
    #cube_big_cfg.func(f"/World/Objects/Big_cube_y", cube_big_cfg, translation=(0, 1, 0.5 + 0.05 + (amount_cubes/10))) # make sure that big cube is at the top
   
    # # spawn a usd file of a table into the scene
    # cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    # cfg.func("/World/Objects/Table", cfg, translation=(0.0, 0.0, 1.05))


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
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # Design scene by spawning assets
    design_scene()

    # Initialize uipc sim
    uipc_cfg = UipcSimCfg()
    uipc_sim = UipcSim(uipc_cfg)

    # spawn uipc cube
    tet_cube_asset_path = pathlib.Path(__file__).parent.resolve() / "assets" / "cube.usd"
    cube_cfg = UipcObjectCfg(
        prim_path="/World/Objects/Cube0",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 1.0], rot=(0.72,-0.3,0.42,-0.45)), #rot=(0.72,-0.3,0.42,-0.45)
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(tet_cube_asset_path),
            scale=(1.0, 1.0, 1.0)
        ),
    )
    cube = UipcObject(cube_cfg, uipc_sim)

    num_cubes = 23
    cubes = []
    for i in range(num_cubes):
        # might lead to intersections due to random pos
        pos = (random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(2.5, 6.0))
        rot = (random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
        cube_cfg = UipcObjectCfg(
            prim_path=f"/World/Objects/Cube{i+1}",
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot), #rot=(0.72,-0.3,0.42,-0.45)
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(tet_cube_asset_path),
                scale=(0.15, 0.15, 0.15)
            ),
        )
        cubeX = UipcObject(cube_cfg, uipc_sim)
        cubes.append(cubeX)

    # Play the simulator
    sim.reset()
    
    # only after Isaac Sim got resetted (= objects init), otherwise wold init is false
    # because _initialize_impl() of the object is called in the sim.reset() method
    # and setup_scene() relies on objects being _intialized_impl()
    uipc_sim.setup_scene()

    # sio = SceneIO(uipc_sim.scene)

    # Now we are ready!
    print("[INFO]: Setup complete...")

    step = 1
    start_uipc_test = False

    # Simulate physics
    while simulation_app.is_running():

        # perform step
        sim.step()
        if start_uipc_test:
            # draw the old points
            # draw.clear_points()
            # points = np.array(gipc_sim.sim.get_vertices())
            # draw.draw_points(points, [(255,0,255,0.5)]*points.shape[0], [30]*points.shape[0])

            print("Step number ", step)
            uipc_sim.step()
            # gipc_sim.render_tet_surface_wireframes(clean_up_first=True)
            uipc_sim.update_render_meshes()
            sim.render()
            
            # sio.write_surface(f"falling_cubes/obj/scene_surface{uipc_sim.world.frame()}.obj")

            # # convert to vtk file
            # reader = vtk.vtkOBJReader()
            # reader.SetFileName(f"falling_cubes/obj/scene_surface{uipc_sim.world.frame()}.obj")
            # reader.Update()
            # obj = reader.GetOutput()

            # writer = vtk.vtkPolyDataWriter()
            # writer.SetFileName(f"falling_cubes/vtk/scene_surface{uipc_sim.world.frame()}.vtk")
            # writer.SetInputData(obj)
            # writer.Write()

            # #draw.clear_points()
            # points = np.array(gipc_sim.sim.get_vertices())
            # draw.draw_points(points, [(255,50 + 50,0,0.5)]*points.shape[0], [30]*points.shape[0])
            # # draw debug view every 5 steps
            # if step % 5 == 0:
            #     # update positions
            #     #gipc_sim.objects["cube"].update_positions(gipc_sim.vertices)
            #     #gipc_sim.objects["cube"].draw_debug_view()

            #     gipc_sim.objects["cube_big"].update_positions(gipc_sim.vertices)
            #     #gipc_sim.objects["cube_big"].draw_debug_view()
            
            step += 1      
        
        # start GIPC sim after pausing and playing the sim
        if sim.is_playing() is False:
            #! test
            start_uipc_test = True
            print("Start uipc simulation by pressing Play")

        # if step % 50 == 0:
        #     print("Reset simulation")
        #     print("Render block ", sim.get_block_on_render())
        #     if start_gipc_test:
        #         gipc_sim.reset()
        #         #gipc_sim.step()
        #         gipc_sim.update_render_meshes()
        #         gipc_sim.render_tet_surface_wireframes(clean_up_first=True)
        #         sim.render()
                
        #         start_gipc_test = False
        #         change_mat_color(stage, "/World/Objects/Big_cube/Looks/OmniPBR/Shader", (0,0,1))

        #         # draw.clear_points()
        #         # draw.clear_lines()
        #     step = 1
          
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()