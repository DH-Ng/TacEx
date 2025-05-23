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
import omni

import omni.isaac.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaacsim.util.debug_draw import _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()

import omni.isaac.core.utils.transformations as tf_utils
from pxr import UsdGeom, Usd, Sdf, PhysxSchema, UsdPhysics, Gf, UsdShade, Vt
#from pxr import UsdGeom, Sdf, Gf, 
import numpy as np
import random
import warp as wp

from gipc_isaac.sim_gipc import SimGIPC, CfgGIPC 
from gipc_isaac.tet_mesh_generation import CfgTetMesh

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
    prim_utils.create_prim("/World/Objects", "Xform")

    own_cube_cfg = sim_utils.UsdFileCfg(usd_path="/workspace/isaaclab/data_storage/TacEx/examples/assets/cube.usd")
    
    amount_cubes = 6
    # spawn multiple cubes
    for i in range(amount_cubes):
        own_cube_cfg.func(f"/World/Objects/Cube{i}", own_cube_cfg, translation=(0.0, 0, 0.05 + i/10), orientation=(0.72,-0.3,0.42,-0.45) )     
        #own_cube_cfg.func(f"/World/Objects/Cube_x{i}", own_cube_cfg, translation=(1, 0, 0.05 + i/10), orientation=(0.72,-0.3,0.42,-0.45) )     
        #own_cube_cfg.func(f"/World/Objects/Cube_y{i}", own_cube_cfg, translation=(0, 1, 0.05 + i/10), orientation=(0.72,-0.3,0.42,-0.45) )     

    cube_big_cfg = sim_utils.UsdFileCfg(usd_path="/workspace/isaaclab/data_storage/TacEx/examples/assets/cube_big.usd")

    cube_big_cfg.func(f"/World/Objects/Big_cube", cube_big_cfg, translation=(0.0, 0, 0.5 + 0.05 + (amount_cubes/10))) # make sure that big cube is at the top
    #cube_big_cfg.func(f"/World/Objects/Big_cube_x", cube_big_cfg, translation=(1, 0, 0.5 + 0.05 + (amount_cubes/10))) # make sure that big cube is at the top
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

    # Design scene by adding assets to it
    design_scene()

    stage = omni.usd.get_context().get_stage() 

    gipc_sim_cfg: CfgGIPC = CfgGIPC(
        friction_rate = 0.4,
        collision_detection_buff_scale = 1,
        debug=True,
        preconditioner_type=0,
    )
    gipc_sim = SimGIPC(gipc_sim_cfg)

    cfg = CfgTetMesh()
    cfg = cfg.replace(stop_quality=1000)
    cfg = cfg.replace(max_its=1000)
    cfg = cfg.replace(edge_length_r=1/2)
    cfg = cfg.replace(epsilon=1e-3)

    gipc_sim.create_objects(prim_paths_expr="/World/Objects/Cube.*", name="cube", debug_view=True, tet_cfg=cfg)
    gipc_sim.create_objects(prim_paths_expr="/World/Objects/Big_cube.*", name="cube_big", debug_view=True, tet_cfg=cfg)

    #print("Num obj cube ", gipc_sim.objects["cube"].num_objects)

    gipc_sim.init_scene()

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Play the simulator
    sim.reset()
    gipc_sim.step()

    step = 1
    start_gipc_test = False

    # Simulate physics
    while simulation_app.is_running():

        # perform step
        sim.step()
        if start_gipc_test:
            # draw the old points
            draw.clear_points()
            points = np.array(gipc_sim.sim.get_vertices())
            draw.draw_points(points, [(255,0,255,0.5)]*points.shape[0], [30]*points.shape[0])

            print("Step number ", step)
            gipc_sim.step()
            gipc_sim.update_render_meshes()
            gipc_sim.render_tet_surface_wireframes(clean_up_first=True)
            
            sim.render()
            gipc_sim.update_render_meshes()

            #draw.clear_points()
            points = np.array(gipc_sim.sim.get_vertices())
            draw.draw_points(points, [(255,50 + 50,0,0.5)]*points.shape[0], [30]*points.shape[0])
            # draw debug view every 5 steps
            if step % 5 == 0:
                # update positions
                #gipc_sim.objects["cube"].update_positions(gipc_sim.vertices)
                #gipc_sim.objects["cube"].draw_debug_view()

                gipc_sim.objects["cube_big"].update_positions(gipc_sim.vertices)
                #gipc_sim.objects["cube_big"].draw_debug_view()
            
            step += 1      
        
        # start GIPC sim after pausing and playing the sim
        if sim.is_playing() is False:
            #! test
            start_gipc_test = True
            print("Start GIPC simulation")
            change_mat_color(stage, "/World/Objects/Big_cube/Looks/OmniPBR/Shader", (0.9,0.17,0.31))
            # change_mat_color(stage, "/World/Objects/Big_cube_x/Looks/OmniPBR/Shader", (0.9,0.17,0.31))
            # change_mat_color(stage, "/World/Objects/Big_cube_y/Looks/OmniPBR/Shader", (0.9,0.17,0.31))

        if step % 50 == 0:
            print("Reset simulation")
            print("Render block ", sim.get_block_on_render())
            if start_gipc_test:
                gipc_sim.reset()
                #gipc_sim.step()
                gipc_sim.update_render_meshes()
                gipc_sim.render_tet_surface_wireframes(clean_up_first=True)
                sim.render()
                
                start_gipc_test = False
                change_mat_color(stage, "/World/Objects/Big_cube/Looks/OmniPBR/Shader", (0,0,1))

                # draw.clear_points()
                # draw.clear_lines()
            step = 1
          
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()