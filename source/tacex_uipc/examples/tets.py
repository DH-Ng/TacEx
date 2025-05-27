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
import omni.usd

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaacsim.util.debug_draw import _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()

from pxr import UsdGeom, Usd, Sdf, PhysxSchema, UsdPhysics, Gf, UsdShade, Vt
#from pxr import UsdGeom, Sdf, Gf, 
import numpy as np
import warp as wp

import carb
import usdrt
from usdrt import Usd, Sdf, UsdGeom, Rt, Gf, Vt

PLANE_SUBDIV = 32
PLANE_EXTENT = 50
PLANE_HEIGHT = 10
PRIM_PATH = f"/World/Plane{PLANE_SUBDIV}x{PLANE_SUBDIV}"


def create_mesh_usdrt(stage: usdrt.Usd.Stage, prim_path: str, num_x_divisions: int, num_z_divisions: int):
    mesh = usdrt.UsdGeom.Mesh.Define(stage, prim_path)

    # Create the vertices and face counts
    vertices = calculate_mesh_vertices(num_x_divisions, num_z_divisions, 0)
    face_vertex_counts = []
    face_vertex_indices = []

    for z in range(num_z_divisions):
        for x in range(num_x_divisions):
            vertex0 = z * (num_x_divisions + 1) + x
            vertex1 = vertex0 + 1
            vertex2 = (z + 1) * (num_x_divisions + 1) + x
            vertex3 = vertex2 + 1
            face_vertex_counts.append(4)
            face_vertex_indices.extend([vertex0, vertex1, vertex3, vertex2])

    # Set the mesh data
    mesh.CreatePointsAttr().Set(usdrt.Vt.Vec3fArray(vertices))
    mesh.CreateFaceVertexCountsAttr().Set(usdrt.Vt.IntArray(face_vertex_counts))
    mesh.CreateFaceVertexIndicesAttr().Set(usdrt.Vt.IntArray(face_vertex_indices))

    prim = mesh.GetPrim()
    # Visibility Attribute
    visibility_attr = prim.CreateAttribute("_worldVisibility", usdrt.Sdf.ValueTypeNames.Bool, True)
    visibility_attr.Set(True)

    # Set the xform
    xformable = usdrt.Rt.Xformable(prim)
    xformable.CreateFabricHierarchyLocalMatrixAttr(usdrt.Gf.Matrix4d(1))
    xformable.CreateFabricHierarchyWorldMatrixAttr(usdrt.Gf.Matrix4d(1))

    # Set the extents
    bound = usdrt.Rt.Boundable(prim)
    world_ext = bound.CreateWorldExtentAttr()
    world_ext.Set(
        usdrt.Gf.Range3d(
            usdrt.Gf.Vec3d(-PLANE_EXTENT, -PLANE_EXTENT, -PLANE_EXTENT),
            usdrt.Gf.Vec3d(PLANE_EXTENT, PLANE_EXTENT, PLANE_EXTENT),
        )
    )
    return mesh

def update_mesh_usdrt(stage: usdrt.Usd.Stage, prim_path: str, num_x_divisions: int, num_z_divisions: int, step: int):
    # Find the prim
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        carb.log_verbose(f"Prim at '{prim_path}' is invalid")
        return

    vertices = calculate_mesh_vertices(num_x_divisions, num_z_divisions, step)

    # Set the mesh data
    mesh = usdrt.UsdGeom.Mesh(prim)
    mesh.CreateVisibilityAttr().Set(True)
    mesh.GetPointsAttr().Set(usdrt.Vt.Vec3fArray(vertices))
    return mesh


def calculate_mesh_vertices(num_x_divisions: int, num_z_divisions: int, step: int) -> [float]:
    x_positions = np.linspace(-PLANE_EXTENT, PLANE_EXTENT, num_x_divisions + 1)
    z_positions = np.linspace(-PLANE_EXTENT, PLANE_EXTENT, num_z_divisions + 1)
    x_grid, z_grid = np.meshgrid(x_positions, z_positions)

    tau = 6.28318
    s = 100.0
    t = step / s
    sx = tau / s
    sz = tau / s
    y_grid = PLANE_HEIGHT * (np.cos(sx * x_grid + t) + np.sin(sz * z_grid + t))

    vertices = np.column_stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten()))
    return vertices.tolist()

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

    # # create a new xform prim for all objects to be spawned under
    # prim_utils.create_prim("/World/Objects", "Xform")

    # own_cube_cfg = sim_utils.UsdFileCfg(usd_path="/workspace/isaaclab/data_storage/TacEx/examples/assets/cube.usd")
    
    # amount_cubes = 6
    # # spawn multiple cubes
    # for i in range(amount_cubes):
    #     own_cube_cfg.func(f"/World/Objects/Cube{i}", own_cube_cfg, translation=(0.0, 0, 0.05 + i/10), orientation=(0.72,-0.3,0.42,-0.45) )     
    #     #own_cube_cfg.func(f"/World/Objects/Cube_x{i}", own_cube_cfg, translation=(1, 0, 0.05 + i/10), orientation=(0.72,-0.3,0.42,-0.45) )     
    #     #own_cube_cfg.func(f"/World/Objects/Cube_y{i}", own_cube_cfg, translation=(0, 1, 0.05 + i/10), orientation=(0.72,-0.3,0.42,-0.45) )     

    # cube_big_cfg = sim_utils.UsdFileCfg(usd_path="/workspace/isaaclab/data_storage/TacEx/examples/assets/cube_big.usd")

    # cube_big_cfg.func(f"/World/Objects/Big_cube", cube_big_cfg, translation=(0.0, 0, 0.5 + 0.05 + (amount_cubes/10))) # make sure that big cube is at the top
    # #cube_big_cfg.func(f"/World/Objects/Big_cube_x", cube_big_cfg, translation=(1, 0, 0.5 + 0.05 + (amount_cubes/10))) # make sure that big cube is at the top
    #cube_big_cfg.func(f"/World/Objects/Big_cube_y", cube_big_cfg, translation=(0, 1, 0.5 + 0.05 + (amount_cubes/10))) # make sure that big cube is at the top
   
    # # spawn a usd file of a table into the scene
    # cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    # cfg.func("/World/Objects/Table", cfg, translation=(0.0, 0.0, 1.05))

### This didnt work
    # stage_id = omni.usd.get_context().get_stage_id()
    # print("Stage ID is ", stage_id)
    # stage = Usd.Stage.Attach(stage_id)

    # prim = stage.DefinePrim("/World/triangle", "Mesh")
    # mesh = UsdGeom.Mesh(prim)

    # points = mesh.CreatePointsAttr()
    # points.Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 0, 0), Gf.Vec3f(0, 1.0, 0), Gf.Vec3f(-1.0, 0, 0)]))
    # face_vc = mesh.CreateFaceVertexCountsAttr()
    # face_vc.Set(Vt.IntArray([3]))
    # face_vi = mesh.CreateFaceVertexIndicesAttr()
    # face_vi.Set(Vt.IntArray([0, 1, 2]))

    # rtbound = Rt.Boundable(prim)
    # world_ext = rtbound.CreateWorldExtentAttr()
    # world_ext.Set(Gf.Range3d(Gf.Vec3d(-1.0, 0, -1.0), Gf.Vec3d(1.0, 1.0, 1.0)))

    # prim = stage.DefinePrim("/World/cube", "Cube")
    # rtbound = Rt.Boundable(prim)
    # world_ext = rtbound.CreateWorldExtentAttr()
    # world_ext.Set(Gf.Range3d(Gf.Vec3d(-0.5, -0.5, -0.5), Gf.Vec3d(0.5, 0.5, 0.5)))

    ctx = omni.usd.get_context()
    stage = usdrt.Usd.Stage.Attach(ctx.get_stage_id())
    create_mesh_usdrt(stage, PRIM_PATH, PLANE_SUBDIV, PLANE_SUBDIV)


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


    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Play the simulator
    sim.reset()

    step = 1
    start_gipc_test = False

    # Simulate physics
    while simulation_app.is_running():

        # perform step
        sim.step()        
        sim.render()
        step += 1

        update_mesh_usdrt(stage, PRIM_PATH, PLANE_SUBDIV, PLANE_SUBDIV, step)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()