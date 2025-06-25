"""Showcase on how to use libuipc with Isaac Sim/Lab.

This example corresponds to 
https://github.com/spiriMirror/libuipc-samples/blob/main/python/1_hello_libuipc/main.py


"""

"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Showcase on how to use libuipc with Isaac Sim/Lab.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.utils.timer import Timer

from pxr import Gf, Sdf, Usd, UsdGeom
import omni.usd
import usdrt

import uipc
from uipc.core import Engine, World, Scene
from uipc.geometry import tetmesh, label_surface, label_triangle_orient, flip_inward_triangles, extract_surface
from uipc.constitution import AffineBodyConstitution
from uipc.unit import MPa, GPa

from uipc import view
from uipc import builtin
from uipc import Animation, Vector3
from uipc.core import Engine, World, Scene, SceneIO
from uipc.constitution import SoftPositionConstraint
from uipc.geometry import GeometrySlot, SimplicialComplex, SimplicialComplexIO

from tacex_uipc import UipcSim, UipcSimCfg, UipcObject, UipcObjectCfg
from tacex_uipc.utils import TetMeshCfg, MeshGenerator

def setup_base_scene(sim: sim_utils.SimulationContext):
    """To make the scene pretty.
    
    """
    # set upAxis to Y to match libuipc-samples
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    # Set main camera
    # sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # Design scene by spawning assets
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func(
        prim_path="/World/defaultGroundPlane", 
        cfg=cfg_ground,
        translation=[0, -1, 0],
        orientation=[0.7071068, -0.7071068, 0, 0]
    )

    # spawn distant light
    cfg_light_dome = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_dome.func("/World/lightDome", cfg_light_dome, translation=(1, 10, 0))

def create_prim_for_uipc_scene_object(uipc_sim, prim_path, uipc_scene_object):
    # spawn a usd mesh in Isaac
    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.Mesh.Define(stage, prim_path)
    
    # get corresponding simplical complex from uipc_scene
    obj_id = uipc_scene_object.geometries().ids()[0]     
    simplicial_complex_slot, _ = uipc_sim.scene.geometries().find(obj_id)
    
    # extract_surface
    surf = extract_surface(simplicial_complex_slot.geometry())
    tet_surf_tri = surf.triangles().topo().view().reshape(-1).tolist()
    tet_surf_points_world = surf.positions().view().reshape(-1,3)
    
    MeshGenerator.update_usd_mesh(prim=prim, surf_points=tet_surf_points_world, triangles=tet_surf_tri)

    # setup mesh updates via Fabric
    fabric_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
    fabric_prim = fabric_stage.GetPrimAtPath(usdrt.Sdf.Path(prim_path))

    # Tell OmniHydra to render points from Fabric
    if not fabric_prim.HasAttribute("Deformable"):
        fabric_prim.CreateAttribute("Deformable", usdrt.Sdf.ValueTypeNames.PrimTypeTag, True)

    # extract world transform
    rtxformable = usdrt.Rt.Xformable(fabric_prim)
    rtxformable.CreateFabricHierarchyWorldMatrixAttr()
    # set world matrix to identity matrix -> uipc already gives us vertices in world frame
    rtxformable.GetFabricHierarchyWorldMatrixAttr().Set(usdrt.Gf.Matrix4d())

    # update fabric mesh with world coor. points
    fabric_mesh_points_attr = fabric_prim.GetAttribute("points")
    fabric_mesh_points_attr.Set(usdrt.Vt.Vec3fArray(tet_surf_points_world))

    # add fabric meshes to uipc sim class for updating the render meshes
    uipc_sim._fabric_meshes.append(fabric_prim)
    
    # save indices to later find corresponding points of the meshes for rendering
    num_surf_points = tet_surf_points_world.shape[0]
    uipc_sim._surf_vertex_offsets.append(
        uipc_sim._surf_vertex_offsets[-1] + num_surf_points
    )
        
def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(
        dt=1/60,
        gravity=[0.0, -9.8, 0.0],
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    setup_base_scene(sim)

    # Initialize uipc sim
    uipc_cfg = UipcSimCfg(
        dt=0.02,
        gravity=[0.0, -9.8, 0.0],
        ground_normal=[0, 1, 0],
        ground_height=-1.0,
        # logger_level="Info",
        contact=UipcSimCfg.Contact(
            default_friction_ratio=0.5,
            default_contact_resistance=1.0,
        )
    )
    uipc_sim = UipcSim(uipc_cfg)

    # create constitution and contact model
    abd = AffineBodyConstitution()

    # friction ratio and contact resistance
    uipc_sim.scene.contact_tabular().default_model(0.5, 1.0 * GPa)
    default_element = uipc_sim.scene.contact_tabular().default_element()

    # create a regular tetrahedron
    Vs = np.array([[0,1,0],
                   [0,0,1],
                   [-np.sqrt(3)/2, 0, -0.5],
                   [np.sqrt(3)/2, 0, -0.5]])
    Ts = np.array([[0,1,2,3]])

    # setup a base mesh to reduce the later work
    base_mesh = tetmesh(Vs, Ts)
    # apply the constitution and contact model to the base mesh
    abd.apply_to(base_mesh, 100 * MPa)
    # apply the default contact model to the base mesh
    default_element.apply_to(base_mesh)

    # label the surface, enable the contact
    label_surface(base_mesh)
    # label the triangle orientation to export the correct surface mesh
    label_triangle_orient(base_mesh)
    # flip the triangles inward for better rendering
    base_mesh = flip_inward_triangles(base_mesh)

    mesh1 = base_mesh.copy()
    pos_view = uipc.view(mesh1.positions())
    # move the mesh up for 1 unit
    pos_view += uipc.Vector3.UnitY() * 1.5

    mesh2 = base_mesh.copy()
    is_fixed = mesh2.instances().find(uipc.builtin.is_fixed)
    is_fixed_view = uipc.view(is_fixed)
    is_fixed_view[:] = 1 # make the second mesh static

    # create objects
    object1 = uipc_sim.scene.objects().create("upper_tet")
    object1.geometries().create(mesh1)  

    object2 = uipc_sim.scene.objects().create("lower_tet")
    object2.geometries().create(mesh2)

    # create prims in Isaac for rendering
    create_prim_for_uipc_scene_object(uipc_sim, prim_path="/World/upper_tet", uipc_scene_object=object1)
    create_prim_for_uipc_scene_object(uipc_sim, prim_path="/World/lower_tet", uipc_scene_object=object2)

    # # Play the simulator
    # sim.reset()

    uipc_sim.setup_sim()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    step = 0

    total_uipc_sim_time = 0.0
    total_uipc_render_time = 0.0

    # Simulate physics
    while simulation_app.is_running():

        # perform Isaac rendering
        sim.render()

        if sim.is_playing():
            print("")
            print("====================================================================================")
            print("====================================================================================")
            print("Step number ", step)
            with Timer("[INFO]: Time taken for uipc sim step.", name="uipc_step"):
                uipc_sim.step()
                # uipc_sim.save_current_world_state()
            with Timer("[INFO]: Time taken for updating the render meshes.", name="render_update"):
                uipc_sim.update_render_meshes()
                sim.render()

            # get time reports
            uipc_sim.get_sim_time_report()
            total_uipc_sim_time += Timer.get_timer_info("uipc_step")
            total_uipc_render_time += Timer.get_timer_info("render_update")

            step += 1      
          
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()