
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

from uipc import view
from uipc import builtin
from uipc import Animation, Vector3
from uipc.core import Engine, World, Scene, SceneIO
from uipc.constitution import SoftPositionConstraint
from uipc.geometry import GeometrySlot, SimplicialComplex, SimplicialComplexIO

from tacex_uipc import UipcSim, UipcSimCfg, UipcObject, UipcObjectCfg
from tacex_uipc.utils import TetMeshCfg

def main():
    """Main function."""
    # Initialize the simulation context
    # render_cfg = sim_utils.RenderCfg(rendering_mode=)
    sim_cfg = sim_utils.SimulationCfg(dt=1/60)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # Design scene by spawning assets
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

    # spawn uipc cube
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

    # For Animation
    spc = SoftPositionConstraint()
    # `apply` has to happen **before** the uipc_scene_object is created! 
    # i.e. before UipcObject._initialize_impl() is called
    spc.apply_to(cube.uipc_meshes[0], 100) # constraint strength ratio

    # tet_ball_asset_path = pathlib.Path(__file__).parent.resolve() / "assets" / "ball.usd"
    # ball_cfg = UipcObjectCfg(
    #     prim_path="/World/Objects/ball",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 1.0]), #rot=(0.72,-0.3,0.42,-0.45)
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(tet_ball_asset_path),
    #         scale=(0.5, 0.5, 0.5)
    #     ),
    #     mesh_cfg=mesh_cfg,
    #     constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg()
    # )
    # ball = UipcObject(ball_cfg, uipc_sim)

    # Play the simulator
    sim.reset()

    # Create Animation -> has to happen after the objects were created in the
    # uipc scene, i.e. after UipcObject._initialize_impl() is called.
    # This is the case after sim.reset().
    animator = uipc_sim.scene.animator()
    def animate_tet(info: Animation.UpdateInfo): # animation function
        geo_slots:list[GeometrySlot] = info.geo_slots() # list of geo_slots -> multiple when uipc_object has instances
        geo: SimplicialComplex = geo_slots[0].geometry()
        rest_geo_slots:list[GeometrySlot] = info.rest_geo_slots()
        rest_geo:SimplicialComplex = rest_geo_slots[0].geometry()

        is_constrained = geo.vertices().find(builtin.is_constrained)
        is_constrained_view = view(is_constrained)
        aim_position = geo.vertices().find(builtin.aim_position)
        aim_position_view = view(aim_position)
        rest_position_view = rest_geo.positions().view()

        is_constrained_view[0] = 1 # animate first vertex of the mesh
        is_constrained_view[1] = 1 # as well as the second vertex

        t = info.dt() * info.frame()
        theta = np.pi * t
        z = -np.sin(theta)

        aim_position_view[0] = rest_position_view[0] + Vector3.UnitZ() * z
    
    animator.insert(cube.uipc_scene_objects[0], animate_tet)

    # only after Isaac Sim got resetted (= objects init), otherwise world init is false
    # because _initialize_impl() of the object is called in the sim.reset() method
    # and setup_scene() relies on objects being _initialize_impl()
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

          
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()