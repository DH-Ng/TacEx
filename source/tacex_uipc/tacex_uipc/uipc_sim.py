from __future__ import annotations

from typing import Any, Dict, Tuple, Union, List, TYPE_CHECKING
import pathlib

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

import usdrt
from pxr import Gf, Sdf, Usd, UsdGeom
import omni.usd

try:
    from isaacsim.util.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
except:
    draw = None

import uipc
from uipc import view
from uipc import Vector3, Transform, Quaternion, AngleAxis
from uipc import Logger, Timer
from uipc.core import Engine, World, Scene, SceneIO
from uipc.geometry import tetmesh, label_surface, label_triangle_orient, flip_inward_triangles, ground
from uipc.unit import MPa, GPa

from uipc.geometry import tetmesh, extract_surface
from tacex_uipc.utils import MeshGenerator

import numpy as np

if TYPE_CHECKING:
    from tacex_uipc import UipcObject

@configclass
class UipcSimCfg:
    device: str | None = "cuda"

    dt: float = 0.01

    sanity_check_enable: bool = True
    sanity_check_mode: str = "quiet"
    """
    Options: "quiet", "normal"
    "quiet" = do not export mesh
    """
    
    workspace: str = str(pathlib.Path().resolve())
    """Path to directory where the libuipc files (e.g. systems.json) should be saved.

    Defaults to current working directory.
    """

    logger_level: str = "Error"

    gravity: tuple = (0.0, 0.0, -9.8)
    ground_height: float = 0.0
    ground_normal: tuple = (0.0, 0.0, 1.0)

    @configclass
    class Newton:
        max_iter: int = 1024

        use_adaptive_tol: bool = False

        # convergence tolerances
        velocity_tol: float = 0.05
        """Convergence tolerance 1)
        max(dx*dt) <= velocity_tol
        """

        ccd_tol: float = 1.0
        """Convergence tolerance 2)
        ccd_toi >= ccd_tol
        """

        transrate_tol: float = 0.1 / 1.0
        """Convergence tolerance 3)
        max dF <=  dF <= transform_tol * dt
        
        e.g. 0.1/1.0 = 10%/s change in transform
        """
    newton: Newton = Newton()

    @configclass
    class LinearSystem:
        solver: str = "linear_pcg"
        """
        Options: linear_pcg, 
        """

        tol_rate: float = 1e-3
    linear_system: LinearSystem = LinearSystem()

    @configclass
    class LineSearch:
        max_iter: int = 8

        report_energy: bool = False
    line_search: LineSearch = LineSearch()

    @configclass
    class Contact:
        enable: bool = True
        
        enable_friction: bool = True

        default_friction_ratio: float = 0.5

        default_contact_resistance: float = 10.0
        """
        in [GPa]
        """

        constitution: str = "ipc"
        
        d_hat: float = 0.001

        eps_velocity: float = 0.01
        """
        in [m/s]
        """
    contact: Contact = Contact()

    collision_detection_method: str = "linear_bvh"
    """
    Options: linear_bvh, informative_linear_bvh
    """

    diff_sim: bool = False

class UipcSim():
    cfg: UipcSimCfg

    def __init__(self, cfg: UipcSimCfg = None):
        """Initialize the uipc simulation.
        
        """
        # will be initialized in `setup_sim()`
        self.isaac_sim = None

        if cfg is None:
            cfg = UipcSimCfg()
        self.cfg = cfg

        Timer.enable_all()
        if self.cfg.logger_level == "Error":
            Logger.set_level(Logger.Error)
        elif self.cfg.logger_level == "Info":
            Logger.set_level(Logger.Info)

        self.engine: Engine = Engine(backend_name=self.cfg.device, workspace=self.cfg.workspace)
        self.world: World = World(self.engine)

        # update uipc default config with our config
        uipc_config = Scene.default_config()
        uipc_config["dt"] = self.cfg.dt
        uipc_config["sanity_check"] = {
            "enable": self.cfg.sanity_check_enable,
            "mode": self.cfg.sanity_check_mode 
        }
        uipc_config["gravity"] = [[self.cfg.gravity[0]], [self.cfg.gravity[1]], [self.cfg.gravity[2]]]
        uipc_config["collision_detection"]["method"] = self.cfg.collision_detection_method
        uipc_config["contact"] = {
            "enable": self.cfg.contact.enable,
            "constitution": self.cfg.contact.constitution,
            "d_hat": self.cfg.contact.d_hat,
            "eps_velocity": self.cfg.contact.eps_velocity,
            "friction": {
                "enable": self.cfg.contact.enable_friction
            }
        }
        uipc_config["diff_sim"] = {
            "enable": self.cfg.diff_sim,
        }
        uipc_config["line_search"] = {
            "max_iter": self.cfg.line_search.max_iter,
            "report_energy": self.cfg.line_search.report_energy,
        }
        uipc_config["linear_system"] = {
            "solver": self.cfg.linear_system.solver,
            "tol_rate": self.cfg.linear_system.tol_rate,
        }
        uipc_config["newton"] = {
            "ccd_tol": self.cfg.newton.ccd_tol,
            "max_iter": self.cfg.newton.max_iter,
            "use_adaptive_tol": self.cfg.newton.use_adaptive_tol,
            "velocity_tol": self.cfg.newton.velocity_tol,
            "transrate_tol": self.cfg.newton.transrate_tol
        }


        self.scene = Scene(uipc_config)

        # create ground
        ground_obj = self.scene.objects().create("ground")
        g = ground(self.cfg.ground_height, self.cfg.ground_normal)
        ground_obj.geometries().create(g)

        # set default friction ratio and contact resistance
        self.scene.contact_tabular().default_model(
            friction_rate=self.cfg.contact.default_friction_ratio, 
            resistance=self.cfg.contact.default_contact_resistance * GPa,
            enable=self.cfg.contact.enable
        )


        # vertex offsets of the subsystems
        self._system_vertex_offsets = {
            "uipc::backend::cuda::GlobalVertexManager": [0], # global vertex offset
            "uipc::backend::cuda::FiniteElementMethod": [0],
            "uipc::backend::cuda::AffineBodyDynamics": [0],
        }

        # for rendering: used to extract which points belong to which surface mesh
        self._surf_vertex_offsets = [0]
        
        self._fabric_meshes = []
        self.uipc_objects: List[UipcObject] = []

    def setup_sim(self):
        self.world.init(self.scene)
        self.world.retrieve()

        # trans = geo_slot.geometry().transforms().view()
        # print("init trans ", trans)
        # for updating render meshes
        self.sio = SceneIO(self.scene)

        # compute the global vertex offset accross the systems
        fem_system = self._system_vertex_offsets["uipc::backend::cuda::FiniteElementMethod"]
        abd_system = self._system_vertex_offsets["uipc::backend::cuda::AffineBodyDynamics"]
        
        #todo figure out how we always get the correct order -> for now we just assume something(ground?) first, then ABD and then FEM system
        #? another idea might be to use the coindices mapping from global_vertex_manager and use it to create mapping for local -> global
        # +1 for total count, like uipc does -> first system is the interABD system I suppose? #todo check if really the case, once docs are updated
        #? or is it just due to the ground?
        self._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"].append(1)

        # second system is the abd system
        global_abd_system = [idx+self._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"][-1] for idx in abd_system[1:]]
        self._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"] += global_abd_system
        print("after abd ", self._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"])        
        
        # add offset from previous system (ABD) to the next system (FEM)
        global_fem_system = [idx+self._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"][-1] for idx in fem_system[1:]]
        self._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"] += global_fem_system
        
        print("after fem ", self._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"])

        #! ehh, pretty complicated/convoluted I would say, can we simplify this?
        # for each obj, compute the global_system_id -> used to infer the objects vertex_offset in the global system
        for uipc_obj in self.uipc_objects:
            if uipc_obj._system_name == "uipc::backend::cuda::AffineBodyDynamics":
                global_id = uipc_obj.local_system_id + 1 # +1, cause uipc global offsets start with 0,1, and comes ABD-sys, FEM-sys offsets #todo adjust, once we find out why it starts with 0,1
            
            if uipc_obj._system_name == "uipc::backend::cuda::FiniteElementMethod":# since FEM system comes after ABD syste,
                global_id = uipc_obj.local_system_id + len(abd_system) # fem system offsets are behind the number of objects in the abd system in the global system
            
            print(f"{uipc_obj.cfg.prim_path} has global id {global_id}")
            uipc_obj.global_system_id = global_id

        # initialize callbacks
        self.isaac_sim: sim_utils.SimulationContext = sim_utils.SimulationContext.instance()
        self.isaac_sim.add_physics_callback("upic_step", self.step)

        # doesnt really help I think, cause we want the mesh update to happen before we render
        # self.isaac_sim.add_render_callback("upic_render_mesh_update", self.update_render_meshes)

        # using #self.isaac_sim.add_physics_callback("upic_render_mesh_update", self.update_render_meshes)
        # -> also didnt help

    def step(self, dt=0):
        self.world.advance()
        self.world.retrieve()

    def reset(self):
        # self.world.recover(0) # go back to frame 0
        # geom = self.scene.geometries()
        # geo_slot, _ = geom.find(1)
        # test = view(geo_slot.geometry().positions())
        # test = self.init_pos
        # self.world.advance()
        
        # new_test = view(geo_slot.geometry().positions())

        # short_cut_trans = geo_slot.geometry().transforms().view()
        self.world.retrieve()
        self.update_render_meshes()

    def update_render_meshes(self, dt=0):
        all_trimesh_points = self.sio.simplicial_surface(2).positions().view().reshape(-1,3)
        #triangles = self.sio.simplicial_surface(2).triangles().topo().view()
        for i, fabric_prim in enumerate(self._fabric_meshes):
            trimesh_points = all_trimesh_points[self._surf_vertex_offsets[i]:self._surf_vertex_offsets[i+1]]
            #draw.draw_points(trimesh_points, [(0,0,255,0.5)]*trimesh_points.shape[0], [30]*trimesh_points.shape[0])
            
            fabric_mesh_points = fabric_prim.GetAttribute("points")
            fabric_mesh_points.Set(usdrt.Vt.Vec3fArray(trimesh_points))

        # draw.clear_points()
        # points = np.array(all_trimesh_points)
        # draw.draw_points(points, [(255,0,255,0.5)]*points.shape[0], [30]*points.shape[0])

        if self.isaac_sim is not None:
            # self.isaac_sim.forward() # this doesnt really help, I think

            # additional render call to somewhat mitigate render delay #todo search for better to fix this
            #? render delay might be solved in newest Isaac Ver with FabricSceneDelegate?
            self.isaac_sim.render()
            

    @staticmethod
    def get_sim_time_report(as_json: bool = False):
        if as_json:
            report = Timer.report_as_json()
        else:
            Timer.report()
    

    def save_frame(self):
        """Saves the current frame into multiple files, which can be retrieved to replay the animation later.

        For replaying use the method `replay_frames`        
        """
        self.world.dump()

    # for replaying already computed uipc simulation results
    def replay_frame(self, frame_num):
        """
        
        Won't work if the loaded tet meshes are different!
        """
        # frame_num = self.world.frame() + 1
        if(self.world.recover(frame_num)):
            self.world.retrieve()
        else:
            print(f"No data for frame {frame_num}.")      

    def init_libuipc_scene_rendering(self):
        num_objs = self.scene.objects().size()
        for obj_id in range(1, num_objs):
            # print("obj id ", obj_id)
            obj = self.scene.objects().find(obj_id)
            obj_geometry_ids = obj.geometries().ids()
            # obj_name = obj.name() #-- doesnt work, get error message (otherwise use this to set prim_path)
            
            # create prim for each geometry
            for id in obj_geometry_ids:
                scene_geom = self.scene.geometries()
                geo_slot, geo_slot_rest = scene_geom.find(id)
                # extract_surface
                geom = geo_slot.geometry()

                # create a mesh for each instance (we keep it simple here and dont clone the prim like in IsaacLab workflow)
                num_geom = geom.instances().size()
                for instance_num in range(num_geom):
                    id += instance_num
                    # spawn a usd mesh in Isaac
                    stage = omni.usd.get_context().get_stage()
                    prim_path = f"/World/uipc_mesh_{id}"
                    prim = UsdGeom.Mesh.Define(stage, prim_path)

                    dim = geom.dim()
                    if dim == 2:
                        # e.g. cloth
                        surf = geom
                    else:
                        # extract_surface only possible for tetrahedra meshes
                        surf = extract_surface(geom)

                    surf_tri = surf.triangles().topo().view().reshape(-1).tolist()
                    surf_points_world = surf.positions().view().reshape(-1,3)

                    MeshGenerator.update_usd_mesh(prim=prim, surf_points=surf_points_world, triangles=surf_tri)

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
                    fabric_mesh_points_attr.Set(usdrt.Vt.Vec3fArray(surf_points_world))

                    # add fabric meshes to uipc sim class for updating the render meshes
                    self._fabric_meshes.append(fabric_prim)

                    # save indices to later find corresponding points of the meshes for rendering
                    num_surf_points = surf_points_world.shape[0]
                    self._surf_vertex_offsets.append(
                        self._surf_vertex_offsets[-1] + num_surf_points
                    )