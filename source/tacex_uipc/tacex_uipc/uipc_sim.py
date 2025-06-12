from __future__ import annotations

from typing import Any, Dict, Tuple, Union, List, TYPE_CHECKING

from isaaclab.utils import configclass

import usdrt

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
    logger_level: str = "Error"

    gravity: tuple = (0.0, 0.0, -9.8)
    ground_height: float = 0.0
    ground_normal: tuple = (0.0, 0.0, 1.0)

    @configclass
    class Newton:
        max_iter: int = 1024

        use_adaptive_tol: bool = False

        velocity_tol: float = 0.05
        """Convergence tolerance
        max(dx*dt) <= velocity_tol
        """

        ccd_tol = 1.0
        """Convergence tolerance
        ccd_toi >= ccd_tol
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

        default_friction_rate: float = 0.5

        default_resistance: float = 10.0
        """
        in [GPa]
        """
        constitution: str = "ipc"
        
        d_hat: float = 0.01

        eps_velocity: float = 0.01
        """
        in [m/s]
        """
    contact: Contact = Contact()

    collision_detection_method: str = "linear_bvh"
    """
    Options: linear_bvh
    """

    diff_sim: bool = False

class UipcSim():
    cfg: UipcSimCfg

    def __init__(self, cfg: UipcSimCfg = None):
        """Initialize the uipc simulation.
        
        """
        if cfg is None:
            cfg = UipcSimCfg()
        self.cfg = cfg

        Timer.enable_all()
        Logger.set_level(Logger.Error)

        self.engine: Engine = Engine(backend_name=self.cfg.device)
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
            "velocity_tol": self.cfg.newton.velocity_tol
        }


        self.scene = Scene(uipc_config)
        
        # create ground
        ground_obj = self.scene.objects().create("ground")
        g = ground(self.cfg.ground_height, self.cfg.ground_normal)
        ground_obj.geometries().create(g)

        # set default friction ratio and contact resistance
        self.scene.contact_tabular().default_model(
            friction_rate=self.cfg.contact.default_friction_rate, 
            resistance=self.cfg.contact.default_resistance * GPa,
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

    def setup_scene(self):
        self.world.init(self.scene)
        self.world.retrieve()

        # trans = geo_slot.geometry().transforms().view()
        # print("init trans ", trans)
        # for updating render meshes
        self.sio = SceneIO(self.scene)

        # compute the global vertex offset accross the systems
        fem_system = self._system_vertex_offsets["uipc::backend::cuda::FiniteElementMethod"]
        abd_system = self._system_vertex_offsets["uipc::backend::cuda::AffineBodyDynamics"]
        
        #todo figure out how we always get the correct order -> for now we just assume FEM first, then abd
        #? another idea might be to use the coindices mapping from global_vertex_manager and use it to create mapping for local -> global
        # self._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"].append(
        #     self._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"][-1]+fem_system
        # )
        self._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"] += fem_system[1:] # append without 0
        print("after fem ", self._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"])        
        
        # # +1 for total count, like uipc does
        self._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"].append(self._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"][-1] + 1)

        # add offset from previous system (FEM system)
        global_abd_system = [idx+self._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"][-1] for idx in abd_system[1:]]
        self._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"] += global_abd_system
        
        print("after abd ", self._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"])

    def step(self):
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

    def update_render_meshes(self):
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

    def simple_update_render_meshes(self):
        pass
    
    def get_time_report(self, as_json: bool = False):
        # self.world.dump() # -> creates files which describe state of the world at this time step [needed if you want to use world.recover()]
        if as_json:
            report = Timer.report_as_json()
        else:
            Timer.report()
        # 

    def save_current_world_state(self):
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