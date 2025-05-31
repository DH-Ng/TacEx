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

    enable_sanity_check: bool = True
    sanity_check_mode: str = "quiet"

    gravity: tuple = (0.0, 0.0, -9.8)
    ground_height: float = 0.0
    ground_normal: tuple = (0.0, 0.0, 1.0)

    @configclass
    class NewtonSolver:
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
    newton_solver: NewtonSolver = NewtonSolver()

    @configclass
    class LinearSystem:
        tol_rate: float = 1e-3

        solver: str = "linear_pcg"
        """
        Options: linear_pcg, 
        """
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
        self.config = Scene.default_config()

        dt = 0.02
        self.config["dt"] = dt
        self.config["sanity_check"] = {
            "enable": True,
            "mode": "quiet" # do not export mesh
        }

        self.config["gravity"] = [[0.0], [0.0], [-9.8]]
        self.scene = Scene(self.config)

        self._fabric_meshes = []
        self._last_point_index = [0]

        # create ground
        ground_obj = self.scene.objects().create("ground")
        g = ground(0.0, [0.0, 0.0, 1.0])
        ground_obj.geometries().create(g)


    def setup_scene(self):
        self.world.init(self.scene)
        self.world.retrieve()

        # for updating render meshes
        self.sio = SceneIO(self.scene)

    def step(self):
        self.world.advance()
        self.world.retrieve()

    def reset(self):
        self.world.recover(1) # go back to frame 1
        self.world.retrieve()
        self.update_render_meshes()

    def update_render_meshes(self):
        all_trimesh_points = self.sio.simplicial_surface(2).positions().view().reshape(-1,3)
        #triangles = self.sio.simplicial_surface(2).triangles().topo().view()
        for i, fabric_prim in enumerate(self._fabric_meshes):
            trimesh_points = all_trimesh_points[self._last_point_index[i]:self._last_point_index[i+1]]
            #draw.draw_points(trimesh_points, [(0,0,255,0.5)]*trimesh_points.shape[0], [30]*trimesh_points.shape[0])
            
            fabric_mesh_points = fabric_prim.GetAttribute("points")
            fabric_mesh_points.Set(usdrt.Vt.Vec3fArray(trimesh_points))

        # draw.clear_points()
        # points = np.array(all_trimesh_points)
        # draw.draw_points(points, [(255,0,255,0.5)]*points.shape[0], [30]*points.shape[0])

    def simple_update_render_meshes(self):
        pass

    def get_time_report(self):
        # self.world.dump() # -> creates files which describe state of the world at this time step [needed if you want to use world.recover()]
        Timer.report()
        # report = Timer.report_as_json()