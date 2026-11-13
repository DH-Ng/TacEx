from __future__ import annotations

import datetime
import pathlib

import omni.log
import omni.usd
import usdrt
import usdrt.Usd
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass


from uipc import Logger, Timer
from uipc.core import Engine, Scene, SceneIO, World
from uipc.geometry import extract_surface, ground
from uipc.unit import GPa

from tacex_uipc.objects import UipcObject
from tacex_uipc.utils import MeshGenerator, add_barycentric_primvar, create_surf_tri_vis_material


@configclass
class UipcSimCfg:
    device: str | None = "cuda"

    update_render_meshes: bool = True

    debug_vis: bool = False
    """Show the triangle outlines of the meshes created by :meth:`UipcSim.init_libuipc_scene_rendering` (libuipc
    sample scenes).

    The simulation meshes of uipc objects are visualized via the objects' ``debug_vis`` (triangle outlines, or the
    simulation mesh wireframe for objects with a separate render mesh).
    """

    dt: float = 0.01

    sanity_check_enable: bool = True
    sanity_check_mode: str = "quiet"
    """
    Options: "quiet", "normal"
    "quiet" = do not export mesh
    """

    workspace: str | None = None
    """Directory for the libuipc workspace files (e.g. ``systems.json``, ``sanity_check/``, frames of
    :meth:`UipcSim.save_frame`).

    If ``None`` (default), a new directory ``logs/uipc/<YYYY-mm-dd_HH-MM-SS>`` in the current working directory is
    created for every :class:`UipcSim` (suffix ``_1``, ``_2``, ... if it exists already). Set it explicitly to replay
    frames saved in a previous run (:meth:`UipcSim.replay_frame`). The resolved path is :attr:`UipcSim.workspace`.
    """

    logger_level: str = "Error"

    gravity: tuple = (0.0, 0.0, -9.8)
    ground_height: float = 0.0
    ground_normal: tuple = (0.0, 0.0, 1.0)

    @configclass
    class Newton:
        max_iter: int = 1024

        min_iter: int = 1

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


class UipcSim:
    cfg: UipcSimCfg

    def __init__(self, cfg: UipcSimCfg = UipcSimCfg()):
        """Initialize the uipc simulation."""
        # will be initialized in `setup_sim()`
        self.isaac_sim = None
        self.cfg: UipcSimCfg = cfg

        Timer.enable_all()
        if self.cfg.logger_level == "Error":
            Logger.set_level(Logger.Error)
        elif self.cfg.logger_level == "Info":
            Logger.set_level(Logger.Info)

        self.workspace: str = self._resolve_workspace()
        """Directory of the libuipc workspace files (see :attr:`UipcSimCfg.workspace`)."""
        omni.log.info(f"UIPC workspace: {self.workspace}")

        self.engine: Engine = Engine(backend_name=self.cfg.device, workspace=self.workspace)
        self.world: World = World(self.engine)

        # update uipc default config with our config
        uipc_config = Scene.default_config()
        uipc_config["dt"] = self.cfg.dt
        uipc_config["sanity_check"] = {"enable": self.cfg.sanity_check_enable, "mode": self.cfg.sanity_check_mode}
        uipc_config["gravity"] = [[self.cfg.gravity[0]], [self.cfg.gravity[1]], [self.cfg.gravity[2]]]
        uipc_config["collision_detection"]["method"] = self.cfg.collision_detection_method
        uipc_config["contact"] = {
            "enable": self.cfg.contact.enable,
            "constitution": self.cfg.contact.constitution,
            "d_hat": self.cfg.contact.d_hat,
            "eps_velocity": self.cfg.contact.eps_velocity,
            "friction": {"enable": self.cfg.contact.enable_friction},
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
            "min_iter": self.cfg.newton.min_iter,
            "use_adaptive_tol": self.cfg.newton.use_adaptive_tol,
            "velocity_tol": self.cfg.newton.velocity_tol,
            "transrate_tol": self.cfg.newton.transrate_tol,
        }

        self.scene = Scene(uipc_config)

        # create ground
        ground_obj = self.scene.objects().create("ground")
        uipc_ground = ground(self.cfg.ground_height, self.cfg.ground_normal)
        ground_obj.geometries().create(uipc_ground)

        # set default friction ratio and contact resistance
        self.scene.contact_tabular().default_model(
            friction_rate=self.cfg.contact.default_friction_ratio,
            resistance=self.cfg.contact.default_contact_resistance * GPa,
            enable=self.cfg.contact.enable,
        )

        # for rendering: used to extract which points belong to which surface mesh
        self._surf_vertex_offsets = [0]

        self._fabric_meshes = []
        # owner (uipc object) of each fabric mesh in `_fabric_meshes`
        self._fabric_mesh_owners: list[UipcObject] = []
        # index in `_fabric_meshes` -> uipc object with a separate render mesh (see `RenderMeshCfg`)
        self._render_mesh_owners: dict[int, UipcObject] = {}
        self.uipc_objects: list[UipcObject] = []

        self.stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())

        # for updating render meshes
        self.sio = SceneIO(self.scene)

    def __del__(self):
        """Unsubscribe from the callbacks."""
        # clear debug visualization
        if self._debug_vis_handle:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    def _resolve_workspace(self) -> str:
        """Creates the workspace directory: the configured one, or a new ``logs/uipc/<timestamp>`` directory."""
        if self.cfg.workspace is not None:
            workspace = pathlib.Path(self.cfg.workspace).expanduser().resolve()
            workspace.mkdir(parents=True, exist_ok=True)
            return str(workspace)

        root = pathlib.Path.cwd() / "logs" / "uipc"
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # unique per UipcSim: mkdir(exist_ok=False) is atomic, so parallel runs never share a directory
        for i in range(1000):
            workspace = root / (stamp if i == 0 else f"{stamp}_{i}")
            try:
                workspace.mkdir()
                return str(workspace)
            except FileExistsError:
                continue
        raise RuntimeError(f"Could not create a UIPC workspace directory in {root}.")

    def setup_sim(self):
        self.world.init(self.scene)
        self.world.retrieve()

        # embed separate render meshes into their simulation meshes (rest configuration as seen by uipc)
        if self._render_mesh_owners:
            all_surf_points = self.sio.simplicial_surface(2).positions().view().reshape(-1, 3)
            for i, owner in self._render_mesh_owners.items():
                owner.init_render_mesh_mapping(
                    all_surf_points[self._surf_vertex_offsets[i] : self._surf_vertex_offsets[i + 1]]
                )

        # initialize callbacks
        self.isaac_sim: sim_utils.SimulationContext = sim_utils.SimulationContext.instance()
        self.isaac_sim.add_physics_callback("uicp_step", self.step)

        if self.cfg.update_render_meshes:
            self.isaac_sim.add_physics_callback("uicp_update_render_meshes", self.update_render_meshes)

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
        all_trimesh_points = self.sio.simplicial_surface(2).positions().view().reshape(-1, 3)
        # triangles = self.sio.simplicial_surface(2).triangles().topo().view()
        for i, fabric_prim in enumerate(self._fabric_meshes):
            trimesh_points = all_trimesh_points[self._surf_vertex_offsets[i] : self._surf_vertex_offsets[i + 1]]
            owner = self._render_mesh_owners.get(i)
            if owner is not None:
                # separate render mesh: reconstruct render vertices from the embedding in the simulation mesh
                owner.update_sim_mesh_vis(trimesh_points)
                trimesh_points = owner.compute_render_points(trimesh_points)

            fabric_mesh_points = fabric_prim.GetAttribute("points")
            fabric_mesh_points.Set(usdrt.Vt.Vec3fArray(trimesh_points))

            # deformation visualization (e.g. elastic energy/forces), see `UipcDeformableObjectCfg.debug_deformation_vis`
            vis_owner = self._fabric_mesh_owners[i] if i < len(self._fabric_mesh_owners) else None
            if vis_owner is not None and getattr(vis_owner.cfg, "debug_deformation_vis", False):
                surf_points = all_trimesh_points[self._surf_vertex_offsets[i] : self._surf_vertex_offsets[i + 1]]
                vis_owner.update_deformation_vis(surf_points, trimesh_points)

            # if self.cfg.debug_vis:
            #     # draw surface mesh
            #     triangles = self.sio.simplicial_surface(2).triangles()
            #     for i in range(0, len(trimesh_points), 3):
            #         test = trimesh_points
            #         # draw.draw_points(tet_points, [(255, 255, 255, 1)] * len(tet_points), [40] * len(tet_points))
            #         # draw.draw_lines(
            #         #     [tet_points[0]] * 2, tet_points[1:], color * 2, [10] * 2
            #         # )  # draw from point 0 to every other point (3 times 0, cause line from 0 to the other 3 points)
            #         # draw.draw_lines([tet_points[1]] * 1, tet_points[2:], color * 1, [10] * 1)

        # if self.isaac_sim is not None:
        #     # additional render call to somewhat mitigate render delay
        #     self.isaac_sim.render()

    @staticmethod
    def get_sim_time_report(as_json: bool = False):
        report = None
        if as_json:
            report = Timer.report_as_json()
        else:
            Timer.report()
        return report

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
        if self.world.recover(frame_num):
            self.world.retrieve()
        else:
            print(f"No data for frame {frame_num}.")

    def init_libuipc_scene_rendering(self):
        """Render method for "standalone" libuipc scenes (e.g. the examples in tacex_uipc/examples/libuipc-samples).

        The method simply creates UsdGeom meshes and updates each mesh in Isaac Sim with the data from libuipc simulation.
        The main difference to the "IsaacLab x UIPC" render method is, that we define each mesh (and not load it from a usd file) and
        that we update **every mesh**.

        With our IsaacLab x UIPC workflow, we have meshes that are not updated by uipc (e.g. robots) and we load the meshes from USD files.
        """
        num_objs = self.scene.objects().size()
        for obj_id in range(1, num_objs):
            # print("obj id ", obj_id)
            obj = self.scene.objects().find(obj_id)
            obj_geometry_ids = obj.geometries().ids()
            obj_name = obj.name()

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
                    prim_path = f"/World/{obj_name}_{id}"
                    prim = UsdGeom.Mesh.Define(stage, prim_path)

                    dim = geom.dim()
                    if dim == 2:
                        # e.g. cloth
                        surf = geom
                    else:
                        # extract_surface only possible for tetrahedra meshes
                        surf = extract_surface(geom)

                    surf_tri = surf.triangles().topo().view().reshape(-1).tolist()
                    surf_points_world = surf.positions().view().reshape(-1, 3)

                    MeshGenerator.update_usd_mesh(gprim=prim, surf_points=surf_points_world, triangles=surf_tri)

                    # setup mesh updates via Fabric
                    fabric_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
                    fabric_prim = fabric_stage.GetPrimAtPath(usdrt.Sdf.Path(prim_path))

                    # Tell OmniHydra to render points from Fabric
                    if not fabric_prim.HasAttribute("Deformable"):
                        fabric_prim.CreateAttribute("Deformable", usdrt.Sdf.ValueTypeNames.PrimTypeTag, True)

                    stage_id = omni.usd.get_context().get_stage_id()
                    fabric_id = fabric_stage.GetFabricId()

                    hier = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(fabric_id, stage_id)
                    hier.set_world_xform(usdrt.Sdf.Path(prim_path), usdrt.Gf.Matrix4d(1))
                    hier.update_world_xforms()

                    if self.cfg.debug_vis:
                        add_barycentric_primvar(prim)
                        mat_path = "/World/Materials/TriangleOutlineMat"
                        stage = omni.usd.get_context().get_stage()
                        mat = stage.GetPrimAtPath(mat_path)
                        if not mat.IsValid():
                            mat = create_surf_tri_vis_material(
                                mat_path=mat_path,
                                primvar_name="baryCoord",
                                outline_color=(0.8, 0.8, 0.8),
                                outline_width=0.05,
                                base_color=(0.0, 0.0, 0.8),
                            )

                        # bind material with fabric
                        rel = fabric_prim.GetRelationship(usdrt.UsdShade.Tokens.materialBinding)
                        rel.SetTargets([mat_path])

                    # add fabric meshes to uipc sim class for updating the render meshes
                    self._fabric_meshes.append(fabric_prim)

                    # save indices to later find corresponding points of the meshes for rendering
                    num_surf_points = surf_points_world.shape[0]
                    self._surf_vertex_offsets.append(self._surf_vertex_offsets[-1] + num_surf_points)
