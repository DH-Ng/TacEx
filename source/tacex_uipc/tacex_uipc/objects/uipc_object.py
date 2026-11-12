from __future__ import annotations

import re
from abc import abstractmethod
from typing import TYPE_CHECKING

import omni.log
import omni.usd
import usdrt
import usdrt.UsdGeom
from isaacsim.core.prims import XFormPrim
import isaacsim.core.utils.prims as prims_utils
from pxr import Gf, UsdGeom, UsdPhysics, Sdf, Usd

try:
    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
except ImportError:
    import warnings

    warnings.warn("_debug_draw failed to import", ImportWarning)
    draw = None

import numpy as np
import torch
import warp as wp
from uipc import builtin
from uipc.geometry import (
    SimplicialComplex,
    SimplicialComplexSlot,
    extract_surface,
)

from isaaclab.assets import AssetBase, AssetBaseCfg
from isaaclab.utils import configclass

wp.init()


from tacex_uipc.utils import (
    MeshGenerator,
    TetMeshCfg,
    VoxelTetMeshCfg,
    add_barycentric_primvar,
    create_surf_tri_vis_material,
)
from tacex_uipc.utils.render_mesh_mapping import RenderMeshMapper


from .constraints import (
    UipcConstraint,
    UipcConstraintCfg,
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
)

if TYPE_CHECKING:
    from tacex_uipc.sim import UipcSim


@configclass
class RenderMeshCfg:
    """Separate render and simulation mesh.

    The render mesh is the original Isaac (USD) mesh: its topology, UVs and material bindings are kept. The UIPC tet mesh is only used for simulation. Every render vertex is embedded once into
    the simulation mesh (rest configuration) and is reconstructed from the deformed simulation mesh each frame,
    see :class:`tacex_uipc.utils.render_mesh_mapping.RenderMeshMapper`.
    """

    prim_name: str | None = None
    """Name of the USD mesh (relative to the uipc object prim) that is rendered.

    If ``None``, the mesh used for creating the tet mesh is rendered (i.e. its original topology is kept).
    If a different mesh is given, the mesh used for the tet mesh is made invisible.
    """

    mapping: str | None = None
    """How render vertices are embedded into the simulation mesh.

    - ``"surface"``: closest simulation surface triangle, barycentrics + normal offset (IsaacIPC, arXiv:2605.24339).
      Suited for render meshes that lie on the simulation surface (e.g. fTetWild meshes of the same USD mesh).
    - ``"tetrahedral"``: enclosing tetrahedron, linear shape functions (Macklin et al. 2020, Sec. 5.3).
      Exact for affine deformations. Suited for coarse tet meshes ("cages") enclosing a detailed render mesh.
    - ``None`` (default): ``"tetrahedral"`` for :class:`VoxelTetMeshCfg` tet meshes, ``"surface"`` otherwise.
    """

    sim_mesh_vis_color: tuple[float, float, float] = (1.0, 0.45, 0.0)
    """Color of the simulation mesh wireframe (``debug_vis``)."""

    sim_mesh_vis_width: float | None = None
    """Line width of the simulation mesh wireframe in world units. If ``None``: 0.3 % of the bounding box diagonal."""

    clear_normals: bool = True
    """Remove authored normals of the render mesh, so that the renderer recomputes them from the deformed points.

    Authored normals stay fixed while the points move, which leads to wrong shading under deformation.
    """


@configclass
class UipcObjectCfg(AssetBaseCfg):
    mesh_cfg: TetMeshCfg | VoxelTetMeshCfg | None = (
        None  # TODO make more General MeshCfg -> want to have Tet and Tri (for cloth) Meshes
    )

    mass_density: float = 1e3

    constraint_cfg: UipcConstraintCfg | None = None

    usd_mesh_prim_name: str | None = None
    """The name of the usd mesh that should be used for the Tet Mesh.

    If `None`, then the first child of the prim at the given prim path is used.
    """

    render_mesh_cfg: RenderMeshCfg | None = None
    """Config for separating render and simulation mesh.

    If ``None`` (default), the USD mesh topology is replaced by the surface of the tet mesh and rendered directly.
    Exception: with a :class:`VoxelTetMeshCfg`, the USD mesh is always kept as render mesh (``RenderMeshCfg()``),
    since rendering the voxel staircase is rarely wanted.
    """


class UipcObject(AssetBase):
    """The base class for UipcObjects."""

    cfg: UipcObjectCfg
    """Configuration instance for the rigid object."""

    def __init__(self, cfg: UipcObjectCfg, uipc_sim: UipcSim):
        """Initialize the uipc object from an USD asset.

        The USD asset should consist of a parent XForm, followed by an USD mesh.
        If the USD mesh name is not given, then the first mesh under the XForm is used.
        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)
        self._uipc_sim: UipcSim = uipc_sim

        prim_paths_expr = self.cfg.prim_path
        print(f"Initializing uipc objects {prim_paths_expr}...")
        self._prim_view = XFormPrim(
            prim_paths_expr=prim_paths_expr, name=f"{prim_paths_expr}", usd=False
        )
        self._prim_view.initialize()

        # the isaac mesh that should be used for creating the Tet mesh
        self._usd_mesh_prim = None

        # check if prim or a child of the uipc_object has PhysX rigid body API applied to it
        root_prim = self._prim_view.prims[0]

        # Check root prim and all its descendants for PhysxAPI and disable if present
        for prim in Usd.PrimRange(root_prim):
            # Disable PhysX rigid body API
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
            # Disable PhysX collisions
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
            # Take first child prim that is a USD mesh, i.e. has "points"-Attribute, as the mesh to create the UIPC mesh
            if self._usd_mesh_prim is None and prim.GetAttribute("points").IsValid():
                self._usd_mesh_prim = prim

        if self.cfg.usd_mesh_prim_name is not None:
            usd_mesh_prim = prims_utils.get_prim_at_path(
                str(self._prim_view.prims[0].GetPath())
                + f"/{self.cfg.usd_mesh_prim_name}"
            )
            if usd_mesh_prim.GetAttribute("points").IsValid():
                self._usd_mesh_prim = usd_mesh_prim

        if self._usd_mesh_prim is None:
            raise RuntimeWarning(
                f"No USD geom mesh found for uipc object at {prim_paths_expr}."
            )

        print(
            "USD mesh that is used for creating the UIPC Mesh: ",
            self._usd_mesh_prim.GetPath(),
        )
        self._usd_geom_mesh = UsdGeom.Mesh(self._usd_mesh_prim)

        self.stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())

        self.uipc_scene_objects = []
        self.geo_slot_list = []

        self.uipc_meshes = []

        self._data = None

        self._state_accessor = None

        # create and setup uipc mesh
        uipc_mesh: SimplicialComplex = self._setup_uipc_mesh()
        self.uipc_meshes.append(uipc_mesh)

        # create uipc scene object
        obj = self._uipc_sim.scene.objects().create(self.cfg.prim_path)
        self.uipc_scene_objects.append(obj)

        # add constraints
        self.constraint: UipcConstraint = None
        if type(self.cfg.constraint_cfg) is UipcConstraintCfg:
            self.constraint: UipcConstraint = UipcConstraint(
                self.cfg.constraint_cfg, self
            )
        elif type(self.cfg.constraint_cfg) is UipcIsaacAttachmentsCfg:
            self.constraint: UipcIsaacAttachments = UipcIsaacAttachments(
                self.cfg.constraint_cfg, self
            )

        obj_geo_slot = self._spawn_uipc_scene_object(obj, uipc_mesh)
        self.geo_slot_list.append(obj_geo_slot)

        # libuipc uses different indexing for the surface topology, so we need to extract it for rendering
        surf = extract_surface(uipc_mesh)
        surf_points_world = surf.positions().view().reshape(-1, 3)
        surf_tri = surf.triangles().topo().view().reshape(-1).tolist()
        surf_tri_orient = surf.triangles().find(builtin.orient).view()

        self.sim_surf_triangles: np.ndarray = np.array(surf_tri, dtype=np.int64).reshape(-1, 3)
        """Triangles of the simulation surface mesh (indices into the surface vertices of the uipc object)."""

        self._render_mapper: RenderMeshMapper | None = None
        self._resolve_render_mesh_cfg()

        # TODO handle multi env
        if self.cfg.render_mesh_cfg is None:
            fabric_prim = self._setup_render_mesh(
                self._usd_geom_mesh, surf_points_world, surf_tri, surf_tri_orient
            )
            self._render_geom_mesh = self._usd_geom_mesh
        else:
            fabric_prim = self._setup_separate_render_mesh(uipc_mesh)
        self.fabric_prim = fabric_prim

        # register the render mesh for embedding (`UipcSim.setup_sim`) and per-frame updates (`update_render_meshes`)
        if self.cfg.render_mesh_cfg is not None:
            self._uipc_sim._render_mesh_owners[len(self._uipc_sim._fabric_meshes)] = self
        # add fabric meshes to uipc sim class for updating the render meshes
        self._uipc_sim._fabric_meshes.append(fabric_prim)

        # save surface offsets for finding corresponding surface points of the meshes for rendering
        num_surf_points = surf_points_world.shape[0]
        self._uipc_sim._surf_vertex_offsets.append(
            self._uipc_sim._surf_vertex_offsets[-1] + num_surf_points
        )

        # simulation mesh visualization (debug_vis). `AssetBase.__init__` already requested the debug vis state
        # before the uipc setup existed -> apply it now.
        self._sim_mesh_vis_prim: UsdGeom.BasisCurves | None = None
        self._sim_mesh_vis_fabric: usdrt.Usd.Prim | None = None
        self._sim_mesh_vis_visible = False
        self._sim_surf_points_w: np.ndarray = np.asarray(surf_points_world)
        self._uipc_setup_done = True
        self._set_debug_vis_impl(getattr(self, "_debug_vis_requested", self.cfg.debug_vis))

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._prim_view.count

    @property
    def uipc_sim(self) -> UipcSim:
        """uipc simulation instance of this uipc object."""
        return self._uipc_sim

    # TODO adjust for multi env
    @property
    def global_vertex_offset(self) -> int:
        geo_slot = self.geo_slot_list[0].geometry()
        global_vertex_offset = geo_slot.meta().find(builtin.global_vertex_offset)

        return global_vertex_offset.view()

    @property
    def vertex_count(self) -> int:
        geo_slot = self.geo_slot_list[0].geometry()
        vertex_count = geo_slot.positions().view().shape[0]

        return vertex_count

    """
    Internal helper.
    """

    @abstractmethod
    def _setup_uipc_mesh(self) -> SimplicialComplex:
        """Generates a mesh inside the uipc simulation.

        Raises:
            NotImplementedError: _description_

        Returns:
            SimplicialComplex: The SimplicialComplex from uipc that contains the mesh data.
        """
        raise NotImplementedError

    def _resolve_render_mesh_cfg(self):
        """Applies the defaults of the render mesh config (separate render mesh + mapping for voxel tet meshes)."""
        is_voxel = isinstance(self.cfg.mesh_cfg, VoxelTetMeshCfg)
        render_cfg = self.cfg.render_mesh_cfg
        if render_cfg is None:
            if not is_voxel:
                return
            render_cfg = RenderMeshCfg()
            omni.log.info(f"Voxel tet mesh for {self.cfg.prim_path}: rendering the original USD mesh.")
        if render_cfg.mapping is None:
            render_cfg = render_cfg.replace(mapping="tetrahedral" if is_voxel else "surface")
        elif is_voxel and render_cfg.mapping == "surface":
            omni.log.warn(
                f"Surface render mapping with a voxel tet mesh for {self.cfg.prim_path}: the voxel staircase leads to"
                " large normal offsets and render artifacts. Consider mapping='tetrahedral'."
            )
        # replace instead of mutating the (possibly shared) user cfg
        self.cfg = self.cfg.replace(render_mesh_cfg=render_cfg)

    def _sim_nodal_positions_w(self) -> np.ndarray:
        """Current positions of all simulation (tet) mesh vertices in world frame. Shape (num_vertices, 3).

        Used for the ``"tetrahedral"`` render mesh mapping.
        """
        raise NotImplementedError(
            f"Tetrahedral render mesh mapping is not supported for {type(self).__name__}. Use mapping='surface'."
        )

    def init_render_mesh_mapping(self, surf_points_w: np.ndarray):
        """Embeds the render mesh into the simulation mesh. Requires a separate render mesh (``render_mesh_cfg``).

        Called by :meth:`UipcSim.setup_sim` after the uipc world is initialized, so that the rest configuration of
        the simulation mesh is taken from the same source as during the simulation.

        Args:
            surf_points_w: Simulation surface vertex positions of this object in world frame (rest configuration).
        """
        self.update_sim_mesh_vis(surf_points_w)
        render_cfg = self.cfg.render_mesh_cfg
        if render_cfg.mapping == "surface":
            sim_points_w, sim_elements = surf_points_w, self.sim_surf_triangles
        else:
            sim_points_w, sim_elements = self._sim_nodal_positions_w(), self._sim_tets

        self._render_mapper = RenderMeshMapper(
            self._render_points_w,
            sim_points_w,
            sim_elements,
            method=render_cfg.mapping,
            device=self._uipc_sim.cfg.device if torch.cuda.is_available() else "cpu",
        )
        omni.log.info(
            f"Render mesh {self._render_geom_mesh.GetPath()} embedded into uipc mesh of {self.cfg.prim_path} "
            f"({render_cfg.mapping}): {self._render_points_w.shape[0]} render vertices, max distance to sim element "
            f"{self._render_mapper.max_offset:.2e} m, {self._render_mapper.num_outside} outside."
        )
        if render_cfg.mapping == "tetrahedral" and self._render_mapper.num_outside > 0:
            omni.log.warn(
                f"{self._render_mapper.num_outside} render vertices of {self._render_geom_mesh.GetPath()} lie outside"
                " of the simulation tet mesh and are extrapolated."
            )

    def compute_render_points(self, surf_points_w: np.ndarray) -> np.ndarray:
        """Computes the render mesh vertex positions (world frame) from the current simulation state.

        Args:
            surf_points_w: Current simulation surface vertex positions of this object in world frame.

        Returns:
            Render vertex positions in world frame, shape (num_render_points, 3), float32.
        """
        if self._render_mapper is None:
            return self._render_points_w.astype(np.float32)
        if self._render_mapper.method == "surface":
            sim_points = surf_points_w
        else:
            sim_points = self._sim_nodal_positions_w()
        return self._render_mapper.map(sim_points).cpu().numpy()

    def _spawn_uipc_scene_object(
        self, obj, uipc_mesh: SimplicialComplex
    ) -> SimplicialComplexSlot:
        # spawn mesh inside uipc simulation
        obj_geo_slot, obj_rest_geo_slot = obj.geometries().create(uipc_mesh)
        self.obj_id = obj_geo_slot.id()
        print(f"obj id of {self.cfg.prim_path}: {self.obj_id} ")

        return obj_geo_slot

    def _setup_render_mesh(
        self,
        gprim: UsdGeom.Mesh,
        surf_points: np.array,
        surf_tri: np.array,
        surf_tri_orient: np.array,
    ) -> usdrt.Usd.Prim:
        usd_mesh_path = str(gprim.GetPath())

        # update the isaac surface mesh with the new topology
        MeshGenerator.update_usd_mesh(
            gprim=gprim, surf_points=surf_points, triangles=surf_tri
        )

        # setup mesh updates via Fabric
        fabric_prim = self.stage.GetPrimAtPath(usdrt.Sdf.Path(usd_mesh_path))
        if not fabric_prim:
            print(f"Prim at path {usd_mesh_path} is not in Fabric")
        if not fabric_prim.HasAttribute("points"):
            print(f"Prim at path {usd_mesh_path} does not have points attribute")

        # Tell OmniHydra to render points from Fabric
        if not fabric_prim.HasAttribute("Deformable"):
            fabric_prim.CreateAttribute(
                "Deformable", usdrt.Sdf.ValueTypeNames.PrimTypeTag, True
            )

        # Set xform transformation to identity, since uipc data is defined in world frame
        stage_id = self.stage.GetStageIdAsStageId()
        fabric_id = self.stage.GetFabricId()
        hier = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
            fabric_id, stage_id
        )
        hier.set_world_xform(usdrt.Sdf.Path(usd_mesh_path), usdrt.Gf.Matrix4d(1))
        hier.update_world_xforms()

        # # update fabric mesh with points defined in world frame from uipc
        # fabric_mesh_points_attr = fabric_prim.GetAttribute("points")
        # fabric_mesh_points_attr.Set(usdrt.Vt.Vec3fArray(surf_points))

        if self.cfg.debug_vis:
            add_barycentric_primvar(gprim)
            mat_path = "/World/Materials/TriangleOutlineMat"
            stage = omni.usd.get_context().get_stage()
            mat = stage.GetPrimAtPath(mat_path)
            if not mat.IsValid():
                mat = create_surf_tri_vis_material(
                    mat_path=mat_path,
                    outline_color=(0.8, 0.8, 0.8),  # white outlines
                    outline_width=0.05,
                    base_color=(0.0, 0.0, 0.8),  # blue mesh
                )

            # bind material with fabric
            rel = fabric_prim.GetRelationship(usdrt.UsdShade.Tokens.materialBinding)
            rel.SetTargets([mat_path])

        return fabric_prim

    def _setup_separate_render_mesh(self, uipc_mesh: SimplicialComplex) -> usdrt.Usd.Prim:
        """Keeps the original USD mesh as render mesh.

        The embedding into the simulation mesh is computed later in :meth:`init_render_mesh_mapping`.
        """
        render_cfg = self.cfg.render_mesh_cfg

        # resolve render mesh prim
        if render_cfg.prim_name is None:
            render_prim = self._usd_mesh_prim
        else:
            render_prim = prims_utils.get_prim_at_path(
                str(self._prim_view.prims[0].GetPath()) + f"/{render_cfg.prim_name}"
            )
            if not render_prim.IsValid() or not render_prim.GetAttribute("points").IsValid():
                raise RuntimeError(
                    f"Render mesh '{render_cfg.prim_name}' of uipc object {self.cfg.prim_path} is not a USD mesh."
                )
            if render_prim != self._usd_mesh_prim:
                # the mesh used for the tet mesh is not updated anymore -> hide it
                UsdGeom.Imageable(self._usd_mesh_prim).MakeInvisible()
        render_mesh = UsdGeom.Mesh(render_prim)
        self._render_geom_mesh = render_mesh

        if render_cfg.clear_normals and render_mesh.GetNormalsAttr().HasAuthoredValue():
            render_mesh.GetNormalsAttr().Set([])
        # face-varying normals primvar is also used by some importers
        pv_api = UsdGeom.PrimvarsAPI(render_mesh)
        if render_cfg.clear_normals and pv_api.HasPrimvar("normals"):
            pv_api.RemovePrimvar("normals")

        # rest configuration of the render mesh in world frame
        render_points = np.array(render_mesh.GetPointsAttr().Get(), dtype=np.float64).reshape(-1, 3)
        tf_world = np.array(omni.usd.get_world_transform_matrix(render_mesh)).T
        self._render_points_w = render_points @ tf_world[:3, :3].T + tf_world[:3, 3]

        if render_cfg.mapping == "tetrahedral":
            # fail early for objects without nodal access (e.g. rigid objects)
            if type(self)._sim_nodal_positions_w is UipcObject._sim_nodal_positions_w:
                self._sim_nodal_positions_w()
            self._sim_tets = np.asarray(uipc_mesh.tetrahedra().topo().view()).reshape(-1, 4)
        elif render_cfg.mapping != "surface":
            raise ValueError(f"Unknown render mesh mapping '{render_cfg.mapping}'. Options: 'surface', 'tetrahedral'.")

        # setup mesh updates via Fabric
        render_path = str(render_prim.GetPath())
        fabric_prim = self.stage.GetPrimAtPath(usdrt.Sdf.Path(render_path))
        if not fabric_prim or not fabric_prim.HasAttribute("points"):
            raise RuntimeError(f"Render mesh {render_path} is not in Fabric or has no points attribute.")
        if not fabric_prim.HasAttribute("Deformable"):
            fabric_prim.CreateAttribute("Deformable", usdrt.Sdf.ValueTypeNames.PrimTypeTag, True)

        # uipc data is defined in world frame -> identity world xform for the render mesh
        hier = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
            self.stage.GetFabricId(), self.stage.GetStageIdAsStageId()
        )
        hier.set_world_xform(usdrt.Sdf.Path(render_path), usdrt.Gf.Matrix4d(1))
        hier.update_world_xforms()

        # initial render points (world frame), otherwise the mesh is shown with the identity xform until the first step
        fabric_prim.GetAttribute("points").Set(usdrt.Vt.Vec3fArray(self._render_points_w.astype(np.float32)))

        return fabric_prim

    """
    Simulation mesh visualization.
    """

    @property
    def sim_mesh_vis_prim_path(self) -> str:
        """Path of the simulation mesh visualization prim (only created with a separate render mesh)."""
        return "/Visuals/UipcSimMeshes/" + re.sub(r"[^A-Za-z0-9_]", "_", self.cfg.prim_path.strip("/"))

    def update_sim_mesh_vis(self, surf_points_w: np.ndarray):
        """Stores the current simulation surface points and updates the visualization if it is shown.

        Called by :meth:`UipcSim.update_render_meshes` for objects with a separate render mesh.
        """
        self._sim_surf_points_w = np.asarray(surf_points_w)
        if self._sim_mesh_vis_visible:
            edge_points = self._sim_surf_points_w[self._sim_mesh_vis_edges].reshape(-1, 3)
            self._sim_mesh_vis_fabric.GetAttribute("points").Set(usdrt.Vt.Vec3fArray(edge_points.astype(np.float32)))

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Shows/hides the simulation mesh (tet mesh surface) as wireframe overlay.

        Only for objects with a separate render mesh: otherwise the rendered mesh is the simulation surface and
        ``debug_vis`` binds the triangle outline material to it at setup instead.
        """
        self._debug_vis_requested = debug_vis
        # called by `AssetBase.__init__` before the uipc setup -> applied at the end of `UipcObject.__init__`
        if not getattr(self, "_uipc_setup_done", False) or self.cfg.render_mesh_cfg is None:
            return
        if debug_vis and self._sim_mesh_vis_prim is None:
            self._create_sim_mesh_vis()
        if self._sim_mesh_vis_prim is None:
            return
        self._sim_mesh_vis_visible = debug_vis
        if debug_vis:
            self._sim_mesh_vis_prim.MakeVisible()
            self.update_sim_mesh_vis(self._sim_surf_points_w)  # might be outdated while hidden
        else:
            self._sim_mesh_vis_prim.MakeInvisible()
        token = usdrt.UsdGeom.Tokens.inherited if debug_vis else usdrt.UsdGeom.Tokens.invisible
        self._sim_mesh_vis_fabric.GetAttribute("visibility").Set(token)

    def _debug_vis_callback(self, event):
        # sim mesh visualization is updated in `UipcSim.update_render_meshes` (with the simulation data)
        pass

    def _create_sim_mesh_vis(self):
        """Creates the simulation mesh wireframe under ``/Visuals`` (outside of the asset).

        A line wireframe (``UsdGeom.BasisCurves``, one segment per surface edge) instead of a mesh, so that the render
        mesh stays visible inside the simulation mesh (e.g. inside a voxel cage).
        """
        render_cfg = self.cfg.render_mesh_cfg
        tris = self.sim_surf_triangles
        edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
        self._sim_mesh_vis_edges = np.unique(np.sort(edges, axis=1), axis=0)
        edge_points = self._sim_surf_points_w[self._sim_mesh_vis_edges].reshape(-1, 3).astype(np.float32)

        width = render_cfg.sim_mesh_vis_width
        if width is None:
            width = 3e-3 * float(np.linalg.norm(np.ptp(self._sim_surf_points_w, axis=0)))

        stage = omni.usd.get_context().get_stage()
        curves = UsdGeom.BasisCurves.Define(stage, self.sim_mesh_vis_prim_path)
        curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        curves.CreateCurveVertexCountsAttr([2] * self._sim_mesh_vis_edges.shape[0])
        curves.CreatePointsAttr(edge_points)
        curves.CreateWidthsAttr([width])
        curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curves.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*render_cfg.sim_mesh_vis_color)])

        self._sim_mesh_vis_prim = curves
        self._sim_mesh_vis_fabric = self.stage.GetPrimAtPath(usdrt.Sdf.Path(self.sim_mesh_vis_prim_path))

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
