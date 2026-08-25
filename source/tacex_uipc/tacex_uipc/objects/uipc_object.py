from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import omni.log
import omni.usd
import usdrt
import usdrt.UsdGeom
from isaacsim.core.prims import XFormPrim
import isaacsim.core.utils.prims as prims_utils
from pxr import UsdGeom, UsdPhysics, Sdf

try:
    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
except ImportError:
    import warnings

    warnings.warn("_debug_draw failed to import", ImportWarning)
    draw = None

import numpy as np
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
    add_barycentric_primvar,
    create_surf_tri_vis_material,
)


from .constraints import UipcConstraint, UipcConstraintCfg, UipcIsaacAttachments, UipcIsaacAttachmentsCfg

if TYPE_CHECKING:
    from tacex_uipc.sim import UipcSim


@configclass
class UipcObjectCfg(AssetBaseCfg):
    mesh_cfg: TetMeshCfg | None = None  # TODO make more General MeshCfg -> want to have Tet and Tri (for cloth) Meshes

    mass_density: float = 1e3

    constraint_cfg: UipcConstraintCfg | None = None

    usd_mesh_prim_name: str | None = None
    """The name of the usd mesh that should be used for the Tet Mesh.

    If `None`, then the first child of the prim at the given prim path is used.
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
        self._prim_view = XFormPrim(prim_paths_expr=prim_paths_expr, name=f"{prim_paths_expr}", usd=False)
        self._prim_view.initialize()

        # check if prim of uipc_object has PhysX rigid body API applied to it
        if UsdPhysics.RigidBodyAPI(self._prim_view.prims[0]):
            # if yes disable it, otherwise render errors
            UsdPhysics.RigidBodyAPI(self._prim_view.prims[0]).GetRigidBodyEnabledAttr().Set(False)

            # disable collisions
            for prim_child in self._prim_view.prims[0].GetChildren():  # todo properly deal with multiple meshs
                if UsdPhysics.CollisionAPI(prim_child):
                    UsdPhysics.CollisionAPI(prim_child).GetCollisionEnabledAttr().Set(False)

        # the isaac mesh that should be used for creating the Tet mesh
        if self.cfg.usd_mesh_prim_name is not None:
            self._usd_mesh_prim = prims_utils.get_prim_at_path(
                str(self._prim_view.prims[0].GetPath()) + f"/{self.cfg.usd_mesh_prim_name}"
            )
        else:
            # Take first child prim that is a USD mesh, i.e. has "points"-Attribute
            prim_children = self._prim_view.prims[0].GetChildren()
            for prim in prim_children:
                if prim.GetAttribute("points").IsValid():
                    self._usd_mesh_prim = prim
                    break

        print("USD mesh that is used for creating the UIPC Mesh: ", self._usd_mesh_prim.GetPath())
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
            self.constraint: UipcConstraint = UipcConstraint(self.cfg.constraint_cfg, self)
        elif type(self.cfg.constraint_cfg) is UipcIsaacAttachmentsCfg:
            self.constraint: UipcIsaacAttachments = UipcIsaacAttachments(self.cfg.constraint_cfg, self)

        obj_geo_slot = self._spawn_uipc_scene_object(obj, uipc_mesh)
        self.geo_slot_list.append(obj_geo_slot)

        # libuipc uses different indexing for the surface topology, so we need to extract it for rendering
        surf = extract_surface(uipc_mesh)
        surf_points_world = surf.positions().view().reshape(-1, 3)
        surf_tri = surf.triangles().topo().view().reshape(-1).tolist()
        surf_tri_orient = surf.triangles().find(builtin.orient).view()

        # TODO handle multi env
        fabric_prim = self._setup_render_mesh(self._usd_geom_mesh, surf_points_world, surf_tri, surf_tri_orient)
        self.fabric_prim = fabric_prim

        # add fabric meshes to uipc sim class for updating the render meshes
        self._uipc_sim._fabric_meshes.append(fabric_prim)

        # save surface offsets for finding corresponding surface points of the meshes for rendering
        num_surf_points = surf_points_world.shape[0]
        self._uipc_sim._surf_vertex_offsets.append(self._uipc_sim._surf_vertex_offsets[-1] + num_surf_points)

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

    def _spawn_uipc_scene_object(self, obj, uipc_mesh: SimplicialComplex) -> SimplicialComplexSlot:
        # spawn mesh inside uipc simulation
        obj_geo_slot, obj_rest_geo_slot = obj.geometries().create(uipc_mesh)
        self.obj_id = obj_geo_slot.id()
        print(f"obj id of {self.cfg.prim_path}: {self.obj_id} ")

        return obj_geo_slot

    def _setup_render_mesh(
        self, gprim: UsdGeom.Mesh, surf_points: np.array, surf_tri: np.array, surf_tri_orient: np.array
    ) -> usdrt.Usd.Prim:
        usd_mesh_path = str(gprim.GetPath())

        # update the isaac surface mesh with the new topology
        MeshGenerator.update_usd_mesh(gprim=gprim, surf_points=surf_points, triangles=surf_tri)

        # setup mesh updates via Fabric
        fabric_prim = self.stage.GetPrimAtPath(usdrt.Sdf.Path(usd_mesh_path))
        if not fabric_prim:
            print(f"Prim at path {usd_mesh_path} is not in Fabric")
        if not fabric_prim.HasAttribute("points"):
            print(f"Prim at path {usd_mesh_path} does not have points attribute")

        # Tell OmniHydra to render points from Fabric
        if not fabric_prim.HasAttribute("Deformable"):
            fabric_prim.CreateAttribute("Deformable", usdrt.Sdf.ValueTypeNames.PrimTypeTag, True)

        # Set xform transformation to identity, since uipc data is defined in world frame
        stage_id = self.stage.GetStageIdAsStageId()
        fabric_id = self.stage.GetFabricId()
        hier = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(fabric_id, stage_id)
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

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
