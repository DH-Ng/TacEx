"""UIPC x Isaac attachment via a UIPC rigid "carrier" body.

Alternative to :class:`UipcIsaacAttachments`, whose ``SoftPositionConstraint`` pulls the attached UIPC vertices to
their aim positions with a stiffness proportional to the (tiny) vertex masses, so a soft gel slides on the sensor
under load. Here the UIPC soft body is stitched to a UIPC affine body (the "carrier") with the shape of the Isaac
body, and the robot moves the carrier:

    Isaac body pose ---SoftTransformConstraint---> carrier (UIPC affine body) ---stitch---> soft body vertices

Both links are still soft (libuipc has no hard moving constraint), but stiffer at the same solver cost: the robot
drives one 12-DOF body whose mass is chosen freely (the transform constraint is mass-weighted), and the stitch
stiffness is not scaled by the soft body's vertex masses.

Carrier geometry: by default the convex hull of every enabled collision mesh of the Isaac body (the approximation
PhysX uses for them), tetrahedralized as a fan from the hull centroid. The carrier does not collide with UIPC bodies
(it touches/overlaps the soft body by construction; IPC needs an intersection-free start).

Stitch (``stitch_type``):

- ``vertex_triangle`` (default): ``SoftVertexTriangleStitch``. Every attached soft body vertex is paired with the
  carrier surface triangles within ``attachment_points_radius``; each pair forms an elastic tetrahedron
  (``stitch_youngs_modulus``) whose rest shape keeps the vertex's offset to the triangle. No extra carrier vertices.
- ``vertex``: ``SoftVertexStitch`` (zero rest length, ``stitch_kappa``). One extra carrier vertex is added at every
  attached soft body vertex (not part of any tet, moves rigidly with the carrier) as its stitch partner.

Create the attachment in ``_setup_scene`` **after** all other UIPC objects (the carrier registers its surface
vertices last in ``UipcSim._surf_vertex_offsets``) and before the UIPC world is set up.
"""

from __future__ import annotations

import numpy as np
import torch
import weakref
from typing import TYPE_CHECKING, Literal

import omni.timeline
import omni.usd
import usdrt
import usdrt.hierarchy
from pxr import Gf, Sdf, Usd, UsdGeom
from scipy.spatial import ConvexHull, QhullError

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils import configclass
from isaaclab.sim.utils import find_matching_prim_paths

from uipc import Animation, builtin, view
from uipc.constitution import (
    AffineBodyConstitution,
    ElasticModuli,
    SoftTransformConstraint,
    SoftVertexStitch,
    SoftVertexTriangleStitch,
)
from uipc.geometry import closest_vertex_triangle_pairs, extract_surface, label_surface, label_triangle_orient, tetmesh
from uipc.unit import MPa

from ...utils import MeshGenerator, TetMeshCfg
from .geometric_attachment import _collision_prims, _to_world, _world_matrix, compute_attachment_data_geometric

if TYPE_CHECKING:
    from ...sim import UipcSim
    from ..uipc_object import UipcObject


@configclass
class UipcIsaacCarrierAttachmentsCfg:
    """Config of :class:`UipcIsaacCarrierAttachments`."""

    isaaclab_rigid_body_prim_path: str = None
    """Prim path (expression) of the Isaac rigid object/articulation, e.g. ``/World/envs/env_.*/Robot``."""

    body_name: str | None = None
    """Body of the articulation the soft body is attached to (e.g. ``gelsight_mini_case``)."""

    attachment_points_radius: float = 5e-4
    """Soft body vertices within this distance of the body's collision shapes are stitched to the carrier [m]."""

    carrier_shape: Literal["convex_hull", "mesh"] = "convex_hull"
    """Carrier geometry per collision mesh of the body: its convex hull, or the tetrahedralized mesh itself."""

    carrier_mesh_cfg: TetMeshCfg = TetMeshCfg()
    """Tet mesh settings for ``carrier_shape="mesh"``."""

    carrier_stiffness: float = 100.0
    """Stiffness of the carrier affine body [MPa]."""

    carrier_mass_density: float = 1e4
    """Mass density of the carrier [kg/m^3]. Heavier = holds its pose better against contact forces."""

    transform_strength_ratio: float = 100.0
    """Strength of the transform constraint (translation and rotation), relative to the carrier's mass.

    The effective stiffness is ``carrier_mass_density * transform_strength_ratio``. 1e4 * 100 keeps the gel's underside
    within ~0.04 mm of the case under load; 1e4 * 1000 makes the UIPC solver stall when the robot rotates the sensor
    while the gel is in contact (ball rolling demo, hand rotation at goal 3).
    """

    stitch_type: Literal["vertex_triangle", "vertex"] = "vertex_triangle"
    """How the soft body is stitched to the carrier (see module docstring)."""

    stitch_youngs_modulus: float = 1e7
    """Young's modulus of the vertex-triangle stitch tetrahedra [Pa] (``stitch_type="vertex_triangle"``)."""

    stitch_min_separate_distance: float = 1e-4
    """Vertex-triangle stitch: if a soft body vertex is closer than this to its triangle's plane at rest, its rest
    position is moved to this distance (degenerate tetrahedra). Small = the soft body is not pushed away from the
    carrier, but thinner (stiffer) tetrahedra [m]."""

    stitch_kappa: float = 1e8
    """Stiffness of the vertex-vertex stitch (``stitch_type="vertex"``)."""

    debug_vis: bool = False
    """Render the carrier in Isaac Sim as a plain mesh (its UIPC surface, updated every step when
    ``UipcSimCfg.update_render_meshes`` is on)."""

    debug_vis_color: tuple[float, float, float] = (0.9, 0.5, 0.1)
    """Color of the carrier render mesh."""


class UipcIsaacCarrierAttachments:
    """Attaches a UIPC soft body to an Isaac body through a UIPC affine "carrier" body (see module docstring)."""

    def __init__(
        self,
        cfg: UipcIsaacCarrierAttachmentsCfg,
        uipc_object: UipcObject,
        isaaclab_rigid_object: RigidObject | Articulation,
    ):
        cfg.validate()
        self.cfg = cfg.copy()
        self.uipc_object = uipc_object
        uipc_sim = uipc_object.uipc_sim
        self.uipc_sim = uipc_sim
        self.isaaclab_rigid_object = isaaclab_rigid_object
        self.rigid_body_id = None
        self._is_initialized = False

        body_path = cfg.isaaclab_rigid_body_prim_path + (f"/.*{cfg.body_name}" if cfg.body_name else "")
        soft_points = np.asarray(uipc_object.uipc_meshes[0].positions().view()).reshape(-1, 3)

        # --- soft body vertices touching the body (indices into the soft body's vertices)
        _, idx, matching, attached_pos, _ = compute_attachment_data_geometric(
            body_path, soft_points, sphere_radius=cfg.attachment_points_radius
        )
        self.attachment_points_idx = np.asarray(idx, dtype=np.int32)
        self.attachment_points_init_pos = np.asarray(attached_pos, dtype=np.float64)
        body_prim = matching[0]

        # --- carrier pose at spawn (USD, column-vector convention), same source as the carrier points
        self._T_body_spawn = np.array(omni.usd.get_world_transform_matrix(body_prim), dtype=np.float64).T

        # --- carrier mesh (world points, identity instance transform); vertex stitch: + extra stitch vertices
        carrier_points, carrier_tets = self._carrier_geometry(body_prim)
        num_shape_points = carrier_points.shape[0]
        points = carrier_points
        if cfg.stitch_type == "vertex":
            points = np.vstack([carrier_points, self.attachment_points_init_pos])
        carrier = tetmesh(points.copy(), carrier_tets.copy())
        label_surface(carrier)
        label_triangle_orient(carrier)
        AffineBodyConstitution().apply_to(carrier, cfg.carrier_stiffness * MPa, mass_density=cfg.carrier_mass_density)
        SoftTransformConstraint().apply_to(
            carrier, np.array([cfg.transform_strength_ratio, cfg.transform_strength_ratio], dtype=np.float64)
        )

        # the carrier touches/overlaps the soft body by construction -> no contact with UIPC bodies
        scene = uipc_sim.scene
        tabular = scene.contact_tabular()
        carrier_element = tabular.create("carrier")
        tabular.insert(carrier_element, tabular.default_element(), 0.0, 0.0, False)
        tabular.insert(carrier_element, carrier_element, 0.0, 0.0, False)
        carrier_element.apply_to(carrier)

        name = f"{uipc_object.cfg.prim_path}_carrier"
        self.carrier_object = scene.objects().create(name)
        self.carrier_slot, _ = self.carrier_object.geometries().create(carrier)
        self.carrier_obj_id = self.carrier_slot.id()
        # register the carrier's surface vertices (the scene surface is indexed per object in creation order)
        surf = extract_surface(carrier)
        surf_points = np.asarray(surf.positions().view()).reshape(-1, 3)
        uipc_sim._surf_vertex_offsets.append(uipc_sim._surf_vertex_offsets[-1] + surf_points.shape[0])
        self.render_prim_path = None
        if cfg.debug_vis:
            self._setup_render_mesh(name, surf_points, np.asarray(surf.triangles().topo().view()).reshape(-1))

            # # Hide original isaac lab rigid body meshes to show UIPC carrier mesh instead
            # stage = omni.usd.get_context().get_stage()
            # prim_paths = find_matching_prim_paths(self.isaaclab_rigid_object.cfg.prim_path)

            # for prim_path in prim_paths:
            #     prim = stage.GetPrimAtPath(prim_path)
            #     if prim.IsValid():
            #         UsdGeom.Imageable(prim).MakeInvisible()

        # --- stitch soft body <-> carrier
        soft_slot, soft_rest_slot = scene.geometries().find(uipc_object.obj_id)
        if cfg.stitch_type == "vertex_triangle":
            stitch = self._vertex_triangle_stitch(carrier, soft_slot, soft_rest_slot)
        elif cfg.stitch_type == "vertex":
            pairs = np.stack(
                [
                    self.attachment_points_idx,
                    num_shape_points + np.arange(len(self.attachment_points_idx), dtype=np.int32),
                ],
                axis=1,
            ).astype(np.int32)
            stitch = SoftVertexStitch().create_geometry((soft_slot, self.carrier_slot), pairs, cfg.stitch_kappa)
            self.num_stitch_pairs, self.num_stitched_vertices = len(pairs), len(pairs)
        else:
            raise ValueError(f"Unknown stitch_type '{cfg.stitch_type}', expected 'vertex_triangle' or 'vertex'.")
        self.stitch_object = scene.objects().create(f"{name}_stitch")
        self.stitch_object.geometries().create(stitch)

        print(
            f"[INFO] UIPC carrier attachment for {uipc_object.cfg.prim_path} -> {body_prim.GetPath()}:"
            f" {len(self.attachment_points_idx)} attached vertices, {self.num_stitched_vertices} stitched"
            f" ({self.num_stitch_pairs} {cfg.stitch_type} pairs), carrier with {num_shape_points} points /"
            f" {carrier_tets.shape[0]} tets ({cfg.carrier_shape})"
        )

        self._create_animation()
        timeline = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        self._initialize_handle = timeline.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY),
            lambda event, obj=weakref.proxy(self): obj._initialize_callback(event),
            order=10,
        )
        self._invalidate_initialize_handle = timeline.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP),
            lambda event, obj=weakref.proxy(self): obj._invalidate_initialize_callback(event),
            order=10,
        )

    def __del__(self):
        for handle in ("_initialize_handle", "_invalidate_initialize_handle"):
            if getattr(self, handle, None) is not None:
                getattr(self, handle).unsubscribe()
                setattr(self, handle, None)

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    """
    Carrier pose.
    """

    def body_pose_matrix(self) -> np.ndarray:
        """Current pose (4, 4, column-vector convention) of the Isaac body, from the physics simulation."""
        if isinstance(self.isaaclab_rigid_object, Articulation):
            poses = self.isaaclab_rigid_object.root_physx_view.get_link_transforms().clone()
            pose = poses[0, self.rigid_body_id[0]]
        else:
            pose = self.isaaclab_rigid_object.root_physx_view.get_transforms().clone()[0]
        pos = pose[:3]
        quat = math_utils.convert_quat(pose[3:7], to="wxyz")
        T = np.eye(4)
        T[:3, :3] = math_utils.matrix_from_quat(quat).cpu().numpy()
        T[:3, 3] = pos.cpu().numpy()
        return T

    def aim_transform(self) -> np.ndarray:
        """Aim transform of the carrier: its points are given in world at the body's spawn pose."""
        return self.body_pose_matrix() @ np.linalg.inv(self._T_body_spawn)

    def aim_positions(self) -> np.ndarray:
        """Where the attached soft body vertices should be (world), given the current body pose."""
        T = self.aim_transform()
        return self.attachment_points_init_pos @ T[:3, :3].T + T[:3, 3]

    """
    Internal helpers.
    """

    def _carrier_geometry(self, body_prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
        """Points (world) and tets of the carrier: one convex hull (or tet mesh) per collision mesh, merged."""
        points, tets = [], []
        offset = 0
        for prim in _collision_prims(body_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if self.cfg.carrier_shape == "convex_hull":
                local = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
                p, t = _hull_tets(_to_world(local, _world_matrix(prim)))
            else:
                p, t, _, _ = MeshGenerator(config=self.cfg.carrier_mesh_cfg).generate_tet_mesh_for_prim(
                    UsdGeom.Mesh(prim)
                )
                p = _to_world(np.asarray(p, dtype=np.float64), _world_matrix(prim))
                t = np.asarray(t, dtype=np.int32).reshape(-1, 4)
            points.append(p)
            tets.append(t + offset)
            offset += p.shape[0]
        if not points:
            raise RuntimeError(f"UIPC carrier attachment: '{body_prim.GetPath()}' has no collision meshes.")
        return np.vstack(points), np.vstack(tets).astype(np.int32)

    def _vertex_triangle_stitch(self, carrier, soft_slot, soft_rest_slot):
        """SoftVertexTriangleStitch between the attached soft body vertices and the carrier's surface triangles."""
        cfg = self.cfg
        # restrict the pair search to the attached soft body vertices and the carrier's surface triangles (a tet mesh's
        # triangles also contain the interior faces)
        group = "carrier_stitch_group"
        tri_group = carrier.triangles().create(group, np.int32(0))
        view(tri_group)[:] = np.asarray(view(carrier.triangles().find(builtin.is_surf))).reshape(-1)
        soft_mesh = self.uipc_object.uipc_meshes[0]
        vert_group = soft_mesh.vertices().find(group) or soft_mesh.vertices().create(group, np.int32(0))
        selected = np.zeros(soft_mesh.vertices().size(), dtype=np.int32)
        selected[self.attachment_points_idx] = 1
        view(vert_group)[:] = selected

        pairs = closest_vertex_triangle_pairs(soft_mesh, carrier, cfg.attachment_points_radius + 1e-5, "", group)
        topo = np.asarray(view(pairs.instances().find("topo"))).reshape(-1, 2)
        if len(topo) == 0:
            raise RuntimeError(
                "UIPC carrier attachment: no soft body vertex lies within"
                f" {cfg.attachment_points_radius} m of a carrier surface triangle."
            )
        self.num_stitch_pairs = len(topo)
        self.num_stitched_vertices = len(np.unique(topo[:, 0]))

        _, carrier_rest_slot = self.uipc_sim.scene.geometries().find(self.carrier_obj_id)
        return SoftVertexTriangleStitch().create_geometry(
            (soft_slot, self.carrier_slot),
            (soft_rest_slot, carrier_rest_slot),
            pairs,
            ElasticModuli.youngs_poisson(cfg.stitch_youngs_modulus, 0.49),
            cfg.stitch_min_separate_distance,
        )

    def _setup_render_mesh(self, name: str, surf_points: np.ndarray, surf_triangles: np.ndarray):
        """Plain USD mesh of the carrier surface, updated from UIPC every step by ``UipcSim.update_render_meshes``.

        Same Fabric setup as ``UipcObject._setup_render_mesh`` (without its debug material): the mesh's points are the
        carrier's UIPC surface points in world coordinates, so its world transform is the identity.
        """

        uipc_sim = self.uipc_sim
        # update_render_meshes pairs _fabric_meshes[i] with _surf_vertex_offsets[i]:[i + 1]: the carrier (registered
        # last in the offsets) must also be the last render mesh
        if len(uipc_sim._fabric_meshes) != len(uipc_sim._surf_vertex_offsets) - 2:
            raise RuntimeError(
                "UIPC carrier render mesh: create the carrier attachment after all other UIPC objects"
                f" ({len(uipc_sim._fabric_meshes)} render meshes, {len(uipc_sim._surf_vertex_offsets) - 1} surfaces)."
            )
        stage = omni.usd.get_context().get_stage()
        path = "/Visuals/UipcCarrier/" + "".join(c if c.isalnum() else "_" for c in name.strip("/"))
        gprim = UsdGeom.Mesh.Define(stage, path)
        MeshGenerator.update_usd_mesh(gprim=gprim, surf_points=surf_points, triangles=surf_triangles.tolist())
        gprim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*self.cfg.debug_vis_color)])
        # like the gelpad mesh (Gelpad_*.usd): not seen by the sensor camera, no shadows on the gel
        primvars = UsdGeom.PrimvarsAPI(gprim)
        primvars.CreatePrimvar("invisibleToSecondaryRays", Sdf.ValueTypeNames.Bool).Set(True)
        primvars.CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)

        fabric_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
        fabric_prim = fabric_stage.GetPrimAtPath(usdrt.Sdf.Path(path))
        if not fabric_prim.HasAttribute("Deformable"):  # tell OmniHydra to render the points from Fabric
            fabric_prim.CreateAttribute("Deformable", usdrt.Sdf.ValueTypeNames.PrimTypeTag, True)
        hier = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
            fabric_stage.GetFabricId(), fabric_stage.GetStageIdAsStageId()
        )
        hier.set_world_xform(usdrt.Sdf.Path(path), usdrt.Gf.Matrix4d(1))
        hier.update_world_xforms()

        uipc_sim._fabric_meshes.append(fabric_prim)
        self.render_prim_path = path

    def _create_animation(self):
        animator = self.uipc_sim.scene.animator()

        def animate_carrier(info: Animation.UpdateInfo):
            if not self._is_initialized:
                return
            geo = info.geo_slots()[0].geometry()
            view(geo.instances().find(builtin.is_constrained))[0] = 1
            view(geo.instances().find(builtin.aim_transform))[0] = self.aim_transform()

        animator.insert(self.carrier_object, animate_carrier)

    def _initialize_callback(self, event):
        if self._is_initialized:
            return

        if isinstance(self.isaaclab_rigid_object, Articulation):
            if self.cfg.body_name is None:
                raise ValueError("UIPC carrier attachment to an articulation needs `body_name`.")
            self.rigid_body_id, _ = self.isaaclab_rigid_object.find_bodies(self.cfg.body_name)

        self._is_initialized = True

    def _invalidate_initialize_callback(self, event):
        self._is_initialized = False


def _hull_tets(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convex hull of the points, tetrahedralized as a fan from its centroid. Returns (points, tets), positively oriented."""
    try:
        hull = ConvexHull(points)
    except QhullError:
        hull = ConvexHull(points, qhull_options="QJ")
    hull_pts = points[hull.vertices]
    remap = {v: i for i, v in enumerate(hull.vertices)}
    centroid = hull_pts.mean(axis=0)
    pts = np.vstack([hull_pts, centroid[None]])
    c = len(hull_pts)
    tets = []
    for tri in hull.simplices:
        a, b, d = (remap[v] for v in tri)
        tet = [c, a, b, d]
        vol = np.dot(np.cross(pts[a] - pts[c], pts[b] - pts[c]), pts[d] - pts[c])
        if vol < 0:
            tet = [c, a, d, b]
        tets.append(tet)
    return pts, np.asarray(tets, dtype=np.int32)
