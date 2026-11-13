"""Geometric computation of UIPC x Isaac attachment data (alternative to the PhysX sweep).

:meth:`UipcIsaacAttachments.compute_attachment_data` finds the UIPC vertices that touch an Isaac rigid body with PhysX
sphere sweeps. These need PhysX scene queries, which Isaac Lab disables by default
(``SimulationCfg.enable_scene_query_support``), and they only see the body after PhysX has loaded the stage.

:func:`compute_attachment_data_geometric` computes the same data from the USD stage alone: it reads the body's
collision shapes, evaluates the collision approximation PhysX would use, and selects every vertex within
``sphere_radius`` of (or inside) a shape. No scene queries and no PhysX scene are needed, and the result is the same
headless or with a GUI.

Supported collision shapes:

- ``UsdGeom.Mesh`` with ``physics:approximation``

  - ``convexHull`` (and ``convexDecomposition``, approximated by the hull of the whole mesh, i.e. slightly larger),
  - ``boundingCube`` (hull of the local bounding box), ``boundingSphere`` (sphere around the local bounding box),
  - ``none`` / ``meshSimplification`` / ``sdf``: exact distance to the triangle mesh (via warp).

- ``UsdGeom.Cube`` and ``UsdGeom.Sphere``.

Other shapes raise ``NotImplementedError``.
"""

from __future__ import annotations

import numpy as np
import torch

import omni.usd
import warp as wp
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial import ConvexHull, QhullError

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils


def compute_attachment_data_geometric(
    isaac_mesh_path: str,
    tet_points: np.ndarray,
    tet_indices: np.ndarray | None = None,
    sphere_radius: float = 5e-4,
    max_dist: float = 1e-5,
):
    """Finds the UIPC vertices attached to an Isaac rigid body, from the body's USD collision shapes.

    Same interface and return values as :meth:`UipcIsaacAttachments.compute_attachment_data`, so it can be used as a
    drop-in replacement. A vertex is attached if its distance to a collision shape of the body is at most
    ``sphere_radius + max_dist`` (the reach of the PhysX sweep: a sphere of ``sphere_radius`` swept by ``max_dist``),
    or if it lies inside the shape.

    Args:
        isaac_mesh_path: Prim path (expression) of the rigid body. The first matching prim is used.
        tet_points: World positions of the UIPC vertices, shape (N, 3).
        tet_indices: Unused, for interface compatibility.
        sphere_radius: Attachment distance in meters.
        max_dist: Additional distance, for parity with the PhysX sweep.

    Returns:
        attachment_offsets (K, 3): attached vertex positions in the body frame,
        attachment_indices (list of K ints): indices into ``tet_points``,
        matching_prims: prims matching ``isaac_mesh_path``,
        attachment_points_positions (K, 3): world positions of the attached vertices,
        obj_pos (3,): world position of the body.

    Raises:
        RuntimeError: If no prim matches or no vertex is attached.
    """
    matching_prims = sim_utils.find_matching_prims(isaac_mesh_path)
    if len(matching_prims) == 0:
        raise RuntimeError(
            f"Could not find prim with path {isaac_mesh_path}. The body_name in the cfg might not exist."
        )
    body_prim = matching_prims[0]

    # body pose (same convention as the PhysX version)
    pose = omni.usd.get_world_transform_matrix(body_prim)
    obj_pos = np.array(pose.ExtractTranslation())
    q = pose.ExtractRotation().GetQuaternion()
    obj_quat = torch.tensor([[q.GetReal(), *q.GetImaginary()]], dtype=torch.float32)

    points = np.asarray(tet_points, dtype=np.float64).reshape(-1, 3)
    reach = sphere_radius + max_dist

    colliders = _collision_prims(body_prim)
    if not colliders:
        raise RuntimeError(f"UIPC x Isaac attachment: '{body_prim.GetPath()}' has no enabled collision shapes.")

    # distance of every vertex to the closest collision shape (<= 0 inside)
    dist = np.full(points.shape[0], np.inf)
    for prim in colliders:
        dist = np.minimum(dist, _distance_to_collider(prim, points))
    idx = np.nonzero(dist <= reach)[0]

    if len(idx) == 0:
        raise RuntimeError(
            f"UIPC x Isaac attachment: no UIPC vertex is within {reach} m of a collision shape of"
            f" '{body_prim.GetPath()}' (closest: {dist.min():.6f} m), so the UIPC object would not be attached."
            " Check that the UIPC mesh touches the body's collision shapes."
        )

    attachment_points_positions = points[idx]
    offsets = torch.tensor(attachment_points_positions - obj_pos, dtype=torch.float32)
    attachment_offsets = math_utils.quat_apply_inverse(obj_quat.expand(len(idx), 4), offsets).numpy()

    return attachment_offsets, idx.tolist(), matching_prims, attachment_points_positions, obj_pos


"""
Collision shapes.
"""


def _collision_prims(body_prim: Usd.Prim) -> list[Usd.Prim]:
    """Enabled collision shapes of the body (its descendants, including instance proxies, excluding nested bodies)."""
    colliders = []
    it = iter(Usd.PrimRange(body_prim, Usd.TraverseInstanceProxies()))
    for prim in it:
        if prim != body_prim and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            it.PruneChildren()  # shapes of a nested rigid body belong to that body
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if enabled is False:
            continue
        colliders.append(prim)
    return colliders


def _world_matrix(prim: Usd.Prim) -> np.ndarray:
    """Local-to-world transform (4, 4) in USD's row-vector convention: ``[p, 1] @ M``."""
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64)


def _to_world(local_points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([local_points, np.ones((local_points.shape[0], 1))])
    return (homogeneous @ matrix)[:, :3]


def _distance_to_collider(prim: Usd.Prim, points: np.ndarray) -> np.ndarray:
    """Distance (N,) of the points to the collision approximation of ``prim`` (<= 0 inside, for convex shapes)."""
    matrix = _world_matrix(prim)

    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        local_pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        approximation = "none"
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or "none"

        if approximation in ("convexHull", "convexDecomposition"):
            return _distance_to_hull(_to_world(local_pts, matrix), points)
        if approximation == "boundingCube":
            return _distance_to_hull(_to_world(_box_corners(local_pts.min(0), local_pts.max(0)), matrix), points)
        if approximation == "boundingSphere":
            lo, hi = local_pts.min(0), local_pts.max(0)
            return _distance_to_sphere((lo + hi) / 2, np.linalg.norm(hi - lo) / 2, matrix, points)
        if approximation in ("none", "meshSimplification", "sdf"):
            faces = _triangulate(
                np.asarray(mesh.GetFaceVertexCountsAttr().Get()), np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            )
            return _distance_to_triangle_mesh(_to_world(local_pts, matrix), faces, points)
        raise NotImplementedError(f"Collision approximation '{approximation}' of '{prim.GetPath()}' is not supported.")

    if prim.IsA(UsdGeom.Cube):
        half = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
        return _distance_to_hull(_to_world(_box_corners(np.full(3, -half), np.full(3, half)), matrix), points)

    if prim.IsA(UsdGeom.Sphere):
        return _distance_to_sphere(np.zeros(3), UsdGeom.Sphere(prim).GetRadiusAttr().Get(), matrix, points)

    raise NotImplementedError(f"Collision shape '{prim.GetTypeName()}' of '{prim.GetPath()}' is not supported.")


def _box_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _distance_to_hull(hull_points: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Signed distance to the convex hull of ``hull_points``: exact inside and to faces; near edges and corners the
    largest face-plane distance is used, which underestimates the true distance by little for small reaches."""
    try:
        hull = ConvexHull(hull_points)
    except QhullError:  # flat or degenerate shapes
        hull = ConvexHull(hull_points, qhull_options="QJ")
    normals, offsets = hull.equations[:, :3], hull.equations[:, 3]  # unit outward normals: n . x + d <= 0 inside
    return (points @ normals.T + offsets).max(axis=1)


def _distance_to_sphere(center_local: np.ndarray, radius: float, matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    center = _to_world(center_local[None], matrix)[0]
    scale = np.linalg.norm(matrix[:3, :3], axis=1).max()  # uniform scale assumed (largest axis)
    return np.linalg.norm(points - center, axis=1) - radius * scale


def _triangulate(face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    """Fan triangulation of polygon faces, (T, 3)."""
    triangles, start = [], 0
    for count in face_counts:
        for k in range(1, count - 1):
            triangles.append((face_indices[start], face_indices[start + k], face_indices[start + k + 1]))
        start += count
    return np.asarray(triangles, dtype=np.int32)


@wp.kernel
def _closest_distance_kernel(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), max_dist: float, out: wp.array(dtype=float)):
    tid = wp.tid()
    p = points[tid]
    query = wp.mesh_query_point_no_sign(mesh, p, max_dist)
    if query.result:
        closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        out[tid] = wp.length(p - closest)
    else:
        out[tid] = max_dist


def _distance_to_triangle_mesh(vertices: np.ndarray, triangles: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Unsigned distance to a triangle mesh (exact, via a warp BVH), capped at 1 m."""
    wp.init()
    device = "cpu"
    mesh = wp.Mesh(
        points=wp.array(vertices.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(triangles.reshape(-1), dtype=wp.int32, device=device),
    )
    pts = wp.array(points.astype(np.float32), dtype=wp.vec3, device=device)
    out = wp.zeros(points.shape[0], dtype=float, device=device)
    wp.launch(_closest_distance_kernel, dim=points.shape[0], inputs=[mesh.id, pts, 1.0, out], device=device)
    return out.numpy().astype(np.float64)
