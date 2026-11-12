"""Voxel-based tetrahedral meshing (alternative to fTetWild).

The input surface is voxelized on a regular grid, every kept voxel is split into 6 tetrahedra (Kuhn/Freudenthal
subdivision along the main diagonal, conforming between neighboring voxels), and optionally the boundary vertices are
moved onto the input surface (without inverting tets).

With ``conservative=True`` every voxel that intersects the surface is kept, so the tet mesh encloses the whole input
surface: a coarse "cage" that is meant to be combined with the tetrahedral render mesh mapping
(Macklin et al. 2020, Sec. 5.3; see :mod:`tacex_uipc.utils.render_mesh_mapping`).

This module only depends on numpy and warp, so it can be used/tested without Isaac Sim.
"""

from __future__ import annotations

import numpy as np
import warp as wp

# Kuhn subdivision of a voxel: corner c = 4*a + 2*b + d for offsets (a, b, d) in {0, 1}^3.
# Each tet follows a monotone path c0 -> c_i -> c_j -> c7 along the main diagonal.
_KUHN_TETS = np.array([[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]])
_CORNER_OFFSETS = np.array([[a, b, d] for a in (0, 1) for b in (0, 1) for d in (0, 1)])
# outward faces of a positively oriented tet (a, b, c, d)
_TET_FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])


@wp.kernel
def _query_kernel(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    max_dist: float,
    sign: wp.array(dtype=float),
    dist: wp.array(dtype=float),
    closest: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    p = points[tid]
    q = wp.mesh_query_point_sign_winding_number(mesh, p, max_dist)
    if q.result:
        c = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        sign[tid] = q.sign
        dist[tid] = wp.length(p - c)
        closest[tid] = c
    else:
        sign[tid] = 1.0
        dist[tid] = max_dist
        closest[tid] = p


def _query_surface(points: np.ndarray, surf_points: np.ndarray, triangles: np.ndarray, device: str):
    """Inside/outside sign (winding number, robust for non-watertight meshes), distance and closest surface point."""
    mesh = wp.Mesh(
        points=wp.array(surf_points.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(triangles.astype(np.int32).reshape(-1), dtype=wp.int32, device=device),
        support_winding_number=True,
    )
    n = points.shape[0]
    sign = wp.zeros(n, dtype=float, device=device)
    dist = wp.zeros(n, dtype=float, device=device)
    closest = wp.zeros(n, dtype=wp.vec3, device=device)
    max_dist = float(np.linalg.norm(np.ptp(np.vstack([points, surf_points]), axis=0))) * 2.0 + 1.0
    wp.launch(
        _query_kernel,
        dim=n,
        inputs=[mesh.id, wp.array(points.astype(np.float32), dtype=wp.vec3, device=device), max_dist],
        outputs=[sign, dist, closest],
        device=device,
    )
    return sign.numpy(), dist.numpy().astype(np.float64), closest.numpy().astype(np.float64)


def tet_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Signed volumes of the tetrahedra."""
    a, b, c, d = (points[tets[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def boundary_faces(tets: np.ndarray) -> np.ndarray:
    """Outward oriented boundary triangles of a positively oriented tet mesh."""
    faces = tets[:, _TET_FACES].reshape(-1, 3)
    keys = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return faces[counts[inverse.reshape(-1)] == 1]


def voxel_tet_mesh(
    points: np.ndarray,
    triangles: np.ndarray,
    voxel_size: float | None = None,
    resolution: int = 10,
    conservative: bool = True,
    snap_to_surface: bool = False,
    min_volume_ratio: float = 0.1,
    snap_rounds: int = 5,
    device: str = "cpu",
):
    """Creates a voxel-based tet mesh for a triangle surface mesh.

    Args:
        points: Surface mesh vertices. Shape (N, 3).
        triangles: Surface mesh triangles. Shape (F, 3).
        voxel_size: Edge length of a voxel (same unit as ``points``). If ``None``, it is derived from ``resolution``.
        resolution: Number of voxels along the longest bounding box axis (used if ``voxel_size`` is ``None``).
        conservative: Keep all voxels touching the surface (tet mesh encloses the input surface). Otherwise only
            voxels whose center lies inside the surface are kept.
        snap_to_surface: Move the boundary vertices onto the input surface. A vertex is only moved as far as no
            adjacent tet gets smaller than ``min_volume_ratio`` of a regular voxel tet. Reduces the staircase of the
            contact surface, but the tet mesh does no longer enclose the input surface completely.
        min_volume_ratio: Min. allowed tet volume after snapping, relative to the volume of an unsnapped tet.
        snap_rounds: Number of snapping rounds. Each round moves every boundary vertex towards the surface as far as
            possible (backtracking); later rounds can move vertices further after their neighbors have moved.
        device: Warp device for the surface queries.

    Returns:
        Tuple of tet points (M, 3), tet indices (T, 4), surface points (S, 3) and surface triangles (B, 3). The surface
        triangles index into the surface points.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    lo, hi = points.min(0), points.max(0)
    extent = hi - lo
    h = float(voxel_size) if voxel_size is not None else float(extent.max()) / max(int(resolution), 1)
    if h <= 0:
        raise ValueError("Voxel size must be positive.")

    # grid (padded by one voxel), centered around the bounding box
    dims = np.ceil(extent / h).astype(np.int64) + 2
    origin = (lo + hi) / 2.0 - dims * h / 2.0
    ijk = np.stack(np.meshgrid(*(np.arange(n) for n in dims), indexing="ij"), -1).reshape(-1, 3)
    centers = origin + (ijk + 0.5) * h

    sign, dist, _ = _query_surface(centers, points, triangles, device)
    keep = sign < 0
    if conservative:
        # a voxel intersecting the surface has its center within half a voxel diagonal of the surface
        keep |= dist <= 0.5 * np.sqrt(3.0) * h * (1.0 + 1e-6)
    ijk = ijk[keep]
    if ijk.shape[0] == 0:
        raise RuntimeError("Voxelization is empty, decrease the voxel size.")

    # global grid vertex ids of the voxel corners -> Kuhn tets
    vdims = dims + 1
    corners = ijk[:, None, :] + _CORNER_OFFSETS[None]  # (V, 8, 3)
    corner_ids = (corners[..., 0] * vdims[1] + corners[..., 1]) * vdims[2] + corners[..., 2]
    tets_global = corner_ids[:, _KUHN_TETS].reshape(-1, 4)
    used, tets = np.unique(tets_global, return_inverse=True)
    tets = tets.reshape(-1, 4)
    grid = np.stack(np.unravel_index(used, tuple(vdims)), -1)
    tet_points = origin + grid * h

    # positive orientation
    neg = tet_volumes(tet_points, tets) < 0
    tets[neg] = tets[neg][:, [0, 1, 3, 2]]

    if snap_to_surface:
        tet_points = _snap_boundary(tet_points, tets, points, triangles, h, min_volume_ratio, snap_rounds, device)

    faces = boundary_faces(tets)
    surf_ids, surf_tris = np.unique(faces, return_inverse=True)
    return tet_points, tets, tet_points[surf_ids], surf_tris.reshape(-1, 3)


def _snap_boundary(tet_points, tets, points, triangles, h, min_volume_ratio, rounds, device, backtracking=12):
    """Moves boundary vertices onto the surface without creating tets smaller than ``min_volume_ratio``.

    Every round, each boundary vertex tries to move the remaining way to its closest surface point. The steps of
    vertices adjacent to too small tets are halved (backtracking) until all tets are valid. Accepted moves persist,
    so later rounds can continue where neighbors have made room.
    """
    bnd = np.unique(boundary_faces(tets))
    _, _, target = _query_surface(tet_points[bnd], points, triangles, device)
    min_vol = min_volume_ratio * h**3 / 6.0  # every Kuhn tet has volume h^3 / 6
    pos = np.full(tet_points.shape[0], -1)
    pos[bnd] = np.arange(bnd.shape[0])
    p = tet_points.copy()

    def involved(bad):
        idx = pos[tets[bad]].reshape(-1)
        return np.unique(idx[idx >= 0])

    for _ in range(rounds):
        step = target - p[bnd]
        alpha = np.ones(bnd.shape[0])
        for it in range(backtracking + 1):
            q = p.copy()
            q[bnd] += alpha[:, None] * step
            bad = tet_volumes(q, tets) < min_vol
            if not bad.any():
                break
            alpha[involved(bad)] *= 0.5
        # cleanup: reset the remaining offenders (terminates, since the state at the round start is valid)
        while bad.any():
            alpha[involved(bad)] = 0.0
            q = p.copy()
            q[bnd] += alpha[:, None] * step
            bad = tet_volumes(q, tets) < min_vol
        p = q
    return p
