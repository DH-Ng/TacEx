"""Embedding of a (high-res) render mesh into a UIPC simulation mesh.

The simulation mesh (tet mesh / its surface) is decoupled from the render mesh (the original Isaac mesh with its
own topology, UVs and material bindings). Each render vertex is embedded once in the rest configuration and is
reconstructed from the deformed simulation mesh every frame.

Two mappings are supported:

``"surface"`` - Dual-mesh mapper of IsaacIPC (arXiv:2605.24339, Sec. 3.2):
    Each render vertex ``p`` is assigned to its closest simulation surface triangle ``(v0, v1, v2)`` and stored as
    plane barycentrics ``(u, v, w)`` plus a signed offset ``delta`` along the triangle normal:

        p  = u v0  + v v1  + w v2  + delta n
        p' = u v0' + v v1' + w v2' + delta n'

``"tetrahedral"`` - Deformable embedding of Macklin et al. 2020 (https://doi.org/10.1145/3384538 , Sec. 5.3) / McAdams et al. 2011:
    Each render vertex is assigned to its enclosing tetrahedron and stored in linear shape function coordinates,
    i.e. ``x_m = M D^-1 (x_s - v0_s) + v0_m`` inverted to ``p' = sum_k b_k x_k'``. Vertices outside the tet mesh
    use the tetrahedron with the least violated barycentric coordinate (linear extrapolation).
    Example use case: Coarse tet mesh, thats bigger than the render mesh (see Fig. 9. of Macklin paper).

Both mappings reproduce the render mesh exactly in the rest configuration and are invariant to rigid motions;
the tetrahedral mapping is additionally exact for any affine deformation.

Choose the mapping based on your tetrahedralization method.

Rule of thumb
- fTetWild / fine tets that follow the surface: surface mapping (the default).
- Voxel cages or other deliberately coarse tet meshes: tetrahedral mapping (default for voxel meshes).

In both cases contact is computed only on the sim mesh surface.
The mapping affects only how the detailed mesh is drawn, not contact accuracy.

This module only depends on numpy, torch and warp, so it can be used/tested without Isaac Sim.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import warp as wp

_EPS = 1e-12


@wp.kernel
def _closest_face_kernel(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    max_dist: float,
    faces: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    query = wp.mesh_query_point_no_sign(mesh, points[tid], max_dist)
    if query.result:
        faces[tid] = query.face
    else:
        faces[tid] = -1


class RenderMeshMapper:
    """Maps simulation mesh vertex positions onto the vertices of an embedded render mesh."""

    def __init__(
        self,
        render_points: np.ndarray | torch.Tensor,
        sim_points: np.ndarray | torch.Tensor,
        sim_elements: np.ndarray | torch.Tensor,
        method: Literal["surface", "tetrahedral"] = "surface",
        device: str = "cpu",
        chunk_budget: int = 2**24,
    ):
        """Computes the embedding of the render vertices in the rest configuration.

        Args:
            render_points: Render mesh vertex positions in the rest configuration (world frame). Shape (N, 3).
            sim_points: Simulation mesh vertex positions in the rest configuration (world frame). Shape (M, 3).
                For ``"surface"`` these are the surface vertices, for ``"tetrahedral"`` all tet mesh vertices.
            sim_elements: Surface triangles (F, 3) for ``"surface"`` or tetrahedra (T, 4) for ``"tetrahedral"``.
                Indices refer to ``sim_points``.
            method: ``"surface"`` or ``"tetrahedral"``.
            device: Torch device used for the embedding and the per-frame mapping.
            chunk_budget: Max. number of (point, tet) pairs evaluated at once by the brute force tet search.
        """
        self.method = method
        self.max_offset: float = 0.0
        """Max. distance between a render vertex and its simulation element in the rest configuration
        (surface: normal offset ``|delta|``; tetrahedral: approx. distance to the tet, 0 inside)."""
        self.num_outside: int = 0
        """Number of render vertices outside their element (surface: projection outside the triangle;
        tetrahedral: outside of the tet mesh), whose coordinates are extrapolated."""
        self.device = device
        self.num_render_points = int(render_points.shape[0])

        p = torch.as_tensor(np.asarray(render_points), dtype=torch.float64, device=device).reshape(-1, 3)
        x = torch.as_tensor(np.asarray(sim_points), dtype=torch.float64, device=device).reshape(-1, 3)

        if method == "surface":
            tri = torch.as_tensor(np.asarray(sim_elements), dtype=torch.long, device=device).reshape(-1, 3)
            self._embed_surface(p, x, tri)
        elif method == "tetrahedral":
            tet = torch.as_tensor(np.asarray(sim_elements), dtype=torch.long, device=device).reshape(-1, 4)
            self._embed_tetrahedral(p, x, tet, chunk_budget)
        else:
            raise ValueError(f"Unknown render mesh mapping '{method}'. Options: 'surface', 'tetrahedral'.")

    """
    Operations.
    """

    def map(self, sim_points: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Reconstructs the render vertex positions from the (deformed) simulation vertex positions.

        Args:
            sim_points: Current simulation vertex positions, same indexing as used for the embedding. Shape (M, 3).

        Returns:
            Render vertex positions. Shape (N, 3), dtype float32.
        """
        x = torch.as_tensor(np.asarray(sim_points) if not torch.is_tensor(sim_points) else sim_points)
        x = x.to(device=self.device, dtype=torch.float32).reshape(-1, 3)

        if self.method == "surface":
            v = x[self.elements]  # (N, 3, 3)
            e1 = v[:, 1] - v[:, 0]
            e2 = v[:, 2] - v[:, 0]
            n = torch.linalg.cross(e1, e2)
            n = n / n.norm(dim=-1, keepdim=True).clamp_min(_EPS)
            c = self.coords
            return v[:, 0] + c[:, 1:2] * e1 + c[:, 2:3] * e2 + self.offsets.unsqueeze(-1) * n

        # tetrahedral: linear shape functions
        return (x[self.elements] * self.coords.unsqueeze(-1)).sum(dim=1)

    def interpolate(self, sim_values: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Interpolates per simulation vertex values (e.g. energies) onto the render vertices.

        Uses the barycentric coordinates of the embedding (surface: triangle, tetrahedral: tet). The normal offset of
        the surface mapping is ignored, i.e. a render vertex gets the value of its closest point on the sim surface.

        Args:
            sim_values: Values per simulation vertex, same indexing as used for the embedding. Shape (M,) or (M, C).

        Returns:
            Values per render vertex. Shape (N,) or (N, C), dtype float32.
        """
        v = torch.as_tensor(np.asarray(sim_values) if not torch.is_tensor(sim_values) else sim_values)
        v = v.to(device=self.device, dtype=torch.float32)
        c = self.coords if v.dim() == 1 else self.coords.unsqueeze(-1)
        return (v[self.elements] * c).sum(dim=1)

    """
    Internal helper.
    """

    def _embed_surface(self, p: torch.Tensor, x: torch.Tensor, tri: torch.Tensor):
        # closest triangle via warp BVH query (float32 is sufficient for selecting the face)
        wp_device = str(self.device) if (str(self.device).startswith("cuda") and wp.is_cuda_available()) else "cpu"
        mesh = wp.Mesh(
            points=wp.array(x.cpu().numpy().astype(np.float32), dtype=wp.vec3, device=wp_device),
            indices=wp.array(tri.cpu().numpy().astype(np.int32).reshape(-1), dtype=wp.int32, device=wp_device),
        )
        points_wp = wp.array(p.cpu().numpy().astype(np.float32), dtype=wp.vec3, device=wp_device)
        faces_wp = wp.zeros(p.shape[0], dtype=wp.int32, device=wp_device)
        max_dist = float((torch.cat([p, x]).amax(0) - torch.cat([p, x]).amin(0)).norm()) * 10.0 + 1.0
        wp.launch(
            _closest_face_kernel,
            dim=p.shape[0],
            inputs=[mesh.id, points_wp, max_dist],
            outputs=[faces_wp],
            device=wp_device,
        )
        faces = torch.as_tensor(faces_wp.numpy(), dtype=torch.long, device=self.device)
        if (faces < 0).any():
            raise RuntimeError("Closest triangle query failed for some render vertices.")

        elements = tri[faces]  # (N, 3)
        v = x[elements]
        e1 = v[:, 1] - v[:, 0]
        e2 = v[:, 2] - v[:, 0]
        d = p - v[:, 0]
        n = torch.linalg.cross(e1, e2)
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(_EPS)

        # plane barycentrics from 2x2 normal equations (exact reconstruction in the rest configuration)
        a11 = (e1 * e1).sum(-1)
        a12 = (e1 * e2).sum(-1)
        a22 = (e2 * e2).sum(-1)
        b1 = (d * e1).sum(-1)
        b2 = (d * e2).sum(-1)
        det = a11 * a22 - a12 * a12
        det = torch.where(det.abs() < _EPS, torch.full_like(det, _EPS), det)
        bv = (a22 * b1 - a12 * b2) / det
        bw = (a11 * b2 - a12 * b1) / det

        self.elements = elements
        self.coords = torch.stack([1.0 - bv - bw, bv, bw], dim=-1).to(torch.float32)
        self.offsets = (d * n).sum(-1).to(torch.float32)

        # diagnostics
        self.max_offset = float(self.offsets.abs().max()) if self.num_render_points else 0.0
        self.num_outside = int((self.coords.amin(-1) < -1e-4).sum())

    def _embed_tetrahedral(self, p: torch.Tensor, x: torch.Tensor, tet: torch.Tensor, chunk_budget: int):
        v0 = x[tet[:, 0]]  # (T, 3)
        D = torch.stack([x[tet[:, 1]] - v0, x[tet[:, 2]] - v0, x[tet[:, 3]] - v0], dim=-1)  # columns = edges
        det = torch.linalg.det(D)
        valid = det.abs() > _EPS * max(1.0, float(D.abs().max()) ** 3)
        D_safe = torch.where(valid[:, None, None], D, torch.eye(3, dtype=D.dtype, device=D.device).expand_as(D))
        Dinv = torch.linalg.inv(D_safe)  # (T, 3, 3)

        num_tets = tet.shape[0]
        chunk = max(1, chunk_budget // max(num_tets, 1))
        best_tet = torch.empty(p.shape[0], dtype=torch.long, device=self.device)
        best_bary = torch.empty(p.shape[0], 4, dtype=torch.float64, device=self.device)
        best_score = torch.empty(p.shape[0], dtype=torch.float64, device=self.device)
        for start in range(0, p.shape[0], chunk):
            pc = p[start : start + chunk]  # (C, 3)
            rel = pc[:, None, :] - v0[None, :, :]  # (C, T, 3)
            b123 = torch.einsum("tij,ctj->cti", Dinv, rel)  # (C, T, 3)
            b0 = 1.0 - b123.sum(-1, keepdim=True)
            bary = torch.cat([b0, b123], dim=-1)  # (C, T, 4)
            # score = least violated barycentric coordinate; > 0 means inside
            score = bary.amin(-1)
            score = torch.where(valid[None, :], score, torch.full_like(score, -torch.inf))
            s, idx = score.max(dim=1)
            best_tet[start : start + chunk] = idx
            best_score[start : start + chunk] = s
            best_bary[start : start + chunk] = bary[torch.arange(pc.shape[0], device=self.device), idx]

        self.elements = tet[best_tet]  # (N, 4)
        self.coords = best_bary.to(torch.float32)
        self.offsets = None

        # diagnostics: distance to the tet approximated via clamped + renormalized barycentrics
        clamped = best_bary.clamp_min(0.0)
        clamped = clamped / clamped.sum(-1, keepdim=True).clamp_min(_EPS)
        dist = (p - (x[self.elements] * clamped.unsqueeze(-1)).sum(1)).norm(dim=-1)
        self.num_outside = int((best_score < -1e-4).sum())
        self.max_offset = float(dist.max()) if self.num_render_points else 0.0
