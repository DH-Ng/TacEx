# Tip: Node id names can differe depending on your application.
#
# To find out what node id's you have, use this:
# from pxr import Sdr
# reg = Sdr.Registry()
# print("\n".join(sorted(reg.GetNodeNames())))

# -> Run this script inside the Isaac Sim script editor

from __future__ import annotations

from dataclasses import dataclass
from PIL import Image
from pathlib import Path

from pxr import UsdGeom, UsdShade, Gf, Sdf
import omni.usd
from isaaclab.sim.utils import bind_visual_material
import math
import time
import omni.kit.app
import omni.timeline

import numpy as np
import torch
from scipy.linalg import svd

DEFAULT_RAMP_IMG_PATH = Path(__file__).resolve().parent / "color_ramp.png"
"""Default color ramp (blue -> green -> red) used by the deformation visualization material."""



def create_ramp_image(ramp_path=None, width=256, height=4):
    """Create a horizontal gradient image: left=blue, middle=green, right=red.

    ramp_path: output file path (if None, :data:`DEFAULT_RAMP_IMG_PATH` is used)
    width: img width
    height: img height

    Returns: Full path to created PNG image.
    """

    ramp_path = Path(ramp_path if ramp_path is not None else DEFAULT_RAMP_IMG_PATH).resolve()
    img = Image.new("RGB", (width, height))
    px = img.load()

    # Define three control colors as RGB tuples (0..255)
    left = (0, 0, 255)  # blue
    mid = (0, 255, 0)  # green
    right = (255, 0, 0)  # red

    mid_x = (width - 1) / 2.0
    for x in range(width):
        if x <= mid_x:
            # interpolate left -> mid
            t = x / mid_x
            r = int(left[0] + (mid[0] - left[0]) * t)
            g = int(left[1] + (mid[1] - left[1]) * t)
            b = int(left[2] + (mid[2] - left[2]) * t)
        else:
            # interpolate mid -> right
            t = (x - mid_x) / (width - 1 - mid_x) if (width - 1 - mid_x) != 0 else 1.0
            r = int(mid[0] + (right[0] - mid[0]) * t)
            g = int(mid[1] + (right[1] - mid[1]) * t)
            b = int(mid[2] + (right[2] - mid[2]) * t)
        for y in range(height):
            px[x, y] = (r, g, b)

    # Ensure directory exists
    ramp_path.parent.resolve().mkdir(parents=True, exist_ok=True)

    img.save(ramp_path, format="PNG")
    return str(ramp_path)


def add_deform_primvar(gprim: UsdGeom.Mesh, primvar_name="deformValue"):
    """
    Adds a float primvar per-vertex that represents a value for "deformation" per-vertex.
    The value could, for example, be:
    - vertex displacement from rest position
    - velocity
    - strain

    The primvar should be updated each frame.
    """
    # Get indices and points
    points_attr = gprim.GetPointsAttr()
    points = points_attr.Get()
    if points is None:
        print("Mesh has no points; cannot add barycentric primvar.")
        return None

    num_verts = len(points)
    primvarsAPI = UsdGeom.PrimvarsAPI(gprim)
    primvar = primvarsAPI.CreatePrimvar(primvar_name, Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
    primvar.Set([0.0] * num_verts)
    return primvar


def create_deform_vis_material(
    mat_path="/World/Materials/DeformationVisMat",
    primvar_name="deformValue",
    ramp_img_path=None,
):
    """Creates a material that colors a mesh with a per-vertex float primvar in [0, 1].

    The primvar value is used as texture coordinate into a color ramp (0 = blue, 0.5 = green, 1 = red).
    Values outside of [0, 1] are clamped, i.e. normalization has to be done before writing the primvar
    (see :func:`normalize_deform_values`).
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage available. Open a stage in Isaac Sim first.")

    material = UsdShade.Material.Define(stage, mat_path)

    # Read values of the primvar
    primvar_reader = UsdShade.Shader.Define(stage, mat_path + "/PrimvarReader_float")
    primvar_reader.CreateIdAttr("ND_UsdPrimvarReader_float")
    primvar_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(primvar_name)

    # Clamp (ensure values are within [0,1])
    clamp_min = UsdShade.Shader.Define(stage, mat_path + "/ClampMin")
    clamp_min.CreateIdAttr("ND_max_float")
    clamp_min.CreateInput("in1", Sdf.ValueTypeNames.Float).Set(0.0)
    clamp_min.CreateInput("in2", Sdf.ValueTypeNames.Float).ConnectToSource(primvar_reader.ConnectableAPI(), "result")

    clamp_max = UsdShade.Shader.Define(stage, mat_path + "/ClampMax")
    clamp_max.CreateIdAttr("ND_min_float")
    clamp_max.CreateInput("in1", Sdf.ValueTypeNames.Float).Set(1.0)
    clamp_max.CreateInput("in2", Sdf.ValueTypeNames.Float).ConnectToSource(clamp_min.ConnectableAPI(), "out")

    # Build texcoord vec2 from scalar (s,0)
    make_uv = UsdShade.Shader.Define(stage, mat_path + "/MakeUV")
    make_uv.CreateIdAttr("ND_combine2_vector2")
    make_uv.CreateInput("in1", Sdf.ValueTypeNames.Float).ConnectToSource(clamp_max.ConnectableAPI(), "out")
    make_uv.CreateInput("in2", Sdf.ValueTypeNames.Float).Set(0.01)

    # Image texture node (UsdUVTexture) for mapping deformation value to color
    img = UsdShade.Shader.Define(stage, mat_path + "/RampTex")
    img.CreateIdAttr("ND_UsdUVTexture")

    # check if color ramp image exists, otherwise create it
    if ramp_img_path is None:
        ramp_img_path = DEFAULT_RAMP_IMG_PATH
    try:
        with Image.open(ramp_img_path) as ramp_img:
            ramp_img.verify()
    except (OSError, SyntaxError):
        ramp_img_path = create_ramp_image(ramp_img_path)
    ramp_img_path = str(ramp_img_path)

    # set file to the ramp image asset (horizontal gradient)
    img.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(ramp_img_path)
    # connect UV coords
    img.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(make_uv.ConnectableAPI(), "out")
    img.CreateInput("wrapS", Sdf.ValueTypeNames.String).Set("clamp")
    img.CreateInput("wrapT", Sdf.ValueTypeNames.String).Set("clamp")

    if not img.GetPrim().IsValid():
        raise RuntimeError("RampTex prim invalid")

    # PBR surface -> USD preview surface won't work with standalone IsaacLab script
    pbr = UsdShade.Shader.Define(stage, mat_path + "/SurfaceShader")
    pbr.CreateIdAttr("ND_open_pbr_surface_surfaceshader")

    # uncomment if you want to check if primvar interpolation = vertex (and not constant)
    # test = UsdShade.Shader.Define(stage, mat_path + "/Test")
    # test.CreateIdAttr("ND_combine3_color3")
    # test.CreateInput("in1", Sdf.ValueTypeNames.Float).ConnectToSource(primvar_reader.ConnectableAPI(), "result")
    # test.CreateInput("in2", Sdf.ValueTypeNames.Float).ConnectToSource(primvar_reader.ConnectableAPI(), "result")
    # test.CreateInput("in3", Sdf.ValueTypeNames.Float).ConnectToSource(primvar_reader.ConnectableAPI(), "result")
    # pbr.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(test.ConnectableAPI(), "out")

    # Connect image sampler output to pbr base_color
    pbr.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(img.ConnectableAPI(), "rgb")
    pbr.CreateInput("base_diffuse_roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    pbr.CreateInput("base_metalness", Sdf.ValueTypeNames.Float).Set(0.0)

    # Bind shading outputs
    material.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "out")

    return material


def assign_material_to_mesh_with_usd(gprim: UsdGeom.Mesh, material=None):
    """Binds the given material to the mesh via the IsaacLab MaterialBind API.
    If materials is None, then a surface triangle vis material will be created
    and then attached to the mesh.

    Note:
    The USD/IsaacLab API will not work properly with usdrt.
    To assign material we need to use usdrt api -> https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/usd_fabric_usdrt.html#id12

    Reason:
    With standard usd-material binding the transformation of the xform will be set to be equal
    to the initial USD xform transformation and then its overwriting the usdrt xform transformation
    """
    # Create material if not provided
    if material is None:
        material = create_deform_vis_material()

    # Create material binding
    # UsdShade.MaterialBindingAPI(gprim).Bind(material)
    bind_visual_material(gprim.GetPath(), material.GetPath(), omni.usd.get_context().get_stage())

    print(f"Assigned material to {gprim.GetPath()}")


# --- Utils for visualization


@dataclass
class ElasticState:
    """Elastic quantities of a tet mesh, see :class:`StableNeoHookeanField`."""

    energy_density: torch.Tensor
    """Elastic strain energy density per vertex [J/m^3 = Pa], volume weighted average of the adjacent tets. Shape (V,)."""
    forces: torch.Tensor
    """Elastic (internal) nodal forces ``-dE/dx`` [N]. Shape (V, 3)."""
    total_energy: float
    """Total elastic strain energy of the mesh [J]."""


class StableNeoHookeanField:
    """Elastic strain energy and nodal forces of a tet mesh with the Stable Neo-Hookean model of libuipc.

    libuipc does not export the elastic energy of finite elements, so it is recomputed here from the nodal positions
    with the same energy density as the libuipc cuda backend (``stable_neo_hookean_3d_snk1.inl``)::

        psi(F) = 0.5 * (mu * (tr(F^T F) - 3) + lambda * (det(F) - 1 - mu / lambda)^2)

    The constant ``psi(I) = mu^2 / (2 lambda)`` is subtracted, so that the energy of the rest shape is zero.
    Unlike the libuipc energies, the results are not scaled with ``dt^2``, i.e. they are in physical units.
    Computations are done in float64, since the strain is small for stiff materials.
    """

    def __init__(
        self,
        rest_points: np.ndarray | torch.Tensor,
        tets: np.ndarray | torch.Tensor,
        youngs_modulus: float,
        poisson_rate: float,
        device: str = "cpu",
    ):
        """
        Args:
            rest_points: Vertex positions of the tet mesh in the rest configuration [m]. Shape (V, 3).
            tets: Tetrahedra (vertex indices). Shape (T, 4).
            youngs_modulus: Young's modulus [Pa].
            poisson_rate: Poisson ratio, < 0.5.
            device: Torch device.
        """
        self.device = device
        self.dtype = torch.float64
        X = torch.as_tensor(np.asarray(rest_points), dtype=self.dtype, device=device).reshape(-1, 3)
        self.tets = torch.as_tensor(np.asarray(tets), dtype=torch.long, device=device).reshape(-1, 4)
        self.num_vertices = X.shape[0]

        Dm = self._edge_matrix(X)
        self.rest_volumes = torch.linalg.det(Dm).abs() / 6.0
        self.Dm_inv = torch.linalg.inv(Dm)

        E, nu = float(youngs_modulus), float(poisson_rate)
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self._psi_rest = 0.5 * self.mu**2 / self.lam

        # volume associated with each vertex (1/4 of each adjacent tet)
        self.vertex_volumes = self._scatter_tets_to_vertices(self.rest_volumes / 4.0).clamp_min(1e-30)

    def compute(self, points: np.ndarray | torch.Tensor) -> ElasticState:
        """Computes the elastic quantities for the current (deformed) vertex positions.

        Args:
            points: Current vertex positions [m], same indexing as the rest points. Shape (V, 3).
        """
        x = torch.as_tensor(np.asarray(points) if not torch.is_tensor(points) else points)
        x = x.to(device=self.device, dtype=self.dtype).reshape(-1, 3)

        F = self._edge_matrix(x) @ self.Dm_inv  # (T, 3, 3)
        f0, f1, f2 = F[..., 0], F[..., 1], F[..., 2]
        cof = torch.stack(
            [torch.linalg.cross(f1, f2), torch.linalg.cross(f2, f0), torch.linalg.cross(f0, f1)], dim=-1
        )  # dJ/dF
        J = (f0 * cof[..., 0]).sum(-1)
        Ic = (F * F).sum(dim=(-2, -1))
        Jm = J - 1.0 - self.mu / self.lam

        psi = 0.5 * (self.mu * (Ic - 3.0) + self.lam * Jm * Jm) - self._psi_rest  # (T,)
        tet_energy = psi * self.rest_volumes

        # first Piola-Kirchhoff stress -> gradient w.r.t. x1, x2, x3 (columns), x0 = -sum
        P = self.mu * F + (self.lam * Jm)[:, None, None] * cof
        H = self.rest_volumes[:, None, None] * (P @ self.Dm_inv.transpose(-2, -1))
        tet_grad = torch.cat([-H.sum(dim=-1, keepdim=True), H], dim=-1).transpose(-2, -1)  # (T, 4, 3)
        grad = torch.zeros((self.num_vertices, 3), dtype=self.dtype, device=self.device)
        grad.index_add_(0, self.tets.reshape(-1), tet_grad.reshape(-1, 3))

        energy_density = self._scatter_tets_to_vertices(tet_energy / 4.0) / self.vertex_volumes
        return ElasticState(
            energy_density=energy_density, forces=-grad, total_energy=float(tet_energy.sum().item())
        )

    def _edge_matrix(self, x: torch.Tensor) -> torch.Tensor:
        v = x[self.tets]  # (T, 4, 3)
        return (v[:, 1:] - v[:, :1]).transpose(-2, -1)  # columns: x1 - x0, x2 - x0, x3 - x0

    def _scatter_tets_to_vertices(self, tet_values: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.num_vertices, dtype=self.dtype, device=self.device)
        out.index_add_(0, self.tets.reshape(-1), tet_values.unsqueeze(-1).expand(-1, 4).reshape(-1))
        return out


def normalize_deform_values(
    values: torch.Tensor,
    value_range: tuple[float, float] | None = None,
    log_scale: bool = False,
    log_decades: float = 3.0,
    auto_range_floor: float = 0.0,
) -> torch.Tensor:
    """Maps non-negative values to [0, 1] for the color ramp of the deformation vis material.

    Args:
        values: Values to normalize.
        value_range: (min, max) mapped to (0, 1). If None, ``(0, max(values.max(), auto_range_floor))`` is used.
        log_scale: Use a logarithmic mapping. The lower bound is then raised to ``max * 10^-log_decades`` if needed.
        log_decades: Number of decades shown with ``log_scale``.
        auto_range_floor: Min. upper bound of the automatic range, so that numerical noise of an undeformed body is
            not stretched over the whole color ramp.

    Returns:
        Normalized values in [0, 1], same shape as ``values``.
    """
    values = values.clamp_min(0.0)
    if value_range is None:
        vmin, vmax = 0.0, max(float(values.max().item()) if values.numel() else 0.0, auto_range_floor)
    else:
        vmin, vmax = float(value_range[0]), float(value_range[1])
    if vmax <= 0.0 or vmax <= vmin:
        return torch.zeros_like(values)

    if log_scale:
        lo = max(vmin, vmax * 10.0 ** (-log_decades))
        out = (torch.log10(values.clamp_min(lo)) - np.log10(lo)) / (np.log10(vmax) - np.log10(lo))
    else:
        out = (values - vmin) / (vmax - vmin)
    return out.clamp(0.0, 1.0)


# TODO pytorch, batched implementation
def kabsch_rigid_transform(A, B):
    """Compute rotation R (3x3) and translation t (3,) such that R @ A + t approximates B.
    A, B: (N,3) corresponding points. Uses Kabsch (centroid removal + SVD).
    Returns R, t
    """
    assert A.shape == B.shape

    # compute centroids
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    # center points
    X = A - centroid_A
    Y = B - centroid_B

    # covariance matrix
    H = X.T @ Y
    U, S, VT = svd(H)

    # optimal rotation
    R = VT.T @ U.T
    if np.linalg.det(R) < 0:
        # Reflection case: correct rotation matrix sotat determinant = 1
        VT[-1, :] *= -1
        R = VT.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t


def compute_residuals_and_magnitudes(rest_points, current_points):
    """Align rest_points -> current_points rigidly

    Returns
    - residuals = Vt - (R @ V0 + t)
    - mags: magnitudes per-vertex
    - rest_points_aligned: the aligned points
    """
    R, t = kabsch_rigid_transform(rest_points, current_points)
    # map current (deformed) back into rest frame for computing local displacement
    # current_in_rest_frame = (R.T @ (current_points - t).T).T
    # residuals = current_in_rest_frame - rest_points

    rest_points_aligned = (R @ rest_points.T).T + t  # shape (n,3)
    residuals = current_points - rest_points_aligned
    mags = np.linalg.norm(residuals, axis=1)
    return residuals, mags, rest_points_aligned


# --- Helper methods for testing the material ---

# omniverse callbacks
_update_sub = None
_deform_sub = None


def create_cube(mesh_path="/World/CubeSurface"):
    # --- Cube vertices (unit cube) ---
    vertices = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
        (0.0, 0.0, 1.0),  # 4
        (1.0, 0.0, 1.0),  # 5
        (1.0, 1.0, 1.0),  # 6
        (0.0, 1.0, 1.0),  # 7
    ]

    # --- Tetrahedralization of the cube (5 tets) ---
    tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (1, 4, 5, 6), (3, 4, 6, 7)]

    # Helper: build outer surface triangles by collecting all faces and removing internal ones
    from collections import Counter

    # Collect face occurrences
    face_counts = Counter()
    face_map = {}  # map sorted face -> original winding (we'll pick one)
    for tet in tets:
        # original oriented faces for visualization (consistent winding)
        a, b, c, d = tet
        oriented_faces = [(a, b, c), (a, d, b), (b, d, c), (c, d, a)]
        for of in oriented_faces:
            key = tuple(sorted(of))
            face_counts[key] += 1
            # store the oriented version (last wins) for the key
            face_map[key] = of

    # Keep only boundary faces (count == 1)
    boundary_faces = [face_map[k] for k, v in face_counts.items() if v == 1]

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage available. Open a stage in Isaac Sim first.")

    # Create mesh prim for boundary surface
    mesh_path = Sdf.Path(mesh_path)
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)

    # Points
    points = [Gf.Vec3f(x, y, z) for (x, y, z) in vertices]
    mesh.GetPointsAttr().Set(points)

    # Build face topology (triangles)
    faceVertexCounts = [3] * len(boundary_faces)
    faceVertexIndices = []
    for tri in boundary_faces:
        for vi in tri:
            faceVertexIndices.append(int(vi))
    mesh.GetFaceVertexCountsAttr().Set(faceVertexCounts)
    mesh.GetFaceVertexIndicesAttr().Set(faceVertexIndices)

    return mesh


def start_deformation_updater(mesh: UsdGeom.Mesh, primvar_name="deformValue"):
    global _update_sub
    prim = mesh.GetPrim()

    # for testing purposes we use custom restPoints attribute
    rest_pts = np.array(prim.GetAttribute("debug:restPoints").Get())
    if rest_pts is None:
        raise RuntimeError("Rest positions not found; call prepare_deform_primvar first")

    from .debug_draw import DebugDraw

    draw = DebugDraw(f"/Visuals/DeformationUpdater{str(mesh.GetPath()).replace('/', '_')}")

    def _tick(event):
        if not omni.timeline.get_timeline_interface().is_playing():
            return

        dt = getattr(event, "dt", None) or getattr(event, "deltaSeconds", None) or 1.0 / 60.0
        cur_pts = np.array(mesh.GetPointsAttr().Get())
        if cur_pts is None or len(cur_pts) != len(rest_pts):
            return
        # compute magnitudes
        # residuals, mags, rest_points_aligned = compute_residuals_and_magnitudes(rest_pts, cur_pts)

        mags = np.linalg.norm(cur_pts - rest_pts, axis=1)

        # # normalize to [0,1]
        # mags_min = np.min(mags)
        # mags_max = np.max(mags)
        # if mags_max == mags_min:
        #     mags = np.zeros_like(mags)
        # else:
        #     mags = (mags - mags_min) / (mags_max - mags_min)

        # write into primvar
        # prim.GetAttribute(f"primvars:{primvar_name}").Set(mags)
        # mags = [1, 1, 1, 1, 0, 0, 0, 0]
        UsdGeom.PrimvarsAPI(mesh).GetPrimvar(f"{primvar_name}").Set(mags)
        # print("test ", UsdGeom.PrimvarsAPI(mesh).GetPrimvar(f"{primvar_name}").Get())

        # draw current mesh points
        center = np.mean(cur_pts, axis=0, keepdims=True)
        sizes = np.r_[np.full(len(cur_pts), draw.point_radius), 2 * draw.point_radius]  # center drawn bigger
        draw.set_points(np.vstack([cur_pts, center]), colors=(0.0, 1.0, 0.0), sizes=sizes)

        # # draw the transformed rest point pos
        # # rest_points_aligned = rest_pts
        # draw.draw_points(
        #     rest_points_aligned,
        #     [(255, 0, 255, 0.5)] * rest_points_aligned.shape[0],
        #     [30] * rest_points_aligned.shape[0],
        # )
        # draw.draw_points([np.mean(rest_points_aligned, axis=0)], [(255, 0, 255, 1.0, 0)], [30])

    # subscribe to app update
    if _update_sub is not None:
        _update_sub.cancel()
    app = omni.kit.app.get_app()
    _update_sub = app.get_update_event_stream().create_subscription_to_push(lambda event: _tick(event))


def start_sine_deformation(
    mesh: UsdGeom.Mesh, amplitude=0.95, frequency=0.1, axis=(0.0, 0.0, 1.0), primvar_name="deformValue"
):
    """
    Apply a time-varying sine displacement to mesh points.
    - amplitude: peak displacement
    - frequency: Hz
    - axis: displacement direction (tuple length 3)
    Requires:
      - mesh has attribute "debug:restPoints" (set by add_deform_primvar)
      - start_deformation_updater(mesh, primvar_name) has been called (or call it here)
    """
    global _deform_sub
    prim = mesh.GetPrim()
    rest_pts = prim.GetAttribute("debug:restPoints").Get()
    if rest_pts is None:
        raise RuntimeError("Rest positions not found; call add_deform_primvar(mesh) first")

    axis_vec = Gf.Vec3f(axis[0], axis[1], axis[2])
    if axis_vec.GetLength() == 0:
        axis_vec = Gf.Vec3f(0.0, 0.0, 1.0)
    axis_vec.Normalize()

    # ensure deformation primvar updater is running
    try:
        start_deformation_updater(mesh, primvar_name=primvar_name)
    except Exception:
        pass

    start_time = time.time()

    def _tick(event):
        if not omni.timeline.get_timeline_interface().is_playing():
            return

        # compute elapsed time
        t = time.time() - start_time
        cur_pts = mesh.GetPointsAttr().Get()
        if cur_pts is None:
            return
        # Create displaced points
        displaced = []
        for i, rest in enumerate(rest_pts):
            newp = rest
            # move first 4 points:
            if i < 4:
                offset = amplitude * math.sin(2.0 * math.pi * frequency * t)
                newp += axis_vec * offset

            displaced.append(newp)
        # write new points to mesh
        # note: the deformation updater will read current points and update primvar deformValue
        mesh.GetPointsAttr().Set(displaced)

    # unsubscribe previous
    if _deform_sub is not None:
        _deform_sub.cancel()
    app = omni.kit.app.get_app()
    # subscribe to update event stream
    _deform_sub = app.get_update_event_stream().create_subscription_to_push(_tick)
    return _deform_sub


if __name__ == "__main__":
    # Example usage:

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage available. Open a stage in Isaac Sim first.")

    mesh_path = "/World/MeshExample"
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    if not mesh_prim.IsValid():
        mesh = create_cube(mesh_path=mesh_path)
    else:
        mesh = UsdGeom.Mesh(mesh_prim)

    # store rest positions on the mesh prim (as a custom attribute) for visualizing vertex displacements
    points = mesh.GetPointsAttr().Get()
    mesh.GetPrim().CreateAttribute("debug:restPoints", Sdf.ValueTypeNames.Point3fArray).Set(points)

    # ensure per-vertex primvar exists and rest points are stored
    primvar = add_deform_primvar(mesh, primvar_name="deformValue")
    if primvar is None:
        print("Primvar creation failed.")

    ramp_img_path = create_ramp_image(ramp_path="./color_ramp.png")

    mat = create_deform_vis_material(
        mat_path="/World/Materials/DeformVisMat",
        primvar_name="deformValue",
        ramp_img_path=ramp_img_path,
    )
    assign_material_to_mesh_with_usd(mesh, material=mat)

    # Test the material by deforming mesh and updating the deformValue primvar
    start_deformation_updater(mesh)
    start_sine_deformation(mesh)
