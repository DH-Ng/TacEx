"""Unit tests for the render/simulation mesh embedding (``tacex_uipc.utils.render_mesh_mapping``).

Pure numpy/torch/warp test, no Isaac Sim app needed:
    python -m pytest source/tacex_uipc/test/render_mesh/test_render_mesh_mapping.py -v
"""

import importlib.util
import pathlib

import numpy as np
import pytest
import torch

# load the module directly: importing the `tacex_uipc` package pulls in Isaac Sim, so we avoid it.
_spec = importlib.util.spec_from_file_location(
    "render_mesh_mapping",
    pathlib.Path(__file__).resolve().parents[2] / "tacex_uipc" / "utils" / "render_mesh_mapping.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
RenderMeshMapper = _mod.RenderMeshMapper

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def _cube_tet_mesh(n: int = 3):
    """Regular grid of the unit cube, each hex split into 6 tets. Returns points, tets, surface triangles."""
    g = np.linspace(0.0, 1.0, n + 1)
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    idx = lambda i, j, k: (i * (n + 1) + j) * (n + 1) + k  # noqa: E731
    tets = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c = [idx(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]
                # Kuhn subdivision along main diagonal c[0] -> c[7]
                for path in ([1, 3], [1, 5], [2, 3], [2, 6], [4, 5], [4, 6]):
                    tets.append([c[0], c[path[0]], c[path[1]], c[7]])
    tets = np.array(tets)
    # boundary faces = faces used by exactly one tet
    faces = {}
    for t in tets:
        for f in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]):
            key = tuple(sorted(t[f]))
            faces[key] = faces.get(key, 0) + 1
    tri = np.array([f for f, c in faces.items() if c == 1])
    # surface vertex subset with local indexing (like extract_surface)
    surf_ids = np.unique(tri)
    remap = -np.ones(pts.shape[0], dtype=int)
    remap[surf_ids] = np.arange(surf_ids.shape[0])
    return pts, tets, surf_ids, remap[tri]


def _render_points(rng, num=500):
    """Points on/near the cube surface, incl. some slightly outside and inside (like a finer render mesh)."""
    p = rng.uniform(0.0, 1.0, (num, 3))
    axis = rng.integers(0, 3, num)
    side = rng.integers(0, 2, num).astype(float)
    p[np.arange(num), axis] = side + rng.uniform(-0.01, 0.01, num)
    return p


def _rigid(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q, rng.normal(size=3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", ["surface", "tetrahedral"])
def test_rest_and_rigid_motion_exact(device, method):
    rng = np.random.default_rng(0)
    pts, tets, surf_ids, tri = _cube_tet_mesh()
    render = _render_points(rng)
    sim_rest = pts[surf_ids] if method == "surface" else pts
    elements = tri if method == "surface" else tets

    mapper = RenderMeshMapper(render, sim_rest, elements, method=method, device=device)

    # rest configuration is reproduced
    np.testing.assert_allclose(mapper.map(sim_rest).cpu().numpy(), render, atol=1e-5)

    # rigid motion is reproduced
    R, t = _rigid(rng)
    out = mapper.map(sim_rest @ R.T + t).cpu().numpy()
    np.testing.assert_allclose(out, render @ R.T + t, atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_tetrahedral_affine_exact(device):
    rng = np.random.default_rng(1)
    pts, tets, _, _ = _cube_tet_mesh()
    render = _render_points(rng)
    mapper = RenderMeshMapper(render, pts, tets, method="tetrahedral", device=device)
    A = np.eye(3) + 0.3 * rng.normal(size=(3, 3))
    out = mapper.map(pts @ A.T).cpu().numpy()
    np.testing.assert_allclose(out, render @ A.T, atol=1e-4)
    assert mapper.num_outside > 0  # some render points lie slightly outside -> extrapolated


@pytest.mark.parametrize("method", ["surface", "tetrahedral"])
def test_local_deformation_follows(method):
    """Indenting the top face moves render points on the top face down, leaves the bottom face untouched."""
    rng = np.random.default_rng(2)
    pts, tets, surf_ids, tri = _cube_tet_mesh(n=4)
    render = _render_points(rng, 1000)
    render = render[(render[:, 2] > 0.999) | (render[:, 2] < 0.001)]  # top/bottom face points only
    sim_rest = pts[surf_ids] if method == "surface" else pts
    elements = tri if method == "surface" else tets
    mapper = RenderMeshMapper(render, sim_rest, elements, method=method)

    deformed = sim_rest.copy()
    on_top = deformed[:, 2] > 1 - 1e-9
    deformed[on_top, 2] -= 0.1 * np.exp(-((deformed[on_top, 0] - 0.5) ** 2 + (deformed[on_top, 1] - 0.5) ** 2) / 0.05)
    out = mapper.map(deformed).cpu().numpy()

    top = render[:, 2] > 0.5
    assert np.all(out[top, 2] <= render[top, 2] + 1e-5)
    assert (render[top, 2] - out[top, 2]).max() > 0.05
    np.testing.assert_allclose(out[~top], render[~top], atol=1e-5)


def test_invalid_method():
    pts, tets, _, _ = _cube_tet_mesh(1)
    with pytest.raises(ValueError):
        RenderMeshMapper(pts, pts, tets, method="foo")
