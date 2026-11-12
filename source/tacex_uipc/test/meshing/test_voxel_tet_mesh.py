"""Unit tests for the voxel-based tet mesh generation (``tacex_uipc.utils.voxel_tet_mesh``).

Pure numpy/warp test, no Isaac Sim app needed:
    python -m pytest source/tacex_uipc/test/meshing/test_voxel_tet_mesh.py -v
"""

import importlib.util
import pathlib

import numpy as np
import pytest

# load the module directly: importing the `tacex_uipc` package pulls in Isaac Sim
_spec = importlib.util.spec_from_file_location(
    "voxel_tet_mesh",
    pathlib.Path(__file__).resolve().parents[2] / "tacex_uipc" / "utils" / "voxel_tet_mesh.py",
)
vtm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vtm)


def _icosphere(radius=1.0, subdiv=3):
    """Closed, outward oriented triangle sphere."""
    t = (1.0 + 5**0.5) / 2.0
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]  # fmt: skip
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
         [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
         [8, 6, 7], [9, 8, 1]]  # fmt: skip
    v = [np.array(p, dtype=float) / np.linalg.norm(p) for p in v]
    for _ in range(subdiv):
        cache, nf = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[key] = len(v) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = nf
    return np.array(v) * radius, np.array(f)


def _mesh_volume(points, tris):
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0


def _inside_tets(query, tet_points, tets, tol=1e-9):
    """True for every query point that lies in (at least) one tet."""
    v0 = tet_points[tets[:, 0]]
    D = np.stack([tet_points[tets[:, k]] - v0 for k in (1, 2, 3)], axis=-1)
    Dinv = np.linalg.inv(D)
    inside = np.zeros(query.shape[0], dtype=bool)
    for i, p in enumerate(query):
        b = np.einsum("tij,tj->ti", Dinv, p - v0)
        bary = np.concatenate([1 - b.sum(1, keepdims=True), b], 1)
        inside[i] = (bary.min(1) >= -tol).any()
    return inside


def _check_valid(tet_points, tets, surf_points, surf_tris):
    # positive volumes
    assert (vtm.tet_volumes(tet_points, tets) > 0).all()
    # closed, consistently oriented boundary: every directed edge appears exactly once, and its reverse too
    e = np.concatenate([surf_tris[:, [0, 1]], surf_tris[:, [1, 2]], surf_tris[:, [2, 0]]])
    directed = {tuple(x) for x in e}
    assert len(directed) == e.shape[0]
    assert all((b, a) in directed for a, b in directed)
    # outward orientation
    assert _mesh_volume(surf_points, surf_tris) > 0
    np.testing.assert_allclose(_mesh_volume(surf_points, surf_tris), vtm.tet_volumes(tet_points, tets).sum(), rtol=1e-9)


@pytest.fixture(scope="module")
def sphere():
    return _icosphere(radius=0.01)  # 1 cm, like the TacEx assets


def test_conservative_encloses_surface(sphere):
    pts, tris = sphere
    tp, tets, sp, st = vtm.voxel_tet_mesh(pts, tris, resolution=6)
    _check_valid(tp, tets, sp, st)
    assert _inside_tets(pts, tp, tets).all()
    assert vtm.tet_volumes(tp, tets).sum() >= _mesh_volume(pts, tris)


def test_inside_only_smaller(sphere):
    pts, tris = sphere
    tp_c, tets_c, _, _ = vtm.voxel_tet_mesh(pts, tris, resolution=8, conservative=True)
    tp_i, tets_i, sp, st = vtm.voxel_tet_mesh(pts, tris, resolution=8, conservative=False)
    _check_valid(tp_i, tets_i, sp, st)
    assert vtm.tet_volumes(tp_i, tets_i).sum() < vtm.tet_volumes(tp_c, tets_c).sum()
    assert not _inside_tets(pts, tp_i, tets_i).all()  # the inside-only mesh does not enclose the surface


def test_voxel_size_vs_resolution(sphere):
    pts, tris = sphere
    h = 0.02 / 5  # diameter / 5
    a = vtm.voxel_tet_mesh(pts, tris, resolution=5)
    b = vtm.voxel_tet_mesh(pts, tris, voxel_size=h)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    # every tet edge along an axis has length h
    edges = a[0][a[1][:, 1]] - a[0][a[1][:, 0]]
    assert np.isclose(np.abs(edges).max(), h)
    # finer resolution -> more tets
    assert vtm.voxel_tet_mesh(pts, tris, resolution=10)[1].shape[0] > a[1].shape[0]


def test_snap_to_surface(sphere):
    pts, tris = sphere
    h = 0.02 / 6
    tp0, tets0, sp0, _ = vtm.voxel_tet_mesh(pts, tris, resolution=6)
    tp, tets, sp, st = vtm.voxel_tet_mesh(pts, tris, resolution=6, snap_to_surface=True, min_volume_ratio=0.1)
    _check_valid(tp, tets, sp, st)
    np.testing.assert_array_equal(tets, tets0)  # topology unchanged
    assert vtm.tet_volumes(tp, tets).min() >= 0.1 * h**3 / 6 * (1 - 1e-9)
    # boundary vertices closer to the sphere surface
    err0 = np.abs(np.linalg.norm(sp0, axis=1) - 0.01)
    err = np.abs(np.linalg.norm(sp, axis=1) - 0.01)
    assert np.median(err) < 0.35 * np.median(err0)
    assert err.max() <= err0.max() + 1e-12


def test_non_watertight_input(sphere):
    """Winding number keeps the occupancy robust when a few triangles are missing."""
    pts, tris = sphere
    holed = tris[5:]
    tp, tets, sp, st = vtm.voxel_tet_mesh(pts, holed, resolution=6)
    _check_valid(tp, tets, sp, st)
    ref = vtm.voxel_tet_mesh(pts, tris, resolution=6)
    assert abs(tets.shape[0] - ref[1].shape[0]) <= 0.05 * ref[1].shape[0]


def test_invalid_voxel_size(sphere):
    with pytest.raises(ValueError):
        vtm.voxel_tet_mesh(*sphere, voxel_size=0.0)
