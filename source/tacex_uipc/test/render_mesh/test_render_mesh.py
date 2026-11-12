"""Separate render and simulation mesh (``RenderMeshCfg``) in a UIPC scene.

Scene: a UIPC gelpad (StableNeoHookean, surface shows upwards) lying on the UIPC ground, and a UIPC ball (affine body) dropped onto
it. The gel renders its original USD mesh via the tetrahedral embedding, the ball via the surface embedding.

Run with the Isaac Lab python:
    [TACEX_TEST_DEBUG_VIS=1] python -m pytest source/tacex_uipc/test/render_mesh/test_render_mesh.py -v -s
"""

import debug_vis_helper as dv

simulation_app = dv.launch_app()  # headless, or the GUI with TACEX_TEST_DEBUG_VIS=1 (see debug_vis_helper.py)

"""Rest everything follows."""

import numpy as np
import pytest
import warp as wp
from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_uipc import UipcSim, UipcSimCfg
from tacex_uipc.objects import (
    RenderMeshCfg,
    UipcDeformableObject,
    UipcDeformableObjectCfg,
    UipcRigidObject,
    UipcRigidObjectCfg,
)
from tacex_uipc.utils import TetMeshCfg

DT = 0.01
GROUND_Z = 0.0
BALL_RADIUS = 0.005
GEL_HEIGHT = 0.004
NUM_STEPS = 60
BALL_DENSITY = 2e4
"""Heavy ball, so that it dents the gel noticeably."""
GEL_USD = f"{TACEX_ASSETS_DATA_DIR}/Sensors/GelSight_Mini/Gelpad_low_res.usd"
BALL_USD = f"{TACEX_ASSETS_DATA_DIR}/Props/ball_wood.usd"


def _source_topology(usd_path: str) -> tuple[int, list[int]]:
    """(number of points, face vertex indices) of the first mesh under the default prim of the USD file."""
    stage = Usd.Stage.Open(usd_path)
    for prim in Usd.PrimRange(stage.GetDefaultPrim()):
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            return len(mesh.GetPointsAttr().Get()), list(mesh.GetFaceVertexIndicesAttr().Get())
    raise RuntimeError(f"No mesh in {usd_path}")


def _render_points(obj) -> np.ndarray:
    return np.array(obj.fabric_prim.GetAttribute("points").Get(), dtype=np.float64)


@wp.kernel
def _dist_kernel(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    tid = wp.tid()
    q = wp.mesh_query_point(mesh, pts[tid], 1.0e3)
    out[tid] = wp.length(pts[tid] - wp.mesh_eval_position(mesh, q.face, q.u, q.v)) * q.sign


def _signed_dist_to_surface(points: np.ndarray, surf_points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Signed distance of points to the (closed) simulation surface mesh."""
    mesh = wp.Mesh(
        points=wp.array(surf_points.astype(np.float32), dtype=wp.vec3),
        indices=wp.array(triangles.astype(np.int32).reshape(-1), dtype=wp.int32),
    )
    out = wp.zeros(points.shape[0], dtype=float)
    wp.launch(
        _dist_kernel,
        dim=points.shape[0],
        inputs=[mesh.id, wp.array(points.astype(np.float32), dtype=wp.vec3)],
        outputs=[out],
    )
    return out.numpy()


def _surf_points(uipc_sim, obj_idx) -> np.ndarray:
    pts = uipc_sim.sio.simplicial_surface(2).positions().view().reshape(-1, 3)
    o = uipc_sim._surf_vertex_offsets
    return np.asarray(pts[o[obj_idx] : o[obj_idx + 1]])


@pytest.fixture(scope="module")
def result():
    # original topology of the USD assets (before any uipc processing)
    src = {"gel": _source_topology(GEL_USD), "ball": _source_topology(BALL_USD)}

    # Setup scene
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cuda"))
    uipc_sim = UipcSim(UipcSimCfg(dt=DT, ground_height=GROUND_Z, contact=UipcSimCfg.Contact(d_hat=1e-4)))
    gel = UipcDeformableObject(
        UipcDeformableObjectCfg(
            prim_path="/World/Gelpad",
            spawn=sim_utils.UsdFileCfg(usd_path=GEL_USD),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, GROUND_Z + 2e-4)),
            mesh_cfg=TetMeshCfg(stop_quality=8, max_its=100, edge_length_r=1 / 20),
            constitution_cfg=UipcDeformableObjectCfg.StableNeoHookeanCfg(),
            render_mesh_cfg=RenderMeshCfg(mapping="tetrahedral"),
            debug_vis=dv.DEBUG_VIS,
        ),
        uipc_sim,
    )
    ball = UipcRigidObject(
        UipcRigidObjectCfg(
            prim_path="/World/Ball",
            spawn=sim_utils.UsdFileCfg(usd_path=BALL_USD),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, GROUND_Z + GEL_HEIGHT + BALL_RADIUS + 1e-3)),
            mass_density=BALL_DENSITY,
            mesh_cfg=TetMeshCfg(stop_quality=8, max_its=100, edge_length_r=1 / 5),
            constitution_cfg=UipcRigidObjectCfg.AffineBodyConstitutionCfg(),
            render_mesh_cfg=RenderMeshCfg(mapping="surface"),
            debug_vis=dv.DEBUG_VIS,
        ),
        uipc_sim,
    )
    uipc_sim.setup_sim()
    sim.reset()
    dv.setup_view(sim, eye=(0.04, 0.04, 0.03), target=(0.0, 0.0, 0.004))

    rest = {"gel": _render_points(gel), "ball": _render_points(ball)}
    rest_surf = {"gel": _surf_points(uipc_sim, 0), "ball": _surf_points(uipc_sim, 1)}
    for _ in range(NUM_STEPS):
        dv.step(sim)
    dv.hold(sim)
    final = {"gel": _render_points(gel), "ball": _render_points(ball)}
    final_surf = {"gel": _surf_points(uipc_sim, 0), "ball": _surf_points(uipc_sim, 1)}
    topo = {
        "gel": list(UsdGeom.Mesh(gel._render_geom_mesh).GetFaceVertexIndicesAttr().Get()),
        "ball": list(UsdGeom.Mesh(ball._render_geom_mesh).GetFaceVertexIndicesAttr().Get()),
    }
    tris = {"gel": gel.sim_surf_triangles, "ball": ball.sim_surf_triangles}
    return dict(src=src, rest=rest, final=final, rest_surf=rest_surf, final_surf=final_surf, topo=topo, tris=tris)


@pytest.mark.parametrize("name", ["gel", "ball"])
def test_render_topology_kept(result, name):
    num_points, faces = result["src"][name]
    assert result["topo"][name] == faces
    assert result["final"][name].shape == (num_points, 3)
    assert np.isfinite(result["final"][name]).all()


@pytest.mark.parametrize("name", ["gel", "ball"])
def test_render_follows_sim(result, name):
    """Render vertices keep their (signed) distance to the simulation surface while it deforms/moves."""
    rest, final = result["rest"][name], result["final"][name]
    rest_s, final_s, tris = result["rest_surf"][name], result["final_surf"][name], result["tris"][name]
    d_rest = _signed_dist_to_surface(rest, rest_s, tris)
    d_final = _signed_dist_to_surface(final, final_s, tris)
    err = np.abs(d_final - d_rest)
    disp = np.linalg.norm(final_s - rest_s, axis=1).max()
    print(
        f"{name}: max sim displacement {disp * 1e3:.3f} mm, max |dist to sim surface| {np.abs(d_rest).max() * 1e3:.3f}"
        f" mm, max change {err.max() * 1e3:.4f} mm"
    )
    assert disp > 1e-4  # something actually moved
    assert err.max() < 0.1 * disp


def test_ball_render_matches_sim_size(result):
    """Render and simulation mesh of the ball have the same size (sim mesh not scaled twice)."""
    np.testing.assert_allclose(np.ptp(result["rest"]["ball"], 0), np.ptp(result["rest_surf"]["ball"], 0), rtol=0.05)


def test_ball_render_rigid(result):
    """The ball is an affine body: its render mesh moves (almost) rigidly."""
    a, b = result["rest"]["ball"], result["final"]["ball"]
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    residual = np.linalg.norm((a - ca) @ R.T + cb - b, axis=1).max()
    print(f"ball: rigid fit residual {residual * 1e6:.2f} um, translation {np.linalg.norm(cb - ca) * 1e3:.3f} mm")
    assert residual < 1e-5
    assert np.linalg.norm(cb - ca) > 1e-4


def test_gel_dented(result):
    """The render mesh shows the dent of the simulation mesh."""
    rest, final = result["rest"]["gel"], result["final"]["gel"]
    rest_s, final_s = result["rest_surf"]["gel"], result["final_surf"]["gel"]
    sim_dz = (final_s - rest_s)[:, 2].min()
    render_dz = (final - rest)[:, 2].min()
    print(f"gel: min dz sim {sim_dz * 1e3:.3f} mm, render {render_dz * 1e3:.3f} mm")
    assert sim_dz < -5e-5
    assert render_dz < 0.5 * sim_dz
