"""Voxel tet meshes (``VoxelTetMeshCfg``) with the original USD mesh as render mesh.

Scene: a UIPC gelpad (StableNeoHookean) and a UIPC ball (affine body), both with coarse conservative voxel tet meshes
and without ``render_mesh_cfg``: the render mesh + tetrahedral mapping must be enabled automatically.

Run with the Isaac Lab python:
    [TACEX_TEST_DEBUG_VIS=1] python -m pytest source/tacex_uipc/test/render_mesh/test_render_mesh_voxel.py -v -s
"""

import debug_vis_helper as dv

simulation_app = dv.launch_app()  # headless, or the GUI with TACEX_TEST_DEBUG_VIS=1 (see debug_vis_helper.py)

"""Rest everything follows."""

import numpy as np
import pytest
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_uipc import UipcSim, UipcSimCfg
from tacex_uipc.objects import UipcDeformableObject, UipcDeformableObjectCfg, UipcRigidObject, UipcRigidObjectCfg
from tacex_uipc.utils import VoxelTetMeshCfg

DT = 0.01
NUM_STEPS = 60
GEL_Z = 0.005  # above the ground by more than one gel voxel (the conservative cage sticks out of the USD mesh)
BALL_Z = 0.03
BALL_DIAMETER = 0.01


def _render_points(obj) -> np.ndarray:
    return np.array(obj.fabric_prim.GetAttribute("points").Get(), dtype=np.float64)


def _surf_points(uipc_sim, obj_idx) -> np.ndarray:
    pts = uipc_sim.sio.simplicial_surface(2).positions().view().reshape(-1, 3)
    o = uipc_sim._surf_vertex_offsets
    return np.asarray(pts[o[obj_idx] : o[obj_idx + 1]])


@pytest.fixture(scope="module")
def result():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cuda"))
    uipc_sim = UipcSim(UipcSimCfg(dt=DT, ground_height=0.0, contact=UipcSimCfg.Contact(d_hat=1e-4)))
    gel = UipcDeformableObject(
        UipcDeformableObjectCfg(
            prim_path="/World/Gelpad",
            spawn=sim_utils.UsdFileCfg(usd_path=f"{TACEX_ASSETS_DATA_DIR}/Sensors/GelSight_Mini/Gelpad_low_res.usd"),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, GEL_Z)),
            mesh_cfg=VoxelTetMeshCfg(resolution=12),
            constitution_cfg=UipcDeformableObjectCfg.StableNeoHookeanCfg(),
            debug_vis=True,  # shows the voxel cage
        ),
        uipc_sim,
    )
    ball = UipcRigidObject(
        UipcRigidObjectCfg(
            prim_path="/World/Ball",
            spawn=sim_utils.UsdFileCfg(usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/ball_wood.usd"),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, BALL_Z)),
            mesh_cfg=VoxelTetMeshCfg(resolution=4),
            constitution_cfg=UipcRigidObjectCfg.AffineBodyConstitutionCfg(),
            mass_density=2e4,
        ),
        uipc_sim,
    )
    uipc_sim.setup_sim()
    sim.reset()
    dv.setup_view(sim, eye=(0.05, 0.05, 0.04), target=(0.0, 0.0, 0.008))

    objs = {"gel": gel, "ball": ball}
    rest = {k: _render_points(o) for k, o in objs.items()}
    rest_surf = {"gel": _surf_points(uipc_sim, 0), "ball": _surf_points(uipc_sim, 1)}
    for _ in range(NUM_STEPS):
        dv.step(sim)
    final = {k: _render_points(o) for k, o in objs.items()}
    final_surf = {"gel": _surf_points(uipc_sim, 0), "ball": _surf_points(uipc_sim, 1)}

    # sim mesh visualization: points while shown, then toggled off -> no more updates
    vis_points_shown = np.array(gel._sim_mesh_vis_fabric.GetAttribute("points").Get(), dtype=np.float64)
    gel.set_debug_vis(False)
    for _ in range(10):
        dv.step(sim)
    vis = dict(
        points_shown=vis_points_shown,
        points_hidden=np.array(gel._sim_mesh_vis_fabric.GetAttribute("points").Get(), dtype=np.float64),
        surf_after_hidden=_surf_points(uipc_sim, 0),
        visibility_hidden=gel._sim_mesh_vis_prim.ComputeVisibility(),
    )
    gel.set_debug_vis(True)
    dv.hold(sim)
    vis["visibility_shown_again"] = gel._sim_mesh_vis_prim.ComputeVisibility()
    vis["points_shown_again"] = np.array(gel._sim_mesh_vis_fabric.GetAttribute("points").Get(), dtype=np.float64)
    return dict(objs=objs, rest=rest, final=final, rest_surf=rest_surf, final_surf=final_surf, vis=vis, sim=sim)


@pytest.mark.parametrize("name", ["gel", "ball"])
def test_auto_render_mesh_tetrahedral(result, name):
    obj = result["objs"][name]
    assert obj.cfg.render_mesh_cfg is not None
    assert obj.cfg.render_mesh_cfg.mapping == "tetrahedral"
    assert obj._render_mapper.method == "tetrahedral"
    print(f"{name}: {obj._sim_tets.shape[0]} tets, {obj._render_points_w.shape[0]} render vertices")


@pytest.mark.parametrize("name", ["gel", "ball"])
def test_conservative_cage_encloses_render_mesh(result, name):
    assert result["objs"][name]._render_mapper.num_outside == 0


@pytest.mark.parametrize("name", ["gel", "ball"])
def test_render_finite_and_moved(result, name):
    final, rest = result["final"][name], result["rest"][name]
    assert np.isfinite(final).all()
    assert np.linalg.norm(final - rest, axis=1).max() > 1e-4


def test_ball_render_rigid_and_true_size(result):
    a, b = result["rest"]["ball"], result["final"]["ball"]
    # render mesh keeps the size of the USD ball (not of the larger voxel cage)
    np.testing.assert_allclose(np.ptp(a, 0), BALL_DIAMETER, rtol=1e-3)
    assert np.ptp(result["rest_surf"]["ball"], 0).min() > BALL_DIAMETER
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    residual = np.linalg.norm((a - ca) @ R.T + cb - b, axis=1).max()
    print(f"ball: rigid fit residual {residual * 1e6:.2f} um, translation {np.linalg.norm(cb - ca) * 1e3:.3f} mm")
    assert residual < 1e-5


def test_gel_render_dented(result):
    rest, final = result["rest"]["gel"], result["final"]["gel"]
    rest_s, final_s = result["rest_surf"]["gel"], result["final_surf"]["gel"]
    # remove the rigid settling of the gel (it falls onto the ground): compare against the min z of the bottom
    sim_dz = (final_s - rest_s)[:, 2]
    render_dz = (final - rest)[:, 2]
    dent_sim = sim_dz.min() - np.median(sim_dz)
    dent_render = render_dz.min() - np.median(render_dz)
    print(f"gel: dent sim {dent_sim * 1e3:.3f} mm, render {dent_render * 1e3:.3f} mm")
    assert dent_sim < -5e-5
    assert dent_render < 0.3 * dent_sim


def test_sim_mesh_vis(result):
    """debug_vis shows the voxel cage (sim surface wireframe) under /Visuals, tracks the sim, toggles at runtime."""
    gel, ball, vis = result["objs"]["gel"], result["objs"]["ball"], result["vis"]
    stage = result["sim"].stage
    prim = stage.GetPrimAtPath(gel.sim_mesh_vis_prim_path)
    assert prim.IsValid() and gel.sim_mesh_vis_prim_path.startswith("/Visuals/")
    # one line segment per sim surface edge (closed triangle surface: E = 3/2 F)
    edges = gel._sim_mesh_vis_edges
    assert edges.shape[0] * 2 == gel.sim_surf_triangles.shape[0] * 3
    assert len(UsdGeom.BasisCurves(prim).GetCurveVertexCountsAttr().Get()) == edges.shape[0]
    # while shown: equals the current sim surface
    np.testing.assert_allclose(vis["points_shown"], result["final_surf"]["gel"][edges].reshape(-1, 3), atol=1e-6)
    # hidden: invisible and no longer updated
    assert vis["visibility_hidden"] == UsdGeom.Tokens.invisible
    np.testing.assert_allclose(vis["points_hidden"], vis["points_shown"], atol=1e-9)
    # shown again: visible and refreshed immediately
    assert vis["visibility_shown_again"] == UsdGeom.Tokens.inherited
    np.testing.assert_allclose(vis["points_shown_again"], vis["surf_after_hidden"][edges].reshape(-1, 3), atol=1e-6)
    # no debug_vis -> no prim
    assert not stage.GetPrimAtPath(ball.sim_mesh_vis_prim_path).IsValid()
