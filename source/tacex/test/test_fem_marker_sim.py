"""Integration test of the FEM-based (ManiSkill) marker simulation on a UIPC gelpad.

Tests the tactile output only; the UIPC behavior itself (attachment, gel deformation) is tested in
``source/tacex_uipc/test``. Own scene, no demo envs: an Isaac rigid body with the GelSight Mini case geometry carries a
GelSight sensor (``GELSIGHT_MINI_TAXIM_FEM_CFG``: Taxim + FEM markers); the UIPC gelpad is attached to it through a UIPC
rigid carrier body, and the case presses a UIPC ball (gel 3 mm above the ball -> 1.5 mm into it -> hold).

Checked:

- marker grid: centered in the tactile image, regular spacing (marker interval projected at the gel surface),
  inside the image, and current = initial markers right after the start.
- markers move only under contact: without contact they stay at their initial positions, under contact they move.

Run with the Isaac Lab python:
    python -m pytest source/tacex/test/test_fem_marker_sim.py -v -s     (-s prints the measured values)
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import numpy as np
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg

from tacex import GelSightSensor

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.sensors.gelsight_mini import GELSIGHT_MINI_TAXIM_FEM_CFG
from tacex_uipc import UipcSim, UipcSimCfg
from tacex_uipc.objects import UipcDeformableObject, UipcDeformableObjectCfg, UipcRigidObject, UipcRigidObjectCfg
from tacex_uipc.objects.constraints.uipc_carrier_attachment import (
    UipcIsaacCarrierAttachments,
    UipcIsaacCarrierAttachmentsCfg,
)
from tacex_uipc.utils import TetMeshCfg

DT = 0.01
GROUND_Z = 0.005
BALL_RADIUS = 0.005
GELPAD_OFFSET = 0.024  # gelpad underside in the case frame (as in Sensor.usd)
GEL_HEIGHT = 0.004
CASE_QUAT = (0.0, 1.0, 0.0, 0.0)  # (w, x, y, z): 180 deg about x, gel facing down
SENSOR_DIR = f"{TACEX_ASSETS_DATA_DIR}/Sensors/GelSight_Mini"

SETTLE_STEPS = 20
START_GAP = 0.003
PRESS_DEPTH = 0.0015
PRESS_STEPS = 45
HOLD_STEPS = 10
CONTACT_DIST = 2e-4
"""A gel surface vertex closer than this to the ball surface counts as contact (IPC d_hat = 0.1 mm)."""


def _trajectory() -> list[np.ndarray]:
    z0 = GROUND_Z + 2 * BALL_RADIUS + START_GAP + GELPAD_OFFSET + GEL_HEIGHT
    traj = [np.array([0.0, 0.0, z0])] * SETTLE_STEPS
    traj += [np.array([0.0, 0.0, z0 - (START_GAP + PRESS_DEPTH) * i / PRESS_STEPS]) for i in range(1, PRESS_STEPS + 1)]
    return traj + [traj[-1]] * HOLD_STEPS


@pytest.fixture(scope="module")
def run():
    """Builds the scene, runs the trajectory once and records the marker output."""
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cuda"))
    uipc_sim = UipcSim(
        UipcSimCfg(dt=DT, ground_height=GROUND_Z, contact=UipcSimCfg.Contact(d_hat=1e-4, default_friction_ratio=0.95))
    )
    traj = _trajectory()
    z0 = traj[0][2]

    case = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Case",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{SENSOR_DIR}/Case.usd", rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z0), rot=CASE_QUAT),
        )
    )
    gel = UipcDeformableObject(
        UipcDeformableObjectCfg(
            prim_path="/World/Gelpad",
            spawn=sim_utils.UsdFileCfg(usd_path=f"{SENSOR_DIR}/Gelpad_low_res.usd"),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, z0 - GELPAD_OFFSET), rot=CASE_QUAT),
            mesh_cfg=TetMeshCfg(stop_quality=8, max_its=100, edge_length_r=1 / 20),
            constitution_cfg=UipcDeformableObjectCfg.StableNeoHookeanCfg(),
        ),
        uipc_sim,
    )
    ball = UipcRigidObject(
        UipcRigidObjectCfg(
            prim_path="/World/Ball",
            spawn=sim_utils.UsdFileCfg(usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/ball_wood.usd"),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, GROUND_Z + BALL_RADIUS + 0.0005)),
            mesh_cfg=TetMeshCfg(stop_quality=8, max_its=100, edge_length_r=1 / 5),
            constitution_cfg=UipcRigidObjectCfg.AffineBodyConstitutionCfg(),
        ),
        uipc_sim,
    )
    UipcIsaacCarrierAttachments(
        UipcIsaacCarrierAttachmentsCfg(isaaclab_rigid_body_prim_path="/World/Case"), gel, uipc_sim, case
    )
    sensor = GelSightSensor(
        GELSIGHT_MINI_TAXIM_FEM_CFG.replace(prim_path="/World/Case", debug_vis=False), gelpad_obj=gel
    )
    uipc_sim.setup_sim()
    sim.reset()

    fem = sensor.marker_motion_simulator.marker_motion_sim
    quat = torch.tensor([CASE_QUAT], device="cuda")
    # read the outputs via sensor._data: sensor.data is lazy and would trigger another sensor update
    record = {
        "fem": fem,
        "img_res": sensor.cfg.marker_motion_sim_cfg.tactile_img_res,
        "ref_cam": fem.reference_surface_vertices_camera.cpu().numpy(),
        "markers_start": sensor._data.output["marker_motion"][0].cpu().numpy().copy(),
        "steps": [],
    }
    for pos in traj:
        case.write_root_pose_to_sim(torch.cat([torch.tensor([pos], device="cuda", dtype=torch.float32), quat], -1))
        case.write_root_velocity_to_sim(torch.zeros(1, 6, device="cuda"))
        sim.step(render=True)
        case.update(DT)
        gel.update(DT)
        ball.update(DT)
        sensor.update(DT, force_recompute=True)
        surf_w = gel._data.surf_nodal_pos_w.reshape(-1, 3).cpu().numpy()
        ball_dist = np.linalg.norm(surf_w - ball.data.root_pos_w[0].cpu().numpy(), axis=1) - BALL_RADIUS
        record["steps"].append({
            "contact": ball_dist.min() < CONTACT_DIST,
            "markers": sensor._data.output["marker_motion"][0].cpu().numpy().copy(),
        })  # fmt: skip
    return record


def _unique_markers(markers: np.ndarray) -> np.ndarray:
    """Markers without the padding duplicates (the FEM sim pads to num_markers with copies of the last marker)."""
    return np.unique(np.round(markers, 3), axis=0)


def test_marker_grid(run):
    w, h = run["img_res"]
    init, curr = run["markers_start"]
    grid = _unique_markers(init)

    # expected spacing: marker interval at the gel surface (the markers lie on the top surface, z = max)
    fx = run["fem"].camera_intrinsic[0, 0]
    spacing_px = run["fem"].marker_interval_range[0] / 1000 * fx / run["ref_cam"][:, 2].max()
    nn = np.sort(np.linalg.norm(grid[:, None] - grid[None], axis=-1), axis=1)[:, 1]  # nearest-neighbor distance
    center = grid.mean(0)
    span = grid.max(0) - grid.min(0)
    print(f"\n[grid] {len(grid)} markers, center ({center[0]:.1f}, {center[1]:.1f}) px, span {span[0]:.0f} x {span[1]:.0f}"
          f" px, spacing {np.median(nn):.1f} px (expected {spacing_px:.1f})")  # fmt: skip

    assert np.allclose(init, curr), "at the start, current markers must equal the initial ones (no contact)"
    assert len(grid) >= 40, "too few markers on the gel"
    assert (grid[:, 0] > 0).all() and (grid[:, 0] < w).all() and (grid[:, 1] > 0).all() and (grid[:, 1] < h).all()
    assert abs(center[0] - w / 2) < 12 and abs(center[1] - h / 2) < 12, "grid not centered in the image"
    assert span[0] > 0.7 * w and span[1] > 0.7 * h, "grid does not cover the gel"
    assert np.median(nn) == pytest.approx(spacing_px, rel=0.1), "grid spacing does not match the marker interval"
    assert nn.max() < 1.5 * spacing_px, "holes in the grid"


def test_markers_move_only_under_contact(run):
    def motion(step):
        init, curr = step["markers"]
        return np.linalg.norm(curr - init, axis=1).max()

    free = [motion(s) for s in run["steps"][SETTLE_STEPS:] if not s["contact"]]
    loaded = [motion(s) for s in run["steps"] if s["contact"]]
    print(f"\n[markers] without contact ({len(free)} steps): max motion {max(free):.2f} px;"
          f" with contact ({len(loaded)} steps): max motion {max(loaded):.2f} px")  # fmt: skip

    assert len(free) >= 20 and len(loaded) >= 5, "expected phases with and without contact"
    assert max(free) < 1.0, "markers move without contact"
    assert max(loaded) > 3.0, "markers do not move under contact"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
