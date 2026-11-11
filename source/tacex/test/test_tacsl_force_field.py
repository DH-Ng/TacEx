"""Tests the force field output of the TacSL simulation approach (Isaac Lab's TacSL force field in TacEx).

Controlled scene (one module-scoped stage), per env:

- a GelSight sensor body (bare Xform; the sensor only needs it for its camera, which this test does not use)
- the gelpad: a kinematic box mesh (rigid body) whose contact surface lies at z = GEL_TOP_Z, outward normal +z
- the contact object: a kinematic sphere with an SDF collision mesh, pressed PRESS_M into the gelpad in contact
  envs and far away in the no-contact env

Nothing moves, so every tactile point's penetration depth is known analytically: max(0, r - |p - c|).

Run with the Isaac Lab python:  python -m pytest source/tacex/test/test_tacsl_force_field.py -v
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas

from tacex import GelSightSensor
from tacex.simulation_approaches.tacsl.tacsl_sim import TacSLSimulator  # not via the package: no silent ImportError
from tacex.simulation_approaches.tacsl.tacsl_sim_cfg import TacSLForceFieldCfg

from tacex_assets.sensors.gelsight_mini.gsmini_tacsl import GELSIGHT_MINI_TACSL_CFG, GELSIGHT_MINI_TACSL_SIM_CFG

DT = 0.01
NUM_WARMUP_STEPS = 3

GEL_SIZE = (0.025, 0.020, 0.004)  # (x, y, thickness) [m]
GEL_TOP_Z = 0.028  # contact surface, same as the GelSight Mini's undeformed gel surface
SPHERE_RADIUS = 0.005
PRESS_M = 0.001
ARRAY_SIZE = (10, 12)
MARGIN = 0.002

CONTACT_ENVS = (0, 1)
NO_CONTACT_ENV = 2
NUM_ENVS = 3

DEPTH_TOL_M = 2e-4  # SDF resolution and sphere tessellation
SPHERE_CENTER_Z = GEL_TOP_Z + SPHERE_RADIUS - PRESS_M


def _build_scene():
    gel_cfg = sim_utils.MeshCuboidCfg(
        size=GEL_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
    )
    sphere_cfg = sim_utils.MeshSphereCfg(
        radius=SPHERE_RADIUS,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )
    for env_id in range(NUM_ENVS):
        env = f"/World/envs/env_{env_id}"
        sim_utils.create_prim(env, "Xform", translation=(env_id * 1.0, 0.0, 0.0))
        sim_utils.create_prim(f"{env}/Sensor", "Xform")
        gel_cfg.func(f"{env}/Gelpad", gel_cfg, translation=(0.0, 0.0, GEL_TOP_Z - GEL_SIZE[2] / 2))
        sphere_z = SPHERE_CENTER_Z if env_id in CONTACT_ENVS else 0.1
        sphere_cfg.func(f"{env}/Object", sphere_cfg, translation=(0.0, 0.0, sphere_z))
        # TacSL queries the object's SDF; mesh spawners only set analytic approximations
        schemas.define_mesh_collision_properties(
            f"{env}/Object/geometry/mesh", sim_utils.SDFMeshPropertiesCfg(sdf_resolution=256)
        )
    sim_utils.update_stage()


def _sensor_cfg():
    tacsl_cfg = GELSIGHT_MINI_TACSL_SIM_CFG.replace(
        force_field=TacSLForceFieldCfg(
            elastomer_prim_path_expr="{ENV_REGEX_NS}/Gelpad",
            contact_object_prim_path_expr="{ENV_REGEX_NS}/Object",
            contact_normal_local=(0.0, 0.0, 1.0),
            tactile_array_size=ARRAY_SIZE,
            tactile_margin=MARGIN,
        )
    )
    return GELSIGHT_MINI_TACSL_CFG.replace(
        prim_path="/World/envs/env_.*/Sensor",
        optical_sim_cfg=None,
        force_field_sim_cfg=tacsl_cfg,
        data_types=["force_field"],
        debug_vis=False,
    )


@pytest.fixture(scope="module")
def scene():
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cuda"))
    _build_scene()
    sensor = GelSightSensor(_sensor_cfg())
    sim.reset()
    for _ in range(NUM_WARMUP_STEPS):
        sim.step()
        sensor.update(DT, force_recompute=True)
    yield sim, sensor
    # no stage teardown (see test_simulation_approaches.py)


def _refresh(sim, sensor):
    sim.step()
    sensor.update(DT, force_recompute=True)
    return sensor.data.output["force_field"]


def _expected_depth(points_pos_w: torch.Tensor) -> torch.Tensor:
    """Analytic penetration depth of every tactile point for the sphere of its env (world frame)."""
    env_x = torch.arange(NUM_ENVS, device=points_pos_w.device, dtype=points_pos_w.dtype) * 1.0
    centers = torch.zeros(NUM_ENVS, 3, device=points_pos_w.device)
    centers[:, 0] = env_x
    centers[:, 2] = torch.tensor([SPHERE_CENTER_Z if e in CONTACT_ENVS else 0.1 for e in range(NUM_ENVS)])
    dist = (points_pos_w - centers[:, None, :]).norm(dim=-1)
    return (SPHERE_RADIUS - dist).clamp(min=0.0)


def test_simulator_and_output_shapes(scene):
    _, sensor = scene
    assert type(sensor.force_field_simulator) is TacSLSimulator
    assert sensor.optical_simulator is None
    out = sensor.data.output["force_field"]
    num_pts = ARRAY_SIZE[0] * ARRAY_SIZE[1]
    assert out["points_pos_w"].shape == (NUM_ENVS, num_pts, 3)
    assert out["penetration_depth"].shape == (NUM_ENVS, num_pts)
    assert out["normal_force"].shape == (NUM_ENVS, num_pts)
    assert out["shear_force"].shape == (NUM_ENVS, num_pts, 2)
    for value in out.values():
        assert torch.isfinite(value).all()


def test_tactile_points_on_gel_surface(scene):
    sim, sensor = scene
    pts = _refresh(sim, sensor)["points_pos_w"]
    # all points on the contact surface, inside the gelpad minus the margin
    assert torch.allclose(pts[..., 2], torch.full_like(pts[..., 2], GEL_TOP_Z), atol=1e-5)
    env_x = torch.arange(NUM_ENVS, device=pts.device, dtype=pts.dtype)[:, None]
    local_x, local_y = pts[..., 0] - env_x, pts[..., 1]
    assert local_x.abs().max() <= GEL_SIZE[0] / 2 - MARGIN + 1e-6
    assert local_y.abs().max() <= GEL_SIZE[1] / 2 - MARGIN + 1e-6
    # grid is centered on the gelpad
    assert local_x.mean().abs() < 1e-5 and local_y.mean().abs() < 1e-5


def test_penetration_matches_geometry(scene):
    sim, sensor = scene
    out = _refresh(sim, sensor)
    depth, expected = out["penetration_depth"], _expected_depth(out["points_pos_w"])
    assert (depth[NO_CONTACT_ENV] == 0).all()
    for env_id in CONTACT_ENVS:
        assert depth[env_id].max() > 0.5 * PRESS_M, f"env {env_id}: no penetration detected"
        assert torch.allclose(depth[env_id], expected[env_id], atol=DEPTH_TOL_M)


def test_forces(scene):
    sim, sensor = scene
    out = _refresh(sim, sensor)
    depth, normal, shear = out["penetration_depth"], out["normal_force"], out["shear_force"]
    k_n = sensor.force_field_simulator.cfg.force_field.normal_contact_stiffness

    in_contact = depth > 0
    # normal force only where the point penetrates, |F_n| <= k_n * depth (= at points right below the sphere)
    assert (normal[~in_contact] == 0).all()
    assert (normal[in_contact].abs() <= k_n * depth[in_contact] + 1e-9).all()
    deepest = depth.argmax(dim=1)
    for env_id in CONTACT_ENVS:
        assert normal[env_id, deepest[env_id]].abs().item() == pytest.approx(
            k_n * depth[env_id, deepest[env_id]].item(), rel=0.05
        )
    # all normal forces push in the same direction (into the gelpad)
    assert (torch.sign(normal[in_contact]) == torch.sign(normal[in_contact][0])).all()
    # nothing moves -> no friction: the whole force is the penalty force k_n * depth along the sphere's surface
    # normal. TacSL's "shear" is the in-plane part of that force (non-zero where the sphere surface is tilted).
    magnitude = torch.sqrt(normal**2 + (shear**2).sum(-1))
    assert torch.allclose(magnitude[in_contact], k_n * depth[in_contact], rtol=1e-3, atol=1e-9)
    assert (shear[~in_contact] == 0).all()


def test_reset_is_per_env(scene):
    sim, sensor = scene
    out = _refresh(sim, sensor)
    before = {k: v.clone() for k, v in out.items()}
    reset_env, kept_env = CONTACT_ENVS

    sensor.reset(env_ids=[reset_env])
    after = sensor._data.output["force_field"]

    assert after is out  # same dict and tensors (updated in place)
    for key in ("penetration_depth", "normal_force", "shear_force"):
        assert (after[key][reset_env] == 0).all(), key
        assert torch.equal(after[key][kept_env], before[key][kept_env]), key
    # the next update recomputes the reset env from the (unchanged) scene
    out = _refresh(sim, sensor)
    assert torch.allclose(out["penetration_depth"][reset_env], before["penetration_depth"][reset_env])


def test_flat_gelpad_needs_contact_normal(scene):
    """Without contact_normal_local the contact side of a flat gelpad cannot be detected: clear error."""
    _, sensor = scene
    simulator = sensor.force_field_simulator
    ff = simulator._force_field
    original = ff.cfg
    ff.cfg = original.replace(contact_normal_local=None)
    try:
        with pytest.raises(ValueError, match="contact_normal_local"):
            ff._generate_tactile_points(list(ARRAY_SIZE), MARGIN, visualize=False)
    finally:
        ff.cfg = original
        ff._generate_tactile_points(list(ARRAY_SIZE), MARGIN, visualize=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
