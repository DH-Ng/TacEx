"""Integration tests for the GelSight simulation approaches.

One scene is built for the whole module (one sim/stage, one RTX start-up). It is fully controlled, with no robot
and no physics contact:

- every approach gets its own group of ``NUM_ENVS`` sensor bodies (bare Xforms at ``/World/<approach>/env_i``);
  the GelSight sensor spawns its depth camera as a child, looking along +Z
  (camera_rot_offset = 180 deg about X, OpenGL convention).
- the undeformed gelpad surface lies at ``clipping_range_m[0] + gelpad_height`` in front of the camera.
- "contact" envs get a visual-only sphere that penetrates the gelpad surface by ``INDENTATION_MM``;
  the "no contact" env has no indenter.

Since nothing moves, the expected height map/indentation per env is known exactly, which lets us check each
approach (outputs, contact vs. no contact) and the per-env reset behaviour.

Note:
    The scene is shared on purpose. Tearing down a stage with a live GelSightSensor and building a new one crashes
    Kit: `GelSightSensor._invalidate_initialize_callback` calls `camera.__del__()` explicitly, and the camera's
    destructor runs again when the object is garbage collected later.

Run with the Isaac Lab python:
    python -m pytest source/tacex/test/test_simulation_approaches.py -v
"""

from isaaclab.app import AppLauncher

# launch omniverse app (cameras are required for the sensor's depth camera)
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import pytest
import torch

import isaaclab.sim as sim_utils

from tacex import GelSightSensor, GelSightSensorCfg
from tacex.simulation_approaches.fots import FOTSMarkerSimulator
from tacex.simulation_approaches.gpu_taxim import TaximSimulator
from tacex.simulation_approaches.pix2pix import Pix2PixSimulator

from tacex_assets.sensors.gelsight_mini import (
    GELSIGHT_MINI_PIX2PIX_CFG,
    GELSIGHT_MINI_TAXIM_CFG,
    GELSIGHT_MINI_TAXIM_FOTS_CFG,
)

DEVICE = "cuda"
DT = 0.01
NUM_WARMUP_STEPS = 5  # RTX needs a few frames before the depth buffers are valid

INDENTATION_MM = 1.0
INDENTATION_TOL_MM = 0.15
INDENTER_RADIUS_M = 0.003

# env layout per approach: env 0 = no contact, env 1 & 2 = identical contact
NUM_ENVS = 3
NO_CONTACT_ENV = 0
CONTACT_ENVS = (1, 2)

CAMERA_CFG = GelSightSensorCfg.SensorCameraCfg(
    prim_name="Camera",
    update_period=0,
    resolution=(320, 240),
    data_types=["depth"],
    clipping_range_m=(0.024, 0.029),
)

CONTACT_MIN_CHANGED_PX = 0.02
"""Min. fraction of changed pixels for a contact to count as visible."""

SAME_MAX_CHANGED_PX = 0.005
"""Max. fraction of changed pixels for two images of the same scene."""

SENTINEL = -1.0
"""Marker value written into a non-reset env's output buffer, to detect writes by ``reset``."""


"""
Sensor configs (one per approach).
"""


def _cfg(base_cfg: GelSightSensorCfg, group: str, **overrides) -> GelSightSensorCfg:
    cfg = base_cfg.replace(
        prim_path=f"/World/{group}/env_.*/Sensor", sensor_camera_cfg=CAMERA_CFG, debug_vis=False, device=DEVICE
    )
    return cfg.replace(**overrides) if overrides else cfg


SENSOR_CFGS: dict[str, GelSightSensorCfg] = {
    "taxim": _cfg(GELSIGHT_MINI_TAXIM_CFG, "taxim", data_types=["tactile_rgb", "height_map"]),
    "pix2pix": _cfg(GELSIGHT_MINI_PIX2PIX_CFG, "pix2pix", data_types=["tactile_rgb", "height_map"]),
    "fots": _cfg(
        GELSIGHT_MINI_TAXIM_FOTS_CFG,
        "fots",
        marker_motion_sim_cfg=GELSIGHT_MINI_TAXIM_FOTS_CFG.marker_motion_sim_cfg.replace(frame_transformer_cfg=None),
        data_types=["tactile_rgb", "marker_motion", "height_map"],
    ),
}

OPTICAL_APPROACHES = {"taxim": TaximSimulator, "pix2pix": Pix2PixSimulator}


"""
Helpers.
"""


def _build_group(group_idx: int, group: str, cfg: GelSightSensorCfg):
    """Spawns NUM_ENVS sensor bodies for one approach, plus an indenter sphere in every contact env."""
    gelpad_surface_z = cfg.sensor_camera_cfg.clipping_range_m[0] + cfg.gelpad_dimensions.height
    indenter_center_z = gelpad_surface_z - INDENTATION_MM / 1000.0 + INDENTER_RADIUS_M
    indenter_cfg = sim_utils.SphereCfg(
        radius=INDENTER_RADIUS_M,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
    )
    for env_id in range(NUM_ENVS):
        # 1 m apart, so no indenter is visible in another sensor's camera
        env_path = f"/World/{group}/env_{env_id}"
        sim_utils.create_prim(env_path, "Xform", translation=(env_id * 1.0, group_idx * 1.0, 0.0))
        sim_utils.create_prim(f"{env_path}/Sensor", "Xform", translation=(0.0, 0.0, 0.0))
        if env_id in CONTACT_ENVS:
            indenter_cfg.func(f"{env_path}/Indenter", indenter_cfg, translation=(0.0, 0.0, indenter_center_z))


def _refresh(sim: sim_utils.SimulationContext, sensor: GelSightSensor):
    """Renders a frame and recomputes the sensor outputs (undoes resets of previous tests)."""
    sim.step()
    sensor.update(DT, force_recompute=True)


def _changed_px(a: torch.Tensor, b: torch.Tensor, rel_thresh: float = 0.1) -> float:
    """Fraction of pixels whose value differs by more than ``rel_thresh`` of the image value range.

    Approach independent (works for [0,1] and [0,255] images). A pixel-fraction metric is used instead of a mean,
    because Taxim amplifies sub-micron height-map noise (from the depth camera) into a few isolated outlier pixels
    in flat regions (ill-defined gradient direction). Measured for a 1 mm sphere indentation at 320x240:
    contact vs. no contact ~6% changed pixels, two identical contact scenes ~0.01%.
    """
    scale = torch.clamp(torch.maximum(a.abs().max(), b.abs().max()), min=1e-6)
    return (((a.float() - b.float()).abs().amax(-1) / scale) > rel_thresh).float().mean().item()


"""
Fixtures.
"""


@pytest.fixture(scope="module")
def scene():
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=DEVICE))
    for group_idx, (group, cfg) in enumerate(SENSOR_CFGS.items()):
        _build_group(group_idx, group, cfg)
    sim_utils.update_stage()

    sensors = {group: GelSightSensor(cfg) for group, cfg in SENSOR_CFGS.items()}
    sim.reset()
    for _ in range(NUM_WARMUP_STEPS):
        sim.step()
        for sensor in sensors.values():
            sensor.update(DT, force_recompute=True)

    yield sim, sensors
    # no stage teardown, see module docstring


"""
Sensor core (height map + indentation): the shared input of every approach.
"""


@pytest.mark.parametrize("group", SENSOR_CFGS.keys())
def test_sensor_initialized(scene, group):
    _, sensors = scene
    sensor = sensors[group]
    assert sensor.is_initialized
    assert sensor.num_instances == NUM_ENVS


@pytest.mark.parametrize("group", SENSOR_CFGS.keys())
def test_height_map_and_indentation(scene, group):
    sim, sensors = scene
    sensor, cfg = sensors[group], SENSOR_CFGS[group]
    _refresh(sim, sensor)

    height_map = sensor.data.output["height_map"]
    assert height_map.shape == (NUM_ENVS, CAMERA_CFG.resolution[1], CAMERA_CFG.resolution[0])
    assert torch.isfinite(height_map).all()
    # values in [-gelpad_height, 0] mm
    assert height_map.max() <= 0.0
    assert height_map.min() >= -cfg.gelpad_dimensions.height * 1000 - 1e-4

    depth = sensor.indentation_depth
    assert (height_map[NO_CONTACT_ENV] == 0).all()
    assert depth[NO_CONTACT_ENV].item() == pytest.approx(0.0, abs=1e-3)
    for env_id in CONTACT_ENVS:
        assert depth[env_id].item() == pytest.approx(INDENTATION_MM, abs=INDENTATION_TOL_MM)


def test_missing_sim_cfg_raises(scene):
    """Requesting an output without a simulator configured for it must fail at construction."""
    base = SENSOR_CFGS["taxim"]
    with pytest.raises(RuntimeError, match="tactile_rgb"):
        GelSightSensor(base.replace(optical_sim_cfg=None, data_types=["tactile_rgb"]))
    with pytest.raises(RuntimeError, match="marker_motion"):
        GelSightSensor(base.replace(data_types=["marker_motion"]))


"""
Optical approaches.
"""


@pytest.mark.parametrize("group", OPTICAL_APPROACHES.keys())
def test_optical_simulation(scene, group):
    sim, sensors = scene
    sensor, cfg = sensors[group], SENSOR_CFGS[group]
    _refresh(sim, sensor)

    assert type(sensor.optical_simulator) is OPTICAL_APPROACHES[group]

    img = sensor.data.output["tactile_rgb"]
    w, h = cfg.optical_sim_cfg.tactile_img_res
    assert img.shape == (NUM_ENVS, h, w, 3)
    assert torch.isfinite(img).all()

    no_contact = img[NO_CONTACT_ENV]
    for env_id in CONTACT_ENVS:
        changed = _changed_px(img[env_id], no_contact)
        assert changed > CONTACT_MIN_CHANGED_PX, f"env {env_id}: contact barely visible ({changed:.4f} changed px)"
    # identical scenes -> (nearly) identical images
    assert _changed_px(img[CONTACT_ENVS[0]], img[CONTACT_ENVS[1]]) < SAME_MAX_CHANGED_PX


@pytest.mark.parametrize("group", OPTICAL_APPROACHES.keys())
def test_optical_reset_is_per_env(scene, group):
    sim, sensors = scene
    sensor = sensors[group]
    _refresh(sim, sensor)

    buffer = sensor.data.output["tactile_rgb"]
    reset_env, kept_env = CONTACT_ENVS
    # sentinel: re-simulating an env would give bit-identical values, so mark the kept env to detect any write
    buffer[kept_env] = SENTINEL
    before = buffer.clone()

    sensor.reset(env_ids=[reset_env])
    after = sensor._data.output["tactile_rgb"]  # not `.data`, which would re-simulate

    # buffer is updated in place (references held by users stay valid)
    assert after.data_ptr() == buffer.data_ptr()
    # reset env shows the no-contact image
    assert _changed_px(after[reset_env], before[NO_CONTACT_ENV]) < SAME_MAX_CHANGED_PX
    # other envs are untouched
    assert (after[kept_env] == SENTINEL).all(), "reset wrote into an env that was not reset"
    assert torch.equal(after[NO_CONTACT_ENV], before[NO_CONTACT_ENV])
    assert sensor.indentation_depth[reset_env].item() == 0.0
    assert sensor.indentation_depth[kept_env].item() == pytest.approx(INDENTATION_MM, abs=INDENTATION_TOL_MM)


"""
Marker motion approaches.
"""


def test_fots_marker_motion(scene):
    sim, sensors = scene
    sensor, cfg = sensors["fots"], SENSOR_CFGS["fots"]
    _refresh(sim, sensor)

    assert type(sensor.marker_motion_simulator) is FOTSMarkerSimulator
    assert type(sensor.optical_simulator) is TaximSimulator

    markers = sensor.data.output["marker_motion"]
    num_markers = cfg.marker_motion_sim_cfg.marker_params.num_markers
    assert markers.shape == (NUM_ENVS, 2, num_markers, 2)
    assert torch.isfinite(markers).all()

    # initial marker positions are identical in every env
    assert torch.allclose(markers[:, 0], markers[0:1, 0].expand_as(markers[:, 0]))

    displacement = (markers[:, 1] - markers[:, 0]).norm(dim=-1)  # (num_envs, num_markers), in px
    assert displacement[NO_CONTACT_ENV].max() < 1e-4, "markers moved without contact"
    for env_id in CONTACT_ENVS:
        assert displacement[env_id].max() > 0.1, f"env {env_id}: markers did not move under contact"


def test_fots_reset_is_per_env(scene):
    sim, sensors = scene
    sensor = sensors["fots"]
    _refresh(sim, sensor)

    buffer = sensor.data.output["marker_motion"]
    reset_env, kept_env = CONTACT_ENVS
    buffer[kept_env] = SENTINEL  # see test_optical_reset_is_per_env
    before = buffer.clone()

    sensor.reset(env_ids=[reset_env])
    after = sensor._data.output["marker_motion"]

    assert after.data_ptr() == buffer.data_ptr()
    # reset env: current markers back at the initial positions, trajectory cleared
    assert torch.allclose(after[reset_env, 1], after[reset_env, 0])
    assert sensor._data.output["traj"][reset_env] == []
    # other envs untouched
    assert (after[kept_env] == SENTINEL).all(), "reset wrote into an env that was not reset"
    assert torch.equal(after[NO_CONTACT_ENV], before[NO_CONTACT_ENV])
    assert len(sensor._data.output["traj"][kept_env]) > 0


@pytest.mark.skip(reason="FEM (ManiSkill) approach needs a UIPC gelpad scene and only writes env 0 so far.")
def test_fem_marker_motion():
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
