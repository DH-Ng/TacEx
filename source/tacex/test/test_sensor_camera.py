"""Checks that the GelSight sensor camera sees exactly the configured physical view.

`GelSightSensor._create_tiled_camera` derives the camera apertures from `SensorCameraCfg`:
    aperture / focal_length = view_size / focus_distance
so at `focus_distance_m` the image should span exactly `view_width_m` x `view_height_m`.

Two independent checks per camera config:
1. intrinsics: focal length in pixels (fx, fy) of the spawned camera -> view size at the focus distance.
2. rendering: a flat target of half the view size, placed just in front of the focus distance (d), must cover
   `0.5 * focus_distance / d` of the image (both axes), centered, at depth d.

Each rendering check also writes a figure (rendered depth with expected/measured target box, and the view footprint
in mm over the gelpad) to `tests/figures/camera_view_<config>.png`, or to `$TACEX_TEST_FIGURE_DIR` if set.

Run:  python -m pytest source/tacex/test/test_sensor_camera.py -v -s     (-s prints the measurement table)
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import os
from pathlib import Path

import pytest
import torch

import isaaclab.sim as sim_utils

from tacex import GelSightSensor, GelSightSensorCfg

from tacex_assets.sensors.gelsight_mini import GELSIGHT_MINI_TAXIM_CFG, GeneralGelSightMiniCfg

DT = 0.01
NUM_ENVS = 2
NUM_WARMUP_STEPS = 5

TARGET_FRACTION = 0.5
"""Target size as fraction of the configured view (both axes)."""
TARGET_THICKNESS_M = 0.0005
TARGET_OFFSET_M = 0.001
"""Target distance = focus distance - offset. A target exactly at the focus distance is not rendered reliably (seen
in several runs), so it is placed slightly closer; a pinhole camera sees `view * d / focus_distance` at distance d,
so the expected pixel extent is scaled by `focus_distance / d`."""

PX_TOL = 1.5
"""Tolerance for rendered extents/center in pixels (edge pixels are partially covered)."""
REL_TOL_INTRINSICS = 1e-3
DEPTH_TOL_M = 1e-4

FIGURE_DIR = Path(os.environ.get("TACEX_TEST_FIGURE_DIR", Path(__file__).resolve().parents[3] / "tests" / "figures"))
"""Where the rendering test writes its figures (repo `tests/` is git-ignored)."""

CAMERA_CFGS: dict[str, GelSightSensorCfg.SensorCameraCfg] = {
    # the camera used by the GelSight Mini presets (view from mm_per_pixel and 320x240 reference image)
    "gsmini_preset": GELSIGHT_MINI_TAXIM_CFG.sensor_camera_cfg,
    # different resolution, focal length, focus distance, scale and view size
    "custom": GelSightSensorCfg.SensorCameraCfg(
        resolution=(200, 100),
        data_types=["depth"],
        clipping_range_m=(0.02, 0.035),
        focal_length_cm=2.0,
        focus_distance_m=0.03,
        mm_per_pixel=0.15,  # -> view 30 x 15 mm
        reference_img_res=(200, 100),
        camera_pos_offset=(0.0, 0.0, 0.0),
    ),
}


def _target_distance(cam_cfg: GelSightSensorCfg.SensorCameraCfg) -> float:
    return cam_cfg.focus_distance_m - TARGET_OFFSET_M


def _sensor_cfg(name: str, cam_cfg: GelSightSensorCfg.SensorCameraCfg) -> GelSightSensorCfg:
    # the target must lie inside the clipping range, otherwise it is not rendered
    near, far = cam_cfg.clipping_range_m
    assert near < _target_distance(cam_cfg) < far, f"{name}: target distance outside clipping range"
    return GeneralGelSightMiniCfg().replace(
        prim_path=f"/World/{name}/env_.*/Sensor",
        sensor_camera_cfg=cam_cfg,
        data_types=["height_map"],  # no simulation approach needed, we read the raw camera depth
        update_period=0.0,
        debug_vis=False,
    )


def _build_scene(group_idx: int, name: str, cam_cfg: GelSightSensorCfg.SensorCameraCfg):
    """Sensor bodies + a flat target at the focus distance, centered on the camera axis."""
    target_cfg = sim_utils.CuboidCfg(
        size=(
            TARGET_FRACTION * cam_cfg.view_width_m,
            TARGET_FRACTION * cam_cfg.view_height_m,
            TARGET_THICKNESS_M,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.9)),
    )
    ox, oy, oz = cam_cfg.camera_pos_offset
    # camera looks along +Z of the sensor frame (camera_rot_offset = 180 deg about X, OpenGL convention);
    # the target's near face lies at _target_distance (just in front of the focus distance)
    target_pos = (ox, oy, oz + _target_distance(cam_cfg) + TARGET_THICKNESS_M / 2)
    for env_id in range(NUM_ENVS):
        env_path = f"/World/{name}/env_{env_id}"
        sim_utils.create_prim(env_path, "Xform", translation=(env_id * 1.0, group_idx * 1.0, 0.0))
        sim_utils.create_prim(f"{env_path}/Sensor", "Xform")
        target_cfg.func(f"{env_path}/Sensor/Target", target_cfg, translation=target_pos)


@pytest.fixture(scope="module")
def sensors():
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cuda"))
    for group_idx, (name, cam_cfg) in enumerate(CAMERA_CFGS.items()):
        _build_scene(group_idx, name, cam_cfg)
    sim_utils.update_stage()

    sensors = {name: GelSightSensor(_sensor_cfg(name, cam_cfg)) for name, cam_cfg in CAMERA_CFGS.items()}
    sim.reset()
    for _ in range(NUM_WARMUP_STEPS):
        sim.step()
        for sensor in sensors.values():
            sensor.update(DT, force_recompute=True)
    yield sensors
    # no stage teardown (see test_simulation_approaches.py)


def _intrinsics_view(sensor: GelSightSensor, cam_cfg: GelSightSensorCfg.SensorCameraCfg) -> tuple[float, float]:
    """View size (m) at the focus distance from the camera's focal length in pixels (pinhole: px * d / f_px)."""
    width_px, height_px = cam_cfg.resolution
    K = sensor.camera.data.intrinsic_matrices[0]
    return width_px * cam_cfg.focus_distance_m / K[0, 0].item(), height_px * cam_cfg.focus_distance_m / K[1, 1].item()


def _save_figure(name: str, sensor: GelSightSensor, depth_m: torch.Tensor, exp_px: tuple, meas_box: tuple,
                 rendered_view_m: tuple, passed: bool) -> Path:
    """Writes the rendered depth (with expected vs. measured target box) and the view footprint in mm."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # validated categorical slots 1-3 (dataviz reference palette, all-pairs safe) + neutral chrome
    c_cfg, c_meas, c_intr = "#2a78d6", "#eb6834", "#1baf7a"
    ink, muted, gel_fill, gel_edge = "#0b0b0b", "#6f6d68", "#efeee9", "#b4b2a8"

    cam_cfg = CAMERA_CFGS[name]
    width_px, height_px = cam_cfg.resolution
    exp_w, exp_h = exp_px
    c0, c1, r0, r1 = meas_box
    intr_w, intr_h = _intrinsics_view(sensor, cam_cfg)

    fig, (ax_img, ax_mm) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.patch.set_facecolor("#fcfcfb")

    # -- left: rendered depth in mm with expected (dashed) and measured (solid) target box, pixel coordinates
    im = ax_img.imshow(depth_m.cpu().numpy() * 1000, cmap="Greys", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.03)
    cbar.set_label("depth (mm)", color=muted)
    cx, cy = (width_px - 1) / 2, (height_px - 1) / 2
    # measured (thick, below) and expected (thin dashed, on top): when both agree, the dashes sit on the solid line
    ax_img.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0 + 1, r1 - r0 + 1, fill=False, ec=c_meas, lw=3.5,
                               label=f"measured {c1 - c0 + 1:.0f} x {r1 - r0 + 1:.0f} px", zorder=3))
    ax_img.add_patch(Rectangle((cx - exp_w / 2, cy - exp_h / 2), exp_w, exp_h, fill=False, ec=c_cfg, lw=1.5, ls="--",
                               label=f"expected {exp_w:.1f} x {exp_h:.1f} px", zorder=4))
    ax_img.plot([cx], [cy], marker="+", ms=10, mew=1.5, color=c_cfg)
    ax_img.set_title(f"Rendered depth, env 0 ({width_px} x {height_px} px)", color=ink, fontsize=11, loc="left")
    ax_img.set_xlabel("x (px)", color=muted)
    ax_img.set_ylabel("y (px)", color=muted)
    ax_img.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.9)

    # -- right: footprint at the focus distance in mm, centred on the camera axis
    gel_len = sensor.cfg.gelpad_dimensions.length * 1000  # along the image width (as in the old view config)
    gel_wid = sensor.cfg.gelpad_dimensions.width * 1000

    def box(w_mm, h_mm, **kw):
        ax_mm.add_patch(Rectangle((-w_mm / 2, -h_mm / 2), w_mm, h_mm, **kw))

    box(gel_len, gel_wid, fc=gel_fill, ec=gel_edge, lw=1, label=f"gelpad {gel_len:.2f} x {gel_wid:.2f} mm")
    vw, vh = cam_cfg.view_width_m * 1000, cam_cfg.view_height_m * 1000
    box(vw, vh, fill=False, ec=c_cfg, lw=2, ls="--", label=f"configured view {vw:.2f} x {vh:.2f} mm")
    box(intr_w * 1000, intr_h * 1000, fill=False, ec=c_intr, lw=2, ls=":",
        label=f"from intrinsics {intr_w * 1000:.2f} x {intr_h * 1000:.2f} mm")
    rw, rh = rendered_view_m[0] * 1000, rendered_view_m[1] * 1000
    box(rw, rh, fc=c_meas, alpha=0.15, ec=c_meas, lw=2, label=f"implied by rendering {rw:.2f} x {rh:.2f} mm")
    half_w = max(gel_len, vw, rw, intr_w * 1000) / 2 * 1.08
    half_h = max(gel_wid, vh, rh, intr_h * 1000) / 2 * 1.08
    ax_mm.set_xlim(-half_w, half_w)
    ax_mm.set_ylim(half_h, -half_h)  # image convention: y down
    ax_mm.set_aspect("equal")
    ax_mm.grid(color="#e1e0d9", lw=0.6)
    ax_mm.set_axisbelow(True)
    ax_mm.set_title(f"View footprint at focus distance {cam_cfg.focus_distance_m * 1000:.1f} mm", color=ink,
                    fontsize=11, loc="left")
    ax_mm.set_xlabel("x (mm)", color=muted)
    ax_mm.set_ylabel("y (mm)", color=muted)
    ax_mm.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, frameon=False)

    for ax in (ax_img, ax_mm):
        ax.tick_params(colors=muted, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#c3c2b7")
    verdict = "PASS" if passed else "FAIL"
    fig.suptitle(f"Sensor camera view: {name}  [{verdict}]  (mm_per_pixel {cam_cfg.mm_per_pixel},"
                 f" reference {cam_cfg.reference_img_res[0]} x {cam_cfg.reference_img_res[1]} px)",
                 x=0.01, ha="left", color=ink, fontsize=12, fontweight="semibold")
    fig.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"camera_view_{name}.png"
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return path


@pytest.mark.parametrize("name", CAMERA_CFGS.keys())
def test_view_size_from_intrinsics(sensors, name):
    cam_cfg = CAMERA_CFGS[name]
    width_px, height_px = cam_cfg.resolution
    view_w, view_h = _intrinsics_view(sensors[name], cam_cfg)
    print(
        f"\n[{name}] intrinsics: view at focus distance = {view_w * 1000:.3f} x {view_h * 1000:.3f} mm"
        f" (cfg {cam_cfg.view_width_m * 1000:.3f} x {cam_cfg.view_height_m * 1000:.3f} mm),"
        f" mm/px = {view_w * 1000 / width_px:.4f} x {view_h * 1000 / height_px:.4f}"
    )
    assert view_w == pytest.approx(cam_cfg.view_width_m, rel=REL_TOL_INTRINSICS), "view width"
    assert view_h == pytest.approx(cam_cfg.view_height_m, rel=REL_TOL_INTRINSICS), "view height"


@pytest.mark.parametrize("name", CAMERA_CFGS.keys())
def test_view_size_rendered(sensors, name):
    cam_cfg = CAMERA_CFGS[name]
    width_px, height_px = cam_cfg.resolution
    far = cam_cfg.clipping_range_m[1]

    depth = sensors[name].camera.data.output["depth"][..., 0]  # (num_envs, H, W), m, background = far clip
    assert depth.shape == (NUM_ENVS, height_px, width_px)

    for env_id in range(NUM_ENVS):
        hit = depth[env_id] < far - 1e-5  # target pixels
        assert hit.any(), f"env {env_id}: target not visible"
        cols = torch.nonzero(hit.any(dim=0)).flatten().float()
        rows = torch.nonzero(hit.any(dim=1)).flatten().float()

        meas_w_px, meas_h_px = len(cols), len(rows)
        # target spans TARGET_FRACTION of the view at the focus distance -> appears larger by focus / d when closer
        scale = cam_cfg.focus_distance_m / _target_distance(cam_cfg)
        exp_w_px, exp_h_px = TARGET_FRACTION * scale * width_px, TARGET_FRACTION * scale * height_px
        center_col, center_row = (cols.min() + cols.max()).item() / 2, (rows.min() + rows.max()).item() / 2
        target_depth = depth[env_id][hit].median().item()

        # physical view size at the focus distance implied by the rendering
        view_w = TARGET_FRACTION * cam_cfg.view_width_m * scale * width_px / meas_w_px
        view_h = TARGET_FRACTION * cam_cfg.view_height_m * scale * height_px / meas_h_px
        if env_id == 0:
            passed = (
                abs(meas_w_px - exp_w_px) <= PX_TOL
                and abs(meas_h_px - exp_h_px) <= PX_TOL
                and abs(center_col - (width_px - 1) / 2) <= PX_TOL
                and abs(center_row - (height_px - 1) / 2) <= PX_TOL
            )
            fig_path = _save_figure(
                name,
                sensors[name],
                depth[env_id],
                (exp_w_px, exp_h_px),
                (cols.min().item(), cols.max().item(), rows.min().item(), rows.max().item()),
                (view_w, view_h),
                passed,
            )
            print(f"\n[{name}] figure: {fig_path}")
            print(
                f"\n[{name}] rendered: target {meas_w_px} x {meas_h_px} px (expected {exp_w_px:.1f} x {exp_h_px:.1f}),"
                f" center ({center_col:.1f}, {center_row:.1f}) (expected ({(width_px - 1) / 2:.1f},"
                f" {(height_px - 1) / 2:.1f})), depth {target_depth * 1000:.3f} mm"
                f" -> implied view {view_w * 1000:.3f} x {view_h * 1000:.3f} mm"
                f" (cfg {cam_cfg.view_width_m * 1000:.3f} x {cam_cfg.view_height_m * 1000:.3f} mm)"
            )

        assert meas_w_px == pytest.approx(exp_w_px, abs=PX_TOL), f"env {env_id}: rendered width"
        assert meas_h_px == pytest.approx(exp_h_px, abs=PX_TOL), f"env {env_id}: rendered height"
        assert center_col == pytest.approx((width_px - 1) / 2, abs=PX_TOL), f"env {env_id}: horizontal center"
        assert center_row == pytest.approx((height_px - 1) / 2, abs=PX_TOL), f"env {env_id}: vertical center"
        assert target_depth == pytest.approx(_target_distance(cam_cfg), abs=DEPTH_TOL_M), f"env {env_id}: depth"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
