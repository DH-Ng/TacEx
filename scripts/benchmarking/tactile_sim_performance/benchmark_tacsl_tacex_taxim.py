"""Benchmark: native Isaac Lab TacSL vs. TacEx Taxim vs. TacEx TacSL (optical tactile images).

Scene: ``num_envs`` envs, each with ``--num_sensors`` GelSight Mini sensors (full asset ``Sensors/GelSight_Mini/Sensor.usd``:
case, glass plate and gelpad mesh; static, without collisions, so the sensor meshes are rendered but do not push the
shapes). A different test shape (``tacex_assets/data/Props/tactile_test_shapes``) presses on every sensor (pose set
every step, sinusoidal press 0..1.5 mm).

Approaches (all use TacEx's GelSight Mini Taxim calibration and the same 20.29 x 15.22 mm view):

- ``native_tacsl``: Isaac Lab ``VisuoTactileSensor`` (TiledCamera + ``GelsightRender``). Always 640x480: the TacSL
  calibration polynomial is defined on the pixel grid of the calibration resolution. Runs once per num_envs.
- ``tacex_taxim``: TacEx ``GelSightSensor`` + Taxim. Runs at every ``--resolutions`` value.
- ``tacex_tacsl``: TacEx ``GelSightSensor`` + TacSL wrapper. Runs at every ``--resolutions`` value.

Resolution (``--resolutions``, TacEx approaches) = camera resolution and tactile image resolution (``tactile_img_res``).
Taxim simulates at that resolution. The TacEx TacSL wrapper still renders at the 640x480 calibration resolution (depth
upsampled, image resized to the output), so for it the resolution only changes camera and resize cost. Taxim keeps the
calibration's mm-per-pixel, so images below 640x480 are for timing, not physically calibrated.

Every (approach, resolution, num_envs) runs in a fresh Isaac Sim process (``--worker``): clean VRAM numbers, and an
out-of-memory run fails on its own. Measured per run: physics+render step time, sensor update time, tactile images per
second, peak VRAM above idle (NVML, device-wide) and PyTorch peak allocation, process CPU %, peak RAM, GPU utilization.

Usage (Isaac Lab python):
    python scripts/benchmarking/tactile_sim_performance/benchmark_tacsl_tacex_taxim.py
    python scripts/benchmarking/tactile_sim_performance/benchmark_tacsl_tacex_taxim.py --num_envs 1 4 16 --num_sensors 4 \
        --approaches native_tacsl tacex_taxim tacex_tacsl --resolutions 32x24 64x48 128x96 320x240 640x480 --steps 300

Results: ``scripts/benchmarking/tactile_sim_performance/logs/tacsl_vs_tacex_taxim/<timestamp>/``
(results.csv, results.json, summary.md, benchmark.png = metrics vs. num_envs at the reference resolution,
resolution.png = metrics vs. resolution at the largest num_envs, samples/*.png).

Watch a run live (Isaac Sim window + a window with env 0's tactile images; one approach, one num_envs and at most one
resolution (default 320x240), runs until the window is closed, writes no results; the printed timings include GUI
rendering and are not benchmark numbers):
    python scripts/benchmarking/tactile_sim_performance/benchmark_tacsl_tacex_taxim.py --debug_vis \
        --approaches tacex_taxim --num_envs 4 --resolutions 64x48
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

APPROACHES = ["native_tacsl", "tacex_taxim", "tacex_tacsl"]
LABELS = {
    "native_tacsl": "Isaac Lab TacSL",
    "tacex_taxim": "TacEx Taxim",
    "tacex_tacsl": "TacEx TacSL",
}
NATIVE_RESOLUTION = "640x480"  # native TacSL only runs at the calibration resolution
DEFAULT_RESOLUTIONS = ["32x24", "64x48", "128x96", "320x240", "640x480"]
REFERENCE_RESOLUTION = "320x240"  # TacEx preset; used for the num_envs plot and as the debug_vis default


def parse_resolution(res: str) -> tuple[int, int]:
    """'WxH' -> (W, H)."""
    try:
        w, h = (int(v) for v in res.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"resolution must be WxH, e.g. 64x48, got {res!r}")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"resolution must be positive, got {res!r}")
    return w, h


def _resolution_arg(res: str) -> str:
    w, h = parse_resolution(res)
    return f"{w}x{h}"


def label(r: dict) -> str:
    return f"{LABELS[r['approach']]} {r['resolution']}"


DEFAULT_SHAPES = [
    "obj_11_sphere.usd",
    "obj_02_hexagon.usd",
    "obj_07_cross_lines.usd",
    "obj_09_dots.usd",
    "obj_20_cone.usd",
    "obj_13_torus.usd",
    "obj_17_pacman.usd",
    "obj_18_wave1.usd",
]

# scene geometry (sensor frame: camera at camera_pos_offset looking along +z; gel surface height from the sensor cfg,
# see _gel_top_z)
PRESS_MAX_M = 0.0015
RETRACT_M = 0.002  # tip this far above the gel = no contact (beyond the far clipping plane)
PRESS_PERIOD_STEPS = 60
SENSOR_SPACING_M = 0.06
ENV_SPACING_M = 0.5
# tactile cameras get their own prim name: the sensor asset already has a placeholder "Camera" prim (from a USD
# reference, so it cannot be deleted)
TACTILE_CAMERA_NAME = "TactileCamera"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approaches", nargs="+", default=APPROACHES, choices=APPROACHES)
    parser.add_argument("--num_envs", nargs="+", type=int, default=[1, 4, 16, 32])
    parser.add_argument("--num_sensors", type=int, default=4, help="Sensors per env (one test shape each).")
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=_resolution_arg,
        default=None,
        help=f"Camera + tactile image resolutions (WxH) for the TacEx approaches. Default: {' '.join(DEFAULT_RESOLUTIONS)}"
        f" ({REFERENCE_RESOLUTION} with --debug_vis). Native TacSL always runs at {NATIVE_RESOLUTION}.",
    )
    parser.add_argument("--steps", type=int, default=300, help="Measured steps per run.")
    parser.add_argument("--warmup", type=int, default=30, help="Unmeasured steps before measuring.")
    parser.add_argument("--timeout", type=int, default=1200, help="Seconds per run before it is killed.")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument(
        "--debug_vis",
        action="store_true",
        help="Watch one run live in the Isaac Sim GUI (one approach, one num_envs). Writes no results.",
    )
    # worker mode (internal)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--approach", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--envs", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--resolution", type=_resolution_arg, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--baseline_vram_mib", type=float, default=0.0, help=argparse.SUPPRESS)
    return parser.parse_args()


"""
Worker: one (approach, num_envs) run in its own Isaac Sim process.
"""


def run_worker(args):
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(headless=not args.debug_vis, enable_cameras=True).app

    import numpy as np
    import psutil
    import pynvml
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sensors import TiledCameraCfg

    from tacex import GelSightSensor

    from tacex_assets import TACEX_ASSETS_DATA_DIR, TACEX_SENSORS_DATA_DIR
    from tacex_assets.sensors.gelsight_mini import GELSIGHT_MINI_TAXIM_CFG
    from tacex_assets.sensors.gelsight_mini.gsmini_tacsl import GELSIGHT_MINI_TACSL_CFG, GELSIGHT_MINI_TACSL_SIM_CFG

    native = args.approach == "native_tacsl"
    resolution = NATIVE_RESOLUTION if native else args.resolution
    res_w, res_h = parse_resolution(resolution)
    result = {
        "approach": args.approach,
        "resolution": resolution,
        "pixels": res_w * res_h,
        "num_envs": args.envs,
        "num_sensors": args.num_sensors,
        "status": "ok",
    }
    process = psutil.Process()
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

    def gpu_used_mib() -> float:
        return pynvml.nvmlDeviceGetMemoryInfo(gpu).used / 2**20

    try:
        t_init = time.perf_counter()
        gel_top_z = _gel_top_z([GELSIGHT_MINI_TAXIM_CFG, GELSIGHT_MINI_TACSL_CFG])
        dt = 1.0 / 60.0
        sim_utils.create_new_stage()
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device="cuda"))

        # -- envs in a grid
        num_envs, num_sensors = args.envs, args.num_sensors
        cols = int(np.ceil(np.sqrt(num_envs)))
        env_origins = torch.zeros(num_envs, 3)
        for e in range(num_envs):
            env_origins[e, 0], env_origins[e, 1] = (e % cols) * ENV_SPACING_M, (e // cols) * ENV_SPACING_M
            sim_utils.create_prim(f"/World/envs/env_{e}", "Xform", translation=tuple(env_origins[e].tolist()))

        # -- GelSight Mini assets (case + glass plate + gelpad, gel surface at gel_top_z), sensor frame = env frame
        # shifted along x. Static and without collisions: rendered by the tactile cameras like on a robot, but the
        # test shapes (pose set every step) pass into the gel without contact forces.
        sensor_asset = sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Sensors/GelSight_Mini/Sensor.usd",  # has a placeholder "Camera" prim
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        )
        for k in range(num_sensors):
            sensor_asset.func(
                f"/World/envs/env_.*/Sensor_{k}", sensor_asset, translation=(k * SENSOR_SPACING_M, 0.0, 0.0)
            )

        # -- one test shape per sensor, flipped (tip down onto the gel). Dynamic bodies without gravity whose pose is
        # set every step: kinematic test shapes crash PhysX's GPU pose write (illegal memory access, Isaac Sim 5.1)
        shape_files = [DEFAULT_SHAPES[k % len(DEFAULT_SHAPES)] for k in range(num_sensors)]
        shapes = []
        for k, shape_file in enumerate(shape_files):
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Shape_{k}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/tactile_test_shapes/{shape_file}",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(k * SENSOR_SPACING_M, 0.0, gel_top_z + RETRACT_M), rot=(0.0, 1.0, 0.0, 0.0)
                ),
            )
            shapes.append(RigidObject(cfg))

        # -- sensors
        sensors = []
        if native:
            from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensor, VisuoTactileSensorCfg

            cam = GELSIGHT_MINI_TACSL_CFG.sensor_camera_cfg
            render_cfg = GELSIGHT_MINI_TACSL_SIM_CFG.render_cfg
            # same scale and pinhole camera as TacEx; far plane = gel surface, so the nominal depth is the gel
            render_cfg = render_cfg.replace(mm_per_pixel=cam.view_width_m * 1000 / render_cfg.image_width)
            gel_surface_dist = GELSIGHT_MINI_TACSL_CFG.get_gel_surface_to_camera_distance_m()
            for k in range(num_sensors):
                sensors.append(
                    VisuoTactileSensor(
                        VisuoTactileSensorCfg(
                            prim_path=f"/World/envs/env_.*/Sensor_{k}/tacsl",
                            render_cfg=render_cfg,
                            enable_camera_tactile=True,
                            enable_force_field=False,
                            tactile_array_size=(1, 1),  # unused without the force field
                            tactile_margin=0.0,
                            camera_cfg=TiledCameraCfg(
                                prim_path=f"/World/envs/env_.*/Sensor_{k}/{TACTILE_CAMERA_NAME}",
                                update_period=0,
                                width=render_cfg.image_width,
                                height=render_cfg.image_height,
                                data_types=["distance_to_image_plane"],
                                offset=TiledCameraCfg.OffsetCfg(
                                    pos=cam.camera_pos_offset, rot=cam.camera_rot_offset, convention="opengl"
                                ),
                                spawn=sim_utils.PinholeCameraCfg(
                                    focal_length=cam.focal_length_cm,
                                    focus_distance=gel_surface_dist,
                                    horizontal_aperture=cam.view_width_m * cam.focal_length_cm / gel_surface_dist,
                                    vertical_aperture=cam.view_height_m * cam.focal_length_cm / gel_surface_dist,
                                    clipping_range=(cam.clipping_range_m[0], gel_surface_dist),
                                ),
                                depth_clipping_behavior="max",
                            ),
                        )
                    )
                )
        else:
            base = {"tacex_taxim": GELSIGHT_MINI_TAXIM_CFG, "tacex_tacsl": GELSIGHT_MINI_TACSL_CFG}[args.approach]
            base = base.replace(
                sensor_camera_cfg=base.sensor_camera_cfg.replace(
                    resolution=(res_w, res_h), prim_name=TACTILE_CAMERA_NAME
                ),
                optical_sim_cfg=base.optical_sim_cfg.replace(tactile_img_res=(res_w, res_h)),
            )
            for k in range(num_sensors):
                sensors.append(
                    GelSightSensor(
                        base.replace(
                            prim_path=f"/World/envs/env_.*/Sensor_{k}", data_types=["tactile_rgb"], debug_vis=False
                        )
                    )
                )

        if args.debug_vis:
            light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
            light.func("/World/DebugVisLight", light)

        sim.reset()

        if args.debug_vis:
            row_center_x = (num_sensors - 1) * SENSOR_SPACING_M / 2
            sim.set_camera_view(
                eye=(row_center_x, -0.15, gel_top_z + 0.09), target=(row_center_x, 0.0, gel_top_z + 0.01)
            )

        if native:
            # TacEx calibration: its matching no-contact frame is in the Taxim dataPack (0.png has another layout)
            f0 = np.load(
                f"{TACEX_SENSORS_DATA_DIR}/GelSight_Mini/calibs/taxim/640x480/dataPack.npz", allow_pickle=True
            )["f0"]
            bg = torch.tensor(np.ascontiguousarray(f0[..., ::-1]), device="cuda")
            for s in sensors:
                s._tactile_rgb_render.background_tensor = bg

        def shape_poses(press_m: float):
            poses = []
            for k in range(num_sensors):
                pos = env_origins.clone()
                pos[:, 0] += k * SENSOR_SPACING_M
                pos[:, 2] = gel_top_z - press_m
                quat = torch.tensor([0.0, 1.0, 0.0, 0.0]).repeat(num_envs, 1)
                poses.append(torch.cat([pos, quat], dim=-1).to("cuda"))
            return poses

        zero_vel = torch.zeros(num_envs, 6, device="cuda")

        def set_press(press_m: float):
            for shape, pose in zip(shapes, shape_poses(press_m)):
                shape.write_root_pose_to_sim(pose)
                shape.write_root_velocity_to_sim(zero_vel)

        def update_sensors():
            for s in sensors:
                s.update(dt, force_recompute=True)

        # -- warmup without contact. Native TacSL must capture its nominal (no-contact) depth before its first update.
        set_press(-RETRACT_M)
        for _ in range(args.warmup):
            sim.step()
            if not native:
                update_sensors()
        if native:
            for s in sensors:
                s.get_initial_render()
            for _ in range(5):
                sim.step()
                update_sensors()
        torch.cuda.synchronize()
        result["init_s"] = time.perf_counter() - t_init

        if args.debug_vis:
            _debug_vis_loop(simulation_app, sim, sensors, native, shape_files, set_press, update_sensors)
            simulation_app.close()
            return

        # -- measurement
        step_ms, update_ms = [], []
        vram_peak, util_samples, rss_peak = gpu_used_mib(), [], process.memory_info().rss
        torch.cuda.reset_peak_memory_stats()
        cpu0, wall0 = process.cpu_times(), time.perf_counter()
        psutil.cpu_percent(interval=None)
        sample_step = min(PRESS_PERIOD_STEPS // 2, args.steps - 1)  # deepest press (or last step, if shorter)
        for i in range(args.steps):
            press = PRESS_MAX_M * 0.5 * (1 - np.cos(2 * np.pi * i / PRESS_PERIOD_STEPS))
            t0 = time.perf_counter()
            set_press(press)
            sim.step()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            update_sensors()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            step_ms.append((t1 - t0) * 1000)
            update_ms.append((t2 - t1) * 1000)
            if i % 10 == 0:
                vram_peak = max(vram_peak, gpu_used_mib())
                util_samples.append(pynvml.nvmlDeviceGetUtilizationRates(gpu).gpu)
                rss_peak = max(rss_peak, process.memory_info().rss)
            if i == sample_step:
                _save_sample(args, sensors, native)
        wall = time.perf_counter() - wall0
        cpu1 = process.cpu_times()

        step_ms, update_ms = np.array(step_ms), np.array(update_ms)
        total_ms = step_ms + update_ms
        images = num_envs * num_sensors
        result.update(
            step_ms_mean=float(step_ms.mean()),
            update_ms_mean=float(update_ms.mean()),
            update_ms_p95=float(np.percentile(update_ms, 95)),
            total_ms_mean=float(total_ms.mean()),
            images_per_s=float(images / (total_ms.mean() / 1000)),
            update_images_per_s=float(images / (update_ms.mean() / 1000)),
            vram_peak_mib=float(vram_peak),
            vram_above_idle_mib=float(vram_peak - args.baseline_vram_mib),
            torch_peak_mib=float(torch.cuda.max_memory_allocated() / 2**20),
            cpu_process_pct=float(((cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)) / wall * 100),
            cpu_system_pct=float(psutil.cpu_percent(interval=None)),
            ram_peak_mib=float(rss_peak / 2**20),
            gpu_util_pct=float(np.mean(util_samples)) if util_samples else None,
        )
    except Exception as e:  # report the failure instead of crashing without a result
        import traceback

        oom = "out of memory" in str(e).lower()
        result.update(status="oom" if oom else "error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc())

    Path(args.result).write_text(json.dumps(result, indent=2))
    # Exit right away: the result is written, and Kit's close() can hang after errors (the process exit frees the GPU).
    sys.stdout.flush()
    os._exit(0)


def _gel_top_z(sensor_cfgs: list) -> float:
    """Height of the undeformed gel surface in the sensor frame, from the sensor cfgs (all must agree, since every
    approach uses the same scene). Assumes the default camera orientation (looking along the sensor frame's +z)."""
    heights = {
        round(cfg.sensor_camera_cfg.camera_pos_offset[2] + cfg.get_gel_surface_to_camera_distance_m(), 9)
        for cfg in sensor_cfgs
    }
    if len(heights) != 1:
        raise ValueError(f"Sensor cfgs disagree on the gel surface height: {sorted(heights)}")
    return heights.pop()


def _env0_images(sensors, native) -> list:
    """Tactile RGB images of env 0, one per sensor, as uint8 HxWx3 arrays."""
    import numpy as np

    imgs = []
    for s in sensors:
        if native:
            img = s.data.tactile_rgb_image[0].float().cpu().numpy() / 255.0
        else:
            img = s.data.output["tactile_rgb"][0].float().cpu().numpy()
        imgs.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
    return imgs


def _save_sample(args, sensors, native):
    """Saves the tactile images of env 0 (all sensors) at the deepest press, to check the shapes really press."""
    import cv2
    import numpy as np

    imgs = [cv2.resize(img, (320, 240)) for img in _env0_images(sensors, native)]
    out = Path(args.result).parent.parent / "samples" / f"{Path(args.result).stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(np.concatenate(imgs, axis=1), cv2.COLOR_RGB2BGR))


def _debug_vis_loop(simulation_app, sim, sensors, native, shape_files, set_press, update_sensors):
    """Runs the press cycle until the window is closed and shows env 0's tactile images in an omni.ui window."""
    import cv2
    import numpy as np
    import omni.ui as ui
    import torch

    tile_w, tile_h = 200, 150
    window = ui.Window("Tactile images (env 0)", width=len(sensors) * (tile_w + 8) + 24, height=tile_h + 60)
    # dock as the active tab of the bottom panel (next to Content/Console), so it does not cover the viewport
    window.deferred_dock_in("Content", ui.DockPolicy.TARGET_WINDOW_IS_ACTIVE)
    providers = []
    with window.frame:
        with ui.HStack(spacing=8):
            for name in shape_files:
                with ui.VStack(width=tile_w):
                    ui.Label(Path(name).stem, height=18)
                    providers.append(ui.ByteImageProvider())
                    ui.ImageWithProvider(providers[-1], width=tile_w, height=tile_h)

    print("[debug_vis] running until the window is closed; timings include GUI rendering, not benchmark numbers")
    i, update_ms = 0, []
    while simulation_app.is_running():
        press = PRESS_MAX_M * 0.5 * (1 - np.cos(2 * np.pi * i / PRESS_PERIOD_STEPS))
        set_press(press)
        sim.step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        update_sensors()
        torch.cuda.synchronize()
        update_ms.append((time.perf_counter() - t0) * 1000)
        for provider, img in zip(providers, _env0_images(sensors, native)):
            rgba = cv2.cvtColor(cv2.resize(img, (tile_w, tile_h)), cv2.COLOR_RGB2RGBA)
            provider.set_bytes_data(rgba.flatten().data, [tile_w, tile_h])
        i += 1
        if i % PRESS_PERIOD_STEPS == 0:
            print(f"[debug_vis] step {i}: sensor update {np.mean(update_ms):.1f} ms (GUI, not representative)")
            update_ms = []


"""
Driver: runs every configuration in its own process, then writes CSV/JSON/summary/plot.
"""


def run_driver(args):
    if args.debug_vis:
        if len(args.approaches) != 1 or len(args.num_envs) != 1 or len(args.resolutions or []) > 1:
            sys.exit(
                "--debug_vis watches one run: pass exactly one --approaches and one --num_envs value, and at most one"
                " --resolutions value."
            )
        args.approach, args.envs = args.approaches[0], args.num_envs[0]
        args.resolution = (args.resolutions or [REFERENCE_RESOLUTION])[0]
        run_worker(args)
        return
    if args.resolutions is None:
        args.resolutions = list(DEFAULT_RESOLUTIONS)

    import pynvml

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (
            Path(__file__).resolve().parent
            / "logs"
            / "tacsl_vs_tacex_taxim"
            / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
    )
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)

    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    system = _system_info(pynvml, gpu)

    # (approach, resolution) configs: native TacSL once at its fixed resolution, TacEx approaches at every resolution
    configs = [
        (approach, res)
        for approach in args.approaches
        for res in ([NATIVE_RESOLUTION] if approach == "native_tacsl" else args.resolutions)
    ]
    print(f"{len(configs) * len(args.num_envs)} runs ({len(configs)} configs x {len(args.num_envs)} num_envs)")

    results = []
    for num_envs in args.num_envs:
        for approach, res in configs:
            baseline = pynvml.nvmlDeviceGetMemoryInfo(gpu).used / 2**20
            run_name = f"{approach}_{res}_envs{num_envs}"
            result_file = out_dir / "runs" / f"{run_name}.json"
            log_file = out_dir / "runs" / f"{run_name}.log"
            cmd = [
                sys.executable, os.path.abspath(__file__), "--worker", "--approach", approach, "--resolution", res,
                "--envs", str(num_envs), "--num_sensors", str(args.num_sensors), "--steps", str(args.steps),
                "--warmup", str(args.warmup), "--result", str(result_file), "--baseline_vram_mib", str(baseline),
            ]  # fmt: skip
            print(f"[{approach:>12} {res:>7} | envs {num_envs:>3}] running ...", end=" ", flush=True)
            t0 = time.perf_counter()
            with open(log_file, "w") as log:
                try:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    pass
            if result_file.exists():
                result = json.loads(result_file.read_text())
            else:  # crashed or timed out before writing a result
                w, h = parse_resolution(res)
                result = {"approach": approach, "resolution": res, "pixels": w * h, "num_envs": num_envs,
                          "num_sensors": args.num_sensors, "status": "timeout/crash",
                          "error": f"no result, see {log_file.name}"}  # fmt: skip
            result["wall_s"] = time.perf_counter() - t0
            results.append(result)
            if result["status"] == "ok":
                print(f"{result['total_ms_mean']:7.1f} ms/step, {result['images_per_s']:7.0f} img/s, "
                      f"VRAM +{result['vram_above_idle_mib'] / 1024:.2f} GiB, CPU {result['cpu_process_pct']:.0f} %")  # fmt: skip
            else:
                print(f"{result['status']}: {result.get('error', '')[:120]}")

    (out_dir / "results.json").write_text(
        json.dumps({"system": system, "args": vars(args), "results": results}, indent=2)
    )
    _write_csv(out_dir / "results.csv", results)
    _write_summary(out_dir / "summary.md", system, args, results)
    _plot_vs_envs(out_dir / "benchmark.png", args, results)
    _plot_vs_resolution(out_dir / "resolution.png", args, results)
    print(f"\nResults: {out_dir}")


def _system_info(pynvml, gpu) -> dict:
    import platform

    import psutil

    return {
        "gpu": pynvml.nvmlDeviceGetName(gpu),
        "gpu_memory_gib": pynvml.nvmlDeviceGetMemoryInfo(gpu).total / 2**30,
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_gib": psutil.virtual_memory().total / 2**30,
    }


COLUMNS = [
    "approach", "resolution", "pixels", "num_envs", "num_sensors", "status", "step_ms_mean", "update_ms_mean", "update_ms_p95",
    "total_ms_mean", "images_per_s", "update_images_per_s", "vram_above_idle_mib", "vram_peak_mib",
    "torch_peak_mib", "cpu_process_pct", "cpu_system_pct", "ram_peak_mib", "gpu_util_pct", "init_s", "error",
]  # fmt: skip


def _write_csv(path: Path, results: list[dict]):
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _write_summary(path: Path, system: dict, args, results: list[dict]):
    lines = [
        "# TacSL vs. Taxim benchmark",
        "",
        f"GPU: {system['gpu']} ({system['gpu_memory_gib']:.1f} GiB, driver {system['driver']}), "
        f"CPU: {system['cpu_cores_physical']} cores / {system['cpu_cores_logical']} threads, RAM {system['ram_gib']:.0f} GiB",
        f"{args.num_sensors} sensors per env, {args.steps} measured steps after {args.warmup} warmup steps.",
        "",
        "Times are per simulation step. *Step* = shape motion + physics + rendering (all cameras); *update* = all "
        "sensors' tactile pipelines. Throughput = tactile images (envs x sensors) per second of step + update. "
        "VRAM = device-wide peak above the idle GPU before the run (includes Isaac Sim itself). CPU = process CPU "
        "time / wall time (100 % = one core).",
        "",
        "| approach | resolution | envs | status | step [ms] | update [ms] | total [ms] | img/s | VRAM +[GiB] "
        "| torch peak [MiB] | CPU [%] | RAM [GiB] | GPU util [%] |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(results, key=lambda r: (r["approach"], r["pixels"], r["num_envs"])):
        head = f"| {LABELS[r['approach']]} | {r['resolution']} | {r['num_envs']} |"
        if r["status"] != "ok":
            lines.append(f"{head} {r['status']} |" + " – |" * 9)
            continue
        lines.append(
            f"{head} ok | {r['step_ms_mean']:.1f} | {r['update_ms_mean']:.1f} "
            f"| {r['total_ms_mean']:.1f} | {r['images_per_s']:.0f} | {r['vram_above_idle_mib'] / 1024:.2f} "
            f"| {r['torch_peak_mib']:.0f} | {r['cpu_process_pct']:.0f} | {r['ram_peak_mib'] / 1024:.1f} "
            f"| {r['gpu_util_pct']:.0f} |"
        )
    failed = [r for r in results if r["status"] != "ok"]
    if failed:
        lines += ["", "## Failed runs", ""]
        lines += [f"- {label(r)}, {r['num_envs']} envs: {r.get('error', '')[:300]}" for r in failed]
    path.write_text("\n".join(lines) + "\n")


# dataviz reference palette (categorical slots 1-3), chart chrome tokens
COLORS = {"native_tacsl": "#2a78d6", "tacex_taxim": "#eb6834", "tacex_tacsl": "#1baf7a"}
MARKERS = {"native_tacsl": "o", "tacex_taxim": "s", "tacex_tacsl": "^"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#6f6d68", "#e1e0d9", "#fcfcfb"
PANELS = [
    ("update_ms_mean", "Sensor update per step [ms]", 1.0),
    ("images_per_s", "Throughput [tactile images / s]", 1.0),
    ("vram_above_idle_mib", "Peak VRAM above idle [GiB]", 1 / 1024),
    ("cpu_process_pct", "Process CPU [% of one core]", 1.0),
]


def _metric_figure(path: Path, title: str, xlabel: str, xticks: list, xticklabels: list[str], series: list[dict]):
    """2x2 metric panels. ``series``: dicts with approach, label, rows (ok results), x (row -> x), failed (results)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, panel_title, scale) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        for sr in series:
            color = COLORS[sr["approach"]]
            if sr["rows"]:
                x = [sr["x"](r) for r in sr["rows"]]
                y = [r[key] * scale for r in sr["rows"]]
                ax.plot(x, y, color=color, marker=MARKERS[sr["approach"]], ms=7, lw=2, label=sr["label"])
                ax.annotate(f"{y[-1]:.3g}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                            fontsize=8, color=INK)  # fmt: skip
            # failed runs (e.g. out of memory): mark at the top of the axis
            for r in sr["failed"]:
                ax.annotate(r["status"], (sr["x"](r), 1.0), xycoords=("data", "axes fraction"), ha="center",
                            va="top", fontsize=7, color=color)  # fmt: skip
        ax.set_xscale("log", base=2)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.minorticks_off()
        ax.set_xlabel(xlabel, color=MUTED)
        ax.set_title(panel_title, loc="left", fontsize=11, color=INK)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side, spine in ax.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_color("#c3c2b7")
        ax.set_ylim(bottom=0)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, 0.995))  # fmt: skip
    fig.suptitle(title, x=0.01, ha="left", y=0.955, fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _plot_vs_envs(path: Path, args, results: list[dict]):
    """Metrics vs. num_envs: native TacSL, and each TacEx approach at the reference resolution (or the largest tested
    one, if the reference resolution was not tested)."""
    tacex_res = REFERENCE_RESOLUTION if REFERENCE_RESOLUTION in args.resolutions else max(
        args.resolutions, key=lambda r: parse_resolution(r)[0] * parse_resolution(r)[1]
    )  # fmt: skip
    series = []
    for approach in args.approaches:
        res = NATIVE_RESOLUTION if approach == "native_tacsl" else tacex_res
        runs = [r for r in results if r["approach"] == approach and r["resolution"] == res]
        series.append({
            "approach": approach,
            "label": f"{LABELS[approach]} {res}",
            "rows": sorted((r for r in runs if r["status"] == "ok"), key=lambda r: r["num_envs"]),
            "failed": [r for r in runs if r["status"] != "ok"],
            "x": lambda r: r["num_envs"],
        })  # fmt: skip
    _metric_figure(path, f"Tactile simulation benchmark ({args.num_sensors} sensors per env)", "num_envs",
                   args.num_envs, [str(n) for n in args.num_envs], series)  # fmt: skip


def _plot_vs_resolution(path: Path, args, results: list[dict]):
    """Metrics vs. resolution (x = pixel count, log scale) at the largest num_envs. Native TacSL is a single point at
    its fixed resolution."""
    num_envs = max(args.num_envs)
    resolutions = sorted(
        set(args.resolutions) | ({NATIVE_RESOLUTION} if "native_tacsl" in args.approaches else set()),
        key=lambda r: parse_resolution(r)[0] * parse_resolution(r)[1],
    )
    pixels = [parse_resolution(r)[0] * parse_resolution(r)[1] for r in resolutions]
    series = []
    for approach in args.approaches:
        runs = [r for r in results if r["approach"] == approach and r["num_envs"] == num_envs]
        series.append({
            "approach": approach,
            "label": LABELS[approach] + (f" ({NATIVE_RESOLUTION} only)" if approach == "native_tacsl" else ""),
            "rows": sorted((r for r in runs if r["status"] == "ok"), key=lambda r: r["pixels"]),
            "failed": [r for r in runs if r["status"] != "ok"],
            "x": lambda r: r["pixels"],
        })  # fmt: skip
    _metric_figure(path, f"Tactile simulation vs. resolution ({num_envs} envs x {args.num_sensors} sensors)",
                   "camera + tactile image resolution", pixels, resolutions, series)  # fmt: skip


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.worker:
        run_worker(cli_args)
    else:
        run_driver(cli_args)
