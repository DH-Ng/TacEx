from isaaclab_contrib.sensors.tacsl_sensor import GelSightRenderCfg

from tacex import GelSightSensorCfg
from tacex.simulation_approaches.tacsl import TacSLSimulatorCfg

from tacex_assets import TACEX_SENSORS_DATA_DIR

from .generic_gsmini_cfg import GeneralGelSightMiniCfg

"""Configuration for simulating the GelSight Mini via TacSL (Isaac Lab's visuo-tactile sensor).

Uses TacEx's GelSight Mini Taxim calibration (same polycalib format as TacSL). The calibration was made at
640x480, so TacSL renders at that resolution; the output is resized to ``tactile_img_res``.

The preset only enables the optical output (``tactile_rgb``), because the force field depends on the scene. To add
it, set the gelpad and contact object and use the same simulator cfg for the force field slot::

    tacsl_cfg = GELSIGHT_MINI_TACSL_SIM_CFG.replace(
        force_field=GELSIGHT_MINI_TACSL_SIM_CFG.force_field.replace(
            elastomer_prim_path_expr="{ENV_REGEX_NS}/Robot/gelsight_mini_gelpad",
            contact_object_prim_path_expr="{ENV_REGEX_NS}/object",  # needs an SDF collision mesh
            contact_normal_local=(0.0, 0.0, -1.0),  # outward normal of the gel surface in the gelpad frame
        )
    )
    sensor_cfg = GELSIGHT_MINI_TACSL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot/gelsight_mini_case",
        optical_sim_cfg=tacsl_cfg,
        force_field_sim_cfg=tacsl_cfg,
        data_types=["tactile_rgb", "height_map", "force_field"],
    )
"""

GELSIGHT_MINI_TACSL_SIM_CFG = TacSLSimulatorCfg(
    render_cfg=GelSightRenderCfg(
        base_data_path=f"{TACEX_SENSORS_DATA_DIR}/GelSight_Mini/calibs/taxim",
        sensor_data_dir_name="640x480",
        background_path="0.png",  # loadable placeholder; replaced by the Taxim background below
        calib_path="polycalib.npz",
        image_width=640,
        image_height=480,
        num_bins=125,
        mm_per_pixel=0.0295,  # calibration value; by default replaced by the sensor view scale
    ),
    # the no-contact frame the calibration belongs to (0.png is a different capture, transposed)
    background_override_path=f"{TACEX_SENSORS_DATA_DIR}/GelSight_Mini/calibs/taxim/640x480/dataPack.npz",
    tactile_img_res=(320, 240),
    device="cuda",
)

GELSIGHT_MINI_TACSL_CFG = GeneralGelSightMiniCfg().replace(
    sensor_camera_cfg=GelSightSensorCfg.SensorCameraCfg(
        prim_name="Camera",
        update_period=0,
        resolution=(320, 240),
        data_types=["depth"],
        clipping_range_m=(0.024, 0.029),
    ),
    update_period=0.01,
    data_types=["tactile_rgb", "height_map"],
    optical_sim_cfg=GELSIGHT_MINI_TACSL_SIM_CFG,
    marker_motion_sim_cfg=None,
    force_field_sim_cfg=None,
    compute_indentation_depth_class="default",
    device="cuda",
)
