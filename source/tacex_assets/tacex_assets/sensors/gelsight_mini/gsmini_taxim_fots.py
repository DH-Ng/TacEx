from dataclasses import MISSING

from tacex import GelSightSensorCfg
from tacex.simulation_approaches.fots import FOTSMarkerSimulatorCfg
from tacex.simulation_approaches.gpu_taxim import TaximSimulatorCfg

from tacex_assets import TACEX_SENSORS_DATA_DIR

from .generic_gsmini_cfg import GeneralGelSightMiniCfg

"""Configuration for simulating the GelSight Mini via GPU-Taxim and FOTS."""

GELSIGHT_MINI_TAXIM_FOTS_CFG = GeneralGelSightMiniCfg()
GELSIGHT_MINI_TAXIM_FOTS_CFG = GELSIGHT_MINI_TAXIM_FOTS_CFG.replace(
    sensor_camera_cfg=GelSightSensorCfg.SensorCameraCfg(
        prim_name="Camera",
        update_period=0,
        resolution=(32, 24),
        data_types=["depth"],
        clipping_range_m=(0.024, 0.029),
    ),
    update_period=0.01,
    data_types=["tactile_rgb", "marker_motion", "height_map"],
    optical_sim_cfg=TaximSimulatorCfg(
        calib_folder_path=f"{TACEX_SENSORS_DATA_DIR}/GelSight_Mini/calibs/taxim/640x480",
        gelpad_height=GELSIGHT_MINI_TAXIM_FOTS_CFG.gelpad_dimensions.height,
        gelpad_to_camera_min_distance=0.024,
        with_shadow=False,
        tactile_img_res=(320, 240),
        device="cuda",
    ),
    marker_motion_sim_cfg=FOTSMarkerSimulatorCfg(
        lamb=[0.00125, 0.00021, 0.00038],
        pyramid_kernel_size=[51, 21, 11, 5],  # [11, 11, 11, 11, 11, 5],
        kernel_size=5,
        marker_params=FOTSMarkerSimulatorCfg.MarkerParams(
            num_markers_col=11,
            num_markers_row=9,
            x0=15,
            y0=26,
            dx=26,
            dy=29,
        ),
        tactile_img_res=(320, 240),
        device="cuda",
        frame_transformer_cfg=MISSING,
    ),
    compute_indentation_depth_class="optical_sim",
    device="cuda",  # use gpu per default #TODO currently gpu mandatory, also enable cpu only usage?
)
