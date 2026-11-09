from tacex.simulation_approaches.pix2pix import Pix2PixSimulatorCfg
from tacex_assets import TACEX_SENSORS_DATA_DIR

from .generic_gsmini_cfg import GeneralGelSightMiniCfg

"""Configuration for simulating the GelSight Mini via pix2pix network."""

GELSIGHT_MINI_PIX2PIX_CFG = GeneralGelSightMiniCfg()
GELSIGHT_MINI_PIX2PIX_CFG = GELSIGHT_MINI_PIX2PIX_CFG.replace(
    sensor_camera_cfg=GELSIGHT_MINI_PIX2PIX_CFG.SensorCameraCfg(
        prim_name="Camera",
        update_period=0,
        resolution=(320, 240),
        data_types=["depth"],
        clipping_range_m=(0.024, 0.034),
    ),
    update_period=0.01,
    data_types=["tactile_rgb", "height_map"],
    optical_sim_cfg=Pix2PixSimulatorCfg(calib_folder_path=f"{TACEX_SENSORS_DATA_DIR}/GelSight_Mini/calibs/pix2pix"),
)
