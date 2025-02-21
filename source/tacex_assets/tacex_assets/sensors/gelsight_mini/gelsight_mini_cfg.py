from dataclasses import dataclass, MISSING
from typing import Any, Dict, Sequence, Tuple, Union, List
from isaaclab.utils import class_to_dict, to_camel_case, configclass
from isaaclab.sensors import SensorBaseCfg, TiledCameraCfg

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex import GelSightSensor, GelSightSensorCfg
from tacex.simulation_approaches.gpu_taxim import TaximSimulator, TaximSimulatorCfg

"""Configuration for the gelsight mini sensor."""
@configclass
class GelSightMiniCfg(GelSightSensorCfg):
    class_type: type = GelSightSensor

    case_dimensions: GelSightSensorCfg.Dimensions = GelSightSensorCfg.Dimensions(
        width=32/1000,
        length=28/1000,
        height=24/1000
    )

    gelpad_dimensions: GelSightSensorCfg.Dimensions = GelSightSensorCfg.Dimensions(
        width=20.75/1000,
        length=25.25/1000,
        height=4.5/1000
    )

    sensor_camera_cfg: GelSightSensorCfg.SensorCameraCfg = GelSightSensorCfg.SensorCameraCfg(
        prim_path_appendix = "/Camera",
        update_period= 0,
        resolution = (160, 120),
        data_types = ["depth"],
        clipping_range = (0.024, 0.034),
    )

    tactile_img_res: Tuple = (480, 640)

    update_period: float = 0.0
    
    data_types: List[str] = ["tactile_rgb", "height_map"]

    optical_sim_cfg = TaximSimulatorCfg(
        calib_folder_path= f"{TACEX_ASSETS_DATA_DIR}/Sensors/GelSight_Mini/calibs/480x640",
        gelpad_height= gelpad_dimensions.height,
        gelpad_to_camera_min_distance= 0.024,
        tactile_img_res= tactile_img_res,
    )

    compute_indentation_depth_class = "optical_sim"

    device: str = "cuda" # use gpu per default #TODO currently gpu mandatory, also enable cpu only usage?    


