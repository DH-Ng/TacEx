from dataclasses import dataclass, MISSING
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple, Union, Literal
from isaaclab.utils import class_to_dict, to_camel_case, configclass
from isaaclab.sensors import SensorBaseCfg, TiledCameraCfg

from .simulation_approaches.optical_sim import OpticalSimulator
from .simulation_approaches.marker_motion_sim import MarkerMotionSimulator

from .gelsight_sensor import GelSightSensor

"""Configuration for a Gelsight tactile sensor."""
@configclass
class GelSightSensorCfg(SensorBaseCfg):
    class_type: type = GelSightSensor

    @configclass
    class Dimensions:
        """Dimensions here are in mm (we assume that the world units are meters)"""
        width: float = 0.0,
        length: float = 0.0,
        height: float = 0.0
    case_dimensions: Dimensions = Dimensions()
    gelpad_dimensions: Dimensions = Dimensions()
 
    @configclass
    class SensorCameraCfg:
        """Configs for the Camera of the GelSight sensor."""
        prim_path_appendix: str = "/Camera",
        update_period: float = 0,
        resolution: Tuple[int] = (160, 120),
        data_types: list[str] = ["depth"],
        clipping_range: Tuple[float] = (0,1),
    sensor_camera_cfg: SensorCameraCfg = SensorCameraCfg()

    data_types: list[str] = ["height_map", "tactile_rgb", "marker_motion"]

    optical_sim_cfg: None = None # freely able to choose what class, but this cfg is also just optionally
    """Cfg class of the optical simulator you want to use."""

    marker_motion_sim_cfg: None = None
    """Cfg class of the marker motion simulator you want to use."""

    compute_indentation_depth_class: Literal["optical_sim", "marker_motion_sim"] = "optical_sim"
    """What class to use for computing the indentation depth.
    
    Maybe you want to use the method via your optical simulation (e.g. Taxim), or use one from your marker motion simulation (e.g. if its FEM based).
    """

    device: str = "cuda"