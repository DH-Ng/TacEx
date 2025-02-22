import torch

from dataclasses import dataclass, MISSING
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple, Union, List
from isaaclab.utils import class_to_dict, to_camel_case, configclass


from .fots_marker_sim import FOTSMarkerSimulator

"""Configuration for a tactile RGB simulation with Taxim."""
@configclass
class FOTSMarkerSimulatorCfg():
    class_type: type = FOTSMarkerSimulator

    calib_folder_path: str = ""

    device: str = "cuda"
    num_envs: int = 0

    with_shadow: bool = False
    tactile_img_res: tuple = (320, 240)
    """Resolution of the Tactile Image.
    
    Can be different from the Sensor Camera.
    If this is the case, then height map from camera is going to be up/down sampled.
    """

    lamb: list[float] = []
    """Parameters for exponential functions used by FOTS for marker simulation"""
    
    # experimental params
    ball_radius = 4.70/2 # mm
    mm_to_pixel = 19.58 # units = pix/mm

    # optical simulation params
    pyramid_kernel_size: list[int] = []
    kernel_size: int = 0

    @configclass
    class MarkerParams:
        """Dimensions here are in mm (we assume that the world units are meters)"""
        height_num_markers: int = 0,
        widht_num_markers: int = 0,
        x0: float = 0,
        y0: float = 0,
        dx: float = 0,
        dy: float = 0

    marker_params: MarkerParams = MarkerParams()


    gelpad_height: float = MISSING
    """Used for computing indentation depth from height map"""

    #### Camera Data
    gelpad_to_camera_min_distance: float = MISSING
    """Min distance of camera to the gelpad. 
    Used for computing the indentation depth out of the 
    camera height map.
    """

    height_map: torch.Tensor = None
    """Reference to buffer of the GelSight sensor which contains height map data"""
