import torch

from dataclasses import dataclass, MISSING
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple, Union, List
from isaaclab.utils import class_to_dict, to_camel_case, configclass


from .taxim_sim import TaximSimulator

"""Configuration for a tactile RGB simulation with Taxim."""
@configclass
class TaximSimulatorCfg():
    class_type: type = TaximSimulator

    calib_folder_path: str = ""

    device: str = "cuda"
    num_envs: int = 0

    with_shadow: bool = False
    tactile_img_res: tuple = (240, 320)
    """Resolution of the Tactile Image.
    
    Can be different from the Sensor Camera.
    If this is the case, then height map from camera is going to be up/down sampled.
    """

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
