from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple, Union

import torch
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import class_to_dict, configclass, to_camel_case

from ..gelsight_simulator_cfg import GelSightSimulatorCfg
from .mani_skill_sim import ManiSkillSimulator

"""Configuration for marker motion simulation via FEM simulation (like ManiSkill-ViTac used)."""
@configclass
class ManiSkillSimulatorCfg(GelSightSimulatorCfg):
    simulation_approach_class: type = ManiSkillSimulator

    calib_folder_path: str = ""

    device: str = "cpu"

    marker_interval_range: Tuple[float, float] = (2.0625, 2.0625)

    marker_rotation_range: float = 0.0

    marker_translation_range: Tuple[float, float] = (0.0, 0.0)

    marker_pos_shift_range: Tuple[float, float] = (0.0, 0.0)

    marker_random_noise: float = 0.0
    
    marker_lose_tracking_probability: float = 0.0

    normalize: bool = False

    marker_flow_size: int = 128
    
    camera_params: Tuple[float, float, float, float, float] = (
        340,
        325,
        160,
        125,
        0.0,
    )

    tactile_img_res: tuple = (240, 320)
    """Resolution of the Tactile Image.

    Can be different from the Sensor Camera.
    If this is the case, then height map from camera is going to be up/down sampled.
    """

    @configclass
    class MarkerParams:
        """Dimensions here are in mm (we assume that the world units are meters)"""
        num_markers_col: int = 63
        num_markers_row: int = 63
        x0: float = 0
        y0: float = 0
        dx: float = 0
        dy: float = 0
    marker_params: MarkerParams = MarkerParams()

    init_marker_pos: tuple = ([[]],[[]])
    """Intial Marker positions.

    Tuple (xx_init pos, yy_init pos):
    - xx_init = inital position of each marker along the "height" of the tactile img (top-down)
        -> for each marker the initial x pos. Shape: (num_markers_row, num_marker_column)
    - yy_init = inital position of each marker along the "width" of the tactile img (left-right)
        -> for each marker the initial y pos. Shape: (num_markers_row, num_marker_column)
    """
