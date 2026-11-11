from dataclasses import MISSING

from isaaclab.utils import configclass

from ..gelsight_simulator_cfg import GelSightSimulatorCfg
from .taxim_sim import TaximSimulator

"""Configuration for a tactile RGB simulation with Taxim."""


@configclass
class TaximSimulatorCfg(GelSightSimulatorCfg):
    """Config for Taxim simulation approach."""

    simulation_approach_class: type = TaximSimulator

    calib_folder_path: str = ""

    device: str = "cuda"

    with_shadow: bool = False

    tactile_img_res: tuple = (320, 240)
    """Resolution (width, height) of the Tactile Image.

    Can be different from the Sensor Camera.
    If this is the case, then height map from camera is up/down sampled.
    """

    gelpad_height: float = 0.0045
    """Used for computing indentation depth from height map"""

    # Asset Data
    gelpad_to_camera_min_distance: float = 0.024
    """Unused (only in commented-out code). The camera-to-gelpad distance is set by
    ``GelSightSensorCfg.gelpad_underside_to_camera_distance_m``."""

    taxim_parameters: dict | None = None
    """Parameters for the Taxim Simulation.

    If `None` then the parameters defined params.json inside the calib_folder_path are used.
    Otherwise, the parameters are overwritten with the values from the given dict.
    An example for the dict:
    {   
        "simulator": {
            "initial_frame_sigma_rel": initial_frame_sigma_rel,
            "diff_threshold": diff_threshold,
            "frame_mixing_percentage": frame_mixing_percentage,
            "contact_scale": contact_scale,
            "deform_final_sigma_rel": deform_final_sigma_rel,
            "deform_pyramid_sigma_rel": deform_pyramid_sigma_rel,
            "shadow_step_rel": shadow_step_rel,
            "shadow_blur_sigma_rel": shadow_blur_sigma_rel,
            "shadow_attachment_kernel_size_rel": shadow_attachment_kernel_size_rel,
            "discretize_precision": discretize_precision,
            "fan_angle": fan_angle,
            "fan_precision": fan_precision,
            "height_precision": height_precision,     
        },
        "sensor": {
            "pixmm": 0.0295,
            "num_bins": 125,
            "w": 640,
            "h": 480
        }
    }
    """
