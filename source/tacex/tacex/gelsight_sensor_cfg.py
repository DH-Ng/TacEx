from typing import Literal

from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils import configclass

from .gelsight_sensor import GelSightSensor
from .simulation_approaches.gelsight_simulator_cfg import GelSightSimulatorCfg

"""Configuration for a Gelsight tactile sensor."""


@configclass
class GelSightSensorCfg(SensorBaseCfg):
    class_type: type = GelSightSensor

    @configclass
    class Dimensions:
        """Dimensions here are in mm (we assume that the world units are meters)"""

        width: float = 0.0
        length: float = 0.0
        height: float = 0.0

        @property
        def as_tuple(self) -> tuple[float, float, float]:
            return self.width, self.length, self.height

    case_dimensions: Dimensions = Dimensions()
    gelpad_dimensions: Dimensions = Dimensions()

    max_indentation_depth: float = 0.0
    """The maximum indentation depth in [m].
    
    Used e.g. for normalizing height map w.r.t to possible indentation depth range.
    """

    @configclass
    class SensorCameraCfg:
        """Configs for the Camera of the GelSight sensor."""

        prim_name: str = "Camera"
        update_period: float = 0
        resolution: tuple[int, int] = (32, 24)
        data_types: list[str] = ["depth"]
        clipping_range_m: tuple[float, float] = (0.024, 0.028)
        update_latest_camera_pose: bool = False

        camera_pos_offset: tuple[float, float, float] = (0.0000, -0.0001, 0.0)
        """Translation of camera w.r.t to sensor frame. 
        
        Even very small changes can lead to relativly big differences in the indentation location.

        Defaults to (0.0000, -0.0001, 0.0) for the GelSight Mini. This offset places the camera in the center of the gelpad frame.
        """

        camera_rot_offset: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (0.0, 1.0, 0.0, 0.0)."""

        mm_per_pixel: float = 0.0634
        """Scale of the real tactile RGB image in mm/pixel, measured on an image with resolution ``reference_img_res``.

        Together with ``reference_img_res`` this defines the physical area the sensor camera sees
        (see ``view_width_m``/``view_height_m``).

        Defaults to 0.0634 - the GelSight Mini value for the 320x240 image after cropping 15% from each border
        (gsrobotics README, measured with a ruler; same value for both axes).

        To measure it yourself, press an object with known dimensions (e.g. a ruler with fine engraved marks or a
        precise grid) onto the gel, capture the image with the same crop/resize as used for the real sensor and divide a
        large known distance (in mm) by its length in pixels. Measure along both axes and near the image center.
        """

        reference_img_res: tuple[int, int] = (320, 240)
        """Resolution (width, height) of the real tactile image for which ``mm_per_pixel`` was measured.

        Independent of the simulated camera ``resolution``: the physical view stays the same,
        whatever resolution is rendered.
        """

        focal_length_cm: float = 1.0
        """Focal length of the sensor camera (in centimeters).
        
        Determines the "zoom" in the image, i.e. how big objects appear. 
        Value should be set so that focus_distance and focal_length combination
        leads to appropriate indentation area of known indenters.
 
        You can use the `taxim_shape_touch.py` script to try out different values.

        Note: This value is not used to compute horizontal/vertical aperture.
        """

        focus_distance_m: float = 0.024 + 0.004
        """Distance from camera to focus plane (in meters). 
        
        Determines where the FOV of the sensor camera is.
        Should be at highest point of the gelpad (e.g. for GsMini thats 2.5cm due to sensor case height + gelpad height 4mm)"""

        @property
        def view_width_m(self) -> float:
            """Physical width of the camera view at ``focus_distance_m`` (in meters).

            Computed as ``reference_img_res[0] * mm_per_pixel``, i.e. the area of the real (already border-cropped)
            tactile image. No border cropping is applied in simulation: the camera renders the cropped view directly.
            """
            return self.reference_img_res[0] * self.mm_per_pixel / 1000.0

        @property
        def view_height_m(self) -> float:
            """Physical height of the camera view at ``focus_distance_m`` (in meters).

            Computed as ``reference_img_res[1] * mm_per_pixel`` (same scale as the width, i.e. square pixels).
            The simulated camera ``resolution`` should have the same aspect ratio as ``reference_img_res``,
            otherwise the renderer derives the height from the width and the resolution aspect.
            """
            return self.reference_img_res[1] * self.mm_per_pixel / 1000.0

    sensor_camera_cfg: SensorCameraCfg = SensorCameraCfg()

    data_types: list[str] = [
        "tactile_rgb",
        "marker_motion",
        "height_map",
        "camera_depth",
        "camera_rgb",
    ]
    """Output of the GelSight Sensor

    - height_map: Generated by the sensor camera. Shape = the camera resolution specified in the SensorCameraCfg.
    - tactile_rgb: Tactile RGB image. Shape = tactile_img_resolution specified in the OpticalSimulator class.
    - marker_motion: Initial and current positions of the gelpad markers.
        Shape (num_envs, 2, num_markers_markers, 2)
        - dim = 1: initial, current markers
        - dim = 2: marker pos (x,y) in the image (x = width, y = height)
    - camera_depth: depth image from the sensor camera
    - camera_rgb: rgb image from the sensor camera
    """

    optical_sim_cfg: GelSightSimulatorCfg = None  # freely choose what simulation class, but can also be left None
    """Cfg class of the optical simulator you want to use."""

    marker_motion_sim_cfg: GelSightSimulatorCfg = None
    """Cfg class of the marker motion simulator you want to use."""

    compute_indentation_depth_class: Literal["optical_sim", "marker_motion_sim", "default"] = "default"
    """Defines which approach is used for computing the indentation depth.

    You can use the method from your optical simulation (e.g. Taxim), or the one from your marker motion simulation (e.g. if its FEM based).
    By default the minimum value of the height map is used as indentation depth.
    """

    device: str = "cuda"
