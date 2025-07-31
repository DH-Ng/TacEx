import numpy as np
from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union


@dataclass
class GelSightSensorData:
    """Data container for a GelSight sensor."""

    position: np.ndarray = None
    """Position of the sensor origin in local frame."""
    orientation: np.ndarray = None
    """Quaternion orientation `(w, x, y, z)` of the sensor origin in local frame."""
    intrinsic_matrix: np.ndarray = None
    """The intrinsic matrix for the camera."""
    image_resolution: Tuple[int, int] = None
    """A tuple containing (height, width) of the camera sensor."""
    output: Dict[str, Any] = None
    """The retrieved sensor data with sensor types as key.

    This is definied inside the correspondig sensor cfg class.
    For GelSight sensors the defaults are "camera_depth", "height_map", "tactile_rgb" and "marker_motion".
    """
