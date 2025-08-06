import numpy as np
from dataclasses import dataclass
from typing import Any


@dataclass
class GelSightSensorData:
    """Data container for a GelSight sensor."""

    position: np.ndarray = None
    """Position of the sensor origin in local frame."""
    orientation: np.ndarray = None
    """Quaternion orientation `(w, x, y, z)` of the sensor origin in local frame."""
    intrinsic_matrix: np.ndarray = None
    """The intrinsic matrix for the camera."""
    image_resolution: tuple[int, int] = None
    """A tuple containing (height, width) of the camera sensor."""
    output: dict[str, Any] = None
    """The retrieved sensor data with sensor types as key.

    This is defined inside the corresponding sensor cfg class.
    For GelSight sensors the defaults are "camera_depth", "height_map", "tactile_rgb" and "marker_motion".
    """
