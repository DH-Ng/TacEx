"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .gelsight_sensor import GelSightSensor
from .gelsight_sensor_cfg import GelSightSensorCfg
from .gelsight_sensor_data import GelSightSensorData

# Register UI extensions.
from .ui_extension_example import *
