import json
from abc import abstractmethod, ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Generic, TypeVar, Union, overload, Tuple

import numpy as np

from .calibration import CALIB_GELSIGHT

ArrayType = TypeVar("ArrayType")
DeviceType = TypeVar("DeviceType")


@dataclass(frozen=True)
class SimulatorParameters:
    kscale: int  # Initial frame sigma
    frame_mixing_percentage: float
    diff_threshold: int
    contact_scale: float
    pyramid_kernel_size: Tuple[int, ...]
    kernel_size: int
    sigma: float
    shadow_step: float
    height_precision: float
    discretize_precision: float
    fan_angle: float
    fan_precision: float


@dataclass(frozen=True)
class SensorParameters:
    w: int
    h: int
    pixmm: float
    num_bins: int

    @property
    def width(self) -> int:
        return self.w

    @property
    def height(self) -> int:
        return self.h

class TaximImpl(Generic[ArrayType, DeviceType], ABC):
    def __init__(self, backend_name: str, device: DeviceType, calib_folder: Path = CALIB_GELSIGHT,
                 params: Optional[Dict[str, Dict[str, Any]]] = None):
        """

        :param calib_folder: Path to the folder with the calibration files.
        :param params:       Simulator parameters. Values set in this dictionary override values set in params.json
                             in the calib_folder.
        :param device:       Device to use.
        """
        with (calib_folder / "params.json").open() as f:
            default_params = json.load(f)

        params = self.__update_dict_recursive(default_params, params) if params is not None else default_params
        sensor_params = params["sensor"]
        simulator_params = params["simulator"]
        simulator_params["sigma"] = float(simulator_params["sigma"])
        simulator_params["pyramid_kernel_size"] = tuple(simulator_params["pyramid_kernel_size"])

        self._sim_params = SimulatorParameters(**simulator_params)
        self._sensor_params = SensorParameters(**sensor_params)
        self._device = device
        self._backend_name = backend_name

    @overload
    def render(self, height_map: ArrayType, with_shadow: bool = True,
               press_depth: Optional[float] = None, orig_hm_fmt: bool = False) -> ArrayType:
        ...

    @overload
    def render(self, height_map: np.ndarray, with_shadow: bool = True,
               press_depth: Optional[float] = None, orig_hm_fmt: bool = False) -> np.ndarray:
        ...

    def render(self, height_map: Union[ArrayType, np.ndarray], with_shadow: bool = True,
               press_depth: Optional[float] = None, orig_hm_fmt: bool = False) -> Union[ArrayType, np.ndarray]:
        """
        Generates a synthetic tactile image from a height map.
        :param height_map:  Height map to generate tactile image for. The values of this height map correspond to the
                            distance of the object to the highest (furthest from the camera) point of the gel in mm.
                            Hence, a value of 0 means that the corresponding point is at the same height as the highest
                            point of the gel. The smaller the values of the height map are, the closer are the
                            corresponding points to the camera. For a point to be in contact with the gel, it has to
                            have a value smaller or equal to zero.
        :param with_shadow: Whether to generate shadows.
        :param press_depth: How deep the object is pressed into the sensor in mm. The height map will be placed such
                            that its closest point is press_depth closer than the furthest point of the gel height map.
                            If this parameter is left empty, then no interaction between the object and the gel is
                            simulated and the height map is not modified.
        :param orig_hm_fmt: Use the original Taxim height map format. Compared to the default, this format is inverted
                            (higher values mean closer to the camera) and it is shifted by the maximum value of the
                            gel pad height map.
        :return: A HxWx3 array containing the resulting image in RGB format.
        """
        if isinstance(height_map, np.ndarray):
            return self.img_to_numpy(
                self._render_impl(self.convert_height_map(height_map), with_shadow, press_depth, orig_hm_fmt))
        return self._render_impl(height_map, with_shadow, press_depth, orig_hm_fmt)

    def render_direct(self, height_map: ArrayType, with_shadow: bool = True,
                      press_depth: Optional[float] = None, orig_hm_fmt: bool = False) -> ArrayType:
        """
        Version of render that does not convert the output to numpy.
        """
        return self._render_impl(height_map, with_shadow, press_depth, orig_hm_fmt)

    @abstractmethod
    def convert_height_map(self, height_map: np.ndarray) -> Any:
        pass

    @abstractmethod
    def img_to_numpy(self, img: ArrayType) -> np.ndarray:
        pass

    @abstractmethod
    def _render_impl(self, height_map: ArrayType, with_shadow: bool = True,
                     press_depth: Optional[float] = None, orig_hm_fmt: bool = False) -> ArrayType:
        pass

    @classmethod
    def __update_dict_recursive(cls, default: Dict, update: Dict) -> Dict:
        """
        Update the default dict with values from the update dict. The update dict cannot contain keys not present in the
        default dict.
        :param default: Dictionary containing the default values.
        :param update:  Dictionary containing the entries to update.
        :return: Recursively updated dictionary.
        """
        unknown_keys = [k for k in update if k not in default]
        if len(unknown_keys) > 0:
            raise ValueError(f"Unknown key(s): {', '.join(map(str, unknown_keys))}")
        return {
            k:
                cls.__update_dict_recursive(default[k], update[k])
                if isinstance(default[k], dict) and k in update else
                update.get(k, default[k])
            for k in default.keys()
        }

    @property
    def width(self) -> int:
        """

        :return: Width of the output image in pixels.
        """
        return self._sensor_params.width

    @property
    def height(self) -> int:
        """

        :return: Height of the output image in pixels.
        """
        return self._sensor_params.height

    @property
    def sim_params(self) -> SimulatorParameters:
        """

        :return: The simulator parameters.
        """
        return self._sim_params

    @property
    def sensor_params(self) -> SensorParameters:
        """

        :return: The sensor parameters.
        """
        return self._sensor_params

    @property
    def device(self) -> DeviceType:
        """

        :return: The device used by this Taxim instance.
        """
        return self._device

    @property
    def backend_name(self) -> str:
        return self._backend_name
