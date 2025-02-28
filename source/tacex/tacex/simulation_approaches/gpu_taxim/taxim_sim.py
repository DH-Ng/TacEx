from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torchvision.transforms.functional as F

from .sim import Taxim
from ..gelsight_simulator import GelSightSimulator
from ...gelsight_sensor import GelSightSensor

if TYPE_CHECKING:
    from .taxim_sim_cfg import TaximSimulatorCfg

class TaximSimulator(GelSightSimulator):
    """Wraps around the Taxim simulation for the optical simulation of GelSight sensors 
    inside Isaac Sim.
    
    """
    cfg: TaximSimulatorCfg

    def __init__(self, sensor: GelSightSensor, cfg: TaximSimulatorCfg):
        self.sensor = sensor

        super().__init__(sensor=sensor, cfg=cfg)

    def _initialize_impl(self):
        calib_folder = Path(self.cfg.calib_folder_path)

        if self.cfg.device is None:
            # use same device as simulation
            self._device = self.sensor.device
        else: 
            self._device = self.cfg.device
            

        self._taxim = Taxim(calib_folder=calib_folder, device=self._device)

        # todo make size adaptable? I mean with env_ids. This way we would always simulate everythings
        self._indentation_depth = torch.zeros((self._num_envs), device=self._device)
        """Indentation depth, i.e. how deep the object is pressed into the gelpad.
        Values are in mm.

        Indentation depth is equal to the maximum pressing depth of the object in the gelpad.
        It is used for shifting the height map for the Taxim simulation.
        """
    
        # if camera resolution is different than the tactile RGB res, scale img

        self.img_res = self.cfg.tactile_img_res

    def optical_simulation(self):
        height_map = self.sensor._data.output["height_map"]
        # up/downscale height map if camera res different than tactile img res
        if height_map.shape != self.cfg.tactile_img_res:
            height_map = F.resize(height_map, self.cfg.tactile_img_res)

        if self._device == "cpu":
            height_map = height_map.cpu()
        
        #todo only render img where indentation_depth > 0
        tactile_rgb_img = self._taxim.render(
            height_map, 
            with_shadow=self.cfg.with_shadow,
            press_depth=self._indentation_depth,
            orig_hm_fmt=False,
        )
        return tactile_rgb_img

    def compute_indentation_depth(self):
        height_map = self.sensor._data.output["height_map"] / 1000 # convert height map from mm to meter
        min_distance_obj = height_map.amin((1,2))
        # smallest distance between object and sensor case
        dist_obj_sensor_case = min_distance_obj - self.cfg.gelpad_to_camera_min_distance

        # print("dist_obj_sensor_case", dist_obj_sensor_case)
        # if (dist_obj_sensor_case < 0):  # object is "inside the sensor", cause the object is closer to the camera than the edge of the sensor
        #     # print("Object is inside the sensor!!! Gelpad would be broken!!!")
        #     dist_obj_sensor_case = 0
        dist_obj_sensor_case = torch.where(dist_obj_sensor_case < 0, 0, dist_obj_sensor_case)

        self._indentation_depth[:] = torch.where(
            dist_obj_sensor_case <= self.cfg.gelpad_height, 
            (self.cfg.gelpad_height - dist_obj_sensor_case)*1000, 
            0
        )

        return self._indentation_depth
    
    def reset(self):
        self._indentation_depth = torch.zeros((self._num_envs), device=self._device)